"""Derive prefix-level labels programmatically from timelines (censoring-aware) + split audit.

For a generation with onset o (token index of first unsafe token; None = no event), length n, finish f,
and a prefix of t response tokens:
  safe prefix      : o is None or o >= t       (only these are used for forecasting)
  y_H = 1          : o is not None and o < t+H
  y_H = 0          : (o is not None and o >= t+H) or (o is None and (f == 'stop' or t+H <= n))
  censored (NaN)   : otherwise (generation hit the cap/failed before t+H without an event)
Cumulative risk is monotone in H by construction (y_H <= y_H' for H < H').
Timing bin (first-event window relative to t): 0:[1,64] 1:[65,128] 2:[129,256] 3:[257,512] 4:none within 512.
"""
import json, numpy as np, pandas as pd

H = [64, 128, 256, 512]
EDGES = [0, 64, 128, 256, 512]


def ylab(o, t, h, n, f):
    if o is not None and o < t + h: return 1.0
    if o is not None: return 0.0
    if f == "stop" or t + h <= n: return 0.0
    return np.nan


def build(gen_path="results/generations.jsonl", ann_path="results/annotations.jsonl", idx_path="results/acts_index.json"):
    fams = {f["family_id"]: f for f in json.load(open("results/families.json"))["families"]}
    gens = {r["family_id"]: r for r in map(json.loads, open(gen_path))}
    anns = {a["family_id"]: a for a in map(json.loads, open(ann_path))}
    idx = json.load(open(idx_path))
    rows = []
    for i, d in enumerate(idx):
        fid, t = d["family_id"], d["t"]
        g, a, fm = gens[fid], anns[fid], fams[fid]
        n, f = len(g["response_ids"]), g["finish_reason"]
        A = a["ann"]
        if "error" in A: continue   # annotation failed -> exclude (reported)
        story = fm["setting"] == "story"
        # primary event: assistant-safety violation (assistant setting) / first fictional-harm description (story)
        o = a.get("fiction_onset_tok") if story else a.get("onset_tok")
        if not story and A.get("assistant_unsafe") and o is None: o = -1  # unsafe but span unmatched -> drop
        r = dict(row=i, family_id=fid, t=t, split=fm["split"], source=fm["source"], setting=fm["setting"], n=n,
                 finish=f, onset=o, event_type=A.get("event_type"), uncertainty=A.get("uncertainty"))
        r["safe_prefix"] = (o is None) or (o >= t)
        r["already_unsafe"] = (o is not None) and (0 <= o < t)
        for h in H: r[f"y{h}"] = ylab(o if (o is None or o >= 0) else None, t, h, n, f) if o != -1 else np.nan
        # timing bin + censoring bin (first bin that is not observed)
        if o is not None and o >= t:
            d_ = o - t
            r["bin"] = next((k for k in range(4) if d_ < EDGES[k + 1]), 4)
        else:
            r["bin"] = 4
        r["cens_from"] = next((k for k, h in enumerate(H) if np.isnan(r[f"y{h}"])), 5)  # bins >= this are unobserved
        # behaviour-type labels within 512
        r["harmrec512"] = float(r["y512"] == 1 and A.get("event_type") == "harmful_recommendation") if not np.isnan(r["y512"]) else np.nan
        r["toxic512"] = float(r["y512"] == 1 and A.get("event_type") == "toxic_language") if not np.isnan(r["y512"]) else np.nan
        ro = a.get("refusal_onset_tok")
        r["refusal_later"] = float(ro is not None and t <= ro < t + 512)
        r["topic128"] = float(any(t < tt <= t + 128 for tt, _ in a.get("topic_toks", [])))
        rows.append(r)
    df = pd.DataFrame(rows)
    # split audit: no family in more than one split; splits fixed pre-generation in families.json
    assert df.groupby("family_id").split.nunique().max() == 1
    # monotonicity audit
    for a_, b_ in zip(H[:-1], H[1:]):
        m = df[[f"y{a_}", f"y{b_}"]].dropna()
        assert (m[f"y{a_}"] <= m[f"y{b_}"]).all()
    return df


if __name__ == "__main__":
    df = build()
    df.to_csv("results/prefix_labels.csv", index=False)
    s = df[df.safe_prefix]
    print(df.groupby(["setting", "split"]).size())
    print("safe-prefix positives per horizon:", {h: (int(s[f'y{h}'].sum()), int(s[f'y{h}'].notna().sum())) for h in H})
    print("censored:", {h: int(s[f'y{h}'].isna().sum()) for h in H})
