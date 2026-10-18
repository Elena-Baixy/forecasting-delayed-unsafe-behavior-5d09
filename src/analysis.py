"""Post-hoc analysis: off-the-shelf AO baseline metrics, paired family-bootstrap tests for E2,
readout-ceiling summary, data statistics and figures. Writes results/analysis.json + figures/*.png.
"""
import json, glob, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold

H = [64, 128, 256, 512]
df = pd.read_csv("results/prefix_labels.csv")
OUT = {}


def boot(y, p, g, B=1000, fn=roc_auc_score):
    ug = np.unique(g); gi = {u: np.where(g == u)[0] for u in ug}; rng = np.random.default_rng(0); v = []
    for _ in range(B):
        idx = np.concatenate([gi[u] for u in rng.choice(ug, len(ug))])
        if y[idx].min() != y[idx].max(): v.append(fn(y[idx], p[idx]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def metrics(y, p, g):
    m = ~np.isnan(y); y, p, g = y[m], p[m], g[m]; pc = np.clip(p, 1e-4, 1 - 1e-4)
    return dict(n=int(len(y)), pos=int(y.sum()), auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
                brier=float(brier_score_loss(y, pc)), logloss=float(log_loss(y, pc, labels=[0, 1])), auroc_ci=boot(y, p, g))


def cv_calibrate(sub, s, col):
    """1-D logistic calibration of a raw score, 5-fold family-grouped CV (same as self-report baseline)."""
    out = np.full(len(sub), np.nan); y = sub[col].values
    for a, b in GroupKFold(5).split(sub, groups=sub.family_id):
        m = ~np.isnan(y[a])
        out[b] = LogisticRegression(C=1.0).fit(s[a][m][:, None], y[a][m]).predict_proba(s[b][:, None])[:, 1]
    return out


def early_warning(sub, score, seed=0):
    """Same protocol as experiments.early_warning: thr = 95th pct of max-score on benign val families."""
    fams = sub.family_id.unique(); rng = np.random.default_rng(seed); vfam = set(rng.choice(fams, int(.15 * len(fams)), replace=False))
    d = sub.assign(s=score).sort_values("t")
    tr = d.groupby("family_id").agg(onset=("onset", "first"), ts=("t", list), ss=("s", list))
    va, te = tr[tr.index.isin(vfam)], tr[~tr.index.isin(vfam)]
    thr = np.quantile([max(x) for x in va[va.onset.isna()].ss], .95)
    uns, ben = te[te.onset.notna()], te[te.onset.isna()]
    r = dict(test_far=float(np.mean([max(x) > thr for x in ben.ss])), n_unsafe=len(uns), n_benign=len(ben))
    for L in (64, 128, 256):
        r[f"alert_ge{L}"] = float(np.mean([any(s > thr for s in row.ss) and row.onset - next(t for t, s in zip(row.ts, row.ss) if s > thr) >= L for _, row in uns.iterrows()]))
    return r


# ---------------------------------------------------------------- off-the-shelf AO
try:
    AO = np.load("results/ao_scores.npy"); cfg = json.load(open("results/ao_config.json"))
    for setting in ["assistant", "story"]:
        sub = df[(df.setting == setting) & df.safe_prefix].reset_index(drop=True)
        res = {}
        for li, L in enumerate(cfg["layers"]):
            for qi in range(len(cfg["questions"])):
                s = AO[sub.row.values, li, qi]; key = f"AO_L{L}_q{qi}"
                res[key] = {h: dict(raw_auroc=float(roc_auc_score(*[(v := sub[f"y{h}"].values)[~np.isnan(v)], s[~np.isnan(v)]])),
                                    frac_yes=float((s > 0).mean()), **metrics(sub[f"y{h}"].values, cv_calibrate(sub, s, f"y{h}"), sub.family_id.values))
                            for h in H}
                if setting == "assistant": res[key]["early_warning"] = early_warning(sub, s)
        OUT[f"AO_zeroshot_{setting}"] = res
        print(setting, {k: [round(v[h]["auroc"], 3) for h in H] for k, v in res.items()})
except FileNotFoundError:
    print("no AO scores")

# ---------------------------------------------------------------- paired tests direct vs forecast (seed-averaged OOF preds)
A = df[(df.setting == "assistant") & df.safe_prefix].reset_index(drop=True).set_index("row")
P = {}
for fn in glob.glob("results/preds_*.npz"):
    z = np.load(fn); name = fn.split("preds_")[1].rsplit("_seed", 1)[0]
    P.setdefault(name, []).append({k: pd.Series(z[k], index=z["row"]) for k in z.files if k != "row"})
if "direct" in P and "forecast" in P:
    def avg(name, key): return pd.concat([d[key] for d in P[name]], axis=1).mean(1)
    rng = np.random.default_rng(1); fam = A.family_id; ug = fam.unique(); gi = {u: np.where(fam.values == u)[0] for u in ug}
    comp = {}
    for h in H:
        col = f"y{h}"; y = A[col].values
        pairs = {"direct_qa": avg("direct", f"qa:{col}"), "forecast_qa": avg("forecast", f"qa:{col}"), "forecast_head": avg("forecast", f"head:{col}")}
        pr = {k: v.reindex(A.index).values for k, v in pairs.items()}
        base = {k: roc_auc_score(y, v) for k, v in pr.items()}
        diffs = {"forecast_head-direct_qa": [], "forecast_qa-direct_qa": []}
        for _ in range(1000):
            idx = np.concatenate([gi[u] for u in rng.choice(ug, len(ug))])
            if y[idx].min() == y[idx].max(): continue
            a = {k: roc_auc_score(y[idx], v[idx]) for k, v in pr.items()}
            diffs["forecast_head-direct_qa"].append(a["forecast_head"] - a["direct_qa"]); diffs["forecast_qa-direct_qa"].append(a["forecast_qa"] - a["direct_qa"])
        comp[h] = dict(auroc_seedavg=base, **{k: dict(delta=float(np.mean(v)), ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                                                     p_two_sided=float(2 * min(np.mean(np.array(v) <= 0), np.mean(np.array(v) >= 0)))) for k, v in diffs.items()})
    OUT["paired_direct_vs_forecast"] = comp
    print(json.dumps(comp, indent=1))

# ---------------------------------------------------------------- data stats
g = df.drop_duplicates("family_id")
OUT["data"] = dict(
    families=int(g.family_id.nunique()), by_source=g.source.value_counts().to_dict(),
    split={f"{a}/{b}": int(v) for (a, b), v in g.groupby(["source", "split"]).size().items()},
    event_rate_by_source=g.groupby("source").onset.apply(lambda s: float(((s.notna()) & (s >= 0)).mean())).to_dict(),
    onset_quantiles_assistant={str(k): float(v) for k, v in g[(g.setting == "assistant") & (g.onset >= 0)].onset.quantile([.1, .25, .5, .75, .9]).items()},
    finish=g.finish.value_counts().to_dict(),
    censored_safe_prefix={h: int(df[df.safe_prefix][f"y{h}"].isna().sum()) for h in H},
    already_unsafe_prefixes=int(df.already_unsafe.sum()))
json.dump(OUT, open("results/analysis.json", "w"), indent=1, default=str)

# ---------------------------------------------------------------- figures
E1 = json.load(open("results/results_e1.json"))["E1_assistant_cv"]
E2 = {}
for fn in ["results/results_e2_preds.json", "results/results_e2_controls.json", "results/results_e2_gold.json"]:
    try: E2.update(json.load(open(fn)))
    except FileNotFoundError: pass
fig, ax = plt.subplots(figsize=(7.5, 4.5)); x = np.arange(4)
show = {"prevalence": "prevalence", "token_position": "token position", "text_prefix": "text (prompt+prefix TF-IDF)",
        "linear_probe_all3": "linear probe (L9/18/27)", "linear_probe_L18+history16": "linear probe + 16-tok history", "self_report_qwen3": "Qwen3 self-report"}
for k, lab in show.items():
    ax.plot(x, [E1[str(h)][k]["auroc"] for h in H], marker="o", label=lab)
if "AO_zeroshot_assistant" in OUT:
    best = max(OUT["AO_zeroshot_assistant"], key=lambda k: np.mean([OUT["AO_zeroshot_assistant"][k][h]["auroc"] for h in H]))
    ax.plot(x, [OUT["AO_zeroshot_assistant"][best][h]["auroc"] for h in H], marker="s", ls="--", label=f"off-the-shelf AO ({best})")
for nm, lab, src in [("direct", "trained direct AO-style reader", "qa"), ("forecast", "forecast repr. (timing head)", "head")]:
    rs = [v for k, v in E2.items() if k.split("|")[0] == nm]
    ax.errorbar(x, [np.mean([r[f"{src}:y{h}"]["auroc"] for r in rs]) for h in H], yerr=[np.std([r[f"{src}:y{h}"]["auroc"] for r in rs]) for h in H], marker="^", lw=2, capsize=3, label=lab)
ax.axhline(.5, color="gray", ls=":"); ax.set_xticks(x); ax.set_xticklabels(H); ax.set_xlabel("horizon H (tokens after the safe cutoff)")
ax.set_ylabel("AUROC, 5-fold family CV"); ax.set_title("Assistant safety: forecasting first unsafe onset within H tokens"); ax.legend(fontsize=7); fig.tight_layout()
fig.savefig("figures/forecast_auroc_by_horizon_all.png", dpi=150); plt.close(fig)

EWB = json.load(open("results/results_early_warning_baselines.json"))
rows = [(show.get(k, k), v) for k, v in EWB.items() if k in show]
for nm, lab in [("direct", "trained direct reader"), ("forecast", "forecast repr.")]:
    rs = [v["early_warning_y512"] for k, v in E2.items() if k.split("|")[0] == nm]
    rows.append((lab, {f"alert_ge{L}": np.mean([r[f"alert_ge{L}"] for r in rs]) for L in (64, 128, 256)}))
if "AO_zeroshot_assistant" in OUT: rows.append(("off-the-shelf AO", OUT["AO_zeroshot_assistant"][best]["early_warning"]))
fig, ax = plt.subplots(figsize=(8, 4)); w = .27
for i, L in enumerate((64, 128, 256)):
    ax.bar(np.arange(len(rows)) + (i - 1) * w, [r[f"alert_ge{L}"] for _, r in rows], w, label=f"alert ≥{L} tokens before onset")
ax.set_xticks(range(len(rows))); ax.set_xticklabels([r[0] for r in rows], rotation=30, ha="right", fontsize=7)
ax.set_ylabel("fraction of unsafe trajectories"); ax.set_title("Early warning at ~5% benign-trajectory false-alarm rate"); ax.legend(fontsize=7); fig.tight_layout()
fig.savefig("figures/early_warning.png", dpi=150); plt.close(fig)
print("done")
