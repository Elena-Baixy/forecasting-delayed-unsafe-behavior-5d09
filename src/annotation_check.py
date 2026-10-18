"""Stratified re-annotation check (proxy for the requested human check, which was not possible in an
automated session): 200 generations (all flagged-unsafe + all story-event + random others), re-annotated
by a stronger, different model (gpt-5.6-terra) with the same rubric; report agreement on the primary
event flag and onset-token differences."""
import json, random, numpy as np
import concurrent.futures as cf
import annotate as an

an.ANNOTATOR = "gpt-5.6-terra"
random.seed(0)
fams = {f["family_id"]: f for f in json.load(open("results/families.json"))["families"]}
gens = {r["family_id"]: r for r in map(json.loads, open("results/generations.jsonl"))}
anns = {a["family_id"]: a for a in map(json.loads, open("results/annotations.jsonl"))}
flag = [k for k, a in anns.items() if a["ann"].get("assistant_unsafe")]
other = [k for k in anns if k not in flag]
sample = flag + random.sample(other, 200 - len(flag))
with cf.ThreadPoolExecutor(64) as ex:
    res = list(ex.map(lambda k: an.process(gens[k], fams[k]["prompt"]), sample))
rows = []
for k, b in zip(sample, res):
    a = anns[k]
    rows.append(dict(family_id=k, a_unsafe=bool(a["ann"].get("assistant_unsafe")), b_unsafe=bool(b["ann"].get("assistant_unsafe")),
                     a_on=a.get("onset_tok"), b_on=b.get("onset_tok"),
                     a_fic=bool(a["ann"].get("fictional_harm")), b_fic=bool(b["ann"].get("fictional_harm"))))
au, bu = np.array([r["a_unsafe"] for r in rows]), np.array([r["b_unsafe"] for r in rows])
both = [abs(r["a_on"] - r["b_on"]) for r in rows if r["a_on"] is not None and r["b_on"] is not None]
from sklearn.metrics import cohen_kappa_score
summ = dict(n=len(rows), n_flagged_primary=int(au.sum()), agree=float((au == bu).mean()), kappa=float(cohen_kappa_score(au, bu)),
            primary_flags_confirmed=float(bu[au].mean()), second_flags_not_in_primary=int((bu & ~au).sum()),
            onset_absdiff_median=float(np.median(both)) if both else None, onset_within_32=float(np.mean(np.array(both) <= 32)) if both else None,
            fiction_agree=float(np.mean([r["a_fic"] == r["b_fic"] for r in rows])))
json.dump(dict(summary=summ, rows=rows), open("results/annotation_check.json", "w"), indent=1)
print(summ)
