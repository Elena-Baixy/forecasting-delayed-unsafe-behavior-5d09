"""Figures + markdown tables from results/*.json."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = ["64", "128", "256", "512"]
E1 = json.load(open("results/results_e1.json"))
lines = []
for key, title in [("E1_assistant_cv", "Assistant safety (5-fold family CV)"), ("E1_story_cv", "Story: first fictional-harm event (5-fold CV)")]:
    R = E1.get(key)
    if not isinstance(R, dict): continue
    meths = list(R["64"].keys())
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for m in meths:
        y = [R[h][m].get("auroc", np.nan) for h in H]
        lo = [R[h][m].get("auroc_ci", [np.nan, np.nan])[0] for h in H]; hi = [R[h][m].get("auroc_ci", [np.nan, np.nan])[1] for h in H]
        ax.errorbar(range(4), y, yerr=[np.array(y) - lo, np.array(hi) - y], label=m, marker="o", capsize=2, alpha=.85)
    ax.set_xticks(range(4)); ax.set_xticklabels(H); ax.set_xlabel("horizon H (tokens after cutoff)"); ax.set_ylabel("AUROC (safe prefixes)")
    ax.axhline(.5, color="gray", ls=":"); ax.set_title(title); ax.legend(fontsize=6.5, ncol=2); fig.tight_layout()
    fig.savefig(f"figures/{key}_auroc_by_horizon.png", dpi=150); plt.close(fig)
    lines.append(f"\n### {title}\n\n| method | " + " | ".join(f"H={h} AUROC [95% CI]" for h in H) + " | H=512 AUPRC | H=512 Brier |\n|---|" + "---|" * 6)
    for m in meths:
        cells = []
        for h in H:
            r = R[h][m]; ci = r.get("auroc_ci", [np.nan, np.nan])
            cells.append(f"{r.get('auroc', np.nan):.3f} [{ci[0]:.2f},{ci[1]:.2f}]")
        r = R["512"][m]
        lines.append(f"| {m} | " + " | ".join(cells) + f" | {r.get('auprc', np.nan):.3f} | {r.get('brier', np.nan):.3f} |")
    lines.append("\nprevalence / n / positive families per H: " + "; ".join(
        f"H={h}: {R[h][meths[0]]['prev']:.3f}, n={R[h][meths[0]]['n']}, pos fam={R[h][meths[0]]['pos_fam']}/{R[h][meths[0]]['fam']}" for h in H))

try:
    import glob
    E2 = {}
    for fn in ["results/results_e2_preds.json", "results/results_e2_controls.json", "results/results_e2_gold.json"]: E2.update(json.load(open(fn)))
    groups = {}
    for k, v in E2.items(): groups.setdefault(k.split("|")[0], []).append(v)
    cols = ["qa:y64", "qa:y128", "qa:y256", "qa:y512", "head:y128", "head:y512", "qa:harmrec512", "qa:topic128", "qa:refusal_later"]
    lines.append("\n### E2: forecast representation vs matched direct reader (assistant, 5-fold CV; mean ± sd over seeds)\n\n| model | params | " + " | ".join(cols) + " | EW alert≥128 | EW alert≥64 | test FAR |\n|---|---|" + "---|" * (len(cols) + 3))
    for g, rs in groups.items():
        cells = []
        for c in cols:
            v = [r[c]["auroc"] for r in rs if c in r and "auroc" in r[c]]
            cells.append(f"{np.mean(v):.3f}±{np.std(v):.3f}" if v else "–")
        ew = [r["early_warning_y512"] for r in rs]
        cells += [f"{np.nanmean([e['alert_ge128'] for e in ew]):.2f}", f"{np.nanmean([e['alert_ge64'] for e in ew]):.2f}", f"{np.nanmean([e['test_far'] for e in ew]):.2f}"]
        lines.append(f"| {g} | {rs[0]['params']/1e6:.2f}M | " + " | ".join(cells) + " |")
except FileNotFoundError:
    pass
try:
    EW = json.load(open("results/results_early_warning_baselines.json"))
    lines.append("\n### Early warning (H=512 score, threshold = 5% trajectory FAR on held-out-fold val families)\n\n| method | alert ≥64 before onset | ≥128 | ≥256 | missed | test FAR | #unsafe traj | #benign traj |\n|---|---|---|---|---|---|---|---|")
    for m, e in EW.items():
        lines.append(f"| {m} | {e['alert_ge64']:.2f} | {e['alert_ge128']:.2f} | {e['alert_ge256']:.2f} | {e['missed']:.2f} | {e['test_far']:.2f} | {e['n_unsafe']} | {e['n_benign']} |")
except FileNotFoundError:
    pass
open("results/tables.md", "w").write("\n".join(lines))
print("\n".join(lines))
