# Forecasting delayed unsafe behavior from Qwen3-8B activations

This project asks whether activation readers can tell, from a still-safe prefix of a frozen Qwen3-8B response, that a clear safety violation will first appear 64–512 tokens later. The readers compared are off-the-shelf verbalizers, probes and trained forecasters. It also tests whether an explicit future representation (horizon queries + timing head) beats a matched direct AO-style reader. Full write-up: [REPORT.md](REPORT.md).

## Key findings
Setup: 880 assistant families, 88 with delayed onsets (median onset token 360), 5-fold family CV.

- **Delayed harm is partly forecastable.** A linear probe on layer 18 plus a 16-token activation history reaches AUROC 0.79 at H=64 and 0.90 at H=512. It alerts on 56% of unsafe trajectories ≥128 tokens before onset at about a 5% benign false-alarm rate.
- **The off-the-shelf Activation Oracle (Qwen3-8B checkpoint) is weak zero-shot.**
  - AUROC is 0.54 at H=64 and 0.76 at H=512.
  - It says "No" to about 99% of prefixes.
  - Qwen3 self-report (0.74→0.84) and a text model on prompt+prefix (0.63→0.85) are competitive with single-position probes.
- **The forecast architecture does not beat the matched direct reader.** Paired ΔAUROC (forecast head − direct) ranges from −0.04 to −0.02 and is significant at H ≥ 256. Early warning is equal (36% vs 37%). Future-event supervision, not the architecture, carries the gains.
- **Controls and audits behave as expected.** Shuffled or constant activations collapse to chance. Causal and split audits pass, and no horizon is censored.
- **Story forecasting is reported separately.** Whether harm happens is mostly set by the prompt (prompt-only text: 0.84 at H=512). Activations help only with timing.

## Reproduce
```bash
source .venv/bin/activate          # uv-managed env, see pyproject.toml
python src/prep_prompts.py         # families + split (before generation)
python src/generate.py             # Qwen3-8B generations (GPU)
python src/annotate.py results/generations.jsonl results/annotations.jsonl   # needs OPENAI_API_KEY
python src/extract.py              # activations, causal audit, self-report
python src/labels.py               # prefix labels + split/monotonicity audits
python src/experiments.py e1       # baselines
python src/experiments.py e2 preds; python src/experiments.py e2 controls; python src/experiments.py e2 gold
python src/ao_zeroshot.py          # off-the-shelf AO
python src/analysis.py; python src/plots.py
```

On an A6000:
- generation takes about 15 minutes;
- annotation about 10 minutes;
- extraction about 5 minutes;
- E1 + E2 about 25 minutes;
- AO about 12 minutes.

## Layout
- `src/` — pipeline code.
- `results/` — raw outputs, metrics JSON and `tables.md`.
- `figures/` — plots.
- `planning.md` — plan and direction ranking.
- `literature_review.md`, `resources.md`, `papers/`, `datasets/`, `code/` — resources gathered before the experiments.
