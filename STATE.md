# Research State

- Current phase: `None`
- Pipeline completed: `True`

## Previous phases

resource_finder (succeeded), experiment_runner (succeeded)

## Current phase context

- Phase: `experiment_runner`
- Status: `completed`
- Started: `2026-10-18T22:49:31.744709Z`
- Next steps:
  - Validate the report and experimental artifacts before finalizing.

## Workspace check

- Root: `/data/smallyan/neurico/random_ideas/forecasting-delayed-unsafe-behavior-5d09`
- Directory usable: `True`

## Output validation

- Valid: `True`
- Expected: `REPORT.md`
- Missing: None
- Outside workspace: None

## Agent notes

<!-- NEURICO_AGENT_NOTES_START -->
### resource_finder
<!-- NEURICO_AGENT_NOTES_START:resource_finder -->
**Phase:** resource_finder. COMPLETE (2026-10-18).

**Done:**
- 24 papers (`papers/`; LatentQA deep-read in full; AO + forecasting papers deep-read by sub-agents → `papers/notes_*.md`).
- 12 datasets in `datasets/` (HF + GitHub CSVs), plus a construction recipe.
- 5 repos in `code/`, including the user-specified LatentQA, which was smoke-tested on GPU: `scripts/latentqa_smoke_test.py`, output in `logs/latentqa_smoke_test.json`.
- `literature_review.md`, `resources.md`, `planning.md` (direction ranking), `code/README.md`, `datasets/README.md`.
- venv `.venv`: torch 2.7.1+cu126, transformers 4.55.2, peft 0.17.1. No flash_attn, so use sdpa.

**Key findings / decisions:**
- No paper forecasts harm 64–512 tokens ahead, and no dataset has such onsets. The experiment runner must CONSTRUCT a delayed-harm set (recipe: `datasets/README.md`), with matched benign/refusal endings to kill the topic shortcut.
- Target model: **Qwen3-8B** (ungated; official AO LoRA `adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B` and `ceselder/qwen3-8b-ao-v3-best`).
- meta-llama/* and gemma return 403. The LatentQA decoder runs on the `NousResearch/Meta-Llama-3-8B-Instruct` mirror (downloaded).
- Building-Better-AOs reports that AOs only tie linear probes, and that a text-only control is similar. Score verbalizers by Yes/No logit-margin AUROC, using ≥20-token activation windows at layers 18/22.
- LatentQA smoke test: the yes/no "will it be harmful later?" question is always answered "safe", even for an explicit harmful request. The open-ended "what will the assistant do later?" question does reveal harmful plans.
- LatentQA upstream bugs (tokenize rename, flash_attn, Encoding.shape) have workarounds in `code/README.md`.

**Direction budget (top 3 kept; see planning.md):**
- D1: lead-distance benchmark + Part 1 baselines.
- D2: Part 2, matched supervision (direct AO vs forecast-vector model vs probes; plus timing head).
- D3: generalization to Qwen3GuardTest `thinking_loc` and held-out behaviors (JBB/StrongREJECT).
- Pruned: MC soft targets (compute), LIT-from-scratch (redundant), steering, multi-turn, FineHarm (gated), scale study.

**Next (experiment_runner):**
1. Generate the delayed-harm data on Qwen3-8B. Split by behavior.
2. Cache layer-18/22 activations every 16 tokens before onset.
3. Run the D1 baselines and decay curves.
4. Train the D2 models with identical budgets.
5. Run the D3 transfer evaluation.

**Risks:**
- Only GPU 2 was free (the others had ~43 GB used).
- /data had ~123 GB free.
- FineHarm and HEx-PHI need the account owner to accept terms (not done).
- Spliced text is off-policy, so include an on-policy prefill subset.
<!-- NEURICO_AGENT_NOTES_END:resource_finder -->

### experiment_runner
<!-- NEURICO_AGENT_NOTES_START:experiment_runner -->
**Phase:** experiment_runner. COMPLETE (2026-10-18). Two sessions.
**Session 1:** families/split, generation, annotation, activations, audits, labels, E1 baselines, E2 (3 seeds), controls. Left without REPORT/figures.
**Session 2:**
- Added off-the-shelf AO zero-shot (`src/ao_zeroshot.py` → `results/ao_scores.npy`).
- Reran E2 main pair, saving OOF preds (`results/results_e2_preds.json`, `preds_*.npz`; replicates session-1 ordering).
- Added gold-readout ceiling (`results/results_e2_gold.json`).
- Paired family bootstrap and data stats (`src/analysis.py` → `results/analysis.json`).
- Figures (`figures/`), tables (`results/tables.md`), REPORT.md, README.md.
**Key findings:**
- History probe AUROC 0.79 (H=64) → 0.90 (H=512); 56% early alerts ≥128 tokens at 5% FAR.
- Off-the-shelf AO 0.54 → 0.76, biased to "No".
- Forecast architecture ≤ matched direct reader (Δ −0.02 to −0.04, significant at H ≥ 256). Supervision, not architecture.
- Audits pass; 0 censored horizons.
**Deviations / unresolved:**
- No human annotation check (LLM–LLM κ = 0.69).
- No repeated-rollout calibration, Track A, RQ4 or RQ6.
- Part 2 uses small attention QA heads, not an LLM decoder.
- peft upgrade pulled torch 2.14 (cu130) into `.venv`.
**Next:** see REPORT.md §7.
<!-- NEURICO_AGENT_NOTES_END:experiment_runner -->

<!-- NEURICO_AGENT_NOTES_END -->
