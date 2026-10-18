# Research State

- Current phase: `None`
- Pipeline completed: `False`

## Previous phases

resource_finder (succeeded), experiment_runner (failed)

## Current phase context

- Phase: `experiment_runner`
- Status: `failed`
- Started: `2026-10-18T21:00:34.648323Z`
- Next steps:
  - Validate the report and experimental artifacts before finalizing.

## Workspace check

- Root: `/data/smallyan/neurico/random_ideas/forecasting-delayed-unsafe-behavior-5d09`
- Directory usable: `True`

## Output validation

- Valid: `False`
- Expected: `REPORT.md`
- Missing: `REPORT.md`
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
**Phase:** experiment_runner. All phases attempted (2026-10-18, ~1 h budget).
**Done:**
- 1000 prompt families, split 80/10/10 before generation (`results/families.json`).
- Fresh Qwen3-8B generations (T=0.7, top_p=0.8, top_k=20, max 768, non-thinking): `results/generations.jsonl`.
- gpt-5.6-luna timelines (`results/annotations.jsonl`); 12 were blocked by the OpenAI bio filter and excluded.
- Activations at layers 9/18/27, cutoffs {32, 64, 128, 192, 256} (`results/acts*.npy`). Causal audit: `results/causal_audit.json`.
- Labels: `results/prefix_labels.csv`. Split and monotonicity audits pass; 0 censored horizons.
- Annotation cross-check (gpt-5.6-terra, n=200): kappa 0.69 (`results/annotation_check.json`).
- E1 baselines: `results/results_e1.json`. E2 runs: `results/results_e2*.json`. Figures and tables in `figures/` and `results/tables.md`. Write-up in REPORT.md.
**Key decisions / deviations:**
- Part 2 is a lightweight analog: small matched heads on cached activations, not an LLM decoder with LoRA.
- Off-the-shelf AO/LatentQA, the repeated-rollout calibration, Track A and RQ4/RQ6 were not run (time).
- No human check; it was replaced by a second-LLM check.
- transformers was upgraded 4.55 → 5.20 by the sentence-transformers install after generation. Extraction ran on 5.20.
**Next:** see REPORT.md "Next steps". Scale positives (more samples per harmful family), run AO zero-shot, add repeated-rollout calibration.
<!-- NEURICO_AGENT_NOTES_END:experiment_runner -->

<!-- NEURICO_AGENT_NOTES_END -->
