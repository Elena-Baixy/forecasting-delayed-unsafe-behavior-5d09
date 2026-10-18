# Resources Catalog

## Summary
Resources for "Forecasting delayed unsafe behavior from model activations with verbalizers":
- 24 papers
- 12 local datasets (9 via HF, 3 via GitHub CSV); 2 more identified but gated
- 5 code repositories, including the user-specified LatentQA, which was smoke-tested on GPU
- A ready venv (`.venv`, torch 2.7.1+cu126, transformers 4.55.2, peft 0.17.1)

Direction ranking and pruning: `planning.md`.

## Papers (24), see papers/README.md
| Title | Year | File | Key info |
|---|---|---|---|
| LatentQA (user-specified) | 2024/ICLR26 | papers/2412.08686_latentqa.pdf | Decoder reads layer-15 acts → future-completion QA; released Llama-3-8B decoder |
| Activation Oracles | 2025 | papers/2512.15674_activation_oracles.pdf | AO baseline; Qwen3-8B ckpts; additive layer-1 injection |
| Building Better AOs | 2026 | papers/2606.02609_building_better_activation_oracles.pdf | AOs tie probes; AUROC on logit margin; layer 22; AObench |
| AO blind spots | 2026 | papers/2607.23379_ao_blind_spots.pdf | Fine-tuned AOs miss concepts that probes catch |
| AO calibration | 2026 | papers/2605.26045_ao_calibration.pdf | Likelihood scoring AUROC 0.92–0.96 |
| Faithful Activation Verbalization | 2026 | papers/2609.34033_faithful_activation_verbalization.pdf | AO hallucination on harmful intent |
| NLA token selection | 2026 | papers/2609.37040_nla_token_selection.pdf | Which positions to read |
| Patchscopes | 2024 | papers/2401.06102_patchscopes.pdf | Training-free baseline |
| SelfIE | 2024 | papers/2403.10949_selfie.pdf | Training-free baseline |
| Future Lens | 2023 | papers/2311.04897_future_lens.pdf | Tokens N ahead; learned soft-prompt decoder |
| Emergent Response Planning | 2025 | papers/2502.06258_emergent_response_planning.pdf | Prompt probes predict response attributes; self-report is weak |
| Do LMs Plan Ahead | 2024 | papers/2404.00859_do_lms_plan_ahead.pdf | Breadcrumbs vs pre-caching |
| Remaining Output Length | 2026 | papers/2607.05316_remaining_output_length.pdf | Time-to-event probe template |
| Streaming Content Monitoring (FineHarm) | 2025 | papers/2506.09996_streaming_content_monitoring.pdf | Token-level harm data, partial detection |
| Predict, Don't React (StreamGuard) | 2026 | papers/2604.03962_predict_dont_react_safety_forecasting.pdf | MC future-risk targets; OnTime/Late/Miss |
| Shallow Safety Alignment | 2024 | papers/2406.05946_shallow_safety_alignment.pdf | Prefill → delayed harm |
| Constitutional Classifiers | 2025 | papers/2501.18837_constitutional_classifiers.pdf | Streaming classifiers |
| Multi-turn activation drift | 2026 | papers/2605.15455_multiturn_activation_drift.pdf | Activation risk surfacing |
| Refusal direction | 2024 | papers/2406.11717_refusal_direction.pdf | 1-D baseline |
| Representation Engineering | 2023 | papers/2310.01405_representation_engineering.pdf | Readout baselines |
| Circuit Breakers | 2024 | papers/2406.04313_circuit_breakers.pdf | Harm representations; dataset |
| Looking Inward | 2024 | papers/2410.13787_looking_inward_introspection.pdf | Self-report baseline |
| HarmBench | 2024 | papers/2402.04249_harmbench.pdf | Behaviors + classifier |
| BeaverTails | 2023 | papers/2307.04657_beavertails.pdf | Harm QA labels |

## Datasets, see datasets/README.md (download + construction recipe)
| Name | Source | Size | Task | Location |
|---|---|---|---|---|
| JBB-Behaviors | HF JailbreakBench/JBB-Behaviors | 100 harmful + 100 benign | held-out test prompts | datasets/jbb_behaviors/ |
| HarmBench | GitHub CSV | 400 behaviors | harmful prompts | datasets/harmbench/ |
| AdvBench | GitHub CSV | 520 | harmful prompts + targets | datasets/advbench/ |
| StrongREJECT | GitHub CSV | 313 | held-out harmful prompts | datasets/strongreject/ |
| harmful_behaviors | HF mlabonne | 520 | harmful prompts | datasets/harmful_behaviors/ |
| harmless_alpaca | HF mlabonne | 31k | benign prompts | datasets/harmless_alpaca/ |
| Dolly-15k | HF databricks | 15k | benign prompts/responses | datasets/dolly15k/ |
| LLM-LAT harmful | HF LLM-LAT/harmful-dataset | 4,948 | harmful continuations + refusals | datasets/llm_lat_harmful/ |
| Circuit-breakers | HF abhayesian/circuit-breakers-dataset | 4,476 | harmful continuations + refusals | datasets/circuit_breakers/ |
| BeaverTails | HF PKU-Alignment | 334k (+30k subset) | harm labels/categories | datasets/beavertails/ |
| PKU-SafeRLHF | HF PKU-Alignment | 82k | safety + severity labels | datasets/pku_saferlhf/ |
| Qwen3GuardTest | HF Qwen/Qwen3GuardTest | 2,441 | token-level unsafe onsets (natural test) | datasets/qwen3guardtest/ |
| FineHarm (gated, not downloaded) | HF liyang-ict/FineHarm | ~29k | sentence/word harm positions | requires owner access |

**No dataset has 64–512-token harm onsets.** The experiment runner must construct one; the recipe is in datasets/README.md.

## Code repositories, see code/README.md
| Name | URL | Purpose | Location |
|---|---|---|---|
| LatentQA (user-specified) | github.com/aypan17/latentqa | LIT decoder train/read/control; smoke-tested | code/latentqa/ |
| Activation Oracles | github.com/adamkarvonen/activation_oracles | AO training/eval, linear-probe baseline | code/activation_oracles/ |
| Probabilistic AOs | github.com/federicotorrielli/probabilistic_activation_oracles | Likelihood-based AO scoring | code/probabilistic_activation_oracles/ |
| Refusal direction | github.com/andyrdt/refusal_direction | 1-D harm/refusal baseline, abliteration | code/refusal_direction/ |
| Shallow-vs-deep alignment | github.com/Unispac/shallow-vs-deep-alignment | Prefilling-attack code | code/shallow-vs-deep-alignment/ |

## Models (HF; access verified 2026-10-18)
| Model | Access | Use |
|---|---|---|
| `Qwen/Qwen3-8B` | ungated (config cached in $HF_HOME) | **Primary target** |
| `adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B` | public LoRA | zero-shot AO; init for the direct AO |
| `ceselder/qwen3-8b-ao-v3-best` | public LoRA | improved AO |
| `NousResearch/Meta-Llama-3-8B-Instruct` | ungated mirror, **downloaded** | LatentQA target |
| `aypan17/latentqa_llama-3-8b-instruct` | public LoRA, **downloaded** | LatentQA decoder |
| `mlabonne/Qwen3-8B-abliterated` | ungated | generator of harmful continuations (optional) |
| `Qwen/Qwen3Guard-Gen-8B` / `Qwen/Qwen3Guard-Stream-8B` | ungated (verified) | onset judge / streaming text baseline |
| `liyang-ict/SCM-7B` | ungated (verified) | streaming harm-detection text baseline |
| `meta-llama/*`, `google/gemma-*` | **403 (gated, no access)** | avoid; use mirrors |

## Resource gathering notes
- **Search strategy:** the paper-finder service was down (localhost:8000), so the search was manual: arXiv API (~30 queries), HF API (models/datasets), GitHub API. Two sub-agents deep-read the AO and forecasting/safety papers. The user-specified LatentQA paper was read fully (all 34 pages).
- **Selection criteria:** direct relevance to verbalizers, future/timing prediction, or streaming harm; datasets that supply harmful requests, harmful continuations, or onset positions.
- **Challenges:**
  - Gated Llama/Gemma/walledai/FineHarm (solved with mirrors and GitHub CSVs; FineHarm is left to the owner).
  - LatentQA upstream bugs with current transformers (documented workarounds).
  - `/data` is 100% used, with ~144 GB free at the start of this phase, so cache activations selectively.
  - Only 1 of 4 GPUs was free.
- **Gaps:** no long-onset dataset exists (a construction recipe is provided); no released verbalizer trained on future timing (expected; this is Part 2).

## Recommendations for experiment design
1. **Primary data:** the constructed delayed-harm set on Qwen3-8B (HarmBench/AdvBench/LLM-LAT/circuit-breaker + generated preambles + matched controls). Natural test set: Qwen3GuardTest `thinking_loc`. Held-out behaviors: JBB / StrongREJECT.
2. **Baselines:** text-only (same LLM; Qwen3Guard-Stream), linear/MLP probes, refusal direction, zero-shot AO ×2, LatentQA decoder (Llama-3-8B), self-report, matched direct AO.
3. **Metrics:** AUROC per lead bucket, time-to-onset MAE / C-index, warning lead at fixed FPR, ECE, bootstrap CIs.
4. **Code to reuse:**
   - `code/activation_oracles` (injection hook, sft.py, linear_probe.py)
   - `code/probabilistic_activation_oracles` (scoring)
   - `scripts/latentqa_smoke_test.py` (working LatentQA read wrapper)
   - `code/refusal_direction`
