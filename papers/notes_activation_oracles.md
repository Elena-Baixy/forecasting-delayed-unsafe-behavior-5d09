# Notes: Activation Oracles (AOs) and related verbalizers

Project: forecasting unsafe content 64-512 tokens ahead from activations; matched-AO baseline vs.
a verbalizer trained with direct future-event + timing supervision. Hardware: 1x A6000 48GB.

## 0. TL;DR for this project

- **Official code:** https://github.com/adamkarvonen/activation_oracles (MIT, ~168 MB repo, 103 stars).
  Use the `paper` branch / `v1.0-paper` tag to reproduce the paper (torch 2.7.1, transformers 4.55.2,
  peft 0.17.1). `main` has moved to transformers 5 / torch 2.12.
  Training entry point: `torchrun --nproc_per_node=N nl_probes/sft.py`; config in
  `nl_probes/configs/sft_config.py`; injection hook in `nl_probes/utils/steering_hooks.py`
  (`get_hf_activation_steering_hook`); prompt builder in `nl_probes/utils/dataset_utils.py`.
  The repo also includes `experiments/linear_probe.py`, a ready-made probe baseline.
- **Best checkpoints that fit 48GB (bf16 base + LoRA):**
  - `adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B` (paper ckpt, 698 MB LoRA, r=64)
  - `adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it` (paper ckpt, 864 MB)
  - `adamkarvonen/checkpoints_latentqa_cls_past_lens_Llama-3_1-8B-Instruct` (post-paper, 671 MB)
  - `adamkarvonen/checkpoints_latentqa_cls_past_lens_Qwen3-4B` (529 MB), `..._Qwen3-14B` (1.03 GB; 14B bf16 ~28 GB, fits)
  - `ceselder/qwen3-8b-ao-v3-best` (Building Better AOs full recipe, 1.40 GB LoRA, r=128 rsLoRA, layers 21-25)
- **No paper here forecasts behavior 64-512 tokens ahead.** The closest are: AO "next-J-token"
  context prediction (J <= 50 tokens, text tokens only, used only as a training task) and AObench's
  **Backtracking** (is the model about to revise its reasoning?) and **MMLU Prediction** (what
  would the model answer if interrupted now?). So our horizon question is open.
- **AO vs linear probes:** Building Better AOs says that narrowly training AOs on the hard CoT-interp
  tasks "consistently matched but never significantly exceeded linear probes". The blind-spots paper
  finds probes decode a hidden concept almost perfectly in cases where AOs fail. Expect probes to be
  a strong baseline. Report AUROC from Yes/No logits, not accuracy.
- **Injection recipe (matched AO):** see Section 1.2. In short: add the norm-matched target
  activation to the residual output of oracle block 1 at each " ?" placeholder, with coefficient 1.0.
  LoRA r=64, alpha=128, all-linear, lr=1e-5, batch size 16, 1 epoch.

## 1. Activation Oracles (Karvonen et al., arXiv 2512.15674): DEEP READ

**Method.** A LoRA fine-tune of the target model itself (the oracle and the target share base
weights; LoRA is disabled for the target forward pass, so activations are generated on the fly).
The oracle answers natural-language questions about K injected activation vectors. This generalizes
LatentQA (Pan et al. 2024) with a more diverse training mixture.

### 1.2 Exact injection recipe (paper Sec 3.1 and App A.5, checked against the repo code)
- Oracle prompt is a single user turn with the chat template, `add_generation_prompt=True`,
  `enable_thinking=False` (Qwen3): `"Layer: {L}\n" + " ?"*K + " \n" + question`.
  The placeholder token is `" ?"`, and its positions are found by token-pattern search.
- Target activations come from the residual-stream output of `model.model.layers[L]` (HF, 0-based),
  taken from the **base** model with the LoRA disabled.
- Injection happens at the output of oracle `model.model.layers[1]` (i.e. after the second block):
  `h'_i = h_i + ||h_i|| * v_i/||v_i|| * coef`, with coef=1.0. This adds, it does not replace. The
  hook fires only on the prefill pass (seq len > 1), and v is detached.
- Why add instead of replace: replacement made placeholder norms grow ~20x (inject at layer 0) and
  ~100,000x (layer 1) during training. Layer 1 beat layer 0 by 1-11% on Llama-70B (Fig 9) and made
  no difference for Qwen3-8B and Gemma-2-9B.
- Loss is computed on assistant answer tokens only (labels before the assistant start are -100).
- Training layers are 25/50/75% depth, one third each. For Qwen3-8B (36 layers) that is layers
  9/18/27. Eval uses 50% (layer 18). Generalizes to unseen layers at 33/66%. Fails at 0-10%.
  On secret-keeping, 50% beats 75%: Taboo 0.205 vs 0.056, SSC 0.59 vs 0.29.
- Hyperparameters (Table 1): LoRA r=64, alpha=128, dropout 0.05, all linear layers; AdamW,
  lr 1e-5, 10% linear warmup then linear decay; batch size 16; 1 epoch. Uses group-by-length batching
  (mega-batch = 16x20) instead of packing. LR is robust for the full mixture, fragile for SPQA-only.
- Compute: Qwen3-8B about 10 H100-hours, Gemma-2-9B about 12 H100-hours (one H100 each);
  Llama-3.3-70B about 90 H200-hours (4xH200, 8-bit). The blind-spots paper reports about 18 h on one
  A100-80GB for Qwen3-8B. **On an A6000 48GB, Qwen3-8B bf16 + LoRA r64 should fit with gradient
  checkpointing at batch size 16 and short sequences. Budget roughly 25-40 h for 1M examples, or
  subsample.**

### 1.3 Training data (~1M examples, ~65M tokens)
- **Context prediction (600k):** 50/50 FineWeb + LMSYS-Chat-1M. Half the examples use a single
  activation, half use 1-50 contiguous activations. Half predict the next J tokens, half the
  previous J tokens, J~U(1,50) (main text says U(1,20)). Input and target spans do not overlap.
  Prompt: "Can you predict the previous/next {J} tokens that came before/after this?"
- **Classification (336k):** 7 datasets x 48k (Geometry of Truth, Relations, SST-2, MD Gender,
  SNLI, NER, Tense), posed as yes/no questions with about 20 Claude-paraphrased templates and
  balanced answers. 2/3 of examples use a single token sampled 1-5 tokens before the end (this
  positional randomness matters: a fixed position generalizes brittlely). 1/3 use windows of 1-50
  tokens ending 1-5 tokens before the end.
- **SPQA (64k):** Pan et al. system-prompt QA. Half use the original full-segment format, half use a
  1-3 token window ending 1-10 tokens before the end.
- Caveat: the HF `ao_config.json` for the released Qwen3-8B ckpt lists past_lens 2x100k, latentqa 100k,
  and 18 cls loaders x 6k. This is smaller than the paper text. Check before claiming an exact match.
- SAE-feature tasks were tried (App G, ~$1k of API for explanations), gave mixed results, and were dropped.

### 1.4 Models, evals, key numbers
- Models: Qwen3-8B, Gemma-2-9B-IT, Llama-3.3-70B-Instruct, Claude Haiku 3.5. The Claude variant
  differs: it replaces at layer 0, does a full fine-tune, uses one middle layer, and does past-only
  context prediction.
- Secret keeping (full-sequence activations, AO vs best white-box vs best black-box):
  Taboo (Gemma-2-9B, 20 words): 0.205 / 0.044 / 0.072. Gender: 0.970 / 0.870 / 0.977.
  SSC (Llama-70B): 0.590 / 0.522 / 0.968. SPQA-only AO: 0.041 / 0.820 / 0.493.
  Taboo improves with a single token at the assistant start-of-turn (Qwen3-8B Taboo 0.495).
- Emergent-misalignment model diffing: the AO gets the activation **difference** (fine-tuned minus
  base) over the first 10 tokens, with the prompt "Can you describe what is happening in this text?".
  It scores 2.00/5, vs ADL 2.03 and black-box 1.50. 2/5 is about the ceiling because the 50%
  chat-mixed fine-tunes barely misalign (only Qwen risky-finance shows 18% EM).
- PersonaQA open-ended: ~5% (base) -> ~14% (full AO) for the open models; Claude 35%. Binary
  version: Qwen 0.696, Llama 0.632. No linear-probe baseline: there were too few attribute instances.
  Li et al. found probes reached only 25% on a related set.
- OOD classification on 7 held-out sets (single token before EOS): the full AO is well above the
  base model and SPQA-only.
- Ablations: 10 of 12 model/eval cells are best with the full mixture. At matched 400k examples,
  adding context prediction still helps (Taboo 0.459 vs 0.302), and 1M beats 400k (0.495).
- Limitations: no calibration (it answers confidently when wrong), black-box QA with no mechanism,
  narrow model organisms. In an anecdotal hallucination-detection test, plain prompting did as well
  (App F). **No probe comparison on the main tasks and no future-behavior prediction.**

## 2. Building Better Activation Oracles (Bauer, De Schamphelaere, Karvonen, Luick, Nanda; arXiv 2606.02609): deep read

- Model: Qwen3-8B only. Starts from Karvonen's recipe and changes four things, with controlled
  ablations at 50M training tokens and 3 seeds:
  1. **New conversational dataset** replaces LatentQA. Sonnet 4.6 splits the target model's own CoT
     into a prefix and a suffix, then writes a question about the suffix that is hard to answer
     from the prefix text but plausibly answerable from the prefix activations. This is
     **essentially "predict what the model will do/say next"**. AObench (chance-adjusted) goes from
     +0.244 to +0.310, the largest single step. HF dataset: `ceselder/cot-oracle-convqa-chunked-sonnet`.
  2. **Layer choice:** the best single layer is 22 of 36 (62%), +0.025 over layer 18. Feeding
     5 contiguous layers (21-25) gives +0.05, with the biggest gain on model-diffing. Advice:
     pick roughly 65% depth if you can only use one layer.
  3. **On-policy past/future-lens corpus** (the model's own CoT rollouts instead of FineWeb):
     +0.244 to +0.274. A modest gain, and confounded with item 1.
  4. **Injection strength 2x** (coef=2 in the same additive norm-matched formula at block 1):
     a small overall gain, but hallucination (not-wrong) improves 79% to 85%. NLA-style
     embedding replacement with a fixed scalar was "significantly worse" on Qwen3-8B.
- Full recipe: +0.244 -> +0.435 (paper text); the HF card says the 4-seed mean is +0.417.
  Vagueness 0.076 -> 0.205. The best ckpt uses LoRA r=128, alpha=16, rsLoRA, lr 3e-5, unsloth.
- **AObench** has 11 evals: Activation Sensitivity (same tokens, different upstream context; a
  clean anti-text-inversion test), Hallucination, Missing Info, Sycophancy (Scruples), Taboo,
  PersonaQA (system prompt), **MMLU Prediction** (the answer the model would give if interrupted
  mid-reasoning), Number Prediction, Vagueness, Domain Confusion, and **Backtracking** (predict
  whether the model is about to revise its reasoning). The last two forecast near-future
  behavior. Eval code: the `cot-oracle` repo.
- **Practical evaluation lessons (very relevant):**
  - Use **ROC AUC on the Yes-vs-No logit margin**, not accuracy. Qwen AOs default to "No".
    On sycophancy, the original AO gets accuracy 0.50 but AUC 0.83.
  - **Sweep the context window.** On Backtracking, the original AO scores 1.26/5 with the last
    token only and 2.10/5 with the last 50 tokens, above the black-box baseline of Qwen3-8B with
    full text. Report windows of >= 20 tokens.
  - Consensus sampling (k=10, agreement >= 0.8) on Taboo: 19.4% coverage at 94.3% precision
    (unfiltered accuracy 46.6%).
  - **Narrow post-training matched linear probes but never significantly exceeded them**
    (footnote: "a very consistent observation"). They recommend: "aim to match the performance of probes".
  - They briefly tried a **text-only control** (same model, no activations, full prior context,
    same objective) and found "performance was similar". They flag this as weak evidence, but
    it is a must-run baseline for us.
  - Most eval uplift appears after about 10% of training. Keep at least 10% mixed-task data at
    every stage to avoid catastrophic forgetting. DPO/GRPO was unstable or mixed.
- Code: https://github.com/japhba/activation_oracles (a fork, ~206 MB), plus
  https://github.com/ceselder/cot-oracle (AObench, ~148 MB). Models: the HF collection
  `ceselder/building-better-activation-oracles`, which contains `ceselder/qwen3-8b-ao-v3-{adam-baseline,
  abl1-sonnet-data, abl2b-layer22, abl2c-multilayer, abl3-cotv5-corpus, abl4c-sonnet-multilayer,
  best, best-dpo-halluc}`.

## 3. AO blind spots (Bersia & Gaintseva, arXiv 2607.23379): skim

- Setup: Qwen3-8B (non-thinking), 5 Taboo concepts x {cooperative, strict} x alpha in
  {0.5, 1.0} UltraChat mix = 20 subjects. AOs trained with the Karvonen recipe (r=64, alpha=128,
  lr 1e-5, 1 epoch, bf16, about 18 h on one A100-80GB) on base-subject activations (base AO) or
  on Taboo-subject activations (FT-AO). Activations come from layer 18 over the assistant response
  span. Note: the paper says it injects "at the AO's layer 18" with lambda=1.0; this conflicts with
  the official layer-1 hook, so treat it as a likely description error.
- Results: the base AO recovers hidden words well (cooperative HINT 93% exact). **FT-AOs become
  concept-specific "anti-readers":** own FT-AO 35% exact vs base AO 94% and cross FT-AO 71%
  (cooperative, alpha=1). Mean-pooled linear probes (logistic regression, layers 9/18/27) decode the
  concept near-perfectly. A probe on AO hidden states (layer 33) still decodes it, but the AO's
  logit lens ranks it poorly. Removing the FT-AO's LoRA from layers 18-23 restores recovery.
- **Implication for us:** training an AO on activations from the same model or distribution where a
  label is always present can teach it NOT to verbalize that label. Our matched AO is trained on
  our own future-unsafe labels with both positives and negatives, so it is less exposed to this.
  Still, compare against a probe on the same activations, and check that the AO does not suppress
  label words.
- Code: none linked in the paper. GitHub search shows `BerTobi/crossmodel-activation-oracles`
  (likely the same first author, a follow-up).

## 4. AO calibration (Torrielli, Schneider-Kamp, Galke Poech; arXiv 2605.26045): skim

- Oracles: **released** Karvonen Qwen3-8B (target layer 18/36) and Gemma-2-9B (target layer
  10/42, 25%), the community Gemma-3-27B oracle (`annasoli/gemma-3-27b-activation-oracle`, layer
  46/62), and a new Qwen3.6-27B oracle (layer 48/64). Injection is at oracle layer 1 with
  `r + s*||r|| h/||h||`, s=1, using the last K=10 target positions. Task: the 20-word Taboo game,
  6,000 samples per method per oracle.
- Five confidence methods, all ranking the same way on all four oracles:
  - **Forced choice** (score each candidate's log-prob under the hook) roughly doubles accuracy
    (Qwen3-8B 0.414 -> 0.646) and gives AUROC 0.92-0.96.
  - Bootstrap agreement over 20 samples at a tuned T is the only free-text method calibrated on
    every oracle (ECE 0.04-0.12).
  - Answer log-prob after temperature/Platt rescaling matches that calibration in one generation.
  - Verbalized numeric confidence is useless (AUROC 0.40-0.53).
  - A constrained 5-label confidence readout recovers some signal (AUROC 0.68-0.82).
- **For us:** score Yes/No (or horizon-bin) tokens with constrained decoding rather than parsing free
  text. This is fair to the AO and gives AUROC and calibration directly. Self-report confidence is a
  weak baseline.
- Code: https://github.com/federicotorrielli/probabilistic_activation_oracles (~166 MB; has the
  patched trainer, which fixes an off-by-one in multi-layer activation slicing and adds a LoRA
  regex for Gated-DeltaNet). The Qwen3.6-27B AO trained on 1 B200 in about 36 h, too big for our GPU.

## 5. Faithful Activation Verbalization / AVPO (Zhao et al., arXiv 2609.34033): skim

- Two-stage. (I) A question-agnostic **inverter** maps the **last-token** activation to text,
  through a Q-Former adapter (M=8 context slots, K=64 learned queries into soft-token prefixes) and
  a LoRA decoder. It is trained by reconstruction SFT, then iterative DPO with reward
  0.35 gist + 0.35 detail + 0.3 anchor-F1 (proper nouns and numbers); Qwen3-32B is judge and QA
  model. (II) A frozen QA model answers questions from the reconstructed text only.
- Donors: Llama-3.1-8B-Instruct and Mistral-Small-24B-Instruct-2501. Decoders: self, Qwen3-4B,
  Qwen3-0.6B. Data: 165,760 train texts (<= 64 tokens) from 6 families. **AO and LatentQA were
  retrained as baselines on their data** (AO = additive steering at block 1).
- Overall QA score on Llama-8B: AO 0.423, LatentQA 0.475, UAV 0.492, AVPO (Qwen3-4B) 0.624.
  On OOD harmful requests (JBB, AdvBench, HarmBench, StrongREJECT, Do-Not-Answer; 250 requests,
  10 categories), harm-intent recovery is AO 0.54 vs AVPO 0.77, and harm category is AO 0.63 vs 0.78.
  AO fabricates names 99% of the time vs AVPO 10%.
- Relevance: the verbalizers read **current** content (an inversion upper bound), not future
  behavior. There are no probe baselines. It is evidence that AOs carry harmful-intent information
  in a single last-token activation. No code link was found in the text.

## 6. NLA token selection (Torrielli, Barmina et al., arXiv 2609.37040): skim

- Uses the four released **Natural Language Autoencoder** verbalizers (Fraser-Taliente et al. 2026,
  Anthropic). Each reads about 2/3 depth: Qwen2.5-7B L20, Gemma-3-12B L32, Gemma-3-27B L41,
  Llama-3.3-70B L53. NLA = activation verbalizer + reconstructor, with an SFT warm-start then
  GRPO on reconstruction MSE, about 130 tokens per explanation. 4.7M explanations over
  OpenPromptInjection, Tensor Trust, Liars' Bench, and Taboo organisms.
- Findings: 13 single-forward-pass signals (surprisal, entropy, attention, activation norm/jump)
  reach median AUROC 0.58 at predicting which positions give on-task explanations. A **chat-structure
  logistic-regression ranker beats them** in 12 of 14 cells. A 5% position budget keeps >= 95.8%
  of audit success on 3 datasets. **Boundary (chat-template) tokens and late-input tokens** are the
  most informative, and they are available *before* the response is generated, so they work for
  pre-emptive monitoring. Pretrained NLAs recover Taboo words from fine-tuned copies at 14-27% of
  positions without retraining.
- **For us:** sample AO/NLA positions at the assistant boundary and at the current position, not
  evenly. The Qwen2.5-7B NLA (layer 20) is a candidate zero-shot verbalizer that fits our GPU.
  Code: https://github.com/federicotorrielli/nla-token-selector.

## 7. Checkpoint inventory (HF, verified via API on 2026-10-18; sizes = LoRA adapter only)

Official collection: https://huggingface.co/collections/adamkarvonen/activation-oracles (13 models)
| HF repo | base model | fits 48GB bf16? |
|---|---|---|
| adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B (698 MB, paper) | Qwen/Qwen3-8B | yes |
| adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it (864 MB, paper) | google/gemma-2-9b-it | yes |
| adamkarvonen/checkpoints_latentqa_cls_past_lens_Llama-3_1-8B-Instruct (671 MB) | meta-llama/Llama-3.1-8B-Instruct | yes |
| adamkarvonen/checkpoints_latentqa_cls_past_lens_Qwen3-4B (529 MB) | Qwen/Qwen3-4B | yes |
| adamkarvonen/checkpoints_cls_latentqa_past_lens_Qwen3-1_7B | Qwen/Qwen3-1.7B | yes (weak OOD) |
| adamkarvonen/checkpoints_cls_latentqa_past_lens_gemma-3-1b-it, ..._Llama-3_2-1B-Instruct | 1B models | yes (weak OOD) |
| adamkarvonen/checkpoints_latentqa_cls_past_lens_Qwen3-14B (1.03 GB) | Qwen/Qwen3-14B | yes (~28 GB) |
| adamkarvonen/checkpoints_latentqa_cls_past_lens_gemma-2-27b-it, ..._gemma-3-27b-it | 27B | no (8-bit only) |
| adamkarvonen/checkpoints_cls_latentqa_past_lens_Qwen3-32B, ..._addition_Qwen3.6-27B | 27-32B | no |
| adamkarvonen/checkpoints_act_cls_latentqa_pretrain_mix_adding_Llama-3_3-70B-Instruct (paper) | 70B | no |
Ablation ckpts on Qwen3-8B, useful as training-mixture baselines: `adamkarvonen/checkpoints_latentqa_only_addition_Qwen3-8B`
(SPQA only), `adamkarvonen/checkpoints_cls_latentqa_only_addition_Qwen3-8B`, `adamkarvonen/checkpoints_cls_only_addition_Qwen3-8B`,
`adamkarvonen/checkpoints_cls_latentqa_past_lens_400k_Qwen3-8B`.
Each ckpt ships an `ao_config.json` with act layers (Qwen3-8B: 9/18/27), hook_onto_layer=1, special token " ?",
and prefix template. Adapter: r=64, alpha=128, q/k/v/o/gate/up/down.
Others: `ceselder/qwen3-8b-ao-v3-*` (Building Better ladder), `nluick/activation-oracle-multilayer*`
(multi-layer Qwen3-4B/8B/14B), `annasoli/gemma-3-27b-activation-oracle` (27B).

Other GitHub repos (API search "activation oracles", 59 hits): `UKGovernmentBEIS/vllm-lens` (135 stars;
vLLM residual extraction and steering, including AO support; useful for fast rollouts),
`ceselder/neural_chameleons_activation_oracles`, `federicotorrielli/activation_oracles`
(Gemma 4 AOs), `zach-perlman/activation_oracles` (training harness), `spencerkitts/fooling-activation-oracles`.

## 8. Design implications for our experiments

1. **Target model:** Qwen3-8B (best-supported: paper AO, Building Better ladder, Taboo organisms,
   calibration testbed) with Llama-3.1-8B-Instruct or Gemma-2-9B-IT as a replication. Read
   activations at layer 18 (50%, official) and about 22 (62%, Building Better best). Optionally
   feed 21-25.
2. **Matched AO baseline:** start from the released Qwen3-8B AO LoRA (or from scratch with the same
   recipe) and fine-tune on our future-unsafe labels phrased as yes/no questions ("Will unsafe
   content appear within the next N tokens?"). Use the identical " ?" placeholders, "Layer: L"
   prefix, block-1 additive norm-matched injection (coef 1, optionally 2), and LoRA r64/alpha128,
   lr 1e-5. Keep >= 10% of the original mixture to avoid forgetting.
3. **Scoring:** use the Yes/No logit margin, then AUROC and ECE after Platt scaling. Use forced
   choice over horizon bins for timing. Sweep the activation window (1 / 20 / 50 tokens).
4. **Baselines are mandatory:** (a) a linear probe on the same layer and positions; Building Better
   says AOs match but do not beat probes. (b) A text-only model with the same context and same labels
   (the text-inversion control). (c) The zero-shot released AO. (d) Self-report.
5. **Positions:** the current token plus the assistant boundary tokens (NLA token-selection paper).
6. **Risk:** the blind-spot effect if training always contains the label; balance positives and
   negatives and check that the label words are not suppressed.
