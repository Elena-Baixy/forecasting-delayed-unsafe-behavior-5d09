# Literature Review: Forecasting Delayed Unsafe Behavior from Activations with Verbalizers

Detailed per-paper notes:
- `papers/notes_activation_oracles.md`: Activation Oracle (AO) family, deep reads.
- `papers/notes_forecasting_safety.md`: future-probing and streaming-harm papers, deep reads.
- LatentQA (user-specified) was read in full; its notes are in this file.

Paper-finder was unavailable (service not running). The search used the arXiv API with ~30 targeted queries plus the HF and GitHub APIs.

## 1. Research area overview
Activation verbalizers are LLMs fine-tuned to answer natural-language questions about another model's activations. The main lines of work:
- **Training-free patching:** Patchscopes, SelfIE.
- **Trained decoders:** LatentQA/LIT (2024) and Activation Oracles (AO, 2025), plus a fast-growing 2026 literature on AO quality, calibration, blind spots and hallucination.
- **Probes for future content:** Future Lens, Emergent Response Planning, myopic-training analysis, remaining-length probes. These show that hidden states carry information about future tokens and global response attributes.
- **Streaming safety monitors:** SCM/FineHarm, StreamGuard, Constitutional Classifiers, Qwen3Guard-Stream. These try to stop harmful output early, but almost all of them *detect harm already present in the prefix*. Only StreamGuard explicitly *forecasts* it, and it is text-only.

**No paper evaluates any verbalizer, or even a probe, on forecasting unsafe content 64–512 tokens ahead.** The longest horizons in the AO literature are next-50-token context prediction (a training task only) and AObench's "Backtracking" and "MMLU Prediction" items. Our Part 1 question is therefore open.

## 2. Key papers

### LatentQA: Teaching LLMs to Decode Activations into Natural Language (Pan, Chen, Steinhardt; ICLR 2026; arXiv 2412.08686). User-specified, deep read
- **Task:** answer open-ended questions about activations. The paper focuses on predicting *qualitative properties of the model's future completion* from activations of the current prompt, which is exactly the framing of our hypothesis, applied there to personas and goals rather than harm timing.
- **Data (Sec. 3, App. A):**
  - o1-preview builds (control, stimulus, completion) dialogs plus QA pairs: 4,670 goals, 3,359 personas and 8,703 extractive-QA items, 16,732 in total.
  - QA comes in two kinds: *description* QA (what is the control) and *reasoning* QA (implications of the control for future behavior).
  - Three data types: control activations; stimulus activations with the control masked; stimulus+completion activations with the control masked.
  - Three design decisions matter for generalization: (1) **mask the control-token activations** so the decoder cannot read the instruction directly and must use what attention has written into later tokens; (2) mix the three data types; (3) make completions faithful (strong control prompts or a stronger generator).
- **Method (LIT):**
  - The decoder is a copy of the target with a LoRA adapter.
  - Activations from target layer k=15 are patched into decoder layer ℓ=0 at placeholder tokens. The patch *replaces* the hidden state.
  - The decoder is trained with cross-entropy on the answer.
  - A layer sweep (Table 6) found read 15 / write 0 best (eval PPL 1.013); read 0 is worst.
  - Control: gradients of the decoder's QA loss are backpropagated into a LoRA on the target (rank 8, layers 0..k).
- **Results:**
  - Hidden system prompt (persona) identification from *user-turn activations only*: 100% (easy) / 92% (hard), vs 97.3% / 73.3% for prompting GPT-4 with the full dialog, and ~24% / ~16% for SelfIE.
  - Relational attribute extraction beats linear probes by 32.2 points and Patchscope by 38.2 points on average.
  - Debiasing (CrowS) is the only statistically significant reduction among the methods tried.
  - Steering with the decoder can elicit harmful knowledge (100% of harmful and 87% of benign prompts after "rogue chemist" steering).
  - Scaling: test loss improves with model size (1B → 70B) and data size. Hard-persona accuracy is 12 / 88 / 92% for 1B / 3B / 8B.
- **Limitations (authors):** narrow data (goals, personas, extractive QA); possible hallucination; no ground truth for latents. Futures are only *qualitative style/goal* properties of the next reply; there are no timing or long-horizon events.
- **Code / checkpoint:**
  - github.com/aypan17/latentqa
  - `aypan17/latentqa_llama-3-8b-instruct` (LoRA)
  - Our smoke test runs (see code/README.md). It reveals a strong "safe" prior on yes/no safety questions, while the open-ended "what will the assistant do later?" question does reveal harmful plans.

### Activation Oracles (Karvonen et al., arXiv 2512.15674): the "AO" baseline
- **Method:**
  - The oracle is a LoRA copy of the target.
  - Prompt: `Layer: L\n ? ? … ?\n<question>`.
  - Target activations from layers at 25/50/75% depth are *added*, norm-matched, at the output of oracle block 1.
- **Training data:** ~1M examples, a mix of next/past-token context prediction (600k), 7 classification datasets (336k) and LatentQA system-prompt QA (64k).
- **Models and checkpoints:** Qwen3-8B, Gemma-2-9B-it, Llama-3.3-70B. LoRA checkpoints are on HF.
- **Evaluations:** secret-keeping model organisms (Taboo, Gender, SSC), emergent-misalignment diffing, PersonaQA, out-of-distribution classification.
- **Findings:** wide training transfers out of distribution. However, the AO paper has *no probe comparison on its main tasks and no future-behavior prediction*.

### Building Better Activation Oracles (arXiv 2606.02609)
- **Recipe (Qwen3-8B):**
  - New conversational data in which the question is about the *suffix* of the model's CoT, asked from prefix activations. This is essentially "predict what the model does next", and it was the largest single gain.
  - Read layer 22 (62% depth), or layers 21–25.
  - On-policy lens corpus.
  - Injection coefficient 2×.
- **Evaluation:** AObench (11 tasks).
- **Critical lessons for us:**
  1. Narrowly trained AOs "consistently matched but never significantly exceeded linear probes".
  2. A text-only control with the same context performed similarly.
  3. Score with AUROC on the Yes/No logit margin. Qwen AOs default to "No".
  4. Feed ≥20–50 tokens of activations, not just the last token.
- **Checkpoint:** `ceselder/qwen3-8b-ao-v3-best`.

### AO calibration; AO blind spots; Faithful Activation Verbalization (AVPO); NLA token selection (2026)
- **Calibration:** candidate-answer likelihood scoring gives AUROC 0.92–0.96; verbalized confidence is useless (0.40–0.53). Code: `code/probabilistic_activation_oracles`.
- **Blind spots:** AOs fine-tuned on a skewed concept stop reporting it (35% vs 94% recovery) while probes still decode it. Balance the labels and always compare against probes.
- **AVPO:** AOs hallucinate on harmful-intent inputs (intent recovery 0.54).
- **NLA token selection:** chat-template boundary tokens and late-input tokens are the most informative read positions.

### Future information in hidden states
- **Future Lens (2311.04897):** a learned soft-prompt decoder predicts tokens 2–4 ahead at ~45% P@1 vs 97% for the next token. A sharp drop, then a plateau. It is a learned "future verbalizer" precursor. Middle layers are best.
- **Emergent Response Planning (2502.06258):** MLP probes on *prompt* activations predict global response attributes (length Spearman 0.85, reasoning steps 0.82, 4-way choice F1 0.84). The signal over the response is U-shaped: high at the start, a dip mid-response, recovery just before the event. **Verbalized self-predictions are near chance.** This is our expected outcome for the self-report baseline.
- **Do LMs Plan Ahead (2404.00859):** future-useful information is mostly "breadcrumbs" (features useful now), with pre-caching that grows with scale. Prediction: long-lead harm forecasting will rely on *intent/topic* features and not encode timing well.
- **Remaining output length (2607.05316):** linear probes encode the number of tokens to EOS (MAE ~30 on 400-token outputs; it beats the constant-median and countdown baselines), and the estimate re-plans at "Wait" tokens. This is the template for our **time-to-onset head and its baselines**.

### Streaming / forecasting harm
- **SCM/FineHarm (2506.09996):**
  - 29K responses with sentence- and word-level harm labels (HF `liyang-ict/FineHarm`, gated click-through).
  - Many responses follow the "safe background → 'However…' → harmful guide" pattern.
  - SCM-7B reaches macro-F1 97.5 after seeing 18% of tokens. It *detects* harm, it does not forecast it.
- **StreamGuard (2604.03962):**
  - The first *forecasting* monitor. The target for each prefix is the Monte Carlo fraction of 16 sampled continuations judged unsafe by Qwen3Guard-Gen-8B.
  - Metrics: OnTime / Late / Miss and detection lead.
  - Text-only.
- **Shallow safety alignment (2406.05946):** safety lives in the first few tokens. Prefilling a compliant prefix gives a 42–57% attack success rate, so prefilling is an on-policy way to create delayed harm.
- **Constitutional Classifiers (2501.18837):** a streaming output classifier with token-level prefix prediction.
- **Multi-turn drift (2605.15455):** surfacing activation-based risk improves user calibration.

### Representation baselines and benchmarks
- **Refusal direction (2406.11717):** one difference-in-means direction mediates refusal. A cheap 1-D baseline.
- **RepE (2310.01405)** and **Circuit Breakers (2406.04313):** harmfulness readout and rerouting.
- **Looking Inward (2410.13787):** fine-tuned self-prediction (introspection) beats cross-model prediction. Basis for the "self-report" baseline.
- **Benchmarks:** HarmBench (2402.04249) and BeaverTails (2307.04657) supply harmful behaviors, categories and classifiers.

## 3. Common methodologies
- Activation patching into a decoder LLM: LatentQA (replace at layer 0), AO (norm-matched add at layer 1), Patchscopes and SelfIE (training-free).
- Linear or 1-hidden-layer MLP probes on residual streams at middle layers (Future Lens, ERP, remaining length, refusal direction).
- Streaming monitors on text prefixes, trained on sentence- or token-level onset labels (SCM, Qwen3Guard-Stream) or on Monte Carlo future-risk targets (StreamGuard).

## 4. Standard baselines (to include)
1. **Text-only forecaster:** the same base LLM, or a small classifier/guard model (Qwen3Guard-Stream-8B, SCM-7B), given the same prefix text.
2. **Linear and MLP probes** per layer and per horizon. The refusal-direction projection is a 1-D probe.
3. **Zero-shot verbalizers:**
   - the official AO (`adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B`);
   - `ceselder/qwen3-8b-ao-v3-best`;
   - the LatentQA decoder (Llama-3-8B via the NousResearch mirror).
4. **Self-report / prompting:** ask the target itself "will your response later contain harmful instructions?" (ERP predicts near chance).
5. **Matched direct AO:** the AO fine-tuned on our future labels (Part 2 control).
6. **Trivial baselines:** prompt-only classifier (does the prompt alone give it away?), constant-median and countdown baselines for timing.

## 5. Evaluation metrics
- **AUROC / AUPRC** for `harm_within(h)` at each lead bucket h ∈ {64, 128, 256, 512}: the decay curve. For verbalizers, use the Yes/No logit margin or candidate-answer likelihood.
- **Time-to-onset MAE** (right-censored; also the C-index) vs the constant-median baseline.
- **Early-warning metrics (StreamGuard):** OnTime / Late / Miss, mean warning lead in tokens, and FPR on matched benign controls at a fixed recall.
- **Calibration:** ECE. **Free-text QA:** LLM-judge correctness.
- **Bootstrap CIs** over prompts/behaviors. Statistical tests across methods use paired bootstrap or DeLong.

## 6. Datasets in the literature
- HarmBench, AdvBench, JBB, StrongREJECT: harmful requests.
- BeaverTails, PKU-SafeRLHF: QA with harm labels and categories.
- FineHarm: token/sentence harm positions (gated).
- Qwen3GuardTest: token spans of unsafe onsets. `thinking_loc` onsets reach ~220 tokens; `response_loc` onsets are nearly all at the start.
- Circuit-breakers / LLM-LAT: harmful plus refusal completions.
- Dolly / Alpaca: benign stimuli (LatentQA uses Dolly/Alpaca).

## 7. Gaps and opportunities
1. No benchmark has controlled harm onsets at long leads. One must be built, and it must include **matched controls** (same prompt and preamble, benign ending) so that the "harmful topic" shortcut is removed.
2. No verbalizer has been trained or evaluated on *timing*. The AO, LatentQA and AObench futures are style, goal, or next ~50 tokens.
3. The AO-vs-probe evidence is mixed-to-negative. Any claim that "verbalizers forecast" must beat probes and text-only baselines on identical inputs.
4. The supervision-vs-architecture question has not been tested: a direct AO trained on the same future labels is the necessary control (as the hypothesis states).

## 8. Recommendations for our experiment
- **Target model:** **Qwen3-8B** (ungated, cached config; official and improved AO checkpoints exist; 16 GB bf16 fits alongside an oracle copy on one 48 GB A6000). Secondary: Llama-3-8B-Instruct via `NousResearch/Meta-Llama-3-8B-Instruct`, for the released LatentQA decoder.
- **Data:** build a delayed-harm set (datasets/README.md recipe).
  - Inputs: HarmBench, AdvBench, LLM-LAT and circuit-breaker prompts and harmful text; benign preambles generated by the target; onsets at L ∈ {0, 64, 128, 256, 512}.
  - Matched benign and refusal endings; on-policy prefill continuations.
  - Split train/val/test **by behavior**. Use JBB and StrongREJECT as held-out behaviors.
  - Natural test: Qwen3GuardTest `thinking_loc`.
- **Read positions:** every 16 tokens before onset. Layers ~18 and 22 (Qwen3-8B). Feed a ≥20-token activation window to the verbalizers.
- **Baselines:** as in §4. Score every verbalizer with logit margins (AUROC), not free text.
- **Part 2:** train (a) a direct AO, i.e. the official AO further fine-tuned on future yes/no + timing QA, and (b) the proposed forecast-vector model (horizon queries → forecast vectors → timing head + QA decoder). Use identical labels, data, steps and LoRA budget. Keep ≥10% of the original AO mixture to avoid forgetting.
- **Pitfalls:**
  - Labels leaking through the preamble or topic.
  - The verbalizers' "No/safe" prior.
  - Hallucination.
  - Off-policy spliced text, so also report on-policy prefill results.
  - Chat-template end-of-turn tokens when reading mid-response.
