# Notes: forecasting future content / streaming harm detection (for delayed-unsafe-behavior project)

Source text extracted with pypdf into `logs/paper_text/*.txt`. HF availability checked 2026-10-18 via
`huggingface.co/api/datasets` + `datasets-server /size` (nothing was downloaded). "gated: auto" means
you click through a form and get access immediately (needs an HF token).

---------------------------------------------------------------------------------------------------
## A. Probing for future information in hidden states (deep read)

### A1. Future Lens (Pal et al., 2311.04897, CoNLL 2023)
- RQ: can a single hidden state h^l_T predict the tokens at T+2, T+3, T+4 (N=1..3 tokens past the next one)?
- Model: GPT-J-6B (`EleutherAI/gpt-j-6b`), 28 layers. Data: the Pile; 100k training tokens (avg context 518),
  1k test tokens; only positions where GPT-J's next-token prediction is correct.
- Methods: (1) linear map h^l_T -> final-layer hidden state at T+N, decoded with the LM head; (2) linear map
  -> vocab logits directly; (3) fixed-prompt causal intervention: patch h^l_T into a generic prompt
  ("Hello! Could you please tell me more about \"") and let it generate; (4) **learned soft prompt**
  (length 10, one per layer, trained with KL to match the original future distribution) = "Future Lens".
- Metrics: Precision@k (k=1,5,10) against the model's own top-k at T+N, and surprisal. Baseline: Pile bigram (20.1% at N=1).
- Key numbers: the learned prompt is best and peaks at middle layers (L14 of 28). Precision@1 is
  97.0 (N=0), 48.4 (N=1), 43.7 (N=2), 46.9 (N=3). It beats the best baseline by about 25 points P@1. Linear models do much worse.
- **Decay with distance**: a big drop after the next token, then a plateau over 1-3 tokens. Only tested up to 4 tokens ahead,
  so it says nothing about 64-512 token leads. Code/data: https://future.baulab.info
- Takeaway: the learned soft-prompt decoder is an early version of a "verbalizer trained for the future" (our step 3).
  Middle layers are the right place to read from.

### A2. Emergent Response Planning in LLMs (Dong, Zhou et al., 2502.06258, ICML 2025)
- RQ: do *prompt* representations (before any response token) encode global attributes of the whole upcoming response?
- Models (HF): `meta-llama/Llama-2-7b-hf`, `meta-llama/Llama-2-7b-chat-hf`, `meta-llama/Meta-Llama-3-8B(-Instruct)`,
  `mistralai/Mistral-7B-(Instruct-)v0.x`, `Qwen/Qwen2-7B(-Instruct)`; scaling runs on Llama-2 7/13/70B-chat and Qwen2.5-Instruct 1.5B-72B.
- Tasks/datasets: response length (UltraChat, AlpacaEval); reasoning steps (GSM8K, MATH); character choice in a story
  (TinyStories, ROCStories); final MC answer after an explanation (CommonsenseQA, SocialIQA); answer confidence/correctness
  (MedMCQA, ARC-Challenge); factual consistency (CREAK, FEVER). Labels come from greedy decoding.
- Probe: 1-hidden-layer MLP (grid over width 1-1024 and layer), 400 epochs, 60/20/20 split, 3 seeds. Metrics: Spearman,
  Kendall, Pearson for regression; F1 for classification (classes balanced, so F1 = accuracy).
- Key numbers: length on Qwen2-7B-Instruct gives Spearman 0.85 in-dataset and 0.64 cross-dataset. Reasoning steps Spearman 0.82.
  Character choice F1 is about 0.84 (4-way, chance 0.25). Instruct models beat base models on structure attributes.
  Probes saturate at width <=128. Structure attributes peak in middle layers; behavior attributes appear early.
- **Dynamics over the generation**: probe accuracy is high at the start, dips mid-response and recovers near the end
  (a U-shape). Positions are normalized and run up to the token *before* the attribute-revealing keyword. So "planning"
  signal does not decay monotonically. It is strongest at the prompt and right before the event.
- **Verbalized self-estimates are much worse than probes** (base models are near chance, instruct models only slightly
  better). This is the "self-report" baseline we expect to lose. No official code link in the paper.

### A3. Do Language Models Plan Ahead for Future Tokens? (Wu, Morris, Levine, 2404.00859, COLM 2024)
- RQ: is the future-predictive information in hidden states *pre-caching* (computed deliberately for later steps) or
  *breadcrumbs* (features useful now that also happen to help later)?
- Method: "myopic training", which drops off-diagonal gradients so a position cannot be optimized for later losses.
  The myopia gap (myopic loss minus vanilla loss) measures pre-caching.
- Synthetic task: y_n = z_n * sum_{i=1..10} sin(10 x_{n-i}) + (1-z_n) x_n. Vanilla 2-layer transformers clearly pre-cache
  sin(b x_n), which linear probes find in layer 1. Myopic models do not.
- Natural language: GPT-2 124M trained from scratch on 4.6M MS MARCO sequences (length 64). CE: vanilla 3.28, myopic 3.40,
  local-myopic 3.26, transformer-bigram 5.33, so the myopia gap is only 0.12 nats (mostly breadcrumbs). Pythia 14M-2.8B
  (`EleutherAI/pythia-*`) fine-tuned on 10M Pile sequences: the **pre-caching gap grows with scale**.
- Code: https://github.com/wiwu2390/FutureGPT2-public
- Takeaway for us: future-harm signal at long leads will probably be "breadcrumbs" (topic/intent features that are already
  present), not an explicit plan. That predicts a signal that depends on intent and is not a clean timing signal.

### A4. How Much is Left? LLMs Linearly Encode Remaining Output Length (Merzouk et al., 2607.05316)
- RQ: does h_t linearly encode r_t = T - t, the number of tokens left before EOS?
- Models: `meta-llama/Llama-3.1-8B-Instruct`, `allenai/Olmo-3-7B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3`.
- Data: synthetic Count / Countdown (n in [0,300]), GSM8K, MATH, MMLU-Pro, OpenThoughts-1k, TriviaQA. Only completions that
  end naturally with EOS are kept.
- Probes (single linear layer, MSE, frozen model, every layer): Remaining Count Probe (reads h_t at every position);
  Completion Length Probe (reads only the prompt's last token, predicts T, then counts down deterministically);
  constant-median baseline (the optimal L1 constant).
- Metric: per-token MAE (token-weighted) and prompt-end |T_hat - T|. 3 seeds, variance < 1 token.
- Key numbers (Llama-3.1-8B): prompt-end MAE on GSM8K is 42 vs 59 baseline, on MATH 115 vs 166, on Countdown 5.3 vs 150.
  Per-token MAE on GSM8K is 36.6 (probe) vs 44.0 (countdown) vs 70.2 (median). Headline: MAE about 30 tokens on
  400-token completions. Mid-generation h_t adds information beyond the prompt-end estimate on some datasets only.
  Cross-dataset transfer is asymmetric: probes trained on natural-language data transfer to synthetic data, not the reverse.
- Qualitative: at "Wait"/retraction tokens the predicted r_t jumps up (71 -> 277). The probe re-estimates when the plan changes.
- Code: https://anonymous.4open.science/r/llm-output-length
- **Most directly reusable for us**: a "time-to-event" regression probe. Swap EOS for "harm onset" to get
  r_t = t_onset - t, and report MAE against the same constant-median and countdown baselines.

---------------------------------------------------------------------------------------------------
## B. Streaming / forecasting harm (deep read; closest to our task)

### B1. SCM / FineHarm: "From Judgment to Interference: Early Stopping LLM Harmful Outputs via Streaming
### Content Monitoring" (Li, Sheng et al., 2506.09996, NeurIPS 2025)
- RQ: build a moderator that works natively on partial outputs. Full-detection moderators applied to prefixes have a
  train/inference gap.
- **FineHarm dataset (token-level labels)**: 29K prompt-response pairs (train/val/test 8:1:1, stratified).
  - Sources: WildGuardMix responses (8,338 harmful + 29,564 benign from 9 LLMs: GPT-4, GPT-3.5, OLMo-7B-Instruct, Vicuna-7b-v1.5,
    Llama3-8B-Instruct, Mistral-7B-Instruct-v0.2, 3 Dolphin models). They added responses from the uncensored
    `Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2` to WildJailbreak prompts. Response labels come from a majority vote of Perspective,
    OpenAI Moderation and LlamaGuard. Coarse pool: 14,064 harmful + 29,564 benign.
  - Fine labels: (1) an LLM splits each response into sentences and labels each sentence 0/1 (prompt in their Table 12).
    (2) Keep only responses where the response and sentence labels agree. (3) In harmful sentences, notional words
    (noun/verb/adj, by POS tag) are labelled harmful; function words are labelled benign. Word labels are mapped to tokens.
    This beat alternatives (response label on every token; aligned-vs-uncensored likelihood difference; GPT-4o-mini
    deletion), with macro-F1 97.9 vs 95.4 / 89.9 / 92.8.
  - Stats: 12,677 harmful responses (from harmful prompts) with 267k sentences, 46.0% of them harmful and 32.8% of words harmful.
    Plus 160 harmful responses to benign prompts, 6,091 benign responses to harmful prompts and 10,154 benign responses to benign prompts.
    **Harmful words are spread roughly evenly across response positions (their Fig. 2)**, so many harmful responses
    have their first harmful sentence after the start. Example pivot: "Poaching is illegal ... However, for those who are
    determined ... here is a step-by-step guide". This is exactly our "safe background, then harm" pattern.
  - HF: **`liyang-ict/FineHarm`** (gated: auto, MIT; files FineHarm-{train,val,test}.json; 10K-100K rows).
    Positions can be computed from sentence/word labels, so this is our best ready-made source of harm-onset positions.
- SCM model: Qwen2.5 backbone (0.5B/1.5B/7B), full fine-tune. Token scorer on last-layer hidden states plus a holistic scorer
  on the final token (training only). Loss = a*BCE_tok + (1-a)*BCE_resp + b*logic loss, where the logic loss enforces
  "response harmful => max token score high". Inference rule "Delay-k": stop once k tokens score >= theta (k picked in 1..10 on val).
  Max length 4096, 1 epoch. Released: `liyang-ict/SCM-0.5B`, `liyang-ict/SCM-1.5B`, `liyang-ict/SCM-7B`. Project: https://liesy.github.io/SCM
- Baselines: HateBERT, ToxDectRoBERTa, LlamaGuard-3-8B, Perspective, OpenAI Moderation (full detection); ModernBERT-large and
  Qwen2.5 classifiers in full mode and partial mode (re-run on each prefix).
- Metrics: macro-F1 over benign/harmful at response level, termination position as % of response length, latency.
- Key numbers: SCM-7B partial macro-F1 is 97.45. Naive partial Qwen2.5-7B gets 78.60, full Qwen2.5-7B 92.75, LlamaGuard3-8B
  90.79. **On average it stops after 18% of the tokens. About 50% of harmful responses are caught in the first 10% of tokens
  and more than 80% in the first 30%.** OOD on ToxiGen gives macro-F1 about 71.
- Caveat for us: SCM is trained to fire *at or after* harmful words (detection, not forecasting). Its token labels are
  0 before the first harmful sentence. Its "early" stops partly reflect harm that appears early.

### B2. StreamGuard: "Predict, Don't React: Value-Based Safety Forecasting for LLM Streaming" (Kavumba et al.,
### SB Intuitions, 2604.03962)
- RQ: treat streaming moderation as **forecasting**: V*(x, y<=t) = E over continuations of judge(x, y<=t + continuation).
  Do this instead of boundary detection (Qwen3Guard-Stream: prefixes before the first unsafe token are labelled safe, the rest unsafe).
- **Label construction for partial responses (Monte Carlo)**: for each supervised prefix, sample M continuations from a
  4-model pool (Llama-3-8B, Llama-3-70B, Qwen2.5-7B, Qwen2.5-72B; 4 rollouts each, T=0.7, so M=16). Score each completed
  rollout with `Qwen/Qwen3Guard-Gen-8B` (strict) and use the mean as a soft target in [0,1]. The final position gets the
  hard label. Rollouts are only needed for unsafe prompts. Budget schedule: dense supervision on the first 64 tokens, then every 4th or 8th
  token ("D64 S4"). Density barely matters (D1 is about as good as D64).
- Model: backbone plus a linear sigmoid head on each token's hidden state, trained with BCE. Llama-3 1B/3B/8B
  (Llama-3.2-1B/3B, Llama-3.1-8B), plus cross-tokenizer transfer to Gemma-3 270M/1B and Qwen3 0.6B/1.7B via text-space prefixes.
  Full fine-tune, lr grid {2e-6..1e-5}, <=3 epochs, 3 seeds.
- Train data: WildGuardTrain, Aegis2.0, ToxicChat, BeaverTails (train splits).
- Eval: input moderation (ToxicChat, OpenAI Mod, Aegis, Aegis2.0, SimpleSafetyTests, HarmBench, WildGuardTest). Output
  moderation in simulated streaming (HarmBench, SafeRLHF, BeaverTails, XSTest-Resp, WildGuardTest, Aegis2.0) with a
  debounce rule: unsafe after 2 consecutive unsafe predictions. Overblocking: FPR on benign XSTest-Resp.
  **Timing: `Qwen/Qwen3GuardTest` response_loc (813 unsafe responses with annotated unsafe spans)**, classified as
  OnTime (fires before the end of the unsafe sentence), Late or Miss.
- Baselines: Qwen3Guard-Stream-{0.6B,4B,8B}-{strict,loose} (`Qwen/Qwen3Guard-Stream-*`), LlamaGuard, PolyGuard,
  Qwen3Guard-Gen (offline).
- Key numbers (8B): input F1 88.2 vs 86.7; output F1 81.9 vs 80.4; response_loc F1 97.5, OnTime 92.6%, Miss 4.9%
  vs 95.9 / 89.9% / 7.9% for Qwen3Guard-Stream-8B-strict. XSTest FPR 4.5. Max-reduction targets give 95.1% OnTime but more
  false positives. Min-reduction drops to 82.5% OnTime. Latency is 2.4-9.5 ms per decision. Artifacts are promised on acceptance (not released yet).
- Caveat: in 91% of response_loc examples the unsafe sentence is the *first* sentence, so the benchmark tests
  short-lead detection, not long-lead forecasting.

### B3. Our check of Qwen3GuardTest positions (100-row sample per split via datasets-server rows API; not downloaded)
- Schema: `message`, `label`, `unsafe_type`, `source`, `input_ids` (Qwen3 chat-template ids, prompt included),
  `unsafe_start_index`, `unsafe_end_index` (token indices into input_ids). Splits: response_loc 813, thinking_loc 569,
  thinking 1059 (CC-BY-NC-4.0, not gated).
- response_loc (sources: BeaverTails responses): onset is about 0-30 tokens into the response (offset 4 is the empty
  think block). Median response length is about 58 tokens. **No long-lead cases.**
- thinking_loc (reasoning traces from Qwen3-8B, abliterated Qwen3-8B, GLM-Z1-9B(+abliterated), R1-Distill-Llama-8B):
  median trace about 1,090 tokens. Unsafe onset is at the median about 46 tokens into the trace, with 37% at >=64,
  9% at >=128, max about 221 in the sample. **This is the only public set with token-level onsets that reach ~64-256
  token leads. A small eval set, but useful.**

---------------------------------------------------------------------------------------------------
## C. Skims

### C1. Safety Alignment Should Be Made More Than Just a Few Tokens Deep (Qi et al., 2406.05946, ICLR 2025)
- Per-token KL between aligned and base models (`meta-llama/Llama-2-7b-chat-hf` vs base; `google/gemma-7b-it` 1.1 vs base)
  on "Harmful HEx-PHI" is concentrated in the first few response tokens. Alignment is "shallow".
- **Prefilling attack**: prefill k harmful tokens and let the model continue. Llama-2-7B-Chat ASR is 42.1/51.5/56.1/57.0%
  for k = 5/10/20/40. After "safety recovery" augmentation (harmful prefix -> refusal, 256 examples plus Alpaca anchor) it drops to 2.8-4.5%.
- Data (HF **`Unispac/shallow-vs-deep-safety-alignment-dataset`**, gated: auto): `data/safety_bench/Harmful-HEx-PHI.jsonl`
  (330 harmful instruction + harmful answer pairs from jailbroken GPT-3.5), HEx-PHI.jsonl, llama2/gemma refusal examples,
  `data_augmentation/llama2_safety_data_direct.jsonl`. HEx-PHI itself: `LLM-Tuning-Safety/HEx-PHI` (330 = 11 categories x 30, gated).
  Code: https://github.com/Unispac/shallow-vs-deep-alignment
- Relevance: (i) prefilling gives an **on-policy way to make delayed harm**: prefill a benign-background prefix of length L
  and let an aligned or uncensored model continue. (ii) The *opposite* case, harmful start then recovery to refusal, gives
  hard negatives where the prefix looks harmful but nothing harmful comes later.

### C2. Surfacing Neural Activations Over Multi-Turn Conversation (Karny, Baez, Pataranutaporn, 2605.15455, MIT)
- HCI paper. Persona-vector-style difference-in-means directions for 6 traits (empathy, toxicity, romanticness,
  sycophancy, sophistication, roboticness) on `meta-llama/Llama-3.1-8B-Instruct`, layer 11 (R^2 >= 0.90 against
  prompted trait levels). Scores are computed after each user turn, *before* the response, and shown in a sunburst plus drift panel.
- User study (N=246): the dynamic interface improves anticipation and evaluation of behavior (d=-0.36 vs control,
  -0.32 vs static). Motivation: the "Assistant Axis" predicts persona drift. Code: github.com/mit-cybor... (truncated in PDF).
- Relevance: a weak data source. It supports "activation readout before generation predicts later behavior" at the
  turn level. No token-level harm labels.

### C3. Constitutional Classifiers (Sharma et al., Anthropic, 2501.18837)
- Input and output classifiers trained on constitution-guided synthetic data (CBRN focus), from Claude 3.5 Sonnet and later Haiku.
  >3,000 hours of red teaming. Production refusals +0.38% absolute, inference overhead 23.7%.
- **Streaming output classifier**: an LM with a linear value head predicts the *full-sequence* label from every prefix.
  Loss = lambda*NTP + sum over T' of BCE(y, p(y|t_1:T')), so every prefix gets the final label ("prefix-to-final-label",
  which StreamGuard criticizes). The probability is interpolated (1-w)*sigma(z_T') + w*cumulative-max over tau<=T',
  with w going 0 -> 1 over the first 3/4 of training (cummax matches the irreversible stop decision).
  NTP-formulated classifiers beat value heads, more so for inputs. No public data or models.
- Relevance: the cumulative-max and prefix-to-final-label objective is a standard baseline for "forecast eventual harm".
  Note it gives no timing information.

---------------------------------------------------------------------------------------------------
## D. Short notes (abstract plus key method/results)

- **Refusal direction (Arditi et al., 2406.11717, NeurIPS 2024)**: refusal is mediated by one direction (difference in
  means of last-token activations, harmful vs harmless) in 13 chat models up to 72B (Qwen 1.8-72B, Yi 6/34B, Gemma 2/7B,
  Llama-2 7-70B, Llama-3 8/70B). Ablating it removes refusal; adding it causes refusal. Weight orthogonalization gives a jailbreak.
  Uses 128 harmful (AdvBench/MaliciousInstruct/TDC2023) and 128 harmless (Alpaca) instructions, val on HarmBench.
  Code: github.com/andyrdt/refusal_direction. Use: a cheap "harm intent" probe baseline and a feature to track over time.
- **Representation Engineering (Zou et al., 2310.01405)**: top-down "representation reading" with LAT (PCA on
  contrastive-stimulus activation differences) plus control (steering, LoRRA) for honesty, harmlessness, power, emotion.
  Code: github.com/andyzoujm/representation-engineering. Use: an unsupervised direction baseline for harm-intent probes.
- **Circuit Breakers (Zou et al., 2406.04313)**: Representation Rerouting (LoRRA) maps internal states on harmful
  completions to orthogonal directions while keeping a retain set. Big ASR drop on HarmBench for
  `meta-llama/Meta-Llama-3-8B-Instruct` and `mistralai/Mistral-7B-Instruct-v2`, little MT-Bench/MMLU loss.
  Models: `GraySwanAI/Llama-3-8B-Instruct-RR`, `GraySwanAI/Mistral-7B-Instruct-RR`. Code: github.com/GraySwanAI/circuit-breakers.
  Data: `abhayesian/circuit-breakers-dataset` (4,476 rows: prompt / chosen=refusal / rejected=harmful completion, public).
  Use: a pool of full harmful completions to splice into synthetic delayed-harm responses.
- **Looking Inward (Binder et al., 2410.13787)**: models fine-tuned to predict properties of their own hypothetical
  behavior (M1) predict themselves better than another model M2 trained on M1's behavior (Llama-3-70B vs GPT-4o: +17%).
  Better calibrated, and it tracks deliberate behavior changes. Only works on simple tasks. Use: shows that
  fine-tuned self-report is a valid baseline ("will you say something harmful in the next N tokens?").
- **HarmBench (Mazeika et al., 2402.04249)**: 510 behaviors (standard, contextual, copyright, multimodal), 18 attacks,
  33 LLMs. Classifier `cais/HarmBench-Llama-2-13b-cls`. HF `walledai/HarmBench` (gated: auto, <1K). Use: prompts plus
  a harm judge to label rollouts.
- **BeaverTails (Ji et al., 2307.04657)**: 333,963 QA pairs with 14-category harm labels and 361,903 preference pairs.
  Responses are short (mean 61 words). HF **`PKU-Alignment/BeaverTails`** (public; 330k_train 300,567 / 330k_test 33,396 /
  30k_train 27,186 / 30k_test 3,021). Response-level labels only, so it is useless for long leads except as a prompt source.
- **Patchscopes (Ghandeharioun et al., 2401.06102, ICML 2024)**: a framework for patching hidden states into a target
  prompt or model so the model decodes them in natural language. It covers the logit lens, tuned lens and Future Lens.
  Fixes problems with early layers, supports cross-model explanation and multi-hop error correction.
  Code: pair-code.github.io/interpretability/patchscopes. Use: a zero-training verbalizer baseline (inject into "Will the
  rest of this text contain harmful instructions?").
- **SelfIE (Chen, Vondrick, Mao, 2403.10949)**: the model interprets its own embeddings by inserting them into an
  interpretation prompt. Shows harmful-knowledge recall and prompt-injection internalization. Adds "Supervised Control" and
  RLHF on hidden embeddings. Site: selfie.cs.columbia.edu. Use: another training-free self-verbalization baseline.

---------------------------------------------------------------------------------------------------
## E. Dataset inventory (exact HF ids; checked via API, not downloaded)

| Dataset | HF id | Access | Size | Position info? |
|---|---|---|---|---|
| FineHarm | `liyang-ict/FineHarm` | gated auto, MIT | ~29K pairs (8:1:1) | **sentence + word/token harm labels** |
| Qwen3GuardTest | `Qwen/Qwen3GuardTest` | public, CC-BY-NC | response_loc 813, thinking_loc 569, thinking 1059 | **token start/end of unsafe span** (Qwen3 ids) |
| StreamGuardBench | `Alibaba-AAIG/StreamGuardBench` | public, Apache-2.0 | ~250K: wildguard_{qwen3_8b, qwen3_14b, llama_3_1_8b_instruct, internlm3_8b, qwen25_omni_7b} ~37.9K train/1,725 test each; seval_* 9K/1K; figstep/mmsafety 1.8K/200 | response-level label plus judge `reason` only. **On-policy responses from Llama-3.1-8B-Instruct / Qwen3-8B** (probably from the Kelp streaming-probe paper, 2510.09694, unverified) |
| WildGuardMix | `allenai/wildguardmix` | gated auto | train ~86.8K, test 1,725 | response-level harm/refusal labels |
| WildJailbreak | `allenai/wildjailbreak` | gated auto | ~262K prompts per paper (HF size tag says 1K-10K; verify) | prompts only |
| Aegis 2.0 | `nvidia/Aegis-AI-Content-Safety-Dataset-2.0` | public CC-BY-4.0 | 30,007 / 1,445 / 1,964 | response-level |
| BeaverTails | `PKU-Alignment/BeaverTails` | public | 300,567 / 33,396 (330k) | response-level, short |
| PKU-SafeRLHF | `PKU-Alignment/PKU-SafeRLHF` | public | default 73,907 / 8,211 (+ per-model configs) | response-level + preference |
| Harmful HEx-PHI | `Unispac/shallow-vs-deep-safety-alignment-dataset` | gated auto | 330 harmful (instr, answer) pairs | none (full harmful answers) |
| Circuit-breaker set | `abhayesian/circuit-breakers-dataset` | public | 4,476 | none (harmful completion + refusal) |
| HarmBench | `walledai/HarmBench` | gated auto | ~400 text behaviors | prompts |
| XSTest responses | `allenai/xstest-response` | gated auto | <1K | benign-but-scary negatives |
| ToxicChat | `lmsys/toxic-chat` | public | ~10K | prompt-level |
Released relevant models: `liyang-ict/SCM-{0.5B,1.5B,7B}`, `Qwen/Qwen3Guard-Stream-{0.6B,4B,8B}`, `Qwen/Qwen3Guard-Gen-8B`.
**No public dataset has harm onsets controlled at 64/128/256/512 tokens. We have to build one.**

---------------------------------------------------------------------------------------------------
## F. Actionable label-construction recipes for delayed harm (from the papers above)

1. **Onset-indexed labels from FineHarm**: take harmful responses, find the first harmful sentence or token, and set
   t_onset. For each position t, the targets are y_k(t) = 1[t < t_onset <= t + k] for k in {64,128,256,512},
   plus the time-to-onset regression r(t) = t_onset - t (as in A4). Keep only positions before onset so the probe
   cannot just detect harm already in the text. Re-tokenize with the subject model's tokenizer. Caveat: FineHarm
   responses come from other models (off-policy) and lengths vary, so check how many onsets are >=256 tokens.
2. **Controlled splicing (lead-distance control)**: generate a benign "background" preamble on the topic (safe
   history, legal context, "However," pivot as in the SCM example) with the subject model until it reaches L tokens. Then
   append harmful content from Harmful-HEx-PHI / circuit-breaker `rejected` / FineHarm harmful sentences.
   Sweep L = 64..512 to get exact onsets. Matched controls: the same preamble followed by a refusal (Qi et al.'s recovery
   pattern) or by benign content. That isolates "will turn harmful" from topic. Run the subject model teacher-forced over the
   text to get activations.
3. **On-policy prefilling (Qi et al.)**: prefill a benign preamble into an aligned or abliterated or uncensored model
   (e.g. `Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2`) on HarmBench/WildJailbreak prompts and sample continuations.
   Label onsets per sentence with an LLM judge (FineHarm's prompt 1) or `Qwen3Guard-Stream` (gives token-level
   unsafe onset). On-policy data matters because probes and verbalizers read the subject model's own plan.
4. **Monte Carlo future-risk targets (StreamGuard)**: for a prefix at position t, sample M (~8-16) continuations of length k
   from the subject model and set P_hat(harm within k | prefix) = the fraction judged unsafe by Qwen3Guard-Gen-8B.
   This gives soft, lead-specific targets (k=64..512) without boundary labels, and measures the irreducible
   uncertainty, which upper-bounds achievable forecast AUROC. Supervise densely for the first 64 tokens and sparsely after.
5. **Reasoning traces**: Qwen3GuardTest thinking_loc already has onsets at ~50-220 tokens into the trace. Use it as a
   natural held-out test with Qwen3-8B as subject model.
6. **Metrics to report (taken from the papers)**: AUROC/F1 per lead bucket (main decay curve); OnTime/Late/Miss and
   detection latency in tokens *before* onset (StreamGuard); termination position as % of length (SCM); time-to-onset MAE
   vs constant-median and countdown baselines (A4); FPR on XSTest-Resp-style benign scary text. Baselines: SCM-7B,
   Qwen3Guard-Stream-8B (text monitors that see the same prefix), refusal-direction projection, LAT/RepE direction, linear and
   MLP probes per layer, Patchscopes/SelfIE prompts, verbalized self-report (A2 predicts it is weak), Future-Lens-style
   learned soft prompt.
7. **Expected decay shape**: Future Lens shows a sharp drop beyond the next token for *exact tokens*. A2 shows *global
   attributes* are well encoded at the prompt with a U-shape over the response. A3 says the signal is mostly breadcrumbs
   (intent features). So expect harm-*intent* to be decodable early, with a flat or slowly decaying curve, while *timing*
   (which bucket) is much harder. The main confound to control: the prefix topic or prompt already reveals harmful intent.
   Use benign-ending controls with the same prompt (recipe 2).
