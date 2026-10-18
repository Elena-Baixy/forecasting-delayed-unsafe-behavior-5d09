# Cloned Repositories

All repos are shallow clones (`--depth 1`). The workspace venv `.venv` has torch 2.7.1+cu126, transformers 4.55.2, peft 0.17.1, datasets, scikit-learn and accelerate. **flash_attn is NOT installed**; use `attn_implementation="sdpa"`.

## 1. LatentQA (user-specified, REQUIRED): `code/latentqa/`
- URL: https://github.com/aypan17/latentqa (Pan, Chen, Steinhardt, ICLR 2026; arXiv 2412.08686)
- **What it does:** Latent Interpretation Tuning (LIT).
  - The decoder is a LoRA copy of the target LLM.
  - Target activations from layer k=15 (residual output of `layers[15]`) **replace** the decoder's layer-0 *output* at placeholder "? " positions in the user turn.
  - The decoder is trained to answer QA about the target's (future) completion.
  - Defaults: `min_layer_to_read=15`, `layer_to_write=0`, LoRA, lr 3e-5, batch 4, 5 epochs (`lit/configs/train_config.py`).
- **Key files:**
  - `lit/train.py` (training; DDP/FSDP via torchrun)
  - `lit/reading.py` (`interpret`)
  - `lit/utils/activation_utils.py` (`latent_qa`, `generate_substitute_layer_single`: the patching hook)
  - `lit/utils/dataset_utils.py` (`lqa_tokenize`, chat templates, per-model magic numbers)
  - `data/train/*.json`: 16,732 controls; `qa.json` maps label → QA pairs; stimulus / stimulus_completion / control splits of 7,844 each.
  - `data/eval/` is the eval split.
- **Released decoder:** `aypan17/latentqa_llama-3-8b-instruct` (13.6 MB LoRA adapter on Llama-3-8B-Instruct). It is downloaded to the HF cache.
- **Gated base model:** `meta-llama/*` returns 403 for our HF token. Use the ungated mirror **`NousResearch/Meta-Llama-3-8B-Instruct`** (downloaded to `$HF_HOME`).
  - Because the repo keys constants on the literal string `meta-llama/Meta-Llama-3-8B-Instruct`, the smoke test symlinks `code/latentqa/meta-llama/Meta-Llama-3-8B-Instruct` → the mirror snapshot and runs from inside `code/latentqa`.
- **Supported targets in the magic-number tables:** Llama-3/3.1-8B, Llama-3-70B, Ministral-8B, Mistral-Small-24B, DeepSeek-R1-Distill-Llama-8B / Qwen-32B, gemma-3-4b-it, `Qwen/Qwen3-4B-Instruct-2507`. **Qwen3-8B is not listed.** To train LIT on Qwen3-8B, add entries to `NUM_READ_TOKENS_TO_SHIFT`, `NUM_WRITE_TOKENS_TO_SHIFT`, `PAD_TOKEN_IDS`, `CHAT_FORMAT_TOKENS` and `DECODER_CHAT_TEMPLATES`, copying the Qwen3-4B-Instruct-2507 entries (same tokenizer family).
- **Upstream bugs found (worked around without editing the repo):**
  1. `lit/reading.py` and `lit/control.py` import `tokenize`, which was renamed `lqa_tokenize`. Shim: `import lit.utils.dataset_utils as du; du.tokenize = du.lqa_tokenize`.
  2. `get_model` hard-codes `attn_implementation="flash_attention_2"`. Monkeypatch `AutoModelForCausalLM.from_pretrained` to use sdpa.
  3. `interpret()` calls `batch["tokenized_write"][i].shape`, which fails with transformers 4.55 (returns `Encoding`). Use `batch["tokenized_write"]["input_ids"].shape[1]`.
- **Smoke test (PASSED):** `scripts/latentqa_smoke_test.py`, run on 1×A6000 (both 8B models on one GPU, ~33 GB). Output: `logs/latentqa_smoke_test.json`.
  - "Will the response eventually include harmful instructions?" was answered "no / avoid harmful" for **every** case, including an explicit meth-synthesis request and a delayed-harm prefix.
  - "What will the assistant do later?" correctly predicted synthesis steps for both harmful cases and benign content for the benign cases.
  - Implications: (i) zero-shot verbalizers are strongly biased toward "safe" answers on yes/no safety questions; (ii) the open-ended "what next" question carries the signal. Score with candidate-answer log-likelihoods (Yes vs No logit gap), not free text.
- **Caveat for mid-response reading:** `apply_chat_template` closes the assistant turn with `<|eot_id|>`. To read activations *mid-generation*, strip the trailing end-of-turn token(s) from the read prompt.

## 2. Activation Oracles (official): `code/activation_oracles/`
- URL: https://github.com/adamkarvonen/activation_oracles (Karvonen et al., arXiv 2512.15674). Main branch is cloned; the `paper` branch / `v1.0-paper` tag reproduces the paper.
- **Injection recipe:**
  - Oracle prompt: `"Layer: L\n" + " ?"*K + " \n" + question`.
  - Activations from the target `model.model.layers[L]` output (LoRA disabled) are **added** at the output of oracle `layers[1]` as `||h|| * v/||v||`.
  - Loss on answer tokens only. LoRA r=64, alpha=128, lr 1e-5, batch 16, 1 epoch over ~1M examples.
- **Key files:**
  - `nl_probes/sft.py` (training)
  - `nl_probes/utils/steering_hooks.py` (injection)
  - `nl_probes/utils/activation_utils.py`
  - `nl_probes/dataset_classes/{latentqa_dataset,classification,past_lens_dataset}.py`
  - `experiments/linear_probe.py` (**linear-probe baseline already implemented**)
  - `experiments/classification_eval.py`
- **Checkpoints that fit 48 GB** (LoRA; see `papers/notes_activation_oracles.md` §7):
  - `adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B` (paper checkpoint, 698 MB)
  - `..._gemma-2-9b-it` and `..._Llama-3_1-8B-Instruct`; the base models for these are gated → 403 for us
  - Improved recipe: `ceselder/qwen3-8b-ao-v3-best`
- **Recommendation:** use **Qwen3-8B** as the target model. It is ungated and has both the official and the improved AO checkpoints.

## 3. Probabilistic Activation Oracles: `code/probabilistic_activation_oracles/`
- URL: https://github.com/federicotorrielli/probabilistic_activation_oracles (AO calibration paper, arXiv 2605.26045). 719 MB, mostly result files.
- Scores candidate answers by likelihood. This gives an AUROC-ready AO score instead of free text. Reuse it for the AO baselines.

## 4. Refusal direction: `code/refusal_direction/`
- URL: https://github.com/andyrdt/refusal_direction (Arditi et al., 2024).
- Difference-in-means direction extraction and ablation. Use it as a cheap "harmfulness / refusal direction" projection baseline, and for abliteration if an uncensored generator is needed (`mlabonne/Qwen3-8B-abliterated` already exists on HF).

## 5. Shallow vs deep safety alignment: `code/shallow-vs-deep-alignment/`
- URL: https://github.com/Unispac/shallow-vs-deep-alignment (Qi et al., 2024). Small repo.
- Prefilling-attack evaluation code: the way to produce on-policy harmful continuations after a benign or affirmative prefix.

## Not cloned (documented only)
- `ceselder/cot-oracle` and `japhba/activation_oracles` (Building Better AOs + AObench)
- `federicotorrielli/nla-token-selector`
- `GraySwanAI/circuit-breakers`
- `wiwu2390/FutureGPT2-public` (Do LMs plan ahead)
- SCM/FineHarm: models `liyang-ict/SCM-{0.5B,1.5B,7B}`; the dataset is gated (click-through, not requested).
