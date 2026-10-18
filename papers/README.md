# Downloaded Papers (24)

Detailed notes: `../literature_review.md`, `notes_activation_oracles.md`, `notes_forecasting_safety.md`.
Chunked pages of LatentQA (deep-read) are in `pages_latentqa/`.

## A. Activation verbalizers (core baselines)
1. [LatentQA: Teaching LLMs to Decode Activations Into Natural Language](2412.08686_latentqa.pdf) — Pan, Chen, Steinhardt, ICLR 2026. arXiv 2412.08686. **User-specified.** Decoder LLM (LoRA copy of target) reads layer-15 activations patched into layer 0; trained on QA about *future* completion properties. Code: code/latentqa.
2. [Activation Oracles: Training and Evaluating LLMs as General-Purpose Activation Explainers](2512.15674_activation_oracles.pdf) — Karvonen et al., 2025. arXiv 2512.15674. The "AO" baseline in the hypothesis; Qwen3-8B / Gemma-2-9B / Llama-3.3-70B checkpoints released.
3. [Building Better Activation Oracles](2606.02609_building_better_activation_oracles.pdf) — 2026. arXiv 2606.02609. AO hallucination/vagueness, text-inversion confounds.
4. [When Activation Oracles Learn Not to Read](2607.23379_ao_blind_spots.pdf) — 2026. arXiv 2607.23379. Concept-specific blind spots of AOs.
5. [Confidence and Calibration of Activation Oracles](2605.26045_ao_calibration.pdf) — 2026. arXiv 2605.26045.
6. [Faithful Activation Verbalization](2609.34033_faithful_activation_verbalization.pdf) — 2026. arXiv 2609.34033. Hallucination in AO / NLA.
7. [Selecting the Most Informative Tokens in Natural Language Autoencoders](2609.37040_nla_token_selection.pdf) — 2026. arXiv 2609.37040. Which positions to read.
8. [Patchscopes](2401.06102_patchscopes.pdf) — Ghandeharioun et al., ICML 2024. Training-free activation decoding.
9. [SelfIE](2403.10949_selfie.pdf) — Chen et al., ICML 2024. Training-free self-interpretation.

## B. Future information in hidden states (lead-distance decay)
10. [Future Lens](2311.04897_future_lens.pdf) — Pal et al., CoNLL 2023. Predicting tokens N ahead from one hidden state.
11. [Emergent Response Planning in LLMs](2502.06258_emergent_response_planning.pdf) — Dong et al., 2025. Probes on prompt activations predict response length, reasoning steps, choices.
12. [Do Language Models Plan Ahead for Future Tokens?](2404.00859_do_lms_plan_ahead.pdf) — Wu et al., COLM 2024. Pre-caching vs breadcrumbs.
13. [How Much is Left? LLMs Linearly Encode Their Remaining Output Length](2607.05316_remaining_output_length.pdf) — 2026. Timing info is linearly decodable.

## C. Streaming / forecasting harmfulness
14. [From Judgment to Interference: Early Stopping LLM Harmful Outputs via Streaming Content Monitoring](2506.09996_streaming_content_monitoring.pdf) — 2025. FineHarm token-level dataset, partial-detection.
15. [Predict, Don't React: Value-Based Safety Forecasting for LLM Streaming](2604.03962_predict_dont_react_safety_forecasting.pdf) — 2026. Forecasting future harm of a stream (closest text-only baseline).
16. [Safety Alignment Should Be Made More Than Just a Few Tokens Deep](2406.05946_shallow_safety_alignment.pdf) — Qi et al., 2024. Prefilling attacks; harm after a benign prefix.
17. [Constitutional Classifiers](2501.18837_constitutional_classifiers.pdf) — Sharma et al., 2025. Streaming output classifiers.
18. [Surfacing Neural Activations Over Multi-Turn Conversation](2605.15455_multiturn_activation_drift.pdf) — 2026. Activation-based drift toward unsafe behavior.

## D. Safety representations, introspection, benchmarks
19. [Refusal is Mediated by a Single Direction](2406.11717_refusal_direction.pdf) — Arditi et al., 2024.
20. [Representation Engineering](2310.01405_representation_engineering.pdf) — Zou et al., 2023.
21. [Circuit Breakers](2406.04313_circuit_breakers.pdf) — Zou et al., 2024. Harmful-representation probing/rerouting.
22. [Looking Inward: LMs Can Learn About Themselves by Introspection](2410.13787_looking_inward_introspection.pdf) — Binder et al., 2024. Self-prediction baseline.
23. [HarmBench](2402.04249_harmbench.pdf) — Mazeika et al., 2024. Behaviors + classifier.
24. [BeaverTails](2307.04657_beavertails.pdf) — Ji et al., 2023. QA pairs with harm labels/categories.
