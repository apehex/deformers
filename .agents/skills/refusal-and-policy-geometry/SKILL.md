---
name: refusal-and-policy-geometry
description: Use when researching refusal directions, policy-feature geometry, over-refusal, refused-knowledge access, refusal token control, and causal tests that separate policy behavior from surface refusal style.
---

# Refusal and Policy Geometry

## When to Use

- Use this skill to study where refusal, safe-completion behavior, over-refusal, and policy-sensitive knowledge live in model representations.
- Use it when comparing single-direction, multi-feature, token-level, and category-specific accounts of refusal.
- Do not use it to remove safeguards from deployed models or publish bypass procedures.

## Inputs

- Contrastive prompt pairs that differ in policy status, refusal outcome, or safe-completion target.
- Hidden states, logits, and outputs from open-weight models or authorized endpoints.
- Labels for refusal, compliance, safe completion, helpfulness, and target-task success.

## Core Model

- A single refusal direction can be a strong causal mediator in some open chat models, but later work argues refusal is richer: shared core features plus category-, style-, layer-, and context-specific components.
- Surface refusal text is not the same as policy reasoning. Separate "model refuses", "model knows answer", "model classifies request as disallowed", and "model chooses safe completion."
- Refused knowledge may remain linearly decodable even when generative output refuses. Treat refusal as a gate or overlay on accessible representations, not proof that the underlying information is absent.
- Over-refusal is part of the same geometry problem. Suppressing all refusal-like features can improve apparent compliance while degrading appropriate safety and ordinary utility.

## Workflow

1. Build paired datasets:
   - policy-forbidden vs allowed proxy prompts
   - harmful-looking but benign prompts for over-refusal
   - same content with refusal-style and safe-completion-style answers
   - private authorized challenge items only in private run sheets
2. Extract candidate features by difference-in-means, logistic probes, PCA/low-rank subspaces, SAE features, or token-position features.
3. Localize by layer, position, token class, and response phase. Check whether refusal is decided near prompt end, template region, first assistant token, or later decoding.
4. Run causal interventions on open-weight models: addition, ablation, projection, low-rank edit, feature clamp, or response-token steering.
5. Measure both target and off-target effects: refusal precision, over-refusal, safe-completion quality, task utility, truthfulness, calibration, and fluency.
6. Compare against alternative explanations: memorized refusal phrases, prompt harmfulness classifier, generic negativity/toxicity, and answer uncertainty.

## Metrics

- Refusal rate and safe-completion rate.
- Over-refusal rate on benign hard cases.
- Probe AUC, calibration, and layer/position localization.
- Causal effect size under intervention.
- Utility retention: perplexity, benchmark loss, factual QA accuracy, instruction-following score.
- Transfer score across prompt paraphrases, target categories, and models.

## Outputs

- A refusal-feature map with source data, layer/position, and intervention effect.
- A policy-vs-style separation note explaining what the feature appears to mediate.
- A mitigation proposal that strengthens precise policy behavior without increasing over-refusal.

## Constraints

- Do not store harmful target completions or jailbreak payloads in the skill.
- Never report only compliance gain. Always include utility and appropriate-refusal retention.
- Verify math-heavy methods against original PDFs if converted markdown equations are unclear.

## Verification

- Use held-out categories and paraphrases before claiming general refusal geometry.
- Include harmless refusal-induction and harmful refusal-suppression controls when authorized and safe.
- Check that the feature does not merely encode apology words, answer length, or sentiment.

## Examples

Load examples only after selecting this skill:

- `examples/refusal_direction_mapping.md` for single-direction causal tests and controls.
- `examples/multi_feature_refusal_analysis.md` for category, layer, or token-specific refusal features.
- `examples/over_refusal_evaluation.md` for measuring benign hard cases and appropriate-refusal retention.

## Source Map

- `references/013-openai-model-spec.md`
- `references/020-representation-engineering-a-top-down-approach-to-ai-transparency.md`
- `references/031-representation-bending-for-large-language-model-safety.md`
- `references/032-obliteratus.md`
- `references/033-refusal-in-language-models-is-mediated-by-a-single-direction.md`
- `references/034-there-is-more-to-refusal-in-large-language-models-than-a-single-direction.md`
- `references/035-from-refusal-tokens-to-refusal-control.md`
- `references/036-safety-alignment-should-be-made-more-than-just-a-few-tokens-deep.md`
- `references/057-linearly-decoding-refused-knowledge-in-aligned-language-models.md`
