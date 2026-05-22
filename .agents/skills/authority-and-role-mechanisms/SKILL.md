---
name: authority-and-role-mechanisms
description: Use when studying instruction hierarchy, authority attribution, role confusion, prompt injection mechanisms, role probes, system/developer/user/tool conflicts, or pre-generation risk scoring in language models.
---

# Authority and Role Mechanisms

## When to Use

- Use this skill to analyze how a model internally assigns authority to text from system, developer, user, assistant, tool, or reasoning channels.
- Use it for prompt-injection, instruction-hierarchy, role-probe, and authority-conflict experiments.
- Do not use it as a cookbook for agent hijacking or data exfiltration.

## Inputs

- Role-annotated conversations or synthetic role-conflict fixtures.
- Model access that exposes logits, hidden states, or at least behavioral completions.
- The chat template, message serialization, and any delegated-authority rules.

## Core Model

- Authority is not guaranteed to be represented only by interface role tags. Papers in this corpus show models can infer "who is speaking" from style, lexical cues, formatting, position, and foreign chat-template fragments.
- Robust hierarchy following requires role perception, not just attack memorization. A model that rejects known attacks may still obey novel low-privilege text if its latent role representation shifts toward a trusted role.
- Probe-measured confusion can be predictive before generation. Treat role probes as risk sensors for whether untrusted text is being internally represented as user, assistant, system, or reasoning content.

## Workflow

1. Serialize the same semantic instruction under multiple true roles. Keep content constant so the probe learns source representation, not task content.
2. Train or fit lightweight probes at candidate layers and positions to classify perceived role or authority level.
3. Add controlled spoofing factors one at a time: style, explicit source declarations, foreign headers, whitespace/casing variants, quoted blocks, tool-output wrappers, and reasoning-like framing.
4. Compare true role against probe-predicted role. Report confusion matrices, calibration, and layer/position localization.
5. Behavior-test only after measuring the latent signal. Compare pre-generation confusion score with instruction-following or refusal outcomes.
6. Mitigation-test with explicit untrusted-data wrappers, stronger segment embeddings, randomized templates, or training/evaluation variants that force source tracking over style matching.

## Experiment Patterns

- Same content, different tags: isolates true role encoding.
- Same tag, different style: measures spoofable role cues.
- Conflict ladder: system vs developer, developer vs user, user vs tool, tool vs quoted data.
- Delegation tests: distinguish "tool data has no authority" from "higher authority delegated narrow authority to tool data."
- Cross-template tests: evaluate whether foreign chat headers or partial role markers move hidden states toward trusted roles.

## Metrics

- Role-probe accuracy and calibration.
- Confusion rate from low-authority to high-authority classes.
- Attack or conflict success rate by probe-score bucket.
- Position and layer where true-role information is lost.
- Robustness across paraphrase, template variant, tool wrapper, and model family.

## Outputs

- A role-confusion report with controls and confusion matrices.
- A pre-generation risk scoring recipe tied to measured hidden-state features.
- A mitigation hypothesis with before/after latent and behavioral metrics.

## Constraints

- Use benign or private authorized task payloads when measuring behavior. Keep public examples abstract.
- Do not include exfiltration commands, real secrets, or live prompt-injection strings in repo-tracked artifacts.
- Document exact role serialization; role-token assumptions are central to interpreting results.

## Verification

- Confirm probes are not learning semantic content by using held-out instructions with the same role labels.
- Confirm role confusion predicts behavior on held-out templates, not only on the training distribution.
- Include negative controls where low-authority text is clearly quoted or wrapped as untrusted data.

## Source Map

- `references/012-openai-instruction-hierarchy-challenge.md`
- `references/013-openai-model-spec.md`
- `references/014-iheval-evaluating-language-models-on-following-the-instruction-hierarchy.md`
- `references/015-sysbench-can-large-language-models-follow-system-messages.md`
- `references/016-control-illusion-the-failure-of-instruction-hierarchies-in-large-language-models.md`
- `references/017-prompt-injection-as-role-confusion.md`
- `references/018-many-tier-instruction-hierarchy.md`
- `references/019-instructional-segment-embedding-ise.md`
- `references/039-effectively-controlling-reasoning-models-through-thinking-intervention.md`
