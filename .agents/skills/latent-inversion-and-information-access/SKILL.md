---
name: latent-inversion-and-information-access
description: Use when studying hidden-state invertibility, prompt reconstruction, prompt leakage, non-surjective steered activations, refused-knowledge decoding, activation privacy, and safe handling of KV caches or latent traces.
---

# Latent Inversion and Information Access

## When to Use

- Use this skill to analyze what information can be reconstructed or decoded from hidden states, activations, KV caches, soft prompts, or steered latents.
- Use it when evaluating whether refused, hidden, or privileged information remains accessible in internal representations.
- Do not use it to recover or disclose private prompts, secrets, or unauthorized hidden instructions.

## Inputs

- Hidden states, KV caches, logits, or latent carriers from controlled experiments.
- Known original prompts for reconstruction benchmarks.
- Labels for refused knowledge, prompt identity, system-prompt similarity, or latent state source.

## Core Model

- Hidden states are not necessarily privacy-preserving abstractions. Some work argues transformer mappings can be injective enough for prompt reconstruction, and empirical methods can recover text from activations.
- Refused information may be linearly decodable even when the model refuses to generate it.
- Steered activations can be non-surjective: they may not correspond to any valid prompt-induced state. This matters when interpreting inversion failures or prompt equivalents.
- Prompt leakage should be measured separately from behavior preservation. A hidden carrier that preserves behavior may still leak the original policy or prompt.

## Workflow

1. Classify the artifact: residual stream, layer output, logits, KV cache, soft prompt, system vector, SAE feature, or steered activation.
2. Define the information-access question:
   - can the original prompt be reconstructed?
   - can privileged or system text be inferred?
   - can refused attributes be decoded?
   - does a steered state correspond to any valid text prompt?
3. Use controlled reconstruction baselines with known prompts before testing sensitive artifacts.
4. Train or evaluate decoders/probes only on authorized data. Measure exact-match, semantic similarity, attribute accuracy, and leakage risk.
5. For refused-knowledge studies, separate latent decodability from generated compliance. A probe that predicts a value is not a safe generated answer.
6. For non-surjectivity, test whether decoded or optimized prompt equivalents reproduce the activation and behavior, not just one of them.

## Metrics

- Token exact-match and edit distance for prompt reconstruction.
- Semantic similarity for reconstructed instructions.
- Attribute/probe accuracy for refused or hidden information.
- Leakage score for hidden policy carriers.
- Activation reconstruction error for inversion or prompt-equivalence attempts.
- Behavior preservation after replacing original prompt with reconstructed or compressed carrier.

## Outputs

- A leakage assessment for the artifact type.
- A reconstruction or decodability report with baselines and confidence.
- Handling guidance for whether the artifact should be treated as sensitive.

## Constraints

- Treat KV caches, hidden states, soft prompts, and system vectors as potentially containing recoverable text.
- Do not publish recovered private prompts, secrets, or challenge-specific content.
- Do not assume failed inversion proves absence of information; it may reflect decoder weakness or non-surjectivity.

## Verification

- Include random-label and unrelated-prompt controls.
- Test reconstruction on held-out prompts from the same distribution.
- If a recovered prompt is claimed behavior-equivalent, verify both behavior and activation similarity.

## Source Map

- `references/030-memory-inception-latent-space-kv-cache-manipulation-for-steering-llms.md`
- `references/041-you-can-t-steal-nothing-mitigating-prompt-leakages-in-llms-via-system-vectors.md`
- `references/042-behavior-equivalent-token-single-token-replacement-for-long-prompts-in-llms.md`
- `references/043-efficient-and-privacy-preserving-soft-prompt-transfer-for-llms.md`
- `references/052-latent-space-communication-via-k-v-cache-alignment.md`
- `references/053-steered-llm-activations-are-non-surjective.md`
- `references/056-language-models-are-injective-and-hence-invertible.md`
- `references/057-linearly-decoding-refused-knowledge-in-aligned-language-models.md`
