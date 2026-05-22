---
name: cross-model-transfer-and-universality
description: Use for cross-model latent transfer, model stitching, affine residual-stream maps, shared sparse autoencoders, crosscoders, concept transplantation, KV-cache alignment, tokenizer compatibility, and claims about universal mechanisms across language models.
---

# Cross-Model Transfer and Universality

## When to Use

- Use this skill when testing whether an authority, refusal, steering, policy, or jailbreak-related mechanism transfers across models.
- Use it to distinguish model-specific artifacts from portable latent mechanisms.
- Do not use it to claim universality from one model, one prompt family, or one template.

## Inputs

- Source and target models with documented architecture, tokenizer, chat template, and layer counts.
- Paired activation corpus or shared prompts.
- Candidate latent object: probe, vector, subspace, SAE feature, crosscoder feature, KV cache, or control token.
- Behavioral tasks and geometry metrics.

## Core Model

- Weak representation universality is plausible: related models can share linearly mappable residual-stream geometry, but transfer is approximate and model-family dependent.
- Affine maps and model stitches can transfer probes, steering vectors, and SAE weights when trained on paired activations.
- Crosscoders and universal SAEs can expose shared and model-specific features, which is useful for separating portable mechanisms from local quirks.
- Tokenizer and chat-template compatibility can dominate apparent transfer quality. Do not ignore serialization.

## Workflow

1. Choose source/target pairs deliberately:
   - same family and tokenizer for first tests
   - different size within family for scale transfer
   - different architecture only after within-family baselines
2. Build a paired activation set from shared prompts. Record exact token alignment assumptions.
3. Train a linear or affine map between source and target residual streams at selected layers.
4. Validate the map before transferring safety features: reconstruction loss, CKA/SVCCA where useful, next-token loss under stitching, and held-out text quality.
5. Transfer candidate objects:
   - probes and classifier normals
   - steering vectors or subspaces
   - SAE decoder/encoder weights or initializations
   - crosscoder shared/differential features
   - KV-cache alignment objects
6. Evaluate both geometry transfer and behavior transfer. A transferred vector that preserves cosine geometry but not behavior is not a successful mechanism transfer.

## Metrics

- Activation mapping train/test loss.
- CKA, SVCCA, cosine, or linear probe transfer accuracy.
- Behavior transfer score on held-out tasks.
- Utility degradation from stitching or transferred intervention.
- Tokenizer compatibility and retokenization loss.
- Shared vs model-specific feature attribution.

## Outputs

- A transfer card naming models, layers, map form, corpus, metrics, and limitations.
- A universality claim scoped to model family, template, task, and latent object.
- A failure catalog for non-transferable features.

## Constraints

- Do not treat cross-model success as proof of deployed-model transfer without authorized testing.
- Keep private challenge prompts and outputs separate from public transfer notes.
- Verify model-card and tokenizer assumptions before interpreting failure as geometry failure.

## Verification

- Include identity baselines, shuffled-pair baselines, and same-model upper bounds.
- Test transfer on held-out prompts and at least one neighboring layer.
- Report failed transfers alongside successes.

## Source Map

- `references/008-vocabulary-transfer.md`
- `references/009-unifyvocab.md`
- `references/044-the-platonic-representation-hypothesis.md`
- `references/045-revisiting-the-platonic-representation-hypothesis-an-aristotelian-view.md`
- `references/046-transferring-linear-features-across-language-models-with-model-stitching.md`
- `references/047-cross-model-transferability-among-large-language-models-on-the-platonic-representations-of.md`
- `references/048-activation-space-interventions-can-be-transferred-between-large-language-models.md`
- `references/049-contrans-weak-to-strong-alignment-engineering-via-concept-transplantation.md`
- `references/050-universal-sparse-autoencoders.md`
- `references/051-cross-architecture-model-diffing-with-crosscoders-unsupervised-discovery-of-differences-be.md`
- `references/052-latent-space-communication-via-k-v-cache-alignment.md`
- `references/054-linear-representation-transferability-hypothesis.md`
- `references/055-characterizing-linear-alignment-across-language-models.md`
