# Mechanistic Safety Research Axis

This note summarizes the state-of-the-art report and translates it into the `deformers` project.

## Summary

The new research axis is not simply "remove refusal." The stronger framing is:

> map the geometry of authority, template anchoring, refusal, and cross-model transfer; then test which mechanisms are causal, robust, safe, and portable.

The recent literature suggests three things at once:

1. refusal and other aligned behaviors can often be steered with compact latent objects
2. those objects are richer than a single universal direction
3. latent interventions do not necessarily correspond to any ordinary text prompt

This matters for `deformers` because the project already separates models into patchable components. That makes it suitable for controlled experiments on where safety-relevant information enters the model, how it propagates through the trunk, and whether it can be translated between models.

## Main Research Threads

### Activation steering

Activation steering has moved from simple vector arithmetic toward structured, layer-aware, and feature-targeted methods.

Important families:

- activation addition
- contrastive activation addition
- mean-centered steering vectors
- conceptor steering
- SAE-targeted steering
- steering tokens
- steering vector fields
- KV-cache steering

Implication for `deformers`:

- implement a generic intervention API before implementing specialized methods
- always compare single-vector, low-rank, SAE-feature, and KV-cache variants
- measure off-target degradation, not only target behavior change

### Refusal and over-refusal

Early work showed a strong one-dimensional refusal direction in several open chat models. Newer work suggests this is an incomplete account: refusal contains a shared core plus category-, style-, and context-specific features.

Implication for `deformers`:

- treat refusal as a family of features
- measure both refusal precision and over-refusal
- test whether changing refusal also changes truthfulness, helpfulness, tone, or instruction following

### Authority and role confusion

Instruction hierarchy is intended to prioritize system, developer, user, and tool instructions. Recent work shows that models may infer authority from style and content, not only from interface role tags.

Implication for `deformers`:

- separate role-token effects from style effects
- train role and authority probes
- test whether untrusted text activates higher-priority role features
- evaluate whether learned segment embeddings or prefix patches can make authority more robust

### Template anchoring

Some aligned models appear to rely too much on the chat-template region for safety decisions. This can create brittle behavior when the template is perturbed, compressed, spoofed, or filtered.

Implication for `deformers`:

- compare activations at template-token positions and content-token positions
- test whether template influence is propagated into later hidden states
- test inversion after removing template-token positions
- avoid assuming that deleted positions leave enough information for exact reconstruction

### Prompt compression and hidden policy carriers

System prompts and policies can sometimes be compressed into vectors, soft prompts, learned tokens, or KV-cache state. This supports the idea that role or policy behavior can survive without visible special tokens, but does not prove that a clean portable text prompt exists.

Implication for `deformers`:

- compare hard tokens, soft prompts, prefix patches, learned tokens, and KV-cache memories
- measure prompt leakage and behavior preservation separately
- test whether compressed carriers survive model transfer

### Cross-model latent translation

Cross-model transfer is increasingly practical, especially with affine maps, model stitching, shared sparse dictionaries, crosscoders, and KV-cache alignment. But it remains approximate and model-family dependent.

Implication for `deformers`:

- treat transfer as a measured alignment problem
- start with paired hidden states and affine maps
- only claim transfer when behavior and geometry both transfer
- keep a failure-case catalog

## Safe Experimental Pattern

Use this pattern for all safety-adjacent experiments.

1. Define a harmless proxy behavior.
2. Build contrastive pairs.
3. Extract candidate latent object.
4. Measure correlation.
5. Intervene causally.
6. Evaluate target behavior.
7. Evaluate off-target utility.
8. Test paraphrase and template robustness.
9. Test transfer to another open model.
10. Document failures.

Do not include operationally harmful content, live jailbreak payloads, or NDA-covered bounty material in public artifacts.

## First Experiments

1. Refusal proxy direction

   - source: open chat model
   - data: harmless refusal-style pairs
   - intervention: add / ablate direction at several layers
   - metric: refusal style change, over-refusal, utility retention

2. Authority probe

   - source: open chat model with known chat template
   - data: benign system/developer/user/tool instruction conflicts
   - probe: predict intended authority level from hidden states
   - metric: role-probe accuracy, confusion matrix, pre-generation risk score

3. Template filtering inversion

   - source: open model with hidden-state capture
   - data: prompts with and without chat-template markers
   - intervention: remove template-token positions before inversion
   - metric: reconstruction accuracy and ambiguity

4. Cross-model direction transfer

   - source / target: related open models
   - data: paired prompt corpus
   - map: affine residual-stream map
   - transfer: refusal proxy or authority direction
   - metric: alignment error and behavior transfer score
