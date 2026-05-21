# Context

## Repository

`apehex/deformers` on Github.

## Models

Mostly `qwen/qwen3.5-9b`.

## Purpose

Patch the layers of pretrained LLMs.

Large foundation models are monolithic: embedding table, trunk, and output head are trained jointly and are tightly coupled.
The goal of this project is to expose clean modular interfaces so that individual components can be replaced or retrained in isolation, without touching the rest.

## Objectives

Experiment with model level composition and test the modularity.

## Directions

- prefix patches (input representation)
- suffix patches (output distribution)
- custom decoder (reversing the auto-regression, toward past tokens)

## Design Principles

- freeze most of the original model
- train only localized modules
- keep the same input partition and tokenization vocabulary
- align patched modules with the pretrained model via distillation or hidden-state matching

## Motivations

- remove dependency on large token embedding tables
- exploit internal structure of token strings
- allow deterministic reconstruction of embeddings
- keep the information on the token composition
- test modular training and composition of sub-models

## Current Experiment: Prefix Patch

The active experiment trains a byte-based prefix module to replace the original embedding layer.

- input: token-piece strings encoded as UTF-8 bytes, padded or truncated to a fixed patch length of 32 bytes
- target: original teacher embedding vectors (frozen)
- trunk and lm_head remain fully frozen; only the prefix parameters are trained

Success is defined in behavioral terms: the patched prefix should produce hidden states and logits that closely match those of the original model.
Perfect embedding reconstruction is not the goal; what matters is that downstream behavior (hidden-state similarity, top-k token agreement, perplexity) is preserved.

## Workflow

1. split pretrained model into prefix / trunk / suffix
2. replace prefix or suffix with experimental module
3. train patch using teacher outputs from the original model
4. evaluate performance relative to baseline model using `scripts/benchmark.py`

---

## New Research Axis: Mechanistic Safety Deformers

This project now also studies whether the same patching interface can be used to analyze, measure, and safely modify safety-relevant latent representations in LLMs.

The core hypothesis is that several high-level behaviors are not only prompt-level phenomena but also have measurable internal structure:

- refusal and over-refusal
- instruction hierarchy and authority attribution
- chat-template and role-token anchoring
- latent prompt compression
- cross-model feature transfer
- prompt or activation inversion

This axis is a natural extension of the existing prefix / trunk / suffix decomposition:

- prefix patches can test how token strings, role markers, and chat-template artifacts are transformed into initial hidden states
- trunk interventions can test whether behavior is mediated by sparse features, directions, subspaces, or KV-cache states
- suffix patches can test whether output-level distributions can preserve useful behavior while changing refusal or compliance tendencies
- custom decoders can test how much text, role, or template information can be reconstructed from hidden states
- cross-model alignment modules can test whether latent features discovered in one open model transfer to another

The goal is not to create public bypasses for deployed models. The goal is to produce controlled, reproducible, white-box science that helps explain where safeguards live, how they fail, and how they can be made more robust.

## Safety and Disclosure Scope

Security research in this repository must stay within defensive and authorized boundaries.

Allowed work:

- open-weight model analysis
- benign synthetic tasks
- instruction-hierarchy benchmarks
- refusal / over-refusal measurement
- latent-space and feature-space diagnostics
- controlled red-team simulations that do not contain operationally harmful content
- responsible-disclosure preparation for programs where the researcher is authorized

Disallowed work:

- publishing or storing live jailbreak payloads for frontier systems
- generating operational bio-risk content
- providing procedures that bypass safeguards on deployed models
- optimizing prompts or agents for real-world harmful capability elicitation
- mixing NDA-covered bounty material into public repository artifacts

## Mechanistic Safety Questions

This axis tracks several concrete research questions.

1. Refusal geometry

   - Is refusal mediated by a single direction, a small shared core, or a richer family of category-specific features?
   - Which layers, token positions, attention heads, MLP features, or SAE latents causally mediate refusal?
   - How can refusal be strengthened or made more precise without degrading usefulness?

2. Authority geometry

   - Does system-level authority correspond to a measurable latent direction, subspace, or feature set?
   - How much of authority attribution comes from explicit role tokens versus style, position, wording, and chat-template structure?
   - Can a model internally confuse untrusted text for higher-priority instructions, and can this be detected before generation?

3. Template anchoring

   - Do safety decisions depend disproportionately on the chat-template region?
   - Does removing, replacing, or compressing special tokens preserve their influence in later hidden states?
   - Can a learned prefix or latent memory reproduce the useful part of a system prompt without exposing raw text?

4. Latent inversion and prompt compression

   - Which hidden states are invertible into the original token sequence?
   - Which projected or edited hidden states are no longer reachable from ordinary text prompts?
   - Can system-prompt behavior be compressed into soft prompts, learned tokens, or latent vectors without leaking the prompt itself?

5. Cross-model latent translation

   - Can features, probes, steering vectors, sparse autoencoder latents, or KV-cache states be translated between models?
   - When does a linear or affine map suffice?
   - Which behaviors are shared across architectures, and which are model-specific?

## Evaluation Principles for Safety Work

Safety experiments should be evaluated with both behavior and geometry metrics.

Behavioral metrics:

- refusal precision on benign and disallowed proxy prompts
- over-refusal rate
- instruction-hierarchy adherence
- prompt-injection resistance on harmless benchmark tasks
- utility retention on ordinary tasks
- stability across paraphrases, contexts, and decoding settings

Geometry metrics:

- activation cosine similarity
- diff-in-means direction strength
- probe accuracy and calibration
- SAE feature activation changes
- causal intervention effect size
- logit KL divergence against teacher outputs
- transfer loss under cross-model mapping
- locality of the intervention across layers and positions

The preferred workflow is to start with benign proxy datasets, validate causal mechanisms on open-weight models, and only then consider responsible-disclosure testing in authorized environments.
