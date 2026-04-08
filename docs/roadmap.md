# Roadmap

This document defines the execution plan of the project.

It combines short-term tasks, mid-term experiments, and longer-term directions.

---

# Phase 0 - Foundations

Objective: establish a stable experimental framework.

## Documentation

- finalize documentation structure
- define invariants and constraints
- document model architecture and patch interfaces

## Infrastructure

- implement model splitting:
  - prefix / trunk / suffix separation
- implement shared trunk execution between models
- implement dataset preprocessing pipeline

## Testing

- define unit tests for prefix patch
- validate tensor shapes and interfaces
- validate compatibility with base model tokenizer

---

# Phase 1 - Prefix Patch (Composite Embedding)

Objective: replace the token embedding layer while preserving model behavior.

## Representation [done]

- encode tokens using UTF-8 byte sequences
- fix maximum token length:
  - `L_max = 32 bytes`
- pad shorter tokens with null bytes (pad_id=128)
- truncate longer tokens
- preprocessing in `src/deformers/patching/bytes.py`

## Architecture [done - Stage A]

- Stage A (implemented):
  - `CompositeBytePrefix` in `src/deformers/layers/prefix.py`
  - `CompositeEmbedding(256, embed_dim, group_dim=G, merge_axes=True)` -> `(B, T, G*embed_dim)`
  - `LayerNorm -> Linear -> SiLU -> Linear -> LayerNorm` projection to `hidden_size`
  - lazy-build, no explicit device args, submodules registered as `self._layers`
- Stage B (planned): add a copied Qwen decoder block inside the prefix
- Stage C (planned): small byte-level transformer over G positions

## Training [done - Stage A]

- training script: `scripts/train_prefix_stage_a.py`
- embedding regression warmup: MSE between prefix output and original embeddings
- hidden-state matching at depth `k` (distillation via `inputs_embeds`)
- trunk and lm_head are frozen; only prefix parameters are trained
- optional KL divergence on logits (planned)

## Integration [done]

- shared transformer trunk via `inputs_embeds` HF interface
- tokenizer partition identical to base model
- output shape `(B, T, hidden_size)` compatible with trunk

## Evaluation

- embedding reconstruction error
- hidden-state similarity
- KL divergence between logits
- qualitative comparison via text generation

---

# Phase 2 - Suffix Patch (Hierarchical Head)

Objective: replace the output projection layer.

## Architecture

- replace:
  - `Linear(hidden_size → vocab_size)`
- with hierarchical prediction:
  - binary decisions along a tree
  - tokens as leaves

## Tree construction

Candidate strategies:

- frequency-based (Huffman)
- embedding-based clustering
- BPE-derived structure (from `merges.txt`)

## Training

- path-based binary cross entropy
- optional distillation from original logits

## Sampling

- iterative traversal from root to leaf
- conditional probability at each node

## Evaluation

- parameter count reduction
- sampling latency
- KL divergence vs original head
- generation quality

---

# Phase 3 - End-to-End Distillation

Objective: align patched model with original model.

## Training signals

- embedding regression (prefix)
- hidden-state matching (intermediate layers)
- KL divergence on logits
- next-token cross entropy

## Training stages

1. prefix-only training
2. suffix-only training
3. combined prefix + suffix training
4. optional partial trunk unfreezing

## Evaluation

- perplexity
- KL divergence
- generation quality

---

# Phase 4 - Extensions

## Reversible suffix

- replace final transformer layers
- explore reversible decoding structures

## Alternative prefix architectures

- convolutional byte encoders
- transformer over byte dimension
- variable-length encoding

## Alternative tree structures

- learned hierarchical clustering
- adaptive trees during training

---

# Phase 5 - Latent Sequence Modeling (Experimental)

Objective: replace token-level modeling with latent sequence modeling over byte representations.

### Motivation

Tokenization provides a good factorization of language by aligning segments with statistical boundaries.

Fixed-size byte patching introduces artificial dependencies across positions:

- a byte block may partially determine the next block
- this reduces sampling flexibility
- this increases inter-token mutual information

This phase explores learning a latent representation that restores a better factorization while keeping byte-level inputs.

### Architecture

Two-stage model:

1. Variational Autoencoder `V`
   - input: byte sequences (fixed length per token or patch)
   - output: latent sequence `z`

2. Autoregressive model `W`
   - operates on latent sequence `z`
   - predicts `z_t` given `z_<t`

Inference:

1. sample latent sequence autoregressively
2. decode full sequence with VAE decoder (non-causal)

Optional extensions:

- masked or non-causal decoding
- diffusion-based refinement in latent or output space

### Latent Space Requirements

The latent representation must satisfy:

#### Factorization

- `p(z_t | z_<t)` must be predictable
- low conditional entropy across positions

#### Locality

- small changes in latent space produce small changes in output

#### Compositionality

- each latent corresponds to a local region of the sequence

#### Redundancy

- representation must tolerate inconsistencies
- decoder can correct errors globally

#### Independence

- reduce mutual information between adjacent latents
- improve sampling flexibility compared to byte patches

### Training

Stage 1 — VAE training

- reconstruction loss on byte sequences
- regularization of latent space (KL)

Stage 2 — Latent AR training

- train autoregressive model on latent sequences

Stage 3 — Optional refinement

- train decoder to correct noisy or inconsistent latent sequences
- optional diffusion or denoising objectives

### Evaluation

- reconstruction quality (bytes → bytes)
- predictability of latent sequence
- mutual information between latent positions
- generation quality after decoding
- comparison with token-based baseline

### Status

Exploratory research direction.

Not required for prefix/suffix patching pipeline.

---

# Open Questions

## Prefix

- optimal byte embedding dimension
- impact of truncation on rare tokens
- best depth for hidden-state alignment

## Suffix

- best tree construction method
- trade-off between tree depth and accuracy
- compatibility with fast sampling

## Training

- minimal compute required for alignment
- optimal loss weighting
- stability of frozen-trunk distillation
