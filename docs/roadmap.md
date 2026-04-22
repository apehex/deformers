# Roadmap

This document defines the execution plan of the project.

It combines short-term tasks, mid-term experiments, and longer-term directions.

---

# Phase 0 - Foundations [done]

Objective: establish a stable experimental framework.

## Documentation [done]

- [x] finalize documentation structure
- [x] define invariants and constraints
- [x] document model architecture and patch interfaces

## Infrastructure [done]

- [x] implement model splitting:
  - [x] prefix / trunk / suffix separation
- [x] implement shared trunk execution between models
- [x] implement dataset preprocessing pipeline

## Testing [done]

- [x] define unit tests for prefix patch
- [x] validate tensor shapes and interfaces
- [x] validate compatibility with base model tokenizer

---

# Phase 1 - Prefix Patch (Composite Embedding)

Objective: replace the token embedding layer while preserving model behavior.

## Representation [done]

- [x] encode tokens using UTF-8 byte sequences
- [x] fix maximum token length:
  - [x] `L_max = 32 bytes`
- [x] pad shorter tokens with null bytes (pad_id=128)
- [x] truncate longer tokens
- [x] preprocessing in `src/deformers/pipelines/patch.py`

## Architecture [done]

- [x] Stage A:
  - [x] `CompositeBytePrefix` in `src/deformers/layers/prefix.py`
  - [x] `CompositeEmbedding(256, embed_dim, group_dim=G, merge_axes=True)` -> `(B, T, G*E)`
  - [x] `LayerNorm -> Linear -> SiLU -> Linear -> LayerNorm` projection to `hidden_size`
  - [x] lazy-build, no explicit device args, submodules registered as `self._layers`
- [ ] Stage B (planned): add a copied Qwen decoder block inside the prefix
- [ ] Stage C (planned): small byte-level transformer over G positions

## Training [~]

- [x] training script: `scripts/train_prefix_stage_a.py`
- [x] embedding regression warmup: MSE between prefix output and original embeddings
- [x] hidden-state matching at depth `k` (distillation via `inputs_embeds`)
- [x] trunk and lm_head are frozen; only prefix parameters are trained
- [x] optional KL divergence on logits (planned)
- [x] learning rate warmup and decay
- [x] apply the attention mask to both hidden and embed losses
- [x] track the KL divergence loss too
- [ ] use `accelerate`
- [ ] two-stage curriculum:
  - [x] train only embedding MSE (set hidden_rate=0, embed_rate=1) until low plateau
  - [x] enable hidden loss (e.g. hidden_rate=1, embed_rate=0.05)
  - [ ] could be extended by increasing the teacher's depth epoch after epoch

## Integration [done]

- [x] shared transformer trunk via `inputs_embeds` HF interface
- [x] tokenizer partition identical to base model
- [x] output shape `(B, T, H)` compatible with trunk

## Monitoring [done]

- [x] add a progress bar during training, with:
  - [x] the epoch number compared to the total epochs
  - [x] the step number compared to the final step of the epoch
  - [x] the current learning rate
  - [x] the embed, hidden and total MSE losses
  - [x] the KL divergence at the target depth
  - [x] the hidden MSE loss on a fixed tensor with the top-k tokens
- [x] add TensorBoard logging:
  - [x] train/loss_total
  - [x] train/loss_hidden
  - [x] train/loss_embed
  - [x] train/lr
  - [x] train/grad_norm
  - [x] train/step_time_ms
  - [x] gpu/memory_allocated_mb
  - [x] gpu/memory_reserved_mb
- [x] log memory and throughput every optimizer step
- [x] keep plain stdout logs for notebook runs

## Evaluation [extended]

Primary objective:
- [ ] match teacher token ranking as closely as possible

Core metrics:
- [x] embedding reconstruction error (MSE)
- [x] hidden-state similarity (MSE at depth `k`)
- [x] KL divergence between teacher and student logits
- [x] top-1 match rate
- [x] top-k set match rate
- [x] top-k exact order match rate (strict)
- [ ] loss on the most prevalent tokens (top-k vocab)
- [ ] delta predictions on a fixed sentence

Secondary checks:
- [ ] qualitative comparison via text generation
- [ ] latency overhead of prefix patching

## Near-term tasks

1. [x] build Stage A evaluation script:
   - [x] `scripts/benchmark.py` - true evaluation entrypoint (no training)
   - [x] fixed validation subset (`train[90%:]`)
   - [x] teacher vs student logits comparison (KL, top-1, top-k set, top-k order)
   - [x] fixed sentence probe: teacher vs student top-k tokens
   - [x] vocab probe: deterministic (B, T) token tensor evaluation
   - [x] shared helpers in `src/deformers/pipelines/eval.py`
2. [ ] define stop criteria:
   - [ ] early stop on KL + top-k exact order plateau
3. [ ] export and load pipeline:
   - [x] save/load prefix checkpoint (`load_prefix_checkpoint` in eval.py)
   - [ ] run end-to-end generation with patched prefix
4. [x] add TensorBoard writer and scalars/histograms
5. [x] add tests for patching and evaluation utilities

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
