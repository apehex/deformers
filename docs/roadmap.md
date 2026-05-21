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

## Architecture [~]

Current implementation:
- [x] `CompositeBytePrefix` in `src/deformers/models/prefix.py`
- [x] byte value + positional encoder (`ByteEncoder`) to shape `(B, T, G, E)`
- [x] stack of byte transformer blocks (`block_num`) with attention on byte axis `G`
- [x] first block is padding-aware, subsequent blocks attend on dense byte representations
- [x] byte mixing + token projection to `(B, T, H)` with lazy build

Architecture experiments (planned):
- [ ] norm variant: replace LayerNorm with RMSNorm
- [ ] no-norm variant: remove normalization layers entirely
- [x] attention on patch axis: self-attention over the G byte-embedding positions within each patch
- [x] padding-aware byte attention in the first block (derived from padding sentinel)
- [ ] residual pre-norm patch block:
  - [ ] `x = x + SelfAttention(RMSNorm(x), paddings)`
  - [ ] `x = x + MLP(RMSNorm(x))`
- [ ] projection head variant: replace `Linear -> SiLU -> Linear` with SwiGLU
- [ ] readout variant: keep flattened readout as baseline
- [ ] readout variant: add masked mean pooling over the byte-patch axis
- [ ] readout variant: add learned attention pooling over the byte-patch axis
- [ ] readout variant: add learned readout token over the byte-patch axis
- [ ] length embedding: add byte-length embedding after readout
- [ ] output calibration: initialize or learn an output scale to match teacher embedding RMS
- [ ] wider MLP: increase intermediate dimension in the projection MLP
- [ ] contextual prefix ablation: optionally add cross-token attention only as a separate experiment, not as the default embedding-table replacement
 

## Training [~]

- [x] training script: `scripts/prefix.py`
- [x] embedding regression warmup: MSE between prefix output and original embeddings
- [x] hidden-state matching at depth `k` (distillation via `inputs_embeds`)
- [x] trunk and lm_head are frozen; only prefix parameters are trained
- [x] learning rate warmup and decay
- [x] apply the attention mask to both hidden and embed losses
- [x] track cosine similarity alongside MSE
- [x] lifecycle-oriented trainer API:
  - [x] trainer owns utility setup from configuration (no prebuilt objects in constructor)
  - [x] shared runner split into `BaseRunner`, `PrefixTrainer`, and `PrefixTester`
  - [x] `setup_global()` creates long-lived utilities (optimizer, scaler, context)
  - [x] `setup_phase()` creates phase-local utilities (scheduler, callbacks)
  - [x] single trainer instance reusable across phases; optimizer persists
  - [x] monotonically increasing global step counter across epochs and phases
- [ ] use `accelerate`
- [ ] curriculum:
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
- [x] cosine similarity between student and teacher embeddings / hidden states
- [x] top-1 match rate
- [x] top-k set match rate
- [x] top-k exact order match rate (strict)
- [x] logit KL divergence (token-probability comparison)
- [ ] loss on the most prevalent tokens (top-k vocab)
- [ ] delta predictions on a fixed sentence

Secondary checks:
- [ ] qualitative comparison via text generation
- [ ] latency overhead of prefix patching

## Near-term tasks

1. [ ] update evaluation script:
   - [ ] rewrite `scripts/benchmark.py` using latest `deformers` and `mlable` APIs
   - [ ] fixed validation subset (`train[90%:]`)
   - [ ] teacher vs student logits comparison (top-1, top-k set, top-k order, logit KL)
   - [ ] cosine similarity and norm metrics on embeddings and hidden states
   - [ ] fixed sentence probe: teacher vs student top-k tokens
   - [ ] vocab probe: deterministic (B, T) token tensor evaluation
   - [ ] shared helpers in `src/deformers/pipelines/eval.py`
2. [ ] define stop criteria:
   - [ ] early stop on cosine similarity + top-k exact order plateau
3. [ ] export and load pipeline:
   - [x] save/load prefix checkpoint (`load_prefix_checkpoint` in eval.py)
   - [ ] run end-to-end generation with patched prefix
4. [x] add TensorBoard writer and scalars/histograms
5. [x] add tests for patching and evaluation utilities

---

# Phase 1.5 - Inspection, Instrumentation, and Calibration

Objective: understand current error distribution before changing the architecture or training recipe.

## Token-wise Error Analysis

- [ ] per-token table: report embed MSE, cosine similarity, hidden MSE, and logit KL for every token in a fixed probe batch
- [ ] break down errors by:
  - [ ] token frequency (common vs rare tokens)
  - [ ] byte length (short tokens, long tokens, truncated tokens)
  - [ ] token type (leading-space tokens, punctuation, long-byte / binary tokens)
- [ ] identify which token classes drive the highest loss

## Geometry Diagnostics

- [ ] norm distribution: compare L2 norms of student vs teacher embeddings and hidden states
- [ ] cosine similarity: distribution of cosine similarities between student and teacher vectors per token
- [ ] anisotropy: measure the degree of isotropy in student vs teacher embedding spaces
- [ ] output scale drift: compare prefix output RMS against the original embedding table RMS
- [ ] byte-length breakdown: compare prefix error by UTF-8 byte length and truncation status
 
## Calibration Experiments

- [ ] controlled noise injection: add Gaussian noise of varying scale to teacher embeddings, then measure the effect on:
  - [ ] hidden-state MSE and cosine similarity at depth k
  - [ ] logit KL divergence and top-k token agreement rate
  - [ ] perplexity on a fixed text probe
- [ ] produce error-vs-noise curves to define the acceptable error scale for the prefix output
- [ ] use these curves to set concrete convergence targets

## Evaluation Improvements

- [ ] multi-depth hidden-state MSE and cosine similarity: track at several trunk depths (not only depth k)
- [ ] add cosine similarity, logit KL, and top-k metrics directly in the training validation loop
- [ ] fixed vocab probe: deterministic (B, T) tensor covering uniform token distribution, tracked every N steps
- [ ] fixed sentence probe: teacher vs student logits on a fixed sentence, tracked every N steps

## Data Strategy

- [ ] explicit distribution mix: decide between uniform vocab coverage dataset and real text dataset
- [ ] phased training:
  - [ ] coverage prephase: train on uniform token-id dataset (all vocab IDs sampled uniformly)
  - [ ] alignment phase: switch to real text (Wikipedia or similar) for contextual alignment
- [ ] document the chosen schedule and rationale once decided

## Architecture Experiments

- [ ] norm variant: RMSNorm instead of LayerNorm in the projection head
- [ ] no-norm variant: remove all normalization layers, rely on weight initialization and learning rate
- [ ] attention on patch axis: self-attention over G byte-embedding positions within each patch, with byte padding passed as `key_padding_mask`
- [ ] residual byte-patch transformer block with pre-norm attention and MLP
- [ ] SwiGLU projection head
- [ ] compare flattened readout against masked mean pooling, learned attention pooling, and learned readout-token pooling
- [ ] add explicit byte-length embedding and measure whether it improves short-token and long-token alignment
- [ ] compare output scale calibration strategies
- [ ] wider MLP: increase intermediate dimension in the projection MLP

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

- which token classes drive the highest embedding and hidden-state error?
- what is the acceptable error scale for the prefix output (calibration target)?
- optimal byte embedding dimension
- impact of truncation on rare tokens
- best depth for hidden-state alignment
- does attention over the patch axis (G byte-embedding positions) improve alignment quality?
- is LayerNorm, RMSNorm, or no normalization best for the projection head?
- does padding-aware byte attention converge faster than implicit padding learning?
- does flattened readout outperform pooled readout when the byte dimension is chosen so that `G*E == H`?
- does learned readout pooling improve embedding geometry compared to flattened byte-slot composition?
- does explicit byte-length conditioning improve the alignment of short, padded, and truncated tokens?
- how closely must the prefix output norm distribution match the teacher embedding table?

## Data

- what is the right balance between uniform vocab coverage and real text distribution?
- does a coverage prephase before alignment phase improve convergence?

## Suffix

- best tree construction method
- trade-off between tree depth and accuracy
- compatibility with fast sampling

## Training

- minimal compute required for alignment
- optimal loss weighting between embedding MSE and hidden-state MSE
- stability of frozen-trunk distillation
- does a wider MLP intermediate dimension improve prefix quality?

---

# Phase 6 - Mechanistic Safety Patching (Experimental)

Objective: study refusal, authority attribution, template anchoring, and cross-model latent transfer through the same modular patching framework used for prefix and suffix experiments.

Status: planned.

## Scope

This phase is defensive research. It should use open-weight models, harmless proxy tasks, and responsible-disclosure workflows. It must not store live jailbreak payloads, operational bio-risk content, or NDA-covered bounty material.

## Phase 6.0 - Literature and Benchmark Map

- [ ] convert the state-of-the-art references into a reading queue
- [ ] tag each reference by mechanism:
  - [ ] activation steering
  - [ ] refusal direction
  - [ ] authority / role confusion
  - [ ] prompt compression
  - [ ] latent inversion
  - [ ] cross-model transfer
- [ ] define safe proxy tasks for:
  - [ ] refusal precision
  - [ ] over-refusal
  - [ ] instruction hierarchy
  - [ ] role confusion
  - [ ] prompt-injection resistance
  - [ ] utility retention
- [ ] implement a benchmark registry under `src/deformers/evals/`
- [ ] document dataset safety filters and excluded content classes

## Phase 6.1 - Instrumentation

- [ ] add hooks for residual stream capture at each layer
- [ ] add hooks for attention outputs and MLP outputs
- [ ] add optional KV-cache capture
- [ ] standardize layer / token-position conventions
- [ ] implement activation patching utilities
- [ ] implement diff-in-means vector extraction
- [ ] implement low-rank direction extraction with SVD
- [ ] implement projection, ablation, and addition interventions
- [ ] add masked metrics for variable-length prompts
- [ ] add artifact logging for directions, probes, and evaluation summaries

## Phase 6.2 - Refusal Geometry

- [ ] reproduce a benign version of the refusal-direction setup on open-weight chat models
- [ ] extract single refusal directions by layer and position
- [ ] compare harmful-proxy / harmless contrast pairs without operational dangerous content
- [ ] measure refusal precision and over-refusal
- [ ] compare:
  - [ ] single vector
  - [ ] multi-vector / low-rank subspace
  - [ ] SAE feature targeting
  - [ ] attention-head or MLP-feature localization
- [ ] measure side effects:
  - [ ] utility loss
  - [ ] fluency degradation
  - [ ] hallucination rate
  - [ ] instruction-following degradation
- [ ] test whether refusal changes are stable across paraphrases and chat templates

## Phase 6.3 - Authority and Instruction Hierarchy Geometry

- [ ] build benign system/developer/user/tool hierarchy tasks
- [ ] construct contrastive pairs that differ only in role, priority, or source
- [ ] separate explicit role-token effects from wording, position, and style effects
- [ ] train probes for inferred speaker role and authority level
- [ ] test whether spoofed low-priority content activates higher-priority role features
- [ ] extract candidate authority directions or subspaces
- [ ] test whether authority features are:
  - [ ] local to template tokens
  - [ ] propagated into later user-token states
  - [ ] recoverable after special-token filtering
  - [ ] entangled with authoritative writing style
- [ ] evaluate mitigation ideas:
  - [ ] stronger segment embeddings
  - [ ] explicit privilege features
  - [ ] role-confusion detectors
  - [ ] template-anchoring reduction

## Phase 6.4 - Template Filtering, Latent Inversion, and Prompt Compression

- [ ] test whether hidden states remain invertible after removing special-token positions
- [ ] compare full hidden-state inversion against projected / position-dropped inversion
- [ ] train a small decoder or VAE to reconstruct safe template structure from hidden states
- [ ] evaluate latent prompt carriers:
  - [ ] soft prompts
  - [ ] learned control tokens
  - [ ] system vectors
  - [ ] prefix patches
  - [ ] KV-cache memories
- [ ] measure:
  - [ ] reconstruction accuracy
  - [ ] behavior preservation
  - [ ] prompt leakage
  - [ ] utility retention
  - [ ] whether the latent state is reachable from discrete text

## Phase 6.5 - Cross-Model Latent Translation

- [ ] choose a source / target open-model pair
- [ ] collect a paired hidden-state corpus with identical prompts
- [ ] train baseline mappings:
  - [ ] mean / scale calibration
  - [ ] orthogonal Procrustes
  - [ ] affine least-squares map
  - [ ] CCA baseline
- [ ] compare richer mappings:
  - [ ] model stitching
  - [ ] shared sparse autoencoder dictionary
  - [ ] crosscoder
  - [ ] KV-cache alignment adapter
- [ ] transfer:
  - [ ] probes
  - [ ] refusal directions
  - [ ] authority directions
  - [ ] steering vectors
  - [ ] soft prompts
- [ ] report:
  - [ ] representation reconstruction error
  - [ ] behavior transfer score
  - [ ] off-target degradation
  - [ ] model-family sensitivity
  - [ ] failure cases

## Phase 6.6 - Defensive Outputs

- [ ] write concise mechanism reports for each experiment
- [ ] produce safe reproductions with harmless proxy prompts
- [ ] create mitigation notes for:
  - [ ] refusal brittleness
  - [ ] template anchoring
  - [ ] role confusion
  - [ ] non-surjective latent interventions
  - [ ] cross-model transfer failure modes
- [ ] prepare private responsible-disclosure templates for authorized programs
- [ ] keep public artifacts free of exploit payloads and operationally harmful content

## Success Criteria

This phase is successful if the project can:

- localize at least one safety-relevant behavior to measurable hidden-state features
- distinguish causal interventions from correlational probes
- preserve utility while changing a targeted benign proxy behavior
- explain when an intervention does or does not transfer between two open models
- document a safe, reproducible workflow for mechanistic safety research

## Open Questions

- is authority a direction, a subspace, a set of sparse features, or a cache-level state?
- can role confusion be detected before generation?
- how much system-prompt behavior survives special-token filtering?
- when does prompt compression preserve behavior without leaking the raw prompt?
- which safety features are shared across model families?
- which latent interventions are unreachable by ordinary text prompts?
- can prefix patches emulate special-token behavior while keeping tokenizer boundaries fixed?
