# Roadmap

This document defines the execution plan of the project.

It combines short-term tasks, mid-term experiments, and longer-term directions.

---

# Phase 0 — Foundations

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

# Phase 1 — Prefix Patch (Composite Embedding)

Objective: replace the token embedding layer while preserving model behavior.

## Representation

- encode tokens using UTF-8 byte sequences
- fix maximum token length:
  - `L_max = 32 bytes`
- pad shorter tokens with null bytes
- truncate longer tokens

## Architecture

- byte embedding:
  - `Embedding(256, d_byte)`
- reshape to token embedding:
  - `(L_max × d_byte = hidden_size)`
- optional projection / normalization layer

## Training

- embedding regression:
  - match original embedding vectors
- hidden-state matching at depth `k`
- optional KL divergence on logits

## Integration

- share transformer trunk with original model
- ensure identical tokenizer partition
- ensure output shape compatibility

## Evaluation

- embedding reconstruction error
- hidden-state similarity
- KL divergence between logits
- qualitative comparison via text generation

---

# Phase 2 — Suffix Patch (Hierarchical Head)

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

# Phase 3 — End-to-End Distillation

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

# Phase 4 — Extensions

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
