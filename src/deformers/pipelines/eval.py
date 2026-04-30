"""
Shared evaluation utilities for prefix/suffix patch experiments.

Assumptions:
- Logits tensors are (B, T, V) float, on the same device.
- Embedding tensors are (B, T, H) float.
- All reductions are over the full (B, T) batch; masking is caller responsibility.
- Vocab probe uses top vocab IDs selected by a deterministic rule (sorted ascending by ID).
- Text probe uses offset-based byte patching aligned to the base tokenizer boundaries.
- Checkpoint format: dict with keys 'config' and 'state'.
"""

import math

import torch
import torch.nn.functional

import deformers.pipelines.patch

# PROBE ########################################################################

def indices_probe(
    vocab_dim: int,
    batch_dim: int,
    sequence_dim: int
) -> list[list[int]]:
    # first indices / tokens of the vocabulary
    __ids = torch.arange(batch_dim * sequence_dim, dtype=torch.long) % vocab_dim
    # (B, T) integers
    return __ids.reshape(batch_dim, sequence_dim).tolist()
