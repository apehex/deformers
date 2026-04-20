"""
Shared evaluation utilities for prefix/suffix patch experiments.

Assumptions:
- Logits tensors are (B, T, V) float, on the same device.
- Embedding tensors are (B, T, H) float.
- All reductions are over the full (B, T) batch; masking is caller responsibility.
- Vocab probe uses top vocab IDs selected by a deterministic rule (sorted ascending by ID).
- Text probe uses offset-based byte patching aligned to the base tokenizer boundaries.
- Checkpoint format: dict with keys 'config' and 'state_dict'.
"""

import math

import torch
import torch.nn.functional

import deformers.pipelines.patch

# METRICS ######################################################################

def kl_divergence(
    teacher_arr: torch.Tensor,
    student_arr: torch.Tensor,
) -> torch.Tensor:
    """KL divergence KL(teacher || student) over (B, T, V) raw logits."""
    __shape = tuple(teacher_arr.shape)
    # merge the batch axes
    __t = teacher_arr.float().reshape(math.prod(__shape[:-1]), __shape[-1])
    __s = student_arr.float().reshape(math.prod(__shape[:-1]), __shape[-1])
    # reduced to a single value
    return torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(__s, dim=-1),
        torch.nn.functional.log_softmax(__t, dim=-1),
        reduction='batchmean',
        log_target=True)

def topk_rate(
    teacher_arr: torch.Tensor,
    student_arr: torch.Tensor,
    k_num: int=10,
) -> torch.Tensor:
    """Fraction of (B, T) positions where teacher top-k and student top-k token sequences match exactly."""
    __t = teacher_arr.topk(k_num, dim=-1).indices
    __s = student_arr.topk(k_num, dim=-1).indices
    # reduced to a Python float
    return (__t == __s).all(dim=-1).float().mean()

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
