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
import os

import torch
import torch.nn.functional

import deformers.layers.prefix
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
    # reduced to a scalar
    return (__t == __s).all(dim=-1).float().mean()

# PROBES #######################################################################

def build_vocab_probe(
    vocab_dim: int,
    batch_dim: int,
    seq_dim: int,
) -> torch.Tensor:
    """Build a deterministic (B, T) token id tensor using consecutive vocab IDs."""
    __total = batch_dim * seq_dim
    __ids = torch.arange(__total, dtype=torch.long) % vocab_dim
    return __ids.reshape(batch_dim, seq_dim)

def build_text_probe(
    texts_arr: list,
    text_tokenizer: object,
    byte_tokenizer: object,
    seq_dim: int=256,
    patch_dim: int=32,
    device_str: str='cpu',
) -> tuple:
    """Build a deterministic fixed probe batch from text samples."""
    __inputs = text_tokenizer(
        texts_arr,
        return_offsets_mapping=True,
        max_length=seq_dim,
        truncation='longest_first',
        padding='max_length')
    __encoded = deformers.pipelines.patch.tokenize_into_bytes(
        texts_arr=texts_arr,
        offsets_arr=__inputs['offset_mapping'],
        patch_dim=patch_dim,
        tokenizer_obj=byte_tokenizer)
    __tokens_arr = torch.tensor(__inputs['input_ids'], dtype=torch.long, device=device_str)
    __mask_arr = torch.tensor(__inputs['attention_mask'], dtype=torch.long, device=device_str)
    __bytes_arr = torch.tensor(__encoded, dtype=torch.long, device=device_str)
    return __tokens_arr, __mask_arr, __bytes_arr

def build_vocab_probe_bytes(
    vocab_ids: torch.Tensor,
    text_tokenizer: object,
    byte_tokenizer: object,
    patch_dim: int=32,
) -> torch.Tensor:
    """Build byte patch tensor (B, T, G) from a (B, T) vocab probe token id tensor."""
    __B, __T = vocab_ids.shape
    __flat = vocab_ids.flatten().tolist()
    # decode each token id to its actual text representation
    __strings = [text_tokenizer.decode([__tid], skip_special_tokens=False) for __tid in __flat]
    # reshape into (B, T) list of lists
    __tokens_2d = [__strings[__i * __T:(__i + 1) * __T] for __i in range(__B)]
    # encode each token string as a fixed-length byte block
    __encoded = deformers.pipelines.patch.encode_into_bytes(
        tokens_arr=__tokens_2d,
        patch_dim=patch_dim,
        tokenizer_obj=byte_tokenizer)
    return torch.tensor(__encoded, dtype=torch.long, device=vocab_ids.device)
