"""
Shared evaluation utilities for prefix/suffix patch experiments.

Assumptions:
- Logits tensors are (B, T, V) float, on the same device.
- Embedding tensors are (B, T, H) float.
- Mask tensors are (B, T) float or long, 1 for real tokens, 0 for padding.
- Vocab probe uses consecutive vocab IDs tiled over (B, T) positions.
- Text probe uses offset-based byte patching aligned to the base tokenizer boundaries.
- Checkpoint format: dict with keys 'config' and 'state'.
"""

import json
import os

import torch
import torch.nn.functional

import mlable.losses
import mlable.metrics

# PROBE ########################################################################

def indices_probe(
    vocab_dim: int,
    batch_dim: int,
    sequence_dim: int
) -> list[list[int]]:
    """Build a deterministic (B, T) token-id tensor cycling over the vocabulary."""
    # first indices / tokens of the vocabulary
    __ids = torch.arange(batch_dim * sequence_dim, dtype=torch.long) % vocab_dim
    # (B, T) integers
    return __ids.reshape(batch_dim, sequence_dim).tolist()

# METRICS ######################################################################

def per_token_metrics(
    teacher_embeds: torch.Tensor,
    student_embeds: torch.Tensor,
    teacher_hidden: torch.Tensor,
    student_hidden: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    mask: torch.Tensor,
    k_num: int=10,
) -> dict:
    """Compute per-position (B, T) unreduced metrics, zeroed outside the mask.

    Returns a dict with keys: embed_mse, embed_cos, hidden_mse, hidden_cos,
    kl, top1, topk.  All values are (B, T) CPU float tensors.
    """
    # cast to float for metric computation
    __te = teacher_embeds.float()
    __se = student_embeds.float()
    __th = teacher_hidden.float()
    __sh = student_hidden.float()
    __tl = teacher_logits.float()
    __sl = student_logits.float()
    return {
        'embed_mse':  mlable.losses.mse_loss(__se, __te, mask_arr=mask, relative_opt=True, reduce_opt=False).cpu(),
        'embed_cos':  mlable.losses.cos_sim(__se, __te, mask_arr=mask, reduce_opt=False).cpu(),
        'hidden_mse': mlable.losses.mse_loss(__sh, __th, mask_arr=mask, relative_opt=True, reduce_opt=False).cpu(),
        'hidden_cos': mlable.losses.cos_sim(__sh, __th, mask_arr=mask, reduce_opt=False).cpu(),
        'kl':         mlable.losses.kl_div(__sl, __tl, mask_arr=mask, reduce_opt=False).cpu(),
        'top1':       mlable.metrics.topk_rate(__sl, __tl, mask_arr=mask, reduce_opt=False, k_num=1).cpu(),
        'topk':       mlable.metrics.topk_rate(__sl, __tl, mask_arr=mask, reduce_opt=False, k_num=k_num).cpu(),}

# STATISTICS ###################################################################

def summary_stats(
    values: torch.Tensor,
    mask: torch.Tensor=None,
) -> dict:
    """Compute mean, median, and p95 of values at masked positions.

    values: (B, T) or (N,) float tensor.
    mask:   (B, T) or (N,) bool/int tensor, 1 = valid position.
            If None, all positions are used.
    Returns dict with float keys 'mean', 'median', 'p95'.
    """
    __v = values.float().flatten()
    if mask is not None:
        __m = mask.bool().flatten()
        __v = __v[__m]
    if __v.numel() == 0:
        return {'mean': 0.0, 'median': 0.0, 'p95': 0.0}
    return {
        'mean':   float(__v.mean().item()),
        'median': float(__v.median().item()),
        'p95':    float(__v.quantile(0.95).item()),}

# TABLE ########################################################################

def token_table(
    token_ids: list,
    token_strings: list,
    metrics: dict,
) -> list:
    """Build a per-token row list from flat lists of ids, strings, and metric values.

    token_ids:     list of N token IDs.
    token_strings: list of N token strings.
    metrics:       dict mapping metric name -> list/tensor of N float values.
    Returns a list of N dicts, one per token position.
    """
    __n = len(token_ids)
    __rows = []
    for __i in range(__n):
        __row = {
            'token_id':     int(token_ids[__i]),
            'token_string': str(token_strings[__i]),
            'byte_length':  len(str(token_strings[__i]).encode('utf-8')),}
        for __key, __vals in metrics.items():
            try:
                __row[__key] = float(__vals[__i])
            except (TypeError, IndexError):
                __row[__key] = 0.0
        __rows.append(__row)
    return __rows

# REPORT #######################################################################

def save_json_report(
    report: dict,
    path: str,
) -> None:
    """Serialize report to a JSON file, creating parent directories as needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as __f:
        json.dump(report, __f, indent=2, ensure_ascii=True)
