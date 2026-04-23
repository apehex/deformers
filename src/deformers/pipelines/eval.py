"""
Shared evaluation utilities for prefix/suffix patch experiments.

Assumptions:
- Logits tensors are (B, T, V) float, on the same device.
- Embedding tensors are (B, T, H) float.
- All reductions are over the (B, T) token positions; mask excludes padding.
- Vocab probe uses consecutive IDs modulo vocab_dim (deterministic, ascending).
- Text probe uses offset-based byte patching aligned to the base tokenizer boundaries.
- Checkpoint format: dict with keys 'config' and 'state'.
"""

import datetime
import json
import os

import torch
import torch.nn.functional

import deformers.layers.prefix
import deformers.pipelines.patch

# PROBE ########################################################################

def indices_probe(
    vocab_dim: int,
    batch_dim: int,
    sequence_dim: int,
) -> list[list[int]]:
    """Build a deterministic (B, T) token-ID list using consecutive vocab IDs."""
    # first indices / tokens of the vocabulary
    __ids = torch.arange(batch_dim * sequence_dim, dtype=torch.long) % vocab_dim
    # (B, T) integers
    return __ids.reshape(batch_dim, sequence_dim).tolist()

# METRICS ######################################################################

def masked_mse(
    predict_arr: torch.Tensor,
    target_arr: torch.Tensor,
    mask_arr: torch.Tensor,
) -> torch.Tensor:
    """Masked per-token MSE over (B, T, H) tensors.

    Computes mean squared error per token position (averaged over the H dim),
    then averages over valid (non-padding) token positions.
    mask_arr is (B, T) binary; zero entries are excluded from the mean.
    Returns a non-negative scalar.
    """
    # (B, T, H) -> (B, T)
    __per_token = ((predict_arr - target_arr) ** 2).mean(dim=-1)
    __mask = mask_arr.float()
    __denom = __mask.sum().clamp(min=1.0)
    return (__per_token * __mask).sum() / __denom


def masked_cosine(
    predict_arr: torch.Tensor,
    target_arr: torch.Tensor,
    mask_arr: torch.Tensor,
) -> torch.Tensor:
    """Masked mean cosine similarity over (B, T, H) tensors.

    Returns a scalar in [-1, 1]; higher means better alignment with the target.
    mask_arr is (B, T) binary; zero entries are excluded from the mean.
    """
    # (B, T)
    __cos = torch.nn.functional.cosine_similarity(predict_arr, target_arr, dim=-1)
    __mask = mask_arr.float()
    __denom = __mask.sum().clamp(min=1.0)
    return (__cos * __mask).sum() / __denom


def kl_divergence(
    student_arr: torch.Tensor,
    teacher_arr: torch.Tensor,
    mask_arr: torch.Tensor=None,
) -> torch.Tensor:
    """Per-token KL divergence KL(teacher || student), averaged over valid positions.

    student_arr, teacher_arr: (B, T, V) logits (not probabilities).
    mask_arr: (B, T) binary; if None, all positions are included.
    Returns a non-negative scalar.
    """
    # log-softmax of student (Q), softmax of teacher (P)
    __log_q = torch.nn.functional.log_softmax(student_arr, dim=-1)
    __p = torch.nn.functional.softmax(teacher_arr, dim=-1)
    # per-logit KL terms -> sum over vocab -> per-token KL: (B, T)
    __per_token = torch.nn.functional.kl_div(__log_q, __p, reduction='none').sum(dim=-1)
    if mask_arr is None:
        return __per_token.mean()
    __mask = mask_arr.float()
    __denom = __mask.sum().clamp(min=1.0)
    return (__per_token * __mask).sum() / __denom


def topk_rate(
    student_arr: torch.Tensor,
    teacher_arr: torch.Tensor,
    mask_arr: torch.Tensor=None,
    k_num: int=10,
    ordered: bool=False,
) -> torch.Tensor:
    """Top-k agreement rate between student and teacher logits.

    student_arr, teacher_arr: (B, T, V) logits.
    ordered=False: fraction of teacher top-k tokens present in student top-k (set match).
    ordered=True:  1.0 iff student and teacher top-k are in the exact same order.
    mask_arr: (B, T) binary; if None, all positions are included.
    Returns a scalar in [0, 1].
    """
    __k = min(k_num, student_arr.shape[-1])
    # (B, T, k) indices
    __t_top = teacher_arr.topk(__k, dim=-1).indices
    __s_top = student_arr.topk(__k, dim=-1).indices
    if ordered:
        # (B, T): all k positions must match
        __match = (__t_top == __s_top).all(dim=-1).float()
    else:
        # (B, T, k, 1) vs (B, T, 1, k): check if each teacher token appears in student top-k
        __overlap = (__t_top.unsqueeze(-1) == __s_top.unsqueeze(-2)).any(dim=-1).float()
        # (B, T): fraction of teacher top-k tokens found in student top-k
        __match = __overlap.mean(dim=-1)
    if mask_arr is None:
        return __match.mean()
    __mask = mask_arr.float()
    __denom = __mask.sum().clamp(min=1.0)
    return (__match * __mask).sum() / __denom


def top1_rate(
    student_arr: torch.Tensor,
    teacher_arr: torch.Tensor,
    mask_arr: torch.Tensor=None,
) -> torch.Tensor:
    """Top-1 agreement rate: fraction of positions where top prediction matches."""
    return topk_rate(student_arr, teacher_arr, mask_arr=mask_arr, k_num=1, ordered=True)

# INSPECTION ###################################################################

def per_token_metrics(
    token_ids_arr: torch.Tensor,
    student_embeds_arr: torch.Tensor,
    teacher_embeds_arr: torch.Tensor,
    student_hidden_arr: torch.Tensor,
    teacher_hidden_arr: torch.Tensor,
    student_logits_arr: torch.Tensor,
    teacher_logits_arr: torch.Tensor,
    mask_arr: torch.Tensor,
) -> list:
    """Build a per-token metric table for all valid (non-masked) positions.

    All tensors share the same (B, T) prefix dimensions.
    Optional tensors (hidden, logits) may be None; their columns are omitted.
    Returns a list of dicts sorted by embed_mse descending (hardest tokens first).
    """
    __B, __T = mask_arr.shape
    # per-token MSE: (B, T)
    __embed_mse = ((student_embeds_arr - teacher_embeds_arr) ** 2).mean(dim=-1)
    # per-token cosine similarity: (B, T)
    __embed_cos = torch.nn.functional.cosine_similarity(
        student_embeds_arr.float(), teacher_embeds_arr.float(), dim=-1)
    # per-token hidden MSE: (B, T)
    __hidden_mse = None
    __hidden_cos = None
    if student_hidden_arr is not None and teacher_hidden_arr is not None:
        __hidden_mse = ((student_hidden_arr - teacher_hidden_arr) ** 2).mean(dim=-1)
        __hidden_cos = torch.nn.functional.cosine_similarity(
            student_hidden_arr.float(), teacher_hidden_arr.float(), dim=-1)
    # per-token KL and top-1 match: (B, T)
    __kl = None
    __top1 = None
    if student_logits_arr is not None and teacher_logits_arr is not None:
        __log_q = torch.nn.functional.log_softmax(student_logits_arr.float(), dim=-1)
        __p = torch.nn.functional.softmax(teacher_logits_arr.float(), dim=-1)
        __kl = torch.nn.functional.kl_div(__log_q, __p, reduction='none').sum(dim=-1)
        __t1_t = teacher_logits_arr.argmax(dim=-1)
        __t1_s = student_logits_arr.argmax(dim=-1)
        __top1 = (__t1_t == __t1_s).float()
    # flatten and filter by mask
    __records = []
    for __b in range(__B):
        for __t in range(__T):
            if mask_arr[__b, __t].item() == 0:
                continue
            __rec = {
                'token_id': int(token_ids_arr[__b, __t].item()),
                'embed_mse': float(__embed_mse[__b, __t].item()),
                'embed_cosine': float(__embed_cos[__b, __t].item()),
            }
            if __hidden_mse is not None:
                __rec['hidden_mse'] = float(__hidden_mse[__b, __t].item())
                __rec['hidden_cosine'] = float(__hidden_cos[__b, __t].item())
            if __kl is not None:
                __rec['kl'] = float(__kl[__b, __t].item())
                __rec['top1_match'] = int(__top1[__b, __t].item())
            __records.append(__rec)
    # sort by embed_mse descending (hardest tokens first)
    __records.sort(key=lambda __r: __r['embed_mse'], reverse=True)
    return __records


def aggregate_metrics(
    values_arr: list,
) -> dict:
    """Compute summary statistics (mean, median, p95) from a list of scalar values.

    Returns a dict with keys 'mean', 'median', 'p95'.
    Returns zeros for an empty list.
    """
    if not values_arr:
        return {'mean': 0.0, 'median': 0.0, 'p95': 0.0}
    __t = torch.tensor(values_arr, dtype=torch.float32)
    return {
        'mean': float(__t.mean().item()),
        'median': float(__t.median().item()),
        'p95': float(__t.quantile(0.95).item()),
    }

# REPORT #######################################################################

def save_report(
    report_dict: dict,
    log_dir: str,
    stem: str='benchmark',
) -> str:
    """Serialize a report dict to a timestamped JSON file in log_dir.

    Returns the path of the saved JSON file.
    Creates log_dir if it does not exist.
    """
    os.makedirs(log_dir, exist_ok=True)
    __ts = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S')
    __json_path = os.path.join(log_dir, f'{stem}-{__ts}.json')
    with open(__json_path, 'w') as __f:
        json.dump(report_dict, __f, indent=2)
    return __json_path

# CHECKPOINT ###################################################################

def load_prefix_checkpoint(
    path: str,
    shape: tuple,
    device: str='cpu',
) -> object:
    """Load a CompositeBytePrefix checkpoint from a local .pt file.

    path:   absolute or relative path to the checkpoint file.
    shape:  input shape tuple (B, T, G) used to build the lazy sub-layers.
    device: target device string ('cpu' or 'cuda').

    Raises AssertionError with a clear diagnostic if the file is missing.
    """
    assert os.path.isfile(path), (
        f'[eval] prefix checkpoint not found: {path}\n'
        f'       In Colab: upload the checkpoint to /content/checkpoints/ first.\n'
        f'       Or pass --checkpoint /path/to/prefix.pt on the command line.')
    return deformers.layers.prefix.CompositeBytePrefix.load_checkpoint(
        path=path,
        shape=shape,
        device=device)
