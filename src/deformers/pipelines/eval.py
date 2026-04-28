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
import math
import os

import torch
import torch.nn
import torch.nn.functional

import mlable.losses
import mlable.metrics

import deformers.models.prefix
import deformers.pipelines.patch

# CHECKPOINT ###################################################################

def load_prefix_checkpoint(
    path: str,
    shape: tuple,
    device: object=None,
) -> object:
    """Load a CompositeBytePrefix from a local checkpoint file.

    Fails fast with a clear error if the file is missing or unreadable.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f'[eval] prefix checkpoint not found: {path}')
    return deformers.models.prefix.CompositeBytePrefix.load_checkpoint(
        path=path, shape=shape, device=device)

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

def vocab_probe_bytes(
    vocab_ids: list[list[int]],
    text_tok: object,
    byte_tok: object,
    patch_dim: int=32,
) -> list[list[list[int]]]:
    """Build byte patches (B, T, G) from a (B, T) vocab probe token-id list.

    Decodes each token ID to its raw token string and encodes it as a
    fixed-length byte block, matching the training preprocessing path.
    """
    # decode each token ID to its raw string (one token at a time, no merging)
    __pad = text_tok.pad_token or ''
    __tokens = [
        [text_tok.decode([__i]).replace(__pad, '') for __i in __row]
        for __row in vocab_ids]
    # encode each token string as a fixed-length byte block
    return deformers.pipelines.patch.encode_into_bytes(
        tokens_arr=__tokens,
        patch_dim=patch_dim,
        tokenizer_obj=byte_tok)

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

# MODEL SUMMARY ################################################################

def model_summary(
    model_obj: object,
    input_shape: tuple,
    name: str='',
    depth: int=0,
) -> tuple:
    """Print a recursive model summary in forward-execution order.

    For each module, prints:
    - module name and class
    - _config (if present)
    - expected input and output shapes
    - own parameters (name, shape, dtype)
    - own buffers (name, shape, dtype)
    - direct sub-module names and classes

    Shape propagation uses output_shape() when available; otherwise the input
    shape is carried forward unchanged (e.g. for torch.nn primitives such as
    RMSNorm that preserve tensor shape).

    Returns the output shape after this module.

    Arguments:
        model_obj:   any torch.nn.Module instance.
        input_shape: (B, T, G) or similar tuple describing the tensor shape fed
                     to this module on a real forward pass.
        name:        name used to label this module (set by the parent).
        depth:       current recursion depth, controls indentation.
    """
    __pad = '  ' * depth
    __cls = type(model_obj).__name__
    __label = f'{name}: {__cls}' if name else __cls

    # compute output shape: use output_shape() when available
    if hasattr(model_obj, 'output_shape'):
        __out = tuple(model_obj.output_shape(input_shape))
    else:
        # propagate through children sequentially for container modules
        __cur = tuple(input_shape)
        for _, __child in model_obj.named_children():
            if hasattr(__child, 'output_shape'):
                __cur = tuple(__child.output_shape(__cur))
        __out = __cur

    # config (custom layers expose _config; nn primitives do not)
    __cfg = getattr(model_obj, '_config', None)

    # own parameters (direct, not recursive)
    __own_params = list(model_obj.named_parameters(recurse=False))
    __own_count = sum(__p.numel() for _, __p in __own_params)
    __total_count = sum(__p.numel() for __p in model_obj.parameters())

    # own buffers (direct, not recursive)
    __own_buffers = list(model_obj.named_buffers(recurse=False))

    # direct children
    __children = list(model_obj.named_children())
    __child_info = [f'{__n} ({type(__m).__name__})' for __n, __m in __children]

    # print header line
    print(f'{__pad}{__label}')
    print(f'{__pad}  in:      {input_shape}')
    print(f'{__pad}  out:     {__out}')

    # config block
    if __cfg is not None:
        print(f'{__pad}  config:  {__cfg}')

    # parameter summary
    print(f'{__pad}  params:  {__total_count:,} total  ({__own_count:,} own)')
    for __pname, __p in __own_params:
        print(f'{__pad}           {__pname}: {tuple(__p.shape)}  dtype={__p.dtype}')

    # buffer summary
    for __bname, __b in __own_buffers:
        print(f'{__pad}  buffer:  {__bname}: {tuple(__b.shape)}  dtype={__b.dtype}')

    # sub-module list (names only at this level; recursion below provides details)
    if __child_info:
        print(f'{__pad}  modules: {__child_info}')

    # recurse into children, carrying the current shape forward
    __cur_shape = tuple(input_shape)
    for __n, __child in __children:
        __cur_shape = model_summary(__child, __cur_shape, name=__n, depth=depth + 1)

    return __out
