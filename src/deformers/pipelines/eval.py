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

import os

import torch
import torch.nn.functional

import deformers.layers.prefix
import deformers.pipelines.patch

# METRICS ######################################################################

def embed_mse(
    teacher_arr: torch.Tensor,
    student_arr: torch.Tensor,
) -> float:
    """MSE between teacher and student embedding tensors (B, T, H)."""
    return torch.nn.functional.mse_loss(
        student_arr.float(),
        teacher_arr.float()).item()


def hidden_mse(
    teacher_arr: torch.Tensor,
    student_arr: torch.Tensor,
) -> float:
    """MSE between teacher and student hidden-state tensors (B, T, H)."""
    return torch.nn.functional.mse_loss(
        student_arr.float(),
        teacher_arr.float()).item()


def kl_divergence(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    reduction: str='batchmean',
) -> float:
    """
    KL divergence KL(teacher || student) over (B, T, V) logits.

    Assumptions:
    - Inputs are raw logits (not log-probs or probs).
    - Reduction is applied over the full (B*T, V) batch.
    - Teacher and student must have identical shapes.
    """
    assert teacher_logits.shape == student_logits.shape, (
        f'shape mismatch: teacher={teacher_logits.shape} student={student_logits.shape}')
    __B, __T, __V = teacher_logits.shape
    __t = teacher_logits.float().reshape(__B * __T, __V)
    __s = student_logits.float().reshape(__B * __T, __V)
    return torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(__s, dim=-1),
        torch.nn.functional.softmax(__t, dim=-1),
        reduction=reduction).item()


def top1_match_rate(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
) -> float:
    """
    Fraction of (B, T) positions where teacher and student argmax tokens match.

    Assumptions:
    - Inputs are raw logits of shape (B, T, V).
    """
    __t = teacher_logits.argmax(dim=-1)
    __s = student_logits.argmax(dim=-1)
    return (__t == __s).float().mean().item()


def topk_set_match_rate(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    k: int=10,
) -> float:
    """
    Fraction of (B, T) positions where teacher top-k and student top-k token sets are identical.

    Assumptions:
    - Inputs are raw logits of shape (B, T, V).
    - Set match: both top-k sets must contain exactly the same token IDs (order ignored).
    """
    __t = teacher_logits.topk(k, dim=-1).indices.sort(dim=-1).values
    __s = student_logits.topk(k, dim=-1).indices.sort(dim=-1).values
    return (__t == __s).all(dim=-1).float().mean().item()


def topk_order_match_rate(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    k: int=10,
) -> float:
    """
    Fraction of (B, T) positions where teacher top-k and student top-k token sequences match exactly (order-sensitive).

    Assumptions:
    - Inputs are raw logits of shape (B, T, V).
    - Exact order match: the top-k ranked token IDs must be identical in the same order.
    """
    __t = teacher_logits.topk(k, dim=-1).indices
    __s = student_logits.topk(k, dim=-1).indices
    return (__t == __s).all(dim=-1).float().mean().item()


# PROBES #######################################################################

def build_vocab_probe(
    vocab_size: int,
    batch_dim: int,
    seq_dim: int,
) -> torch.Tensor:
    """
    Build a deterministic (B, T) token id tensor using consecutive vocab IDs.

    Assumptions:
    - Fills positions with IDs 0, 1, 2, ..., (B*T - 1) mod vocab_size.
    - Deterministic: no randomness, same output for same arguments.
    """
    __total = batch_dim * seq_dim
    __ids = torch.arange(__total, dtype=torch.long) % vocab_size
    return __ids.reshape(batch_dim, seq_dim)


def build_text_probe(
    texts_arr: list,
    text_tokenizer: object,
    byte_tokenizer: object,
    seq_dim: int=256,
    patch_dim: int=32,
    device_str: str='cpu',
) -> tuple:
    """
    Build a deterministic fixed probe batch from text samples.

    Returns:
        tokens_arr: (B, T) long tensor of token ids
        mask_arr: (B, T) long tensor of attention masks
        bytes_arr: (B, T, G) long tensor of byte patches

    Assumptions:
    - Tokenizer boundaries are identical to the base model tokenizer.
    - Padding is right-padded to seq_dim.
    - Byte patch dimension G is patch_dim.
    """
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
    """
    Build byte patch tensor (B, T, G) from a (B, T) vocab probe token id tensor.

    Each token id is decoded to its text string using text_tokenizer, then
    re-encoded as a fixed-length byte block using byte_tokenizer.

    Assumptions:
    - text_tokenizer.decode([id]) returns the actual UTF-8 text for that token.
    - Byte patch dimension G is patch_dim.
    - Special token ids may produce empty or single-char strings.
    """
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


# TEACHER FORWARD ##############################################################

def teacher_embed(
    model: object,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Return the embedding layer output for the given token ids."""
    return model.model.embed_tokens(input_ids)


def teacher_forward(
    model: object,
    embeds: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple:
    """
    Run teacher trunk and lm_head forward with inputs_embeds.

    Returns:
        residuals: (B, T, H) hidden states at the last configured layer
        logits: (B, T, V) lm_head output

    Assumptions:
    - Model has .model(inputs_embeds=...) and .lm_head(...) interfaces.
    - No gradient is tracked (call within torch.no_grad()).
    """
    __residuals = model.model(
        inputs_embeds=embeds,
        attention_mask=attention_mask,
        use_cache=False).last_hidden_state
    __logits = model.lm_head(__residuals)
    return __residuals, __logits


# CHECKPOINT ###################################################################

def load_prefix_checkpoint(
    local_path: str,
    hf_repo: str='',
    hf_filename: str='prefix.pt',
    device_str: str='cpu',
) -> object:
    """
    Load a CompositeBytePrefix from a local checkpoint or HF hub path.

    Assumptions:
    - Checkpoint format: dict with keys 'config' and 'state_dict'.
    - local_path is required; hf_repo is optional override.
    - If hf_repo is non-empty, downloads the file and uses that path instead.
    """
    __path = local_path
    if hf_repo:
        import huggingface_hub
        __path = huggingface_hub.hf_hub_download(
            repo_id=hf_repo,
            filename=hf_filename,
            repo_type='model')
    if not os.path.isfile(__path):
        raise FileNotFoundError(f'prefix checkpoint not found: {__path}')
    __ckpt = torch.load(__path, map_location=device_str, weights_only=True)
    __prefix = deformers.layers.prefix.CompositeBytePrefix(**__ckpt['config'])
    __prefix.load_state_dict(__ckpt['state_dict'])
    __prefix.eval()
    return __prefix
