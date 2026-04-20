import torch

import mlable.losses

import deformers.pipelines.patch
import deformers.tokenizers.generic

# PREPROCESSING ################################################################

def tensors_from_strings(
    text_arr: list[str],
    text_tok: object,
    byte_tok: object,
    sequence_dim: int,
    patch_dim: int,
    dtype_obj: object=torch.long,
    device_str: str='cpu',
) -> tuple[torch.Tensor]:
    # common casting arguments
    __args = {'dtype': dtype_obj, 'device': device_str,}
    # input_ids (B, T) and attention_mask (B, T)
    __inputs = text_tok(
        text_arr,
        return_offsets_mapping=True,
        max_length=sequence_dim,
        truncation='longest_first',
        padding='max_length')
    # byte patches (B, T, G)
    __encoded = deformers.pipelines.patch.tokenize_into_bytes(
        texts_arr=text_arr,
        offsets_arr=__inputs['offset_mapping'],
        patch_dim=patch_dim,
        tokenizer_obj=byte_tok)
    # format as tensors
    __mask_arr = torch.tensor(__inputs['attention_mask'], **__args)
    __indices_arr = torch.tensor(__inputs['input_ids'], **__args)
    __bytes_arr = torch.tensor(__encoded, **__args)
    # (B, T), (B, T), (B, T, G)
    return __mask_arr, __indices_arr, __bytes_arr

def tensors_from_indices(
    indices_arr: list[list[int]],
    text_tok: object,
    byte_tok: object,
    sequence_dim: int,
    patch_dim: int,
    dtype_obj: object=torch.long,
    device_str: str='cpu',
    padding_str: str='',
) -> tuple[torch.Tensor]:
    # common casting arguments
    __args = {'dtype': dtype_obj, 'device': device_str,}
    # input_ids (B, T) and attention_mask (B, T)
    __inputs = text_tok.pad(
        {'input_ids': indices_arr},
        max_length=sequence_dim,
        padding='max_length')
    # translaste the IDs into tokens
    __tokens = [
        [text_tok.decode(__i).replace(text_tok.pad_token, padding_str) for __i in __r]
        for __r in __inputs['input_ids']]
    # byte patches (B, T, G)
    __encoded = deformers.pipelines.patch.encode_into_bytes(
        tokens_arr=__tokens,
        patch_dim=patch_dim,
        tokenizer_obj=byte_tok)
    # format as tensors
    __mask_arr = torch.tensor(__inputs['attention_mask'], **__args)
    __indices_arr = torch.tensor(__inputs['input_ids'], **__args)
    __bytes_arr = torch.tensor(__encoded, **__args)
    # (B, T), (B, T), (B, T, G)
    return __mask_arr, __indices_arr, __bytes_arr

# LOSS #########################################################################

def compute_losses(
    student_0_arr: torch.Tensor,
    student_k_arr: torch.Tensor,
    teacher_0_arr: torch.Tensor,
    teacher_k_arr: torch.Tensor,
    mask_arr: torch.Tensor,
    step_num: int,
    mse_0_rate: float=1.0,
    mse_k_rate: float=1.0,
    kld_0_rate: float=0.0,
    kld_k_rate: float=0.0,
) -> tuple[torch.Tensor]:
    """Compute the combined embedding and hidden-state MSE loss."""
    assert any((__r > 0.0) for __r in [mse_0_rate, mse_k_rate, kld_0_rate, kld_k_rate])
    # default to 0 when a factor is null
    __mse_0, __mse_k, __kld_0, __kld_k = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
    # MSE on the embeddings (depth 0)
    if mse_0_rate > 0.0:
        __mse_0 = mlable.losses.mse_loss(
            predict_arr=student_0_arr.float(),
            target_arr=teacher_0_arr.float(),
            mask_arr=mask_arr)
    # MSE on the hidden states at depth k
    if mse_k_rate > 0.0:
        __mse_k = mlable.losses.mse_loss(
            predict_arr=student_k_arr.float(),
            target_arr=teacher_k_arr.float(),
            mask_arr=mask_arr)
    # KL divergence on the embeddings (depth 0)
    if kld_0_rate > 0.0:
        __kld_0 = mlable.losses.kl_div(
            predict_arr=student_0_arr.float(),
            target_arr=teacher_0_arr.float(),
            mask_arr=mask_arr)
    # MSE on the hidden states at depth k
    if kld_k_rate > 0.0:
        __kld_k = mlable.losses.kl_div(
            predict_arr=student_k_arr.float(),
            target_arr=teacher_k_arr.float(),
            mask_arr=mask_arr)
    # combine the losses
    __loss = mse_0_rate * __mse_0 + mse_k_rate * __mse_k + kld_0_rate * __kld_0 + kld_k_rate * __kld_k
    # average over the gradient accumulation steps
    __factor = float(max(1, step_num))
    # return the components for monitoring
    return (
        __mse_0.detach() / __factor,
        __mse_k.detach() / __factor,
        __kld_0.detach() / __factor,
        __kld_k.detach() / __factor,
        __loss / __factor)
