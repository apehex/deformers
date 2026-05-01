import torch

import mlable.losses
import mlable.utils

import deformers.pipelines.patch

# GENERIC ######################################################################

def embed(
    indices_arr: torch.Tensor,
    model_obj: object,
) -> torch.Tensor:
    return model_obj.model.embed_tokens(indices_arr)

def forward(
    embeds_arr: torch.Tensor,
    mask_arr: torch.Tensor,
    model_obj: object,
) -> torch.Tensor:
    return model_obj.model(
        inputs_embeds=embeds_arr,
        attention_mask=mask_arr,
        use_cache=False).last_hidden_state

# PREPROCESSING ################################################################

def vectorize_strings(
    text_arr: list[str],
    text_tok: object,
    byte_tok: object,
    sequence_dim: int,
    patch_dim: int,
    dtype_obj: object=torch.long,
    device_str: str='cpu',
    left_pad: bool=True,
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
        left_pad=left_pad,
        tokenizer_obj=byte_tok)
    # format as tensors (B, T), (B, T), (B, T, G)
    return (
        torch.tensor(__inputs['attention_mask'], **__args),
        torch.tensor(__inputs['input_ids'], **__args),
        torch.tensor(__encoded, **__args),)

def vectorize_indices(
    indices_arr: list[list[int]],
    text_tok: object,
    byte_tok: object,
    sequence_dim: int,
    patch_dim: int,
    dtype_obj: object=torch.long,
    device_str: str='cpu',
    padding_str: str='',
    left_pad: bool=True,
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
        left_pad=left_pad,
        tokenizer_obj=byte_tok)
    # format as tensors (B, T), (B, T), (B, T, G)
    return (
        torch.tensor(__inputs['attention_mask'], **__args),
        torch.tensor(__inputs['input_ids'], **__args),
        torch.tensor(__encoded, **__args),)

# LOSS #########################################################################

def compute_losses(
    student_0_arr: torch.Tensor,
    student_k_arr: torch.Tensor,
    teacher_0_arr: torch.Tensor,
    teacher_k_arr: torch.Tensor,
    mask_arr: torch.Tensor=None,
    step_num: int=1,
    mse_0_rate: float=1.0,
    mse_k_rate: float=1.0,
    cos_0_rate: float=0.0,
    cos_k_rate: float=0.0,
    relative_opt: bool=True,
) -> tuple[torch.Tensor]:
    """Compute the combined embedding and hidden-state MSE loss."""
    assert any((__r > 0.0) for __r in [mse_0_rate, mse_k_rate, cos_0_rate, cos_k_rate])
    # default to 0 when a factor is null
    __mse_0, __mse_k, __cos_0, __cos_k = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
    # MSE on the embeddings (depth 0)
    if mse_0_rate > 0.0:
        __mse_0 = mlable.losses.mse_loss(
            predict_arr=student_0_arr.float(),
            target_arr=teacher_0_arr.float(),
            mask_arr=mask_arr,
            relative_opt=relative_opt,
            reduce_opt=True)
    # MSE on the hidden states at depth k
    if mse_k_rate > 0.0:
        __mse_k = mlable.losses.mse_loss(
            predict_arr=student_k_arr.float(),
            target_arr=teacher_k_arr.float(),
            mask_arr=mask_arr,
            relative_opt=relative_opt,
            reduce_opt=True)
    # KL divergence on the embeddings (depth 0)
    if cos_0_rate > 0.0:
        __cos_0 = mlable.losses.cos_sim(
            predict_arr=student_0_arr.float(),
            target_arr=teacher_0_arr.float(),
            mask_arr=mask_arr,
            reduce_opt=True)
    # MSE on the hidden states at depth k
    if cos_k_rate > 0.0:
        __cos_k = mlable.losses.cos_sim(
            predict_arr=student_k_arr.float(),
            target_arr=teacher_k_arr.float(),
            mask_arr=mask_arr,
            reduce_opt=True)
    # combine the losses
    __loss = mse_0_rate * __mse_0 + mse_k_rate * __mse_k + cos_0_rate * (1.0 - __cos_0) + cos_k_rate * (1.0 - __cos_k)
    # average over the gradient accumulation steps
    __factor = float(max(1, step_num))
    # return the components for monitoring
    return (
        __mse_0.detach() / __factor,
        __mse_k.detach() / __factor,
        __cos_0.detach() / __factor,
        __cos_k.detach() / __factor,
        __loss / __factor)

# FACTORIES ####################################################################

def prepare_batch_processor(
    text_tok: object,
    byte_tok: object,
    sequence_dim: int,
    patch_dim: int,
    dtype_obj: object=torch.long,
    device_str: str='cpu',
    padding_str: str='',
    pad_left_opt: bool=True,
    from_text_opt: bool=True,
) -> callable:
    # from a batch of integer indices
    def __vectorize(
            indices_arr: list[list[int]],
            state_arr: dict[str, dict],
        ) -> dict[str, dict]:
        # fix all the inputs
        __outputs = vectorize_indices(
            indices_arr=indices_arr,
            text_tok=text_tok,
            byte_tok=byte_tok,
            sequence_dim=sequence_dim,
            patch_dim=patch_dim,
            dtype_obj=dtype_obj,
            device_str=device_str,
            padding_str=padding_str,
            left_pad=pad_left_opt,)
        # update the training artifacts
        state_arr['tensors']['inputs/mask'] = __outputs[0]
        state_arr['tensors']['inputs/indices'] = __outputs[1]
        state_arr['tensors']['inputs/bytes'] = __outputs[2]
        # dict map with all the tensors and scalars
        return state_arr
    # from a batch of text samples
    if from_text_opt:
        def __vectorize(
                texts_arr: list[str],
                state_arr: dict[str, dict],
            ) -> dict[str, dict]:
            # fix all the inputs
            __outputs = vectorize_strings(
                text_arr=texts_arr,
                text_tok=text_tok,
                byte_tok=byte_tok,
                sequence_dim=sequence_dim,
                patch_dim=patch_dim,
                dtype_obj=dtype_obj,
                device_str=device_str,
                left_pad=pad_left_opt,)
            # update the training artifacts
            state_arr['tensors']['inputs/mask'] = __outputs[0]
            state_arr['tensors']['inputs/indices'] = __outputs[1]
            state_arr['tensors']['inputs/bytes'] = __outputs[2]
            # dict map with all the tensors and scalars
            return state_arr
    # specialized function updating the temp artifacts
    return __vectorize

def prepare_forward_processor(
    teacher_mod: object,
    student_mod: object,
    context_obj: object,
) -> callable:
    # calculate the teacher and student internals
    def __forward(
        state_arr: dict[str, dict],
    ) -> dict[str, dict]:
        # mixed precision / no-op context
        with context_obj:
            # teacher forward: get original embeddings and hidden states (no grad)
            with torch.no_grad():
                state_arr['tensors']['outputs/teacher/0'] = embed(
                    indices_arr=state_arr['tensors']['inputs/indices'],
                    model_obj=teacher_mod)
                state_arr['tensors']['outputs/teacher/k'] = forward(
                    embeds_arr=state_arr['tensors']['outputs/teacher/0'],
                    mask_arr=state_arr['tensors']['inputs/mask'],
                    model_obj=teacher_mod)
            # student forward: prefix -> inputs_embeds -> trunk -> hidden_k
            state_arr['tensors']['outputs/student/0'] = student_mod(state_arr['tensors']['inputs/bytes']).to(dtype=state_arr['tensors']['outputs/teacher/0'].dtype)
            state_arr['tensors']['outputs/student/k'] = forward(
                embeds_arr=state_arr['tensors']['outputs/student/0'],
                mask_arr=state_arr['tensors']['inputs/mask'],
                model_obj=teacher_mod)
        # dict map with all the tensors and scalars
        return state_arr
    # specialized function updating the training artifacts
    return __forward

def prepare_loss_processor(
    scaler_obj: object,
    context_obj: object,
    step_num: int=1,
    mse_0_rate: float=1.0,
    mse_k_rate: float=1.0,
    cos_0_rate: float=0.0,
    cos_k_rate: float=0.0,
    relative_opt: bool=True,
) -> callable:
    # updates the state
    def __score(
        state_arr: dict[str, dict],
    ) -> dict[str, dict]:
        # mixed precision / no-op context
        with context_obj:
            # fix all the inputs
            __outputs = compute_losses(
                mask_arr=state_arr['tensors']['inputs/mask'],
                student_0_arr=state_arr['tensors']['outputs/student/0'],
                student_k_arr=state_arr['tensors']['outputs/student/k'],
                teacher_0_arr=state_arr['tensors']['outputs/teacher/0'],
                teacher_k_arr=state_arr['tensors']['outputs/teacher/k'],
                step_num=step_num,
                mse_0_rate=mse_0_rate,
                mse_k_rate=mse_k_rate,
                cos_0_rate=cos_0_rate,
                cos_k_rate=cos_k_rate,
                relative_opt=relative_opt)
        # perform the backward computation
        scaler_obj.scale(__outputs[-1]).backward()
        # update the training state
        state_arr['scalars']['loss/mse/0'] += __outputs[0].item()
        state_arr['scalars']['loss/mse/k'] += __outputs[1].item()
        state_arr['scalars']['loss/cos/0'] += __outputs[2].item()
        state_arr['scalars']['loss/cos/k'] += __outputs[3].item()
        state_arr['scalars']['loss/total'] += __outputs[4].item()
        # dict map with all the tensors and scalars
        return state_arr
    # specialized function updating the training artifacts
    return __score

def prepare_gradient_processor(
    student_mod: object,
    scaler_obj: object,
    scheduler_obj: object,
    optimizer_obj: object,
    norm_max: float,
    acc_num: int,
    ema_num: int=256,
    ema_rate: float=0.99,
) -> callable:
    # update the state
    def __gradient(
        state_arr: dict[str, dict]
    ) -> dict[str, dict]:
        # accumulate the gradients for several steps
        if state_arr['scalars']['step/current'] % acc_num:
            # track the loss EMA, default to the current loss for the first 128 steps
            state_arr['scalars']['loss/ema'] = mlable.utils.ema(
                average=state_arr['scalars']['loss/ema'],
                current=state_arr['scalars']['loss/total'],
                factor=0.99 * float(state_arr['scalars']['step/current'] > 256))
            # gradient clipping; unscale first to get true grad norm
            scaler_obj.unscale_(optimizer_obj)
            state_arr['scalars']['gradient/rate'] = deformers.pipelines.monitor.current_lr(optimizer_obj)
            state_arr['scalars']['gradient/norm'] = torch.nn.utils.clip_grad_norm_(
                student_mod.parameters(),
                max_norm=norm_max).item()
            # update the weights
            scaler_obj.step(optimizer_obj)
            scaler_obj.update()
            scheduler_obj.step()
            optimizer_obj.zero_grad()
        # dict map with all the tensors and scalars
        return state_arr
    # specialized function updating the training artifacts
    return __gradient
