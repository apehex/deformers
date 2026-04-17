"""
Replace

Trains the CompositeBytePrefix module to distill from qwen/qwen3.5-9b by
injecting prefix outputs via inputs_embeds into the frozen trunk.

Assumptions:
- Base model is qwen/qwen3.5-9b with hidden_size=4096.
- Tokenizer boundaries are identical to the base model.
- Trunk is frozen during training; only the prefix parameters are trained.
- Byte block size default follows docs/roadmap.md (L_max=32), configurable.
- The byte tokenizer uses pad_id=128 (as implemented by ByteTokenizer).

Training scheme (Stage A):
- Teacher: qwen/qwen3.5-9b forward with original input_ids (no grad).
- Student: CompositeBytePrefix forward then trunk forward with inputs_embeds.
- Loss: hidden-state MSE at depth k (configurable), optional embedding MSE.
- Only prefix parameters are updated; trunk is frozen.

Monitoring (per optimizer step):
- Progress bar (tqdm) with epoch, step, lr, embed MSE, hidden MSE, total loss, KL.
- TensorBoard scalars: train/loss_total, train/loss_hidden, train/loss_embed,
  train/lr, train/grad_norm, train/step_time_ms, gpu/memory_allocated_mb,
  gpu/memory_reserved_mb.
- KL divergence is computed for monitoring only (not added to the optimization loss).
  It uses lm_head applied to the last micro-batch's hidden states (first item only
  to limit memory overhead).
- Accumulated losses (embed, hidden, total) are unscaled per-step means.
"""

import contextlib
import os
import time

import datasets
import huggingface_hub
import torch
import torch.amp
import torch.nn
import torch.optim
import torch.utils.tensorboard
import tqdm
import transformers

import mlable.models
import mlable.schedulers
import mlable.utils

import deformers.layers.prefix
import deformers.models.generic
import deformers.pipelines.eval
import deformers.pipelines.monitor
import deformers.pipelines.patch
import deformers.tokenizers.byte

# COMMON CONFIG ################################################################

MAIN_CFG = {
    'resume_opt': True,
    'model_str': 'qwen/qwen3.5-9b',
    'device_str': 'cuda' if torch.cuda.is_available() else 'cpu',
    'encoding_str': 'utf-8',
    'seed_num': 1337,
    'batch_dim': 64,
    'sequence_dim': 256,
    'patch_dim': 32,
    'depth_num': 1,
    'epoch_num': 4,
    'accumulation_num': 4,}

# DATA CONFIG ##################################################################

DATASET_CFG = {
    'path': 'wikimedia/wikipedia',
    'name': '20231101.en',
    'split': 'train[:10%]',
    'streaming': False,}

BATCH_CFG = {
    'batch_dim': MAIN_CFG['batch_dim'],
    'sequence_dim': MAIN_CFG['sequence_dim'],
    'patch_dim': MAIN_CFG['patch_dim'],}

# PREPROCESSING CONFIG #########################################################

TOKEN_CFG = {
    'pretrained_model_name_or_path': MAIN_CFG['model_str'],
    'use_fast': True,}

BYTE_CFG = {
    'encoding': MAIN_CFG['encoding_str'],}

# MODEL CONFIG #################################################################

DOWNLOAD_CFG = {
    'repo_id': MAIN_CFG['model_str'],
    'repo_type': 'model',
    'local_dir': os.path.abspath('downloads'),
    'ignore_patterns': ['*.onnx', '*.tflite', '*.msgpack'],}

CONFIG_CFG = {
    'pretrained_model_name_or_path': DOWNLOAD_CFG['local_dir'],
    'trust_remote_code': False,}

MODEL_CFG = {
    'pretrained_model_name_or_path': DOWNLOAD_CFG['local_dir'],
    'trust_remote_code': CONFIG_CFG['trust_remote_code'],
    'dtype': torch.bfloat16,
    'low_cpu_mem_usage': True,
    'ignore_mismatched_sizes': True,}

PREFIX_CFG = {
    'embed_dim': 256, # 32 * 256 = 8192
    'vocab_dim': 256,
    'latent_dim': 4096,
    'group_dim': -1,}

# TRAINING CONFIG ##############################################################

TRAINING_CFG = {
    'epoch_num': MAIN_CFG['epoch_num'],}

OPTIMIZER_CFG = {
    'lr': 1e-4,
    'betas': (0.9, 0.999),
    'weight_decay': 0.01,}

SCALER_CFG = {
    'enabled': MAIN_CFG['device_str'] == 'cuda',}

GRADIENT_CFG = {
    'accumulation_num': MAIN_CFG['accumulation_num'],
    'max_norm': 1.0,}

LOSS_CFG = {
    'embeds_rate': 1.0,
    'residuals_rate': 1.0,}

# OUTPUT CONFIG ################################################################

LOGGING_CFG = {
    'step_num': 32,
    'log_path': os.path.abspath('logs/prefix.log'),}

OUTPUT_CFG = {
    'save_path': os.path.abspath('checkpoints/prefix.pt'),}

# PREPROC UTILS ################################################################

def compute_tensors(
    text_arr: list[str],
    text_tok: object,
    byte_tok: object,
    dtype_obj: object=torch.long,
    sequence_dim: int=BATCH_CFG['sequence_dim'],
    patch_dim: int=BATCH_CFG['patch_dim'],
    device_str: str=MAIN_CFG['device_str']
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
    __tokens_arr = torch.tensor(__inputs['input_ids'], **__args)
    __mask_arr = torch.tensor(__inputs['attention_mask'], **__args)
    __bytes_arr = torch.tensor(__encoded, **__args)
    # (B, T), (B, T), (B, T, G)
    return __mask_arr, __tokens_arr, __bytes_arr

# LOSS UTILS ###################################################################

def compute_losses(
    student_embeds: torch.Tensor,
    teacher_embeds: torch.Tensor,
    student_residuals: torch.Tensor,
    teacher_residuals: torch.Tensor,
    embeds_rate: float=LOSS_CFG['embeds_rate'],
    residuals_rate: float=LOSS_CFG['residuals_rate'],
    accumulation_num: int=GRADIENT_CFG['accumulation_num'],
) -> tuple:
    """Compute the combined embedding and hidden-state MSE loss."""
    # MSE on the embeddings
    __loss_embeds = torch.nn.functional.mse_loss(
        student_embeds.float(),
        teacher_embeds.float())
    # MSE on the hidden states at depth k
    __loss_residuals = torch.nn.functional.mse_loss(
        student_residuals.float(),
        teacher_residuals.float())
    # combine the losses
    __loss = residuals_rate * __loss_residuals + embeds_rate * __loss_embeds
    # average over the gradient accumulation steps
    __factor = float(max(1, accumulation_num))
    # return the components for monitoring
    return (__loss_embeds.detach() / __factor, __loss_residuals.detach() / __factor, __loss / __factor)

# LOGGING UTILS ################################################################

def init_state() -> dict:
    return {
        'train/iter/start': time.monotonic(),
        'train/iter/time': 0.0,
        'train/iter/tps': 0.0,
        'train/gradient/rate': 0.0,
        'train/gradient/norm': 0.0,
        'train/loss/ema': 1.0,
        'train/loss/total': 0.0,
        'train/loss/embed': 0.0,
        'train/loss/hidden': 0.0,
        'train/loss/kldiv': 0.0,
        'train/vocab/seen': 0.0,
        'train/vocab/max': 0.0,
        'gpu/memory/allocated': 0.0,
        'gpu/memory/reserved': 0.0,}

def reset_state(state: dict, ignore: list=[]) -> dict:
    """Reset all the tracked state variables."""
    return {
        __k: __v if (__k in ignore) else (time.monotonic() if (__k == 'train/iter/start') else 0.0)
        for (__k, __v) in state.items()}

def format_state(
    state: dict,
    epoch_num: int,
    epoch_tot: int,
    step_num: int,
    step_tot: int,
) -> dict:
    """Group and format the state variables to export them."""
    return {
        'epoch': f"({epoch_num}/{epoch_tot})",
        'step': f"({step_num}/{step_tot})",
        'loss': f"(ema: {state['train/loss/ema']:.6f} total: {state['train/loss/total']:.6f} embed: {state['train/loss/embed']:.6f} hidden: {state['train/loss/hidden']:.6f} kl: {state['train/loss/kldiv']:.6f})",
        'gradient': f"(rate: {state['train/gradient/rate']:.2e} norm: {state['train/gradient/norm']:.4f})",
        'iter': f"(time: {state['train/iter/time'] * 1000.0:.0f} tok/s: {state['train/iter/tps']:.0f})",
        'vocab': f"(seen: {state['train/vocab/seen'] * 100.0:.1f}% max: {state['train/vocab/max'] * 100.0:.1f}%)",}

def serialize_state(
    state: dict,
    prefix: str='[train] ',
) -> str:
    """Serialize the state variables into a single string."""
    return prefix + ' '.join([
        f'{__k}{state[__k]}'
        for __k in state.keys()])

# DATASET ######################################################################

print('[init] downloading the dataset...')
DATASET_OBJ = datasets.load_dataset(**DATASET_CFG)

print('[init] preprocessing the dataset...')
DATASET_OBJ = DATASET_OBJ.shuffle(seed=MAIN_CFG['seed_num'])

print('[init] calculating the training metadata...')
DATASET_DIM = len(DATASET_OBJ) // BATCH_CFG['batch_dim']
BATCH_LEN = BATCH_CFG['batch_dim'] * BATCH_CFG['sequence_dim'] * GRADIENT_CFG['accumulation_num']

# TOKENIZERS ###################################################################

print('[init] loading the tokenizers...')
TEXT_TOK = transformers.AutoTokenizer.from_pretrained(**TOKEN_CFG)
BYTE_TOK = deformers.tokenizers.byte.ByteTokenizer(**BYTE_CFG)

print('[init] calculating the tokenizer metadata...')
VOCAB_ARR = {__v: __k for (__k, __v) in TEXT_TOK.get_vocab().items()}
VOCAB_LEN = len(VOCAB_ARR)

# MODELS #######################################################################

print('[init] creating the output directories...')
os.makedirs(DOWNLOAD_CFG['local_dir'], exist_ok=True)
os.makedirs(os.path.dirname(LOGGING_CFG['log_path']), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_CFG['save_path']), exist_ok=True)

print('[init] downloading the teacher...')
huggingface_hub.snapshot_download(**DOWNLOAD_CFG)

print('[init] loading the config...')
TRUNK_CFG = transformers.AutoConfig.from_pretrained(**CONFIG_CFG)

print('[init] truncating the config...')
TRUNK_CFG = deformers.models.generic.truncate_config(TRUNK_CFG, layer_num=MAIN_CFG['depth_num'], target_key='text_config')

print('[init] loading the teacher...') # load only the used layers, up to the chose depth
SOURCE_MOD = transformers.AutoModelForCausalLM.from_pretrained(config=TRUNK_CFG, **MODEL_CFG).to(device=MAIN_CFG['device_str'])

print('[init] freezing the teacher...')
SOURCE_MOD.eval()
mlable.models.freeze(SOURCE_MOD)

print('[init] creating the student...')
PREFIX_MOD = deformers.layers.prefix.CompositeBytePrefix(**PREFIX_CFG).to(device=MAIN_CFG['device_str'])
if MAIN_CFG['resume_opt'] and os.path.exists(OUTPUT_CFG['save_path']):
    PREFIX_MOD = deformers.layers.prefix.CompositeBytePrefix.load_checkpoint(
        path=OUTPUT_CFG['save_path'],
        device=MAIN_CFG['device_str'])

print('[init] freeing the unused layers...')
mlable.models.free_memory()

print('[init] building the student...')
PREFIX_MOD.build(
    shape=(BATCH_CFG['batch_dim'], BATCH_CFG['sequence_dim'], BATCH_CFG['patch_dim']),
    device=MAIN_CFG['device_str'],
    dtype=torch.float32)

# OPTIMIZER ####################################################################

print('[init] creating optimizer...')
OPTIMIZER_OBJ = torch.optim.AdamW(PREFIX_MOD.parameters(), **OPTIMIZER_CFG)
SCALER_OBJ = torch.amp.GradScaler(**SCALER_CFG)

print('[init] creating scheduler...')
SCHEDULER_OBJ = mlable.schedulers.WaveLR(
    optimizer_obj=OPTIMIZER_OBJ,
    start_rate=1e-4,
    end_rate=1e-2,
    total_num=(DATASET_DIM * TRAINING_CFG['epoch_num']) // GRADIENT_CFG['accumulation_num'],
    warmup_num=min(128, DATASET_DIM // GRADIENT_CFG['accumulation_num']))

print('[init] enabling mixed precision...')
MIXED_CTX = (
    torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    if SCALER_CFG['enabled']
    else contextlib.nullcontext())

# LOGGING ######################################################################

print('[init] logging to {}...'.format(os.path.dirname(LOGGING_CFG['log_path'])))
LOG_TB = torch.utils.tensorboard.SummaryWriter(log_dir=os.path.dirname(LOGGING_CFG['log_path']))
LOG_FILE = open(LOGGING_CFG['log_path'], 'w')

# ZERO #########################################################################

print('[init] zeroing the state...')
__step = 0
__state = init_state()
__count = torch.zeros(size=(VOCAB_LEN,), dtype=torch.long, device='cpu')

OPTIMIZER_OBJ.zero_grad()

# TRAINING #####################################################################

for __epoch in range(TRAINING_CFG['epoch_num']):
    # create a new iterator since the previous one was exhausted
    __dataset = DATASET_OBJ.iter(batch_size=BATCH_CFG['batch_dim'])

    # progress bar for this epoch (advances every micro-step)
    __pbar = tqdm.tqdm(
        __dataset,
        total=DATASET_DIM,
        desc=f'epoch {__epoch + 1}/{TRAINING_CFG["epoch_num"]}',
        unit='batch',
        leave=True)

    # reset the accumulators and start the timing
    __state = reset_state(state=__state, ignore=['train/loss/ema'])

    for __batch in __pbar:
        # list of plain strings (B,)
        __texts = __batch['text']

        # mask (B, T), tokens (B, T), bytes (B, T, G) integers
        __mask_arr, __tokens_arr, __bytes_arr = compute_tensors(
            text_arr=__texts,
            text_tok=TEXT_TOK,
            byte_tok=BYTE_TOK,
            dtype_obj=torch.long,
            sequence_dim=BATCH_CFG['sequence_dim'],
            patch_dim=BATCH_CFG['patch_dim'],
            device_str=MAIN_CFG['device_str'])

        # track token stats
        __count += torch.bincount(__tokens_arr.flatten().cpu(), minlength=VOCAB_LEN)
        __state['train/vocab/seen'] = float((__count > 0).sum().item()) / VOCAB_LEN
        __state['train/vocab/max'] = float(__count.max().item()) / float(__count.sum().item())

        # teacher forward: get original embeddings and hidden states (no grad)
        with torch.no_grad():
            __teacher_embeds = SOURCE_MOD.model.embed_tokens(__tokens_arr)
            __teacher_residuals = SOURCE_MOD.model(
                inputs_embeds=__teacher_embeds,
                attention_mask=__mask_arr,
                use_cache=False).last_hidden_state

        # student forward: prefix -> inputs_embeds -> trunk -> hidden_k
        with MIXED_CTX:
            __student_embeds = PREFIX_MOD(__bytes_arr).to(dtype=__teacher_embeds.dtype)
            __student_residuals = SOURCE_MOD.model(
                inputs_embeds=__student_embeds,
                attention_mask=__mask_arr,
                use_cache=False).last_hidden_state

            # combination of the MSE at depth 0 and k
            __losses = compute_losses(
                teacher_embeds=__teacher_embeds,
                student_embeds=__student_embeds,
                teacher_residuals=__teacher_residuals,
                student_residuals=__student_residuals,
                embeds_rate=LOSS_CFG['embeds_rate'],
                residuals_rate=LOSS_CFG['residuals_rate'],
                accumulation_num=GRADIENT_CFG['accumulation_num'])

        # perform the backward propagation of the loss
        SCALER_OBJ.scale(__losses[-1]).backward()

        # the total loss is the average loss after N accumulation steps
        __state['train/loss/embed'] += __losses[0].item()
        __state['train/loss/hidden'] += __losses[1].item()
        __state['train/loss/total'] += __losses[-1].item()

        # optimizer step after gradient accumulation
        if (__step + 1) % GRADIENT_CFG['accumulation_num'] == 0:
            # compute KL from the hidden states (monitoring only)
            __state['train/loss/kldiv'] = deformers.pipelines.eval.kl_divergence(__teacher_residuals, __student_residuals).item()

            # track the loss EMA
            __state['train/loss/ema'] = mlable.utils.ema(average=__state['train/loss/ema'], current=__state['train/loss/total'], factor=0.99 * float(__step > 32))

            # gradient clipping; unscale first to get true grad norm
            SCALER_OBJ.unscale_(OPTIMIZER_OBJ)
            __state['train/gradient/rate'] = deformers.pipelines.monitor.current_lr(OPTIMIZER_OBJ)
            __state['train/gradient/norm'] = torch.nn.utils.clip_grad_norm_(
                PREFIX_MOD.parameters(),
                max_norm=GRADIENT_CFG['max_norm']).item()

            # update the weights
            SCALER_OBJ.step(OPTIMIZER_OBJ)
            SCALER_OBJ.update()
            SCHEDULER_OBJ.step()
            OPTIMIZER_OBJ.zero_grad()

            # timing and throughput
            __state['train/iter/time'] = time.monotonic() - __state['train/iter/start']
            __state['train/iter/tps'] = deformers.pipelines.monitor.throughput(BATCH_LEN, __state['train/iter/time'])

            # track the memory consumption too
            __state = {**__state, **deformers.pipelines.monitor.gpu_memory_mb()}

            # format the state for logging
            __stats = format_state(
                state=__state,
                epoch_num=__epoch + 1,
                epoch_tot=TRAINING_CFG["epoch_num"],
                step_num=__step,
                step_tot=DATASET_DIM)

            # write all the stats to the log file
            LOG_FILE.write(serialize_state(state=__stats, prefix='[train] ') + '\n')

            # write all the stats to the tensorboard summary
            deformers.pipelines.monitor.log_scalars(writer=LOG_TB, step=__step, scalars=__state)

            # filter the epoch and step since they are already in the pbar
            __pbar.set_postfix({__k: __v for (__k, __v) in __stats.items() if (__k not in ['epoch', 'step'])})

            # reset the accumulators and start the timing
            __state = reset_state(state=__state, ignore=['train/loss/ema'])

        # track the global step (across epochs)
        __step += 1

    # cleanup
    __pbar.close()

# POST PROCESSING ##############################################################

print('[post] closing the log streams...')
LOG_TB.close()
LOG_FILE.close()

print(f'[post] saving prefix to {OUTPUT_CFG["save_path"]}...')
PREFIX_MOD.save_checkpoint(path=OUTPUT_CFG['save_path'])
