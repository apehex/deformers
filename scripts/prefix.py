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
import functools
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
import deformers.pipelines.prefix
import deformers.tokenizers.byte

# COMMON CONFIG ################################################################

MAIN_CFG = {
    'resume_opt': True,
    'model_str': 'qwen/qwen3.5-9b',
    'device_str': 'cuda' if torch.cuda.is_available() else 'cpu',
    'dtype_obj': torch.bfloat16,
    'encoding_str': 'utf-8',
    'seed_num': 1337,
    'batch_dim': 64,
    'sequence_dim': 256,
    'patch_dim': 32,
    'depth_num': 1,
    'epoch_num': 4,
    'accumulation_num': 4,
    'logging_num': 32,
    'testing_num': 128,
    'checkpoint_num': 128,}

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
    'enabled': MAIN_CFG['dtype_obj'] == torch.float16,}

GRADIENT_CFG = {
    'step_num': MAIN_CFG['accumulation_num'],
    'max_norm': 1.0,}

SCHEDULER_CFG = { # counted in acc steps (not micro steps)
    'start_rate': 1e-4,
    'end_rate': 1e-2,
    'total_num': 4096,
    'warmup_num': 128,}

LOSS_CFG = {
    'mse_0_rate': 1.0,
    'mse_k_rate': 1.0,
    'kld_0_rate': 1.0,
    'kld_k_rate': 1.0,}

# TESTING CONFIG ###############################################################

TESTING_CFG = {
    'step_num': MAIN_CFG['testing_num'],}

# OUTPUT CONFIG ################################################################

STATE_CFG = {
    'train/epoch/total': lambda __x: 0,
    'train/epoch/current': lambda __x: 0,
    'train/step/total': lambda __x: 0,
    'train/step/current': lambda __x: 0,
    'train/iter/start': lambda __x: time.monotonic(), # start of the gradient acc
    'train/iter/time': lambda __x: 0.0,
    'train/iter/tps': lambda __x: 0.0,
    'train/gradient/rate': lambda __x: 0.0,
    'train/gradient/norm': lambda __x: 0.0,
    'train/loss/ema': lambda __x: __x, # keep the current loss EMA
    'train/loss/total': lambda __x: 0.0,
    'train/loss/mse/0': lambda __x: 0.0,
    'train/loss/mse/k': lambda __x: 0.0,
    'train/loss/kldiv/0': lambda __x: 0.0,
    'train/loss/kldiv/k': lambda __x: 0.0,
    'train/vocab/seen': lambda __x: 0.0,
    'train/vocab/max': lambda __x: 0.0,
    'gpu/memory/allocated': lambda __x: 0.0,
    'gpu/memory/reserved': lambda __x: 0.0,}

LOGGING_CFG = {
    'step_num': MAIN_CFG['logging_num'],
    'log_path': os.path.abspath('logs/prefix.log'),}

CHECKPOINT_CFG = {
    'step_num': MAIN_CFG['checkpoint_num'],
    'save_path': os.path.abspath('checkpoints/prefix.pt'),}

# LOGGING UTILS ################################################################

def format_state(state: dict) -> dict:
    """Group and format the state variables to export them."""
    return {
        'epoch': f"({state['train/epoch/current']}/{state['train/epoch/total']})",
        'step': f"({state['train/step/current']}/{state['train/step/total']})",
        'loss': f"(ema: {state['train/loss/ema']:.6f} total: {state['train/loss/total']:.6f} mse(0: {state['train/loss/mse/0']:.6f} k: {state['train/loss/mse/k']:.6f}) kl-div(0: {state['train/loss/kldiv/0']:.6f} k: {state['train/loss/kldiv/k']:.6f}))",
        'gradient': f"(rate: {state['train/gradient/rate']:.2e} norm: {state['train/gradient/norm']:.4f})",
        'iter': f"(time: {state['train/iter/time'] * 1000.0:.0f} tok/s: {state['train/iter/tps']:.0f})",
        'vocab': f"(seen: {state['train/vocab/seen'] * 100.0:.1f}% max: {state['train/vocab/max'] * 100.0:.1f}%)",}

# DATASET ######################################################################

print('[init] downloading the dataset...')
DATASET_OBJ = datasets.load_dataset(**DATASET_CFG)

print('[init] preprocessing the dataset...')
DATASET_OBJ = DATASET_OBJ.shuffle(seed=MAIN_CFG['seed_num'])

print('[init] calculating the training metadata...')
DATASET_DIM = len(DATASET_OBJ) // BATCH_CFG['batch_dim']
BATCH_LEN = BATCH_CFG['batch_dim'] * BATCH_CFG['sequence_dim'] * GRADIENT_CFG['step_num']
SCHEDULER_CFG['total_num'] = (2 * DATASET_DIM) // GRADIENT_CFG['step_num']

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
os.makedirs(os.path.dirname(CHECKPOINT_CFG['save_path']), exist_ok=True)

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
if MAIN_CFG['resume_opt'] and os.path.exists(CHECKPOINT_CFG['save_path']):
    PREFIX_MOD = deformers.layers.prefix.CompositeBytePrefix.load_checkpoint(
        path=CHECKPOINT_CFG['save_path'],
        shape=(BATCH_CFG['batch_dim'], BATCH_CFG['sequence_dim'], BATCH_CFG['patch_dim']),
        device=MAIN_CFG['device_str'])

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
SCHEDULER_OBJ = mlable.schedulers.WaveLR(optimizer_obj=OPTIMIZER_OBJ, **SCHEDULER_CFG)

print('[init] enabling mixed precision...')
MIXED_CTX = (
    torch.amp.autocast(device_type=MAIN_CFG['device_str'], dtype=MAIN_CFG['dtype_obj'])
    if (MAIN_CFG['dtype_obj'] != torch.float32)
    else contextlib.nullcontext())

# LOGGING ######################################################################

print('[init] logging to {}...'.format(os.path.dirname(LOGGING_CFG['log_path'])))
LOG_TB = torch.utils.tensorboard.SummaryWriter(log_dir=os.path.dirname(LOGGING_CFG['log_path']))
LOG_FILE = open(LOGGING_CFG['log_path'], 'w')

# ZERO #########################################################################

print('[init] zeroing the state...')
__step = 0
__state = deformers.pipelines.monitor.reset_state(state={__k: 0.0 for __k in STATE_CFG}, update=STATE_CFG)
__count = torch.zeros(size=(VOCAB_LEN,), dtype=torch.long, device='cpu')

OPTIMIZER_OBJ.zero_grad()

# UTILITIES ####################################################################

print('[init] creating specialized utilities...')

preprocess_s = functools.partial(
    deformers.pipelines.prefix.tensors_from_strings,
    text_tok=TEXT_TOK,
    byte_tok=BYTE_TOK,
    dtype_obj=torch.long,
    sequence_dim=BATCH_CFG['sequence_dim'],
    patch_dim=BATCH_CFG['patch_dim'],
    device_str=MAIN_CFG['device_str'],
    left_pad=True)

preprocess_i = functools.partial(
    deformers.pipelines.prefix.tensors_from_indices,
    text_tok=TEXT_TOK,
    byte_tok=BYTE_TOK,
    dtype_obj=torch.long,
    sequence_dim=BATCH_CFG['sequence_dim'],
    patch_dim=BATCH_CFG['patch_dim'],
    device_str=MAIN_CFG['device_str'],
    left_pad=True)

score = functools.partial(
    deformers.pipelines.prefix.compute_losses,
    step_num=GRADIENT_CFG['step_num'],
    **LOSS_CFG)

def embed(
    indices_arr: torch.Tensor,
    model_obj: object=SOURCE_MOD,
) -> torch.Tensor:
    return model_obj.model.embed_tokens(indices_arr)

def forward(
    embeds_arr: torch.Tensor,
    mask_arr: torch.Tensor,
    model_obj: object=SOURCE_MOD,
) -> torch.Tensor:
    return model_obj.model(
        inputs_embeds=embeds_arr,
        attention_mask=mask_arr,
        use_cache=False).last_hidden_state

# TESTING ######################################################################

print('[init] preparing a batch for testing...')
PROBE_I = deformers.pipelines.eval.indices_probe(
    vocab_dim=VOCAB_LEN,
    batch_dim=BATCH_CFG['batch_dim'],
    sequence_dim=BATCH_CFG['sequence_dim'])

print('[init] encoding the testing batch...')
PROBE_M, PROBE_I, PROBE_B = preprocess_i(PROBE_I)

print('[init] embedding the testing batch...')
with MIXED_CTX:
    with torch.no_grad():
        PROBE_0 = embed(indices_arr=PROBE_I)
        PROBE_K = forward(embeds_arr=PROBE_0, mask_arr=PROBE_M)

# CLEANUP ######################################################################

print('[clean] freeing the unused memory...')
mlable.models.free_memory()

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
    __state = deformers.pipelines.monitor.reset_state(state=__state, update=STATE_CFG)

    for __batch in __pbar:
        # list of plain strings (B,)
        __texts = __batch['text']

        # mask (B, T), tokens (B, T), bytes (B, T, G) integers
        __mask_arr, __indices_arr, __bytes_arr = preprocess_s(text_arr=__texts)

        # compute in bfloat16
        with MIXED_CTX:
            # teacher forward: get original embeddings and hidden states (no grad)
            with torch.no_grad():
                __teacher_0_arr = embed(indices_arr=__indices_arr)
                __teacher_k_arr = forward(embeds_arr=__teacher_0_arr, mask_arr=__mask_arr)

            # student forward: prefix -> inputs_embeds -> trunk -> hidden_k
            __student_0_arr = PREFIX_MOD(__bytes_arr).to(dtype=__teacher_0_arr.dtype)
            __student_k_arr = forward(embeds_arr=__student_0_arr, mask_arr=__mask_arr)

            # combination of the MSE at depth 0 and k
            __losses = score(
                teacher_0_arr=__teacher_0_arr,
                student_0_arr=__student_0_arr,
                teacher_k_arr=__teacher_k_arr,
                student_k_arr=__student_k_arr,
                mask_arr=__mask_arr)

        # perform the backward propagation of the loss
        SCALER_OBJ.scale(__losses[-1]).backward()

        # track the iteration progress
        __state['train/epoch/total'] = TRAINING_CFG['epoch_num']
        __state['train/epoch/current'] = __epoch + 1
        __state['train/step/total'] = DATASET_DIM
        __state['train/step/current'] = __step + 1

        # track token stats
        __count += torch.bincount(__indices_arr.flatten().cpu(), minlength=VOCAB_LEN)
        __state['train/vocab/seen'] = float((__count > 0).sum().item()) / VOCAB_LEN
        __state['train/vocab/max'] = float(__count.max().item()) / float(__count.sum().item())

        # the total loss is the average loss after N accumulation steps
        __state['train/loss/mse/0'] += __losses[0].item()
        __state['train/loss/mse/k'] += __losses[1].item()
        __state['train/loss/kldiv/0'] += __losses[2].item()
        __state['train/loss/kldiv/k'] += __losses[3].item()
        __state['train/loss/total'] += __losses[-1].item()

        # optimizer step after gradient accumulation
        if (__step + 1) % GRADIENT_CFG['step_num'] == 0:
            # track the loss EMA, default to the current loss for the first 128 steps
            __state['train/loss/ema'] = mlable.utils.ema(average=__state['train/loss/ema'], current=__state['train/loss/total'], factor=0.99 * float(__step > 128))

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
            __stats = format_state(state=__state)

            # filter the epoch and step since they are already in the pbar
            __pbar.set_postfix({__k: __v for (__k, __v) in __stats.items() if (__k not in ['epoch', 'step'])})

        # log only a fraction of the steps
        if (__step + 1) % LOGGING_CFG['step_num'] == 0:
            # write all the stats to the log file
            LOG_FILE.write(deformers.pipelines.monitor.serialize_state(state=__stats, prefix='[train] ') + '\n')

            # write all the stats to the tensorboard summary
            deformers.pipelines.monitor.log_scalars(writer=LOG_TB, step=__step, scalars=__state)

        # test the prefix on independent data
        if (__step + 1) % TESTING_CFG['step_num'] == 0:
            with MIXED_CTX:
                with torch.no_grad():
                    # embed the probe with the alternative prefix
                    __probe_0_arr = PREFIX_MOD(PROBE_B).to(dtype=PROBE_0.dtype)
                    __probe_k_arr = forward(embeds_arr=__probe_0_arr, mask_arr=PROBE_M)
                    # combination of the MSE at depth 0 and k
                    __metrics = score(
                        teacher_0_arr=PROBE_0,
                        student_0_arr=__probe_0_arr,
                        teacher_k_arr=PROBE_K,
                        student_k_arr=__probe_k_arr,
                        mask_arr=PROBE_M)
                    # rescale and format all the metrics
                    __metrics = [float(GRADIENT_CFG['step_num']) * float(__m) for __m in __metrics]
                    # log the testing results
                    print(f"[test] loss(total: {__metrics[-1]:.6f} mse(0: {__metrics[0]:.6f} k: {__metrics[1]:.6f}) kl-div(0: {__metrics[2]:.6f} k: {__metrics[3]:.6f}))")

        # write to disk sporadically
        if (__step + 1) % CHECKPOINT_CFG['step_num'] == 0:
            # save the weights and config
            PREFIX_MOD.save_checkpoint(path=CHECKPOINT_CFG['save_path'])

        # reset only after the gradient accumulation
        if (__step + 1) % GRADIENT_CFG['step_num'] == 0:
            # reset the accumulators and start the timing
            __state = deformers.pipelines.monitor.reset_state(state=__state, update=STATE_CFG)

        # track the global step (across epochs)
        __step += 1

    # cleanup
    __pbar.close()

# POST PROCESSING ##############################################################

print('[post] closing the log streams...')
LOG_TB.close()
LOG_FILE.close()

print(f'[post] saving prefix to {CHECKPOINT_CFG["save_path"]}...')
PREFIX_MOD.save_checkpoint(path=CHECKPOINT_CFG['save_path'])
