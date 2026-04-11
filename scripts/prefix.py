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
import torch.utils.tensorboard
import tqdm
import transformers


import deformers.layers.prefix
import deformers.models.generic
import deformers.monitoring
import deformers.pipelines.eval
import deformers.pipelines.patch
import deformers.tokenizers.byte

# COMMON CONFIG ################################################################

MAIN_CFG = {
    'model_str': 'qwen/qwen3.5-9b',
    'device_str': 'cuda' if torch.cuda.is_available() else 'cpu',
    'encoding_str': 'utf-8',
    'seed_num': 1337,
    'batch_dim': 32,
    'sequence_dim': 256,
    'patch_dim': 32,
    'depth_num': 4,
    'epoch_num': 4,
    'accumulation_num': 8,}

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
    'embed_dim': 4096 // BATCH_CFG['patch_dim'],
    'vocab_dim': 256,
    'latent_dim': 4096,
    'group_dim': -1,}

# TRAINING CONFIG ##############################################################

TRAINING_CFG = {
    'epoch_num': MAIN_CFG['epoch_num'],}

OPTIMIZER_CFG = {
    'lr': 1e-4,}

SCALER_CFG = {
    'enabled': MAIN_CFG['device_str'] == 'cuda',}

GRADIENT_CFG = {
    'accumulation_num': MAIN_CFG['accumulation_num'],
    'max_norm': 1.0,}

LOSS_CFG = {
    'embeds_rate': 0.1,
    'residuals_rate': 1.0,}

# OUTPUT CONFIG ################################################################

LOGGING_CFG = {
    'step_num': 32,
    'log_path': os.path.abspath('logs/prefix.log'),}

OUTPUT_CFG = {
    'save_path': os.path.abspath('checkpoints/prefix.pt'),}

# UTILS ########################################################################

def freeze_model(model_obj: torch.nn.Module) -> None:
    """Disable gradients for all model parameters."""
    for __p in model_obj.parameters():
        __p.requires_grad_(False)

def save_checkpoint(
    model_obj: torch.nn.Module,
    path_str: str=OUTPUT_CFG['save_path'],
) -> None:
    torch.save(
        {
            'config': model_obj._config,
            'state_dict': model_obj.state_dict()},
        path_str)

def load_checkpoint(
    file_path: str='prefix.pt',
    device_str: str='cpu',
) -> object:
    """Load a model from a local checkpoint or HF hub path."""
    # check the disk
    assert os.path.isfile(file_path), f'model checkpoint not found: {file_path}'
    # parse the data
    __ckpt = torch.load(file_path, map_location=device_str, weights_only=True)
    # instantiate the model
    __prefix = deformers.layers.prefix.CompositeBytePrefix(**__ckpt['config'])
    # load the weights
    __prefix.load_state_dict(__ckpt['state_dict'])
    # alternative transformer prefix
    return __prefix.to(device=device_str)

def compute_loss(
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
    # scale to the accumulation batch size; return components for monitoring
    return __loss / float(max(1, accumulation_num)), __loss_embeds.detach(), __loss_residuals.detach()

# DATASET ######################################################################

print('[init] downloading the dataset...')
DATASET_OBJ = datasets.load_dataset(**DATASET_CFG)

print('[init] preprocessing the dataset...')
DATASET_OBJ = DATASET_OBJ.shuffle(seed=MAIN_CFG['seed_num'])

# TOKENIZERS ###################################################################

print('[init] loading the tokenizers...')
TEXT_TOK = transformers.AutoTokenizer.from_pretrained(**TOKEN_CFG)
BYTE_TOK = deformers.tokenizers.byte.ByteTokenizer(**BYTE_CFG)

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
freeze_model(SOURCE_MOD)

print('[init] creating the student...')
PREFIX_MOD = deformers.layers.prefix.CompositeBytePrefix(**PREFIX_CFG).to(device=MAIN_CFG['device_str'])

print('[init] freeing the unused layers...')
deformers.models.generic.free_memory()

print('[init] building the student...')
PREFIX_MOD._build(
    shape_arr=(BATCH_CFG['batch_dim'], BATCH_CFG['sequence_dim'], BATCH_CFG['patch_dim']),
    device_str=MAIN_CFG['device_str'])

# OPTIMIZER ####################################################################

print('[init] creating optimizer...')
OPTIMIZER_OBJ = torch.optim.AdamW(PREFIX_MOD.parameters(), **OPTIMIZER_CFG)
SCALER_OBJ = torch.amp.GradScaler(**SCALER_CFG)

print('[init] enabling mixed precision...')
MIXED_CTX = (
    torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    if SCALER_CFG['enabled']
    else contextlib.nullcontext())

# LOGGING ######################################################################

print('[init] logging to {}...'.format(os.path.dirname(LOGGING_CFG['log_path'])))
TB_WRITER = torch.utils.tensorboard.SummaryWriter(log_dir=os.path.dirname(LOGGING_CFG['log_path']))

print('[init] calculating the training metadata...')
DATASET_DIM = len(DATASET_OBJ) // BATCH_CFG['batch_dim']
BATCH_LEN = BATCH_CFG['batch_dim'] * BATCH_CFG['sequence_dim'] * GRADIENT_CFG['accumulation_num']

# ZERO #########################################################################

print('[init] zeroing the state...')
__step = 0
__state = {
    'train/iter/start': 0.0,
    'train/iter/time': 0.0,
    'train/iter/tps': 0.0,
    'train/iter/bps': 0.0,
    'train/gradient/rate': 0.0,
    'train/gradient/norm': 0.0,
    'train/loss/total': 0.0,
    'train/loss/embed': 0.0,
    'train/loss/hidden': 0.0,
    'train/loss/kldiv': 0.0,
    'gpu/memory/allocated': 0.0,
    'gpu/memory/reserved': 0.0,}

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

    # reset per-accumulation-window accumulators
    __state['train/loss/total'] = 0.0
    __state['train/loss/embed'] = 0.0
    __state['train/loss/hidden'] = 0.0
    # start timing the first accumulation window
    __state['train/iter/start'] = time.monotonic()

    for __batch in __pbar:
        __texts = __batch['text']

        # input_ids (B, T) and attention_mask (B, T)
        __inputs = TEXT_TOK(
            __texts,
            return_offsets_mapping=True,
            max_length=BATCH_CFG['sequence_dim'],
            truncation='longest_first',
            padding='max_length')

        # byte patches (B, T, G)
        __encoded = deformers.pipelines.patch.tokenize_into_bytes(
            texts_arr=__texts,
            offsets_arr=__inputs['offset_mapping'],
            patch_dim=BATCH_CFG['patch_dim'],
            tokenizer_obj=BYTE_TOK)

        # format as tensors
        __tokens_arr = torch.tensor(__inputs['input_ids'], dtype=torch.long, device=MAIN_CFG['device_str'])
        __mask_arr = torch.tensor(__inputs['attention_mask'], dtype=torch.long, device=MAIN_CFG['device_str'])
        __bytes_arr = torch.tensor(__encoded, dtype=torch.long, device=MAIN_CFG['device_str'])

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

            # combination of the MSE at depth 0 and k; also returns unscaled components
            __loss, __embed_mse_t, __hidden_mse_t = compute_loss(
                teacher_embeds=__teacher_embeds,
                student_embeds=__student_embeds,
                teacher_residuals=__teacher_residuals,
                student_residuals=__student_residuals,
                embeds_rate=LOSS_CFG['embeds_rate'],
                residuals_rate=LOSS_CFG['residuals_rate'],
                accumulation_num=GRADIENT_CFG['accumulation_num'])

        SCALER_OBJ.scale(__loss).backward()
        # __state['train/loss/total'] is sum of (loss / accumulation_num) = mean(loss) after N steps
        __state['train/loss/total'] += __loss.item()
        __state['train/loss/embed'] += __embed_mse_t.item() / GRADIENT_CFG['accumulation_num']
        __state['train/loss/hidden'] += __hidden_mse_t.item() / GRADIENT_CFG['accumulation_num']

        # optimizer step after gradient accumulation
        if (__step + 1) % GRADIENT_CFG['accumulation_num'] == 0:
            # compute KL from last micro-batch hidden states (monitoring only)
            # uses first batch item only to avoid large logit tensors (B, T, V)
            with torch.no_grad():
                __t_logits = SOURCE_MOD.lm_head(__teacher_residuals[:1])
                __s_logits = SOURCE_MOD.lm_head(__student_residuals[:1].detach())
            __state['train/loss/kldiv'] = deformers.pipelines.eval.kl_divergence(__t_logits, __s_logits).item()
            del __t_logits, __s_logits

            # gradient clipping; unscale first to get true grad norm
            SCALER_OBJ.unscale_(OPTIMIZER_OBJ)
            __state['train/gradient/norm'] = torch.nn.utils.clip_grad_norm_(
                PREFIX_MOD.parameters(),
                max_norm=GRADIENT_CFG['max_norm']).item()
            SCALER_OBJ.step(OPTIMIZER_OBJ)
            SCALER_OBJ.update()
            OPTIMIZER_OBJ.zero_grad()

            # timing and throughput
            __state['train/iter/time'] = time.monotonic() - __state['train/iter/start']
            __state['train/iter/tps'] = deformers.monitoring.throughput(BATCH_LEN, __state['train/iter/time'])

            # __state['train/loss/total'] is already mean(total_loss) over the window
            __state['train/gradient/rate'] = deformers.monitoring.current_lr(OPTIMIZER_OBJ)
            __mem = deformers.monitoring.gpu_memory_mb()

            # stdout: concise line for notebook-visible progress
            print(
                f'[train] epoch={__epoch + 1}/{TRAINING_CFG["epoch_num"]}'
                f' step={__step:04d}/{DATASET_DIM}'
                f' loss={__state['train/loss/total']:.6f}'
                f' embed={__state['train/loss/embed']:.6f}'
                f' hidden={__state['train/loss/hidden']:.6f}'
                f' kl={__state['train/loss/kldiv']:.6f}'
                f' lr={__state['train/gradient/rate']:.2e}'
                f' gnorm={__state['train/gradient/norm']:.4f}'
                f' ms={__state['train/iter/time'] * 1000.0:.0f}'
                f' tok/s={__state['train/iter/tps']:.0f}')

            # TensorBoard: all required tags plus KL and throughput
            deformers.monitoring.log_scalars(writer=TB_WRITER, step=__step, scalars=__state)

            # update progress bar postfix with latest optimizer-step metrics
            __pbar.set_postfix({
                'step': f'{__step}/{DATASET_DIM}',
                'loss': f'{__state['train/loss/total']:.4f}',
                'embed': f'{__state['train/loss/embed']:.4f}',
                'hidden': f'{__state['train/loss/hidden']:.4f}',
                'kl': f'{__state['train/loss/kldiv']:.4f}',
                'lr': f'{__state['train/gradient/rate']:.1e}',})

            # reset accumulators for next window
            __state['train/loss/total'] = 0.0
            __state['train/loss/embed'] = 0.0
            __state['train/loss/hidden'] = 0.0
            __state['train/iter/start'] = time.monotonic()

        # track the global step (across epochs)
        __step += 1

    # cleanup
    __pbar.close()

# close TensorBoard writer
TB_WRITER.close()

# EXPORT #######################################################################

print(f'[train] saving prefix to {OUTPUT_CFG["save_path"]}...')
save_checkpoint(model_obj=PREFIX_MOD, path_str=OUTPUT_CFG['save_path'])
