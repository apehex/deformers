"""
Trains the prefix (student) module to distill from an open source (teacher) model.
The student embeddings are injected into the teacher and compared at depth 0 and K.

Assumptions:
- Base model is qwen/qwen3.5-9b with hidden_size=4096.
- Tokenizer boundaries are identical to the base model.
- Trunk is frozen during training; only the prefix parameters are trained.
- Byte block size default follows docs/roadmap.md (patch_dim=32), configurable.
- The byte tokenizer uses pad_id=128 (as implemented by ByteTokenizer).

Training scheme (Stage A):
- Teacher: qwen/qwen3.5-9b forward with original input_ids (no grad).
- Student: CompositeBytePrefix forward then trunk forward with inputs_embeds.
- Loss: MSE at depth 0 (embed) and K (hidden), plus optional KL-div at depth 0 and K
- Only prefix parameters are updated; trunk is frozen.
- Accumulated losses (embed, hidden, total) are unscaled and averaged.

Monitoring (per optimizer step):
- Progress bar (tqdm), TensorBoard scalars and plain text log file
- Values tracked:
    - loss: embed MSE, hidden MSE, total loss, EMA loss
    - iteration: accumulation time, epoch and step
    - gradients: learning rate, gradient norm
    - GPU: allocated and reserved memory
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

import deformers.datasets.random
import deformers.models.generic
import deformers.models.prefix
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
    'checkpoint_num': 128,
    'learning_rate': 5e-5,}

# DATA CONFIG ##################################################################

DATASET_CFG = {
    'wikipedia': {
        'path': 'wikimedia/wikipedia',
        'name': '20231101.en',
        'split': 'train[:10%]',
        'streaming': False,},
    'random': {
        'dataset_len': 1024 * MAIN_CFG['batch_dim'],
        'sequence_dim': MAIN_CFG['sequence_dim'],
        'vocab_dim': 248320,
        'seed_num': MAIN_CFG['seed_num'],}}

BATCH_CFG = {
    'batch_dim': MAIN_CFG['batch_dim'],
    'sequence_dim': MAIN_CFG['sequence_dim'],
    'patch_dim': MAIN_CFG['patch_dim'],}

PREPROC_CFG = {
    'truncation': 'longest_first',
    'padding': 'max_length',
    'max_length': BATCH_CFG['sequence_dim'],}

VECTORIZE_CFG = {
    'sequence_dim': BATCH_CFG['sequence_dim'],
    'patch_dim': BATCH_CFG['patch_dim'],
    'device_str': MAIN_CFG['device_str'],
    'dtype_obj': torch.long,
    'left_pad': True,}

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
    'patch_dim': -1,
    'hidden_dim': -1,
    'output_dim': 4096,
    'vocab_dim': 256,
    'padding_idx': 128,
    'block_num': 4,
    'head_num': 4,
    'dropout_rate': 0.001,}

# TRAINING CONFIG ##############################################################

TRAINING_CFG = {
    'epoch_num': MAIN_CFG['epoch_num'],}

OPTIMIZER_CFG = {
    'lr': MAIN_CFG['learning_rate'],
    'betas': (0.9, 0.999),
    'weight_decay': 0.01,}

SCALER_CFG = {
    'enabled': MAIN_CFG['dtype_obj'] == torch.float16,}

GRADIENT_CFG = {
    'step_num': MAIN_CFG['accumulation_num'],
    'max_norm': 1.0,}

SCHEDULER_CFG = { # counted in acc steps (not micro steps)
    'start_rate': 1e-4,
    'end_rate': 5e-2,
    'total_num': 4096,
    'warmup_num': 128,}

LOSS_CFG = {
    'mse_0_rate': 10.0,
    'mse_k_rate': 1.0,
    'cos_0_rate': 0.0,
    'cos_k_rate': 0.0,}

# TESTING CONFIG ###############################################################

TESTING_CFG = {
    'step_num': MAIN_CFG['testing_num'],}

# OUTPUT CONFIG ################################################################

STATE_CFG = {
    'switch/train': lambda __x: 1,
    'switch/grad': lambda __x: 0,
    'switch/log': lambda __x: 0,
    'switch/save': lambda __x: 0,
    'epoch/total': lambda __x: TRAINING_CFG['epoch_num'],
    'epoch/current': lambda __x: 1,
    'step/total': lambda __x: 0,
    'step/current': lambda __x: 1,
    'iter/start': lambda __x: time.monotonic(), # start of the gradient acc
    'iter/time': lambda __x: 0.0,
    'iter/tps': lambda __x: 0.0,
    'gradient/rate': lambda __x: 0.0,
    'gradient/norm': lambda __x: 0.0,
    'loss/ema': lambda __x: __x, # keep the current loss EMA
    'loss/total': lambda __x: 0.0,
    'loss/mse/0': lambda __x: 0.0,
    'loss/mse/k': lambda __x: 0.0,
    'loss/cos/0': lambda __x: 0.0,
    'loss/cos/k': lambda __x: 0.0,
    'vocab/seen': lambda __x: 0.0,
    'vocab/max': lambda __x: 0.0,
    'gpu/memory/allocated': lambda __x: 0.0,
    'gpu/memory/reserved': lambda __x: 0.0,}

LOGGING_CFG = {
    'step_num': MAIN_CFG['logging_num'],
    'log_path': os.path.abspath('logs/prefix.log'),}

CHECKPOINT_CFG = {
    'step_num': MAIN_CFG['checkpoint_num'],
    'save_path': os.path.abspath('checkpoints/prefix.pt'),}

# TOKENIZERS ###################################################################

print('[init] loading the tokenizers...')
TEXT_TOK = transformers.AutoTokenizer.from_pretrained(**TOKEN_CFG)
BYTE_TOK = deformers.tokenizers.byte.ByteTokenizer(**BYTE_CFG)

print('[init] calculating the tokenizer metadata...')
VOCAB_ARR = {__v: __k for (__k, __v) in TEXT_TOK.get_vocab().items()}
VOCAB_LEN = len(VOCAB_ARR)
DATASET_CFG['random']['vocab_dim'] = VOCAB_LEN

print('[init] defining a padding token...')
TEXT_TOK.pad_token = TEXT_TOK.eos_token if not bool(TEXT_TOK.pad_token) else TEXT_TOK.pad_token

# DATASET ######################################################################

def preprocess(sample: dict) -> dict:
    return {'indices': TEXT_TOK(sample['text'], **PREPROC_CFG)['input_ids']}

print('[init] downloading the main dataset...')
DATASETS = {'wikipedia': datasets.load_dataset(**DATASET_CFG['wikipedia']).select_columns(['text']),}

print('[init] preprocessing the main dataset...')
DATASETS['wikipedia'] = DATASETS['wikipedia'].map(preprocess, batched=True, remove_columns=['text'])

print('[init] building a random dataset...')
DATASETS['random'] = deformers.datasets.random.build_uniform_dataset(**DATASET_CFG['random'])

print('[init] concatenating the two datasets...')
DATASET_OBJ = datasets.concatenate_datasets([DATASETS['random'], DATASETS['wikipedia']])

print('[init] calculating the training metadata...')
DATASET_DIM = len(DATASET_OBJ) // BATCH_CFG['batch_dim']
BATCH_LEN = BATCH_CFG['batch_dim'] * BATCH_CFG['sequence_dim'] * GRADIENT_CFG['step_num']
SCHEDULER_CFG['total_num'] = (2 * DATASET_DIM) // GRADIENT_CFG['step_num']

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
PREFIX_MOD = deformers.models.prefix.CompositeBytePrefix(**PREFIX_CFG).to(device=MAIN_CFG['device_str'])
if MAIN_CFG['resume_opt'] and os.path.exists(CHECKPOINT_CFG['save_path']):
    PREFIX_MOD = deformers.models.prefix.CompositeBytePrefix.load_checkpoint(
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

vectorize = functools.partial(
    deformers.pipelines.prefix.vectorize_indices,
    text_tok=TEXT_TOK,
    byte_tok=BYTE_TOK,
    **VECTORIZE_CFG)

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

def format_state(state: dict) -> dict:
    """Group and format the state variables to export them."""
    return {
        '': f"[{' '.join(state['switch/train'] * ['train'] + (not state['switch/train']) * ['test'] + state['switch/grad'] * ['grad'] + state['switch/log'] * ['log'] + state['switch/save'] * ['save'])}]",
        'epoch': f"({state['epoch/current']}/{state['epoch/total']})",
        'step': f"({state['step/current']}/{state['step/total']})",
        'loss': f"(ema: {state['loss/ema']:.6f} total: {state['loss/total']:.6f} mse(0: {state['loss/mse/0']:.6f} k: {state['loss/mse/k']:.6f}) cos(0: {state['loss/cos/0']:.6f} k: {state['loss/cos/k']:.6f}))",
        'gradient': f"(rate: {state['gradient/rate']:.2e} norm: {state['gradient/norm']:.4f})",
        'iter': f"(time: {state['iter/time'] * 1000.0:.0f} tok/s: {state['iter/tps']:.0f})",
        'vocab': f"(seen: {state['vocab/seen'] * 100.0:.1f}% min: {state['vocab/min']} max: {state['vocab/max']})",}

# TESTING ######################################################################

print('[init] preparing a batch for testing...')
PROBE_I = deformers.pipelines.eval.indices_probe(
    vocab_dim=VOCAB_LEN,
    batch_dim=BATCH_CFG['batch_dim'],
    sequence_dim=BATCH_CFG['sequence_dim'])

# CLEANUP ######################################################################

print('[clean] freeing the unused memory...')
mlable.models.free_memory()

# SUMMARY ######################################################################

print('[check] showing the prefix architecture...')
print(PREFIX_MOD)

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
        # mask (B, T), tokens (B, T), bytes (B, T, G) integers
        __mask_arr, __indices_arr, __bytes_arr = vectorize(
            __batch['indices'] if __state['switch/train']
            else PROBE_I)

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

        # track token stats
        __count += torch.bincount(__indices_arr.flatten().cpu(), minlength=VOCAB_LEN)
        __state['vocab/seen'] = float((__count > 0).sum().item()) / VOCAB_LEN
        __state['vocab/min'] = int(__count.min().item())
        __state['vocab/max'] = int(__count.max().item())

        # the total loss is the average loss after N accumulation steps
        __state['loss/mse/0'] += __losses[0].item()
        __state['loss/mse/k'] += __losses[1].item()
        __state['loss/cos/0'] += __losses[2].item()
        __state['loss/cos/k'] += __losses[3].item()
        __state['loss/total'] += __losses[-1].item()

        # optimizer step after gradient accumulation
        if __state['switch/grad']: # (could be disabled when the input is the probe batch but whatever)
            # track the loss EMA, default to the current loss for the first 128 steps
            __state['loss/ema'] = mlable.utils.ema(average=__state['loss/ema'], current=__state['loss/total'], factor=0.99 * float(__step > 256))

            # gradient clipping; unscale first to get true grad norm
            SCALER_OBJ.unscale_(OPTIMIZER_OBJ)
            __state['gradient/rate'] = deformers.pipelines.monitor.current_lr(OPTIMIZER_OBJ)
            __state['gradient/norm'] = torch.nn.utils.clip_grad_norm_(
                PREFIX_MOD.parameters(),
                max_norm=GRADIENT_CFG['max_norm']).item()

            # update the weights
            SCALER_OBJ.step(OPTIMIZER_OBJ)
            SCALER_OBJ.update()
            SCHEDULER_OBJ.step()
            OPTIMIZER_OBJ.zero_grad()

            # timing and throughput
            __state['iter/time'] = time.monotonic() - __state['iter/start']
            __state['iter/tps'] = deformers.pipelines.monitor.throughput(BATCH_LEN, __state['iter/time'])

            # track the memory consumption too
            __state = {**__state, **deformers.pipelines.monitor.gpu_memory_mb()}

            # format the state for logging
            __stats = format_state(state=__state)

            # filter the epoch and step since they are already in the pbar
            __pbar.set_postfix({__k: __v for (__k, __v) in __stats.items() if (__k not in ['epoch', 'step'])})

        # log only a fraction of the steps
        if __state['switch/log']:
            # write all the stats to the log file
            LOG_FILE.write(deformers.pipelines.monitor.serialize_state(state=__stats, prefix='') + '\n')

            # write all the stats to the tensorboard summary
            deformers.pipelines.monitor.log_scalars(writer=LOG_TB, step=__step, scalars=__state)

        # write to disk sporadically
        if __state['switch/save']:
            # save the weights and config
            PREFIX_MOD.save_checkpoint(path=CHECKPOINT_CFG['save_path'])

        # reset only after the gradient accumulation
        if __state['switch/grad']:
            # reset the accumulators and start the timing
            __state = deformers.pipelines.monitor.reset_state(state=__state, update=STATE_CFG)

        # track the global step (across epochs)
        __step += 1

        # track the iteration progress
        __state['epoch/total'] = TRAINING_CFG['epoch_num']
        __state['epoch/current'] = __epoch + 1
        __state['step/total'] = DATASET_DIM
        __state['step/current'] = __step + 1

        # check which processes should be run on this step
        __state['switch/train'] = ((__step + 1) % TESTING_CFG['step_num']) != 0
        __state['switch/grad'] = ((__step + 1) % GRADIENT_CFG['step_num']) == 0
        __state['switch/log'] = ((__step + 1) % LOGGING_CFG['step_num']) == 0
        __state['switch/save'] = ((__step + 1) % CHECKPOINT_CFG['step_num']) == 0

    # cleanup
    __pbar.close()

# POST PROCESSING ##############################################################

print('[post] closing the log streams...')
LOG_TB.close()
LOG_FILE.close()

print(f'[post] saving prefix to {CHECKPOINT_CFG["save_path"]}...')
PREFIX_MOD.save_checkpoint(path=CHECKPOINT_CFG['save_path'])

# DATAVIZ ######################################################################

# !tensorboard --logdir=logs
