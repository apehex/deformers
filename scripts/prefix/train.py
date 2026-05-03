"""
Trains the prefix (student) module to distill from an open source (teacher) model.
The student embeddings are injected into the teacher and compared at depth 0 and K.

Assumptions:
- Base model is qwen/qwen3.5-9b with hidden_size=4096.
- Tokenizer boundaries are identical to the base model.
- Trunk is frozen during training; only the prefix parameters are trained.
- Byte block size default follows docs/roadmap.md (patch_dim=32), configurable.
- The byte tokenizer uses pad_id=128 (as implemented by ByteTokenizer).

Training is split into two sequential phases orchestrated by PrefixTrainer:

  Phase 1 - Uniform vocabulary warm-up:
    Dataset : random uniform token indices covering the full vocabulary.
    Purpose : learn a token-agnostic byte-to-embedding mapping before switching
              to natural-language distributions.  Only depth-0 (embedding)
              losses are active; the teacher trunk is never called for depth-k
              hidden states, which avoids expensive trunk forward passes.

  Phase 2 - Wikipedia text fine-tuning:
    Dataset : wikimedia/wikipedia (pre-tokenized by the text tokenizer).
    Purpose : adapt the prefix to natural-language token distributions and
              align deeper hidden states (depth-k) with the teacher.
              Both depth-0 and depth-k losses are active.

The two trainers share the same student model, optimizer, and scaler so that
weight updates and optimizer momentum carry over from phase 1 to phase 2.
Each phase creates its own scheduler and callbacks so that their learning-rate
envelopes, log files, and TensorBoard runs are cleanly separated.

Monitoring (per optimizer step):
- Progress bar (tqdm), TensorBoard scalars and plain text log file
- Values tracked:
    - loss: embed MSE, hidden MSE, total loss, EMA loss
    - iteration: accumulation time, epoch and step
    - gradients: learning rate, gradient norm
    - GPU: allocated and reserved memory
"""

import contextlib
import os

import datasets
import huggingface_hub
import torch
import torch.amp
import torch.optim
import transformers

import mlable.models
import mlable.schedulers

import deformers.datasets.random
import deformers.models.generic
import deformers.models.prefix
import deformers.pipelines.prefix.callbacks
import deformers.pipelines.prefix.trainer
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
        'dataset_len': 1024 * MAIN_CFG['batch_dim'], # pre-batched rows; each row == one mini-batch
        'batch_dim': MAIN_CFG['batch_dim'],
        'sequence_dim': MAIN_CFG['sequence_dim'],
        'vocab_dim': 248320,                          # updated after tokenizer loads
        'seed_num': MAIN_CFG['seed_num'],}}

BATCH_CFG = {
    'batch_dim': MAIN_CFG['batch_dim'],
    'sequence_dim': MAIN_CFG['sequence_dim'],
    'patch_dim': MAIN_CFG['patch_dim'],
    'padding_str': '',
    'left_pad': True,}

PREPROC_CFG = {
    'truncation': 'longest_first',
    'padding': 'max_length',
    'max_length': BATCH_CFG['sequence_dim'],}

# TOKENIZER CONFIG #############################################################

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
    'embed_dim': 128, # 32 * 128 = 4096
    'patch_dim': -1,
    'hidden_dim': 4096,
    'output_dim': 4096,
    'vocab_dim': 256,
    'padding_idx': 128,
    'block_num': 4,
    'head_num': 4,
    'dropout_rate': 0.001,}

# TRAINING UTILITIES CONFIG (shared across phases) #############################

# 'dtype' and 'device' are the exact keys the trainer's step_batch reads.
# Index tensors (mask, input_ids, byte patches) must be torch.long.
# Mixed-precision compute is handled separately via MIXED_CTX.
TRAINING_CFG = {
    'dtype': torch.long,
    'device': MAIN_CFG['device_str'],}

OPTIMIZER_CFG = {
    'lr': MAIN_CFG['learning_rate'],
    'betas': (0.9, 0.999),
    'weight_decay': 0.01,}

# GradScaler is only meaningful for float16; bfloat16 uses a no-op scaler
SCALER_CFG = {
    'enabled': MAIN_CFG['dtype_obj'] == torch.float16,}

GRADIENT_CFG = {
    'every_num': MAIN_CFG['accumulation_num'],
    'max_norm': 1.0,}

TESTING_CFG = {
    'every_num': MAIN_CFG['testing_num'],}

# PHASE 1 CONFIG: Uniform vocabulary warm-up ###################################
#
# Only depth-0 (embedding) losses are active.  Setting mse_k_rate and
# cos_k_rate to zero means the teacher trunk is never called for depth-k
# hidden states, which avoids expensive trunk forward passes during this
# vocabulary coverage phase.
#
# The goal is a token-agnostic byte-to-embedding mapping: every vocabulary
# token should produce a reasonable prefix embedding before we encounter
# natural-language token frequencies in phase 2.

PHASE1_CFG = {
    'name': 'uniform',
    'epoch_num': 2,
    'column_str': 'indices',}

PHASE1_TRAINING_CFG = {
    **TRAINING_CFG,
    'epoch_num': PHASE1_CFG['epoch_num'],}

PHASE1_LOSS_CFG = {
    'mse_0_rate': 1.0,
    'mse_k_rate': 0.0, # skip depth-k: trunk hidden states not needed here
    'cos_0_rate': 1.0,
    'cos_k_rate': 0.0,
    'relative_opt': True,}

PHASE1_SCHEDULER_CFG = {  # total_num updated after datasets are loaded
    'start_rate': 1e-4,
    'end_rate': 5e-2,
    'total_num': 4096,
    'warmup_num': 128,}

PHASE1_EMA_CFG = {
    'every_num': GRADIENT_CFG['every_num'],
    'start_num': 256,
    'smooth_rate': 0.99,}

PHASE1_SPEED_CFG = {
    'every_num': GRADIENT_CFG['every_num'],
    'batch_len': BATCH_CFG['batch_dim'] * BATCH_CFG['sequence_dim'],}

PHASE1_LOGGING_CFG = {
    'every_num': MAIN_CFG['logging_num'],
    'path_str': os.path.abspath('logs/prefix_phase1.log'),}

PHASE1_TBOARD_CFG = {
    'every_num': MAIN_CFG['logging_num'],
    'path_str': os.path.abspath('logs/phase1/'),}

PHASE1_SAVING_CFG = {
    'every_num': MAIN_CFG['checkpoint_num'],
    'path_str': os.path.abspath('checkpoints/prefix.pt'),}

# PHASE 2 CONFIG: Wikipedia text fine-tuning ###################################
#
# Both depth-0 and depth-k losses are active so the student learns to align
# its representations with the teacher at every measured depth.
#
# The optimizer and student model carry over from phase 1: weights and
# optimizer momentum are preserved.  Only the LR schedule resets via a new
# WaveLR instance bound to the same optimizer.

PHASE2_CFG = {
    'name': 'wikipedia',
    'epoch_num': 4,
    'column_str': 'indices',}

PHASE2_TRAINING_CFG = {
    **TRAINING_CFG,
    'epoch_num': PHASE2_CFG['epoch_num'],}

PHASE2_LOSS_CFG = {
    'mse_0_rate': 1.0,
    'mse_k_rate': 0.1,
    'cos_0_rate': 1.0,
    'cos_k_rate': 0.1,
    'relative_opt': True,}

PHASE2_SCHEDULER_CFG = {  # total_num updated after datasets are loaded
    'start_rate': 5e-5,
    'end_rate': 1e-3,
    'total_num': 16384,
    'warmup_num': 256,}

PHASE2_EMA_CFG = {
    'every_num': GRADIENT_CFG['every_num'],
    'start_num': 256,
    'smooth_rate': 0.99,}

PHASE2_SPEED_CFG = {
    'every_num': GRADIENT_CFG['every_num'],
    'batch_len': BATCH_CFG['batch_dim'] * BATCH_CFG['sequence_dim'],}

PHASE2_LOGGING_CFG = {
    'every_num': MAIN_CFG['logging_num'],
    'path_str': os.path.abspath('logs/prefix_phase2.log'),}

PHASE2_TBOARD_CFG = {
    'every_num': MAIN_CFG['logging_num'],
    'path_str': os.path.abspath('logs/phase2/'),}

PHASE2_SAVING_CFG = {
    'every_num': MAIN_CFG['checkpoint_num'],
    'path_str': os.path.abspath('checkpoints/prefix.pt'),}

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

# DATASETS #####################################################################

def preprocess(sample: dict) -> dict:
    return {'indices': TEXT_TOK(sample['text'], **PREPROC_CFG)['input_ids']}

print('[init] downloading the main dataset...')
DATASETS = {'wikipedia': datasets.load_dataset(**DATASET_CFG['wikipedia']).select_columns(['text']),}

print('[init] preprocessing the main dataset...')
DATASETS['wikipedia'] = DATASETS['wikipedia'].map(preprocess, batched=True, remove_columns=['text'])

print('[init] building a random dataset...')
DATASETS['random'] = deformers.datasets.random.build_uniform_dataset(**DATASET_CFG['random'])

# compute step counts per epoch (used to calibrate scheduler total_num)
# the random dataset rows are pre-batched so each row == one training step
# the wikipedia dataset rows are single sequences; divide by batch_dim
RANDOM_DIM = len(DATASETS['random'])
WIKI_DIM = len(DATASETS['wikipedia']) // BATCH_CFG['batch_dim']

print('[init] calibrating scheduler step counts...')
PHASE1_SCHEDULER_CFG['total_num'] = max(1, (PHASE1_CFG['epoch_num'] * RANDOM_DIM) // GRADIENT_CFG['every_num'])
PHASE2_SCHEDULER_CFG['total_num'] = max(1, (PHASE2_CFG['epoch_num'] * WIKI_DIM) // GRADIENT_CFG['every_num'])

# MODELS #######################################################################

print('[init] creating the output directories...')
os.makedirs(DOWNLOAD_CFG['local_dir'], exist_ok=True)

print('[init] downloading the teacher...')
huggingface_hub.snapshot_download(**DOWNLOAD_CFG)

print('[init] loading the config...')
TRUNK_CFG = transformers.AutoConfig.from_pretrained(**CONFIG_CFG)

print('[init] truncating the config...')
TRUNK_CFG = deformers.models.generic.truncate_config(TRUNK_CFG, layer_num=MAIN_CFG['depth_num'], target_key='text_config')

print('[init] loading the teacher...') # load only the used layers, up to the chosen depth
SOURCE_MOD = transformers.AutoModelForCausalLM.from_pretrained(config=TRUNK_CFG, **MODEL_CFG).to(device=MAIN_CFG['device_str'])

print('[init] freezing the teacher...')
SOURCE_MOD.eval()
mlable.models.freeze(SOURCE_MOD)

print('[init] creating the student...')
PREFIX_MOD = deformers.models.prefix.CompositeBytePrefix(**PREFIX_CFG).to(device=MAIN_CFG['device_str'])
if MAIN_CFG['resume_opt'] and os.path.exists(PHASE2_SAVING_CFG['path_str']):
    PREFIX_MOD = deformers.models.prefix.CompositeBytePrefix.load_checkpoint(
        path=PHASE2_SAVING_CFG['path_str'],
        shape=(BATCH_CFG['batch_dim'], BATCH_CFG['sequence_dim'], BATCH_CFG['patch_dim']),
        device=MAIN_CFG['device_str'])

print('[init] building the student...')
PREFIX_MOD.build(
    shape=(BATCH_CFG['batch_dim'], BATCH_CFG['sequence_dim'], BATCH_CFG['patch_dim']),
    device=MAIN_CFG['device_str'],
    dtype=torch.float32)

# OPTIMIZER (shared across phases) #############################################

print('[init] creating optimizer and scaler...')
OPTIMIZER_OBJ = torch.optim.AdamW(PREFIX_MOD.parameters(), **OPTIMIZER_CFG)
SCALER_OBJ = torch.amp.GradScaler(**SCALER_CFG)

print('[init] enabling mixed precision context...')
MIXED_CTX = (
    torch.amp.autocast(device_type=MAIN_CFG['device_str'], dtype=MAIN_CFG['dtype_obj'])
    if (MAIN_CFG['dtype_obj'] != torch.float32)
    else contextlib.nullcontext())

OPTIMIZER_OBJ.zero_grad()

# CLEANUP ######################################################################

print('[clean] freeing the unused memory...')
mlable.models.free_memory()

# SUMMARY ######################################################################

print('[check] showing the prefix architecture...')
print(PREFIX_MOD)

# DATASET WRAPPER ##############################################################

class BatchedDataset:
    """Wraps a HuggingFace Dataset to expose batch-level len() and iter().

    The trainer's init_epoch expects an object where len() returns the number
    of training steps (batches) and iter() yields one batch dict per step.
    A raw HuggingFace Dataset yields individual rows; this wrapper corrects
    that for datasets where each row is a single tokenized sequence (e.g.
    Wikipedia after preprocessing).

    The random dataset rows are already pre-batched (each row == one
    mini-batch of batch_dim sequences), so it is passed to the trainer
    directly without wrapping.
    """
    def __init__(self, dataset: object, batch_dim: int) -> None:
        self._dataset = dataset
        self._batch_dim = int(batch_dim)

    def __len__(self) -> int:
        return len(self._dataset) // self._batch_dim

    def __iter__(self) -> object:
        return self._dataset.iter(batch_size=self._batch_dim)

# PHASE 1: UNIFORM VOCABULARY WARM-UP ##########################################
#
# Use a fresh scheduler calibrated to the phase 1 step budget.
# The random dataset is passed directly: each of its rows is a pre-batched
# mini-batch so init_epoch sees len(DATASETS['random']) steps per epoch.

print('[phase1] creating scheduler...')
PHASE1_SCHEDULER = mlable.schedulers.WaveLR(optimizer_obj=OPTIMIZER_OBJ, **PHASE1_SCHEDULER_CFG)

print('[phase1] creating callbacks...')
PHASE1_CALLBACKS = [
    deformers.pipelines.prefix.callbacks.prepare_speed_callback(**PHASE1_SPEED_CFG),
    deformers.pipelines.prefix.callbacks.prepare_ema_callback(**PHASE1_EMA_CFG),
    deformers.pipelines.prefix.callbacks.prepare_logging_callback(**PHASE1_LOGGING_CFG),
    deformers.pipelines.prefix.callbacks.prepare_tensorboard_callback(**PHASE1_TBOARD_CFG),
    deformers.pipelines.prefix.callbacks.prepare_saving_callback(model_obj=PREFIX_MOD, **PHASE1_SAVING_CFG),]

print('[phase1] creating trainer...')
PHASE1_TRAINER = deformers.pipelines.prefix.trainer.PrefixTrainer(
    text_tok=TEXT_TOK,
    byte_tok=BYTE_TOK,
    teacher_mod=SOURCE_MOD,
    student_mod=PREFIX_MOD,
    optimizer_obj=OPTIMIZER_OBJ,
    scheduler_obj=PHASE1_SCHEDULER,
    scaler_obj=SCALER_OBJ,
    context_obj=MIXED_CTX,
    batch_cfg=BATCH_CFG,
    loss_cfg=PHASE1_LOSS_CFG,
    gradient_cfg=GRADIENT_CFG,
    training_cfg=PHASE1_TRAINING_CFG,
    logging_cfg=PHASE1_LOGGING_CFG,
    saving_cfg=PHASE1_SAVING_CFG,
    testing_cfg=TESTING_CFG,
    callbacks_arr=PHASE1_CALLBACKS,)

print('[phase1] training on uniform vocabulary coverage...')
PHASE1_TRAINER.run_phase(
    epoch_tot=PHASE1_CFG['epoch_num'],
    dataset_obj=DATASETS['random'],
    column_str=PHASE1_CFG['column_str'])

print('[phase1] cleaning up...')
PHASE1_TRAINER.cleanup_callbacks()

# PHASE 2: WIKIPEDIA TEXT FINE-TUNING ##########################################
#
# Create a fresh scheduler calibrated to the phase 2 step budget.  The
# optimizer and student model carry over from phase 1 so weights and momentum
# are preserved; only the LR schedule restarts.
#
# Wikipedia rows are single sequences so BatchedDataset wraps the dataset to
# group rows into mini-batches of batch_dim sequences each.

print('[phase2] creating scheduler...')
PHASE2_SCHEDULER = mlable.schedulers.WaveLR(optimizer_obj=OPTIMIZER_OBJ, **PHASE2_SCHEDULER_CFG)

print('[phase2] creating callbacks...')
PHASE2_CALLBACKS = [
    deformers.pipelines.prefix.callbacks.prepare_speed_callback(**PHASE2_SPEED_CFG),
    deformers.pipelines.prefix.callbacks.prepare_ema_callback(**PHASE2_EMA_CFG),
    deformers.pipelines.prefix.callbacks.prepare_logging_callback(**PHASE2_LOGGING_CFG),
    deformers.pipelines.prefix.callbacks.prepare_tensorboard_callback(**PHASE2_TBOARD_CFG),
    deformers.pipelines.prefix.callbacks.prepare_saving_callback(model_obj=PREFIX_MOD, **PHASE2_SAVING_CFG),]

print('[phase2] creating trainer...')
PHASE2_TRAINER = deformers.pipelines.prefix.trainer.PrefixTrainer(
    text_tok=TEXT_TOK,
    byte_tok=BYTE_TOK,
    teacher_mod=SOURCE_MOD,
    student_mod=PREFIX_MOD,
    optimizer_obj=OPTIMIZER_OBJ,
    scheduler_obj=PHASE2_SCHEDULER,
    scaler_obj=SCALER_OBJ,
    context_obj=MIXED_CTX,
    batch_cfg=BATCH_CFG,
    loss_cfg=PHASE2_LOSS_CFG,
    gradient_cfg=GRADIENT_CFG,
    training_cfg=PHASE2_TRAINING_CFG,
    logging_cfg=PHASE2_LOGGING_CFG,
    saving_cfg=PHASE2_SAVING_CFG,
    testing_cfg=TESTING_CFG,
    callbacks_arr=PHASE2_CALLBACKS,)

print('[phase2] training on Wikipedia...')
PHASE2_TRAINER.run_phase(
    epoch_tot=PHASE2_CFG['epoch_num'],
    dataset_obj=BatchedDataset(DATASETS['wikipedia'], BATCH_CFG['batch_dim']),
    column_str=PHASE2_CFG['column_str'])

print('[phase2] cleaning up...')
PHASE2_TRAINER.cleanup_callbacks()

# DATAVIZ ######################################################################

# %load_ext tensorboard
# %tensorboard --logdir=logs
