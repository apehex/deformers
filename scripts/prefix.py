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
"""

import contextlib
import os

import datasets
import huggingface_hub
import torch
import torch.amp
import torch.nn
import transformers

import deformers.layers.prefix
import deformers.models.generic
import deformers.pipelines.patching
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
    'epoch_num': 256,}

# DATA CONFIG ##################################################################

DATASET_CFG = {
    'path': 'wikimedia/wikipedia',
    'name': '20231101.en',
    'split': 'train',
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

# MODEL_CFG = {
#     'pretrained_model_name_or_path': MAIN_CFG['model_str'],
#     'device_map': MAIN_CFG['device_str'],
#     'dtype': torch.bfloat16,}

MODEL_CFG = {
    'pretrained_model_name_or_path': DOWNLOAD_CFG['local_dir'],
    'trust_remote_code': CONFIG_CFG['trust_remote_code'],
    'torch_dtype': torch.bfloat16,
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
    'lr': 3e-4,}

SCALER_CFG = {
    'enabled': MAIN_CFG['device_str'] == 'cuda',}

GRADIENT_CFG = {
    'accumulation_num': 4,
    'max_norm': 1.0,}

LOSS_CFG = {
    'hidden_rate': 1.0,
    'embed_rate': 0.1,}

# OUTPUT CONFIG ################################################################

LOGGING_CFG = {
    'step_num': 32,}

OUTPUT_CFG = {
    'save_path': os.path.abspath('checkpoints/prefix.pt'),}

# UTILS ########################################################################

def freeze_model(model: torch.nn.Module) -> None:
    """Disable gradients for all model parameters."""
    for __p in model.parameters():
        __p.requires_grad_(False)

def embed_tokens(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Return the embedding layer output for the given token ids."""
    return model.model.embed_tokens(input_ids)

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

# print('[init] truncating the teacher...')
# SOURCE_MOD = deformers.models.generic.truncate_model(SOURCE_MOD, layer_num=MAIN_CFG['depth_num'])

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

# ZERO #########################################################################

print('[init] zeroing the state...')
__step = 0
__accum_loss = 0.0

OPTIMIZER_OBJ.zero_grad()

# TRAINING #####################################################################

for __epoch in range(TRAINING_CFG['epoch_num']):
    # create a new iterator since the previous one was exhausted
    __dataset = DATASET_OBJ.iter(batch_size=BATCH_CFG['batch_dim'])

    # iterate on batches
    for __batch in __dataset:
        __texts = __batch['text']

        # input_ids (B, T) and attention_mask (B, T)
        __inputs = TEXT_TOK(
            __texts,
            return_offsets_mapping=True,
            max_length=BATCH_CFG['sequence_dim'],
            truncation='longest_first',
            padding='max_length')

        # byte patches (B, T, G)
        __encoded = deformers.pipelines.patching.tokenize_into_bytes(
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
            __teacher_embeds = embed_tokens(SOURCE_MOD, __tokens_arr)
            __teacher_residuals = SOURCE_MOD.model(
                inputs_embeds=__teacher_embeds,
                attention_mask=__mask_arr,
                use_cache=False).last_hidden_state

        # student forward: prefix -> inputs_embeds -> trunk -> hidden_k
        with MIXED_CTX:
            __student_embeds = PREFIX_MOD(__bytes_arr)
            __student_residuals = SOURCE_MOD.model(
                inputs_embeds=__student_embeds,
                attention_mask=__mask_arr,
                use_cache=False).last_hidden_state

            # hidden-state MSE at depth k
            __loss_hidden = torch.nn.functional.mse_loss(__student_residuals, __teacher_residuals)
            # optional embedding MSE warmup
            __loss_embed = torch.nn.functional.mse_loss(__student_embeds.float(), __teacher_embeds.float())
            # total loss
            __loss = LOSS_CFG['hidden_rate'] * __loss_hidden + LOSS_CFG['embed_rate'] * __loss_embed
            __loss = __loss / GRADIENT_CFG['accumulation_num']

        SCALER_OBJ.scale(__loss).backward()
        __accum_loss += __loss.item()

        # optimizer step after gradient accumulation
        if (__step + 1) % GRADIENT_CFG['accumulation_num'] == 0:
            SCALER_OBJ.unscale_(OPTIMIZER_OBJ)
            torch.nn.utils.clip_grad_norm_(PREFIX_MOD.parameters(), max_norm=GRADIENT_CFG['max_norm'])
            SCALER_OBJ.step(OPTIMIZER_OBJ)
            SCALER_OBJ.update()
            OPTIMIZER_OBJ.zero_grad()

            print(f"[train] step={((__step + 1) // GRADIENT_CFG['accumulation_num']):04d} loss={__accum_loss:.6f}")
            __accum_loss = 0.0

        # track the global step (across epochs)
        __step += 1

# save prefix weights
torch.save({'config': PREFIX_MOD._config, 'state_dict': PREFIX_MOD.state_dict()}, OUTPUT_CFG['save_path'])
print(f"[train] saved prefix to {OUTPUT_CFG['save_path']}")
