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
import torch
import torch.amp
import torch.nn
import transformers

import deformers.layers.prefix
import deformers.pipelines.patching
import deformers.tokenizers.byte

# COMMON CONFIG ################################################################

MAIN_CFG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'encoding': 'utf-8',
    'seed': 1337,
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
    'pretrained_model_name_or_path': 'qwen/qwen3.5-9b',
    'use_fast': True,}

BYTE_CFG = {
    'encoding': MAIN_CFG['encoding'],}

# MODEL CONFIG #################################################################

MODEL_CFG = {
    'pretrained_model_name_or_path': 'qwen/qwen3.5-9b',
    'device_map': MAIN_CFG['device'],
    'torch_dtype': torch.bfloat16,}

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
    'enabled': MAIN_CFG['device'] == 'cuda',}

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
    'save_path': 'checkpoints/prefix.pt',}

# UTILS ########################################################################

def freeze_model(model: torch.nn.Module) -> None:
    """Disable gradients for all model parameters."""
    for __p in model.parameters():
        __p.requires_grad_(False)

def embed_tokens(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Return the embedding layer output for the given token ids."""
    return model.model.embed_tokens(input_ids)

# DATASET ######################################################################

print('[init] downloading dataset...')
DATASET_OBJ = datasets.load_dataset(**DATASET_CFG)

print('[init] preprocessing dataset...')
DATASET_OBJ = DATASET_OBJ.shuffle(seed=MAIN_CFG['seed']).iter(batch_size=BATCH_CFG['batch_dim'])

# TOKENIZERS ###################################################################

print('[init] loading tokenizers...')
TEXT_TOK = transformers.AutoTokenizer.from_pretrained(**TOKEN_CFG)
BYTE_TOK = deformers.tokenizers.byte.ByteTokenizer(encoding=MAIN_CFG['encoding'])

# MODELS #######################################################################

print('[init] loading models...')
ORIGIN_MOD = transformers.AutoModelForCausalLM.from_pretrained(**MODEL_CFG).to(device=MAIN_CFG['device'])
PREFIX_MOD = deformers.layers.prefix.CompositeBytePrefix(**PREFIX_CFG).to(device=MAIN_CFG['device'])

print('[init] freezing teacher...')
ORIGIN_MOD.eval()
freeze_model(ORIGIN_MOD)

print('[init] building the student...')
PREFIX_MOD._build(
    shape_arr=(BATCH_CFG['batch_dim'], BATCH_CFG['sequence_dim'], BATCH_CFG['patch_dim']),
    device_str=MAIN_CFG['device'])

# OPTIMIZER ####################################################################

print('[init] creating optimizer...')
OPTIMIZER_OBJ = torch.optim.AdamW(PREFIX_MOD.parameters(), **OPTIMIZER_CFG)
SCALER_OBJ = torch.amp.GradScaler(**SCALER_CFG)

# INIT #########################################################################

print('[init] zeroing the state...')
__step = 0
__accum_loss = 0.0

OPTIMIZER_OBJ.zero_grad()

# TRAINING #####################################################################

for __epoch in range(TRAINING_CFG['epoch_num']):
    for __batch in iter(DATASET_OBJ):
        # sample a batch
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
        __tokens_arr = torch.tensor(__inputs['input_ids'], dtype=torch.long, device=MAIN_CFG['device'])
        __mask_arr = torch.tensor(__inputs['attention_mask'], dtype=torch.long, device=MAIN_CFG['device'])
        __bytes_arr = torch.tensor(__encoded, dtype=torch.long, device=MAIN_CFG['device'])

        # mixed precision
        __amp_ctx = (
            torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
            if SCALER_CFG['enabled']
            else contextlib.nullcontext())

        # teacher forward: get original embeddings and hidden states (no grad)
        with torch.no_grad():
            __teacher_embeds = embed_tokens(ORIGIN_MOD, __tokens_arr)
            __teacher_out = ORIGIN_MOD(
                input_ids=__tokens_arr,
                attention_mask=__mask_arr,
                output_hidden_states=True,
                use_cache=False)
            __teacher_hidden_k = __teacher_out.hidden_states[MAIN_CFG['depth_num']].detach()
            __teacher_embeds = __teacher_embeds.detach()

        # student forward: prefix -> inputs_embeds -> trunk -> hidden_k
        with __amp_ctx:
            __student_embeds = PREFIX_MOD(__bytes_arr)
            __student_out = ORIGIN_MOD(
                inputs_embeds=__student_embeds,
                attention_mask=__mask_arr,
                output_hidden_states=True,
                use_cache=False)
            __student_hidden_k = __student_out.hidden_states[MAIN_CFG['depth_num']]

            # hidden-state MSE at depth k
            __loss_hidden = torch.nn.functional.mse_loss(__student_hidden_k, __teacher_hidden_k)
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
os.makedirs(os.path.dirname(OUTPUT_CFG['save_path']), exist_ok=True)
torch.save({'config': PREFIX_MOD._config, 'state_dict': PREFIX_MOD.state_dict()}, OUTPUT_CFG['save_path'])
print(f"[train] saved prefix to {OUTPUT_CFG['save_path']}")
