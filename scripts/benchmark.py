"""
Evaluation script for prefix patch experiments.

Loads a teacher model and an alternative prefix from a checkpoint, then
computes and prints summary metrics over a fixed evaluation split.

Assumptions:
- Base model is qwen/qwen3.5-9b with hidden_size=4096.
- Tokenizer boundaries are identical to the base model.
- Trunk is frozen; prefix checkpoint is required (local path or HF repo).
- Byte block size default follows docs/roadmap.md (L_max=32), configurable.
- The byte tokenizer uses pad_id=128 (as implemented by ByteTokenizer).
- Memory-safe defaults: small batch, mixed precision on CUDA.

Metrics computed:
- embedding MSE
- hidden-state MSE at the configured trunk depth
- KL divergence (teacher logits vs student logits)
- top-k match rate (ordered)

Optional probes:
- fixed sentence probe: teacher vs student top-k tokens for the same contexts
- vocab probe: metrics on a deterministic (B, T) token tensor
"""

import contextlib
import os

import datasets
import huggingface_hub
import torch
import torch.amp
import torch.nn
import torch.nn.functional
import transformers

import deformers.models.generic
import deformers.pipelines.eval
import deformers.pipelines.patch
import deformers.tokenizers.byte

# COMMON CONFIG ################################################################

MAIN_CFG = {
    'model_str': 'qwen/qwen3.5-9b',
    'device_str': 'cuda' if torch.cuda.is_available() else 'cpu',
    'encoding_str': 'utf-8',
    'seed_num': 1337,
    'batch_dim': 4,
    'sequence_dim': 256,
    'patch_dim': 32,
    'depth_num': 4,
    'batch_num': 16,
    'topk_num': 10,}

# DATA CONFIG ##################################################################

DATASET_CFG = {
    'path': 'wikimedia/wikipedia',
    'name': '20231101.en',
    'split': 'train[90%:]',  # small bounded eval split
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
    'torch_dtype': torch.bfloat16,
    'low_cpu_mem_usage': True,
    'ignore_mismatched_sizes': True,}

# CHECKPOINT CONFIG ############################################################

CHECKPOINT_CFG = {
    'local_path': os.path.abspath('checkpoints/prefix.pt'),
    'hf_repo': '',       # optional HF repo id; overrides local_path if set
    'hf_filename': 'prefix.pt',}

# EVAL CONFIG ##################################################################

EVAL_CFG = {
    'batch_num': MAIN_CFG['batch_num'],
    'topk_num': MAIN_CFG['topk_num'],
    'probe_sentences': [
        'The quick brown fox jumps over the lazy dog.',
        'In the beginning was the Word, and the Word was with God.',],
    'vocab_probe': True,}

# UTILS ########################################################################

def freeze_model(model: torch.nn.Module) -> None:
    """Disable gradients for all model parameters."""
    for __p in model.parameters():
        __p.requires_grad_(False)

# MIXED PRECISION ##############################################################

MIXED_CTX = (
    torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    if MAIN_CFG['device_str'] == 'cuda'
    else contextlib.nullcontext())

# DATASET ######################################################################

print('[eval] downloading the dataset...')
DATASET_OBJ = datasets.load_dataset(**DATASET_CFG)

print('[eval] preprocessing the dataset...')
DATASET_OBJ = DATASET_OBJ.shuffle(seed=MAIN_CFG['seed_num'])

# TOKENIZERS ###################################################################

print('[eval] loading the tokenizers...')
TEXT_TOK = transformers.AutoTokenizer.from_pretrained(**TOKEN_CFG)
BYTE_TOK = deformers.tokenizers.byte.ByteTokenizer(**BYTE_CFG)

# MODELS #######################################################################

print('[eval] creating the output directories...')
os.makedirs(DOWNLOAD_CFG['local_dir'], exist_ok=True)

print('[eval] downloading the teacher...')
huggingface_hub.snapshot_download(**DOWNLOAD_CFG)

print('[eval] loading the config...')
TRUNK_CFG = transformers.AutoConfig.from_pretrained(**CONFIG_CFG)

print('[eval] truncating the config...')
TRUNK_CFG = deformers.models.generic.truncate_config(
    TRUNK_CFG, layer_num=MAIN_CFG['depth_num'], target_key='text_config')

print('[eval] loading the teacher...')  # load only the used layers up to the chosen depth
SOURCE_MOD = transformers.AutoModelForCausalLM.from_pretrained(
    config=TRUNK_CFG, **MODEL_CFG).to(device=MAIN_CFG['device_str'])

print('[eval] freezing the teacher...')
SOURCE_MOD.eval()
freeze_model(SOURCE_MOD)

print('[eval] freeing unused memory...')
deformers.models.generic.free_memory()

print('[eval] loading the prefix checkpoint...')
PREFIX_MOD = deformers.pipelines.eval.load_prefix_checkpoint(
    local_path=CHECKPOINT_CFG['local_path'],
    hf_repo=CHECKPOINT_CFG['hf_repo'],
    hf_filename=CHECKPOINT_CFG['hf_filename'],
    device_str=MAIN_CFG['device_str'])
PREFIX_MOD = PREFIX_MOD.to(device=MAIN_CFG['device_str'])

print('[eval] building the prefix...')
PREFIX_MOD._build(
    shape_arr=(BATCH_CFG['batch_dim'], BATCH_CFG['sequence_dim'], BATCH_CFG['patch_dim']),
    device_str=MAIN_CFG['device_str'])

# ACCUMULATORS #################################################################

__n_batches = 0
__sum_embed_mse = 0.0
__sum_hidden_mse = 0.0
__sum_kl = 0.0
__sum_topk = 0.0

# EVALUATION LOOP ##############################################################

print('[eval] starting evaluation...')
__dataset = DATASET_OBJ.iter(batch_size=BATCH_CFG['batch_dim'])

for __batch in __dataset:
    if __n_batches >= EVAL_CFG['batch_num']:
        break

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

    with torch.no_grad():
        # teacher forward: embeddings, residuals and logits (no grad)
        __teacher_embeds = deformers.pipelines.eval.teacher_embed(SOURCE_MOD, __tokens_arr)
        __teacher_residuals, __teacher_logits = deformers.pipelines.eval.teacher_forward(
            SOURCE_MOD, __teacher_embeds, __mask_arr)

        # student forward: prefix -> inputs_embeds -> trunk -> logits
        with MIXED_CTX:
            __student_embeds = PREFIX_MOD(__bytes_arr)
            __student_residuals, __student_logits = deformers.pipelines.eval.teacher_forward(
                SOURCE_MOD, __student_embeds, __mask_arr)

    # accumulate metrics
    __sum_embed_mse += torch.nn.functional.mse_loss(__teacher_embeds.float(), __student_embeds.float()).item()
    __sum_hidden_mse += torch.nn.functional.mse_loss(__teacher_residuals.float(), __student_residuals.float()).item()
    __sum_kl += deformers.pipelines.eval.kl_divergence(__teacher_logits.float(), __student_logits.float()).item()
    __sum_topk += deformers.pipelines.eval.topk_rate(__teacher_logits, __student_logits, k_num=EVAL_CFG['topk_num']).item()

    __n_batches += 1
    if __n_batches % 4 == 0:
        print(f'[eval] batch {__n_batches}/{EVAL_CFG["batch_num"]}')

# SUMMARY ######################################################################

if __n_batches > 0:
    print('\n[eval] === summary metrics ===')
    print(f'[eval] batches evaluated : {__n_batches}')
    print(f'[eval] embed MSE         : {__sum_embed_mse / __n_batches:.6f}')
    print(f'[eval] hidden MSE        : {__sum_hidden_mse / __n_batches:.6f}')
    print(f'[eval] KL divergence     : {__sum_kl / __n_batches:.6f}')
    print(f'[eval] top-k match       : {__sum_topk / __n_batches:.4f} (k={EVAL_CFG["topk_num"]})')

# FIXED SENTENCE PROBE #########################################################

if EVAL_CFG['probe_sentences']:
    print('\n[eval] === fixed sentence probe ===')
    __probe_tokens, __probe_mask, __probe_bytes = deformers.pipelines.eval.build_text_probe(
        texts_arr=EVAL_CFG['probe_sentences'],
        text_tokenizer=TEXT_TOK,
        byte_tokenizer=BYTE_TOK,
        seq_dim=BATCH_CFG['sequence_dim'],
        patch_dim=BATCH_CFG['patch_dim'],
        device_str=MAIN_CFG['device_str'])

    with torch.no_grad():
        __p_teacher_embeds = deformers.pipelines.eval.teacher_embed(SOURCE_MOD, __probe_tokens)
        __p_teacher_residuals, __p_teacher_logits = deformers.pipelines.eval.teacher_forward(
            SOURCE_MOD, __p_teacher_embeds, __probe_mask)
        with MIXED_CTX:
            __p_student_embeds = PREFIX_MOD(__probe_bytes)
            __p_student_residuals, __p_student_logits = deformers.pipelines.eval.teacher_forward(
                SOURCE_MOD, __p_student_embeds, __probe_mask)

    __k = EVAL_CFG['topk_num']
    for __i, __sent in enumerate(EVAL_CFG['probe_sentences']):
        # report at the last real token position
        __pos = int(__probe_mask[__i].sum().item()) - 1
        __t_top = __p_teacher_logits[__i, __pos].topk(__k).indices.tolist()
        __s_top = __p_student_logits[__i, __pos].topk(__k).indices.tolist()
        __t_toks = TEXT_TOK.convert_ids_to_tokens(__t_top)
        __s_toks = TEXT_TOK.convert_ids_to_tokens(__s_top)
        print(f'[eval] sentence {__i}: "{__sent[:60]}"')
        print(f'[eval] teacher top-{__k}: {__t_toks}')
        print(f'[eval] student top-{__k}: {__s_toks}')

# VOCAB PROBE ##################################################################

if EVAL_CFG['vocab_probe']:
    print('\n[eval] === vocab probe ===')
    # resolve vocab size from text_config if available (multimodal models)
    __vocab_size = (
        SOURCE_MOD.config.text_config.vocab_size
        if hasattr(SOURCE_MOD.config, 'text_config')
        else SOURCE_MOD.config.vocab_size)
    __vocab_ids = deformers.pipelines.eval.build_vocab_probe(
        vocab_size=__vocab_size,
        batch_dim=BATCH_CFG['batch_dim'],
        seq_dim=BATCH_CFG['sequence_dim']).to(device=MAIN_CFG['device_str'])

    # build corresponding byte patches by decoding each token to its text
    __vocab_bytes = deformers.pipelines.eval.build_vocab_probe_bytes(
        vocab_ids=__vocab_ids,
        text_tokenizer=TEXT_TOK,
        byte_tokenizer=BYTE_TOK,
        patch_dim=BATCH_CFG['patch_dim'])

    __vocab_mask = torch.ones_like(__vocab_ids)

    with torch.no_grad():
        __v_teacher_embeds = deformers.pipelines.eval.teacher_embed(SOURCE_MOD, __vocab_ids)
        __v_teacher_residuals, __v_teacher_logits = deformers.pipelines.eval.teacher_forward(
            SOURCE_MOD, __v_teacher_embeds, __vocab_mask)
        with MIXED_CTX:
            __v_student_embeds = PREFIX_MOD(__vocab_bytes)
            __v_student_residuals, __v_student_logits = deformers.pipelines.eval.teacher_forward(
                SOURCE_MOD, __v_student_embeds, __vocab_mask)

    print(f'[eval] vocab embed MSE   : {torch.nn.functional.mse_loss(__v_teacher_embeds.float(), __v_student_embeds.float()).item():.6f}')
    print(f'[eval] vocab hidden MSE  : {torch.nn.functional.mse_loss(__v_teacher_residuals.float(), __v_student_residuals.float()).item():.6f}')
    print(f'[eval] vocab KL          : {deformers.pipelines.eval.kl_divergence(__v_teacher_logits.float(), __v_student_logits.float()):.6f}')
    print(f'[eval] vocab top-k       : {deformers.pipelines.eval.topk_rate(__v_teacher_logits, __v_student_logits):.4f}')
