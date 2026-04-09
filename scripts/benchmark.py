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

import deformers.layers.prefix
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

REPOSITORY_CFG = {
    'repo_path': '',}

CHECKPOINT_CFG = {
    'file_path': os.path.abspath('checkpoints/prefix.pt'),
    'device_str': MAIN_CFG['device_str'],}

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

def save_checkpoint(
    model_obj: torch.nn.Module,
    path_str: str=CHECKPOINT_CFG['file_path'],
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

def build_vocab_probe(
    vocab_dim: int,
    batch_dim: int,
    seq_dim: int,
) -> torch.Tensor:
    """Build a deterministic (B, T) token id tensor using consecutive vocab IDs."""
    __total = batch_dim * seq_dim
    __ids = torch.arange(__total, dtype=torch.long) % vocab_dim
    return __ids.reshape(batch_dim, seq_dim)

def build_text_probe(
    texts_arr: list,
    text_tokenizer: object,
    byte_tokenizer: object,
    seq_dim: int=256,
    patch_dim: int=32,
    device_str: str='cpu',
) -> tuple:
    """Build a deterministic fixed probe batch from text samples."""
    __inputs = text_tokenizer(
        texts_arr,
        return_offsets_mapping=True,
        max_length=seq_dim,
        truncation='longest_first',
        padding='max_length')
    __encoded = deformers.pipelines.patch.tokenize_into_bytes(
        texts_arr=texts_arr,
        offsets_arr=__inputs['offset_mapping'],
        patch_dim=patch_dim,
        tokenizer_obj=byte_tokenizer)
    __tokens_arr = torch.tensor(__inputs['input_ids'], dtype=torch.long, device=device_str)
    __mask_arr = torch.tensor(__inputs['attention_mask'], dtype=torch.long, device=device_str)
    __bytes_arr = torch.tensor(__encoded, dtype=torch.long, device=device_str)
    return __tokens_arr, __mask_arr, __bytes_arr

def build_vocab_probe_bytes(
    vocab_ids: torch.Tensor,
    text_tokenizer: object,
    byte_tokenizer: object,
    patch_dim: int=32,
) -> torch.Tensor:
    """Build byte patch tensor (B, T, G) from a (B, T) vocab probe token id tensor."""
    __B, __T = vocab_ids.shape
    __flat = vocab_ids.flatten().tolist()
    # decode each token id to its actual text representation
    __strings = [text_tokenizer.decode([__tid], skip_special_tokens=False) for __tid in __flat]
    # reshape into (B, T) list of lists
    __tokens_2d = [__strings[__i * __T:(__i + 1) * __T] for __i in range(__B)]
    # encode each token string as a fixed-length byte block
    __encoded = deformers.pipelines.patch.encode_into_bytes(
        tokens_arr=__tokens_2d,
        patch_dim=patch_dim,
        tokenizer_obj=byte_tokenizer)
    return torch.tensor(__encoded, dtype=torch.long, device=vocab_ids.device)

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

if REPOSITORY_CFG['repo_path']:
    print('[init] downloading the prefix checkpoint...')
    huggingface_hub.hf_hub_download(
        repo_id=REPOSITORY_CFG['repo_path'],
        filename=os.path.basename(CHECKPOINT_CFG['file_path']),
        local_dir=os.path.dirname(CHECKPOINT_CFG['file_path']),
        repo_type='model')

print('[eval] loading the prefix weights...')
PREFIX_MOD = load_checkpoint(**CHECKPOINT_CFG)
PREFIX_MOD.eval()

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
        __teacher_embeds = SOURCE_MOD.model.embed_tokens(__tokens_arr)
        __teacher_residuals = SOURCE_MOD.model(inputs_embeds=__teacher_embeds, attention_mask=__mask_arr, use_cache=False).last_hidden_state
        __teacher_logits = SOURCE_MOD.lm_head(__teacher_residuals)

        # student forward: prefix -> inputs_embeds -> trunk -> logits
        with MIXED_CTX:
            __student_embeds = PREFIX_MOD(__bytes_arr)
        __student_residuals = SOURCE_MOD.model(inputs_embeds=__student_embeds, attention_mask=__mask_arr, use_cache=False).last_hidden_state
        __student_logits = SOURCE_MOD.lm_head(__student_residuals)

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
    __probe_tokens, __probe_mask, __probe_bytes = build_text_probe(
        texts_arr=EVAL_CFG['probe_sentences'],
        text_tokenizer=TEXT_TOK,
        byte_tokenizer=BYTE_TOK,
        seq_dim=BATCH_CFG['sequence_dim'],
        patch_dim=BATCH_CFG['patch_dim'],
        device_str=MAIN_CFG['device_str'])

    with torch.no_grad():
        __p_teacher_embeds = SOURCE_MOD.model.embed_tokens(__probe_tokens)
        __p_teacher_residuals = SOURCE_MOD.model(inputs_embeds=__p_teacher_embeds, attention_mask=__probe_mask, use_cache=False).last_hidden_state
        __p_teacher_logits = SOURCE_MOD.lm_head(__p_teacher_residuals)
        with MIXED_CTX:
            __p_student_embeds = PREFIX_MOD(__probe_bytes)
            __p_student_residuals = SOURCE_MOD.model(inputs_embeds=__p_student_embeds, attention_mask=__probe_mask, use_cache=False).last_hidden_state
            __p_student_logits = SOURCE_MOD.lm_head(__p_student_residuals)

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
    __vocab_ids = build_vocab_probe(
        vocab_size=__vocab_size,
        batch_dim=BATCH_CFG['batch_dim'],
        seq_dim=BATCH_CFG['sequence_dim']).to(device=MAIN_CFG['device_str'])

    # build corresponding byte patches by decoding each token to its text
    __vocab_bytes = build_vocab_probe_bytes(
        vocab_ids=__vocab_ids,
        text_tokenizer=TEXT_TOK,
        byte_tokenizer=BYTE_TOK,
        patch_dim=BATCH_CFG['patch_dim']).to(device=MAIN_CFG['device_str'])

    __vocab_mask = torch.ones_like(__vocab_ids)

    with torch.no_grad():
        __v_teacher_embeds = SOURCE_MOD.model.embed_tokens(__vocab_ids)
        __v_teacher_residuals = SOURCE_MOD.model(inputs_embeds=__v_teacher_embeds, attention_mask=__vocab_mask, use_cache=False).last_hidden_state
        __v_teacher_logits = SOURCE_MOD.lm_head(__v_teacher_residuals)
        with MIXED_CTX:
            __v_student_embeds = PREFIX_MOD(__vocab_bytes)
            __v_student_residuals = SOURCE_MOD.model(inputs_embeds=__v_student_embeds, attention_mask=__vocab_mask, use_cache=False).last_hidden_state
            __v_student_logits = SOURCE_MOD.lm_head(__v_student_residuals)

    print(f'[eval] vocab embed MSE   : {torch.nn.functional.mse_loss(__v_teacher_embeds.float(), __v_student_embeds.float()).item():.6f}')
    print(f'[eval] vocab hidden MSE  : {torch.nn.functional.mse_loss(__v_teacher_residuals.float(), __v_student_residuals.float()).item():.6f}')
    print(f'[eval] vocab KL          : {deformers.pipelines.eval.kl_divergence(__v_teacher_logits.float(), __v_student_logits.float()):.6f}')
    print(f'[eval] vocab top-k       : {deformers.pipelines.eval.topk_rate(__v_teacher_logits, __v_student_logits):.4f}')
