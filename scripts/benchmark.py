"""
Evaluation script for prefix patch experiments.

Loads a frozen teacher model and a trained prefix checkpoint, then computes
detailed metrics over a fixed evaluation split and two deterministic probes.

Assumptions:
- Base model is qwen/qwen3.5-9b with hidden_size=4096.
- Tokenizer boundaries are identical to the base model.
- Trunk is frozen; prefix checkpoint is required (local path or HF repo).
- Byte block size default follows docs/roadmap.md (patch_dim=32), configurable.
- The byte tokenizer uses pad_id=128 (as implemented by ByteTokenizer).
- Memory-safe defaults: small batch, mixed precision on CUDA.
- Colab: upload checkpoint to /content/checkpoints/prefix.pt before running.

Metrics computed:
- eval loop (real text): embedding MSE/cosine, hidden-state MSE/cosine, logit KL, top-1/top-k
- sentence and vocab probes: embedding MSE and cosine similarity only (hidden-state and
  logit metrics are unreliable on fixed token sequences without global context)

Probes:
- fixed sentence probe: embed MSE and cosine per sentence
- vocab probe: deterministic (B, T) token tensor; per-token embed MSE/cosine table

Outputs:
- detailed stdout logs
- TensorBoard scalars under benchmark/ in .logs/
- JSON report artifact in .logs/benchmark_<timestamp>.json
"""

import contextlib
import datetime
import functools
import json
import os

import datasets
import huggingface_hub
import torch
import torch.amp
import torch.nn
import torch.utils.tensorboard
import transformers

import mlable.losses
import mlable.metrics
import mlable.models

import deformers.datasets.random
import deformers.models.generic
import deformers.models.prefix
import deformers.pipelines.eval
import deformers.pipelines.patch
import deformers.pipelines.prefix
import deformers.tokenizers.byte

# COMMON CONFIG ################################################################

MAIN_CFG = {
    'model_str': 'qwen/qwen3.5-9b',
    'device_str': 'cuda' if torch.cuda.is_available() else 'cpu',
    'dtype_obj': torch.bfloat16,
    'encoding_str': 'utf-8',
    'seed_num': 1337,
    'batch_dim': 4,
    'sequence_dim': 256,
    'patch_dim': 32,
    'depth_num': 1,
    'batch_num': 16,
    'topk_num': 10,}

# DATA CONFIG ##################################################################

DATASET_CFG = {
    'path': 'wikimedia/wikipedia',
    'name': '20231101.en',
    'split': 'train[90%:]',
    'streaming': False,}

BATCH_CFG = {
    'batch_dim': MAIN_CFG['batch_dim'],
    'sequence_dim': MAIN_CFG['sequence_dim'],
    'patch_dim': MAIN_CFG['patch_dim'],}

# PREPROCESSING CONFIG #########################################################

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
    'embed_dim': 128,
    'patch_dim': -1,
    'hidden_dim': 4096,
    'output_dim': 4096,
    'vocab_dim': 256,
    'padding_idx': 128,
    'block_num': 4,
    'head_num': 4,
    'dropout_rate': 0.001,}

# CHECKPOINT CONFIG ############################################################

REPOSITORY_CFG = {
    'repo_path': '',}  # optional HF repo to download checkpoint from

CHECKPOINT_CFG = {
    'path': os.environ.get('BENCHMARK_CHECKPOINT', '/content/checkpoints/prefix.pt'),
    'device': MAIN_CFG['device_str'],}
# NOTE: default path /content/checkpoints/prefix.pt targets Colab.
# Override via: BENCHMARK_CHECKPOINT=/path/to/prefix.pt python scripts/benchmark.py

# EVAL CONFIG ##################################################################

EVAL_CFG = {
    'batch_num': MAIN_CFG['batch_num'],
    'topk_num': MAIN_CFG['topk_num'],
    'probe_sentences': [
        'The quick brown fox jumps over the lazy dog.',
        'In the beginning was the Word, and the Word was with God.',],
    'vocab_probe': True,}

# OUTPUT CONFIG ################################################################

_TIMESTAMP = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H%M%SZ')

LOGGING_CFG = {
    'log_dir': os.path.abspath('.logs'),
    'report_path': os.path.abspath(f'.logs/benchmark_{_TIMESTAMP}.json'),}

# MIXED PRECISION ##############################################################

MIXED_CTX = (
    torch.amp.autocast(device_type='cuda', dtype=MAIN_CFG['dtype_obj'])
    if MAIN_CFG['device_str'] == 'cuda'
    else contextlib.nullcontext())

# TOKENIZERS ###################################################################

print('[init] loading the tokenizers...')
TEXT_TOK = transformers.AutoTokenizer.from_pretrained(**TOKEN_CFG)
BYTE_TOK = deformers.tokenizers.byte.ByteTokenizer(**BYTE_CFG)

print('[init] defining a padding token...')
TEXT_TOK.pad_token = TEXT_TOK.pad_token or TEXT_TOK.eos_token

print('[init] calculating the tokenizer metadata...')
VOCAB_LEN = len(TEXT_TOK.get_vocab())

# DATASET ######################################################################

print('[init] downloading the dataset...')
DATASET_OBJ = datasets.load_dataset(**DATASET_CFG)

print('[init] shuffling the dataset...')
DATASET_OBJ = DATASET_OBJ.shuffle(seed=MAIN_CFG['seed_num'])

print('[init] preprocessing the dataset...')
DATASET_OBJ = DATASET_OBJ.map(
    lambda __s: {'indices': TEXT_TOK(__s['text'], **PREPROC_CFG)['input_ids']},
    batched=True,
    remove_columns=['text'])

# MODELS #######################################################################

print('[init] creating the output directories...')
os.makedirs(DOWNLOAD_CFG['local_dir'], exist_ok=True)
os.makedirs(LOGGING_CFG['log_dir'], exist_ok=True)

print('[init] downloading the teacher...')
huggingface_hub.snapshot_download(**DOWNLOAD_CFG)

print('[init] loading the config...')
TRUNK_CFG = transformers.AutoConfig.from_pretrained(**CONFIG_CFG)

print('[init] truncating the config...')
TRUNK_CFG = deformers.models.generic.truncate_config(
    TRUNK_CFG, layer_num=MAIN_CFG['depth_num'], target_key='text_config')

print('[init] loading the teacher...')
SOURCE_MOD = transformers.AutoModelForCausalLM.from_pretrained(
    config=TRUNK_CFG, **MODEL_CFG).to(device=MAIN_CFG['device_str'])

print('[init] freezing the teacher...')
SOURCE_MOD.eval()
mlable.models.freeze(SOURCE_MOD)

print('[init] freeing unused memory...')
mlable.models.free_memory()

# CHECKPOINT ###################################################################

if REPOSITORY_CFG['repo_path']:
    print('[init] downloading the prefix checkpoint...')
    huggingface_hub.hf_hub_download(
        repo_id=REPOSITORY_CFG['repo_path'],
        filename=os.path.basename(CHECKPOINT_CFG['path']),
        local_dir=os.path.dirname(CHECKPOINT_CFG['path']),
        repo_type='model')

print(f'[init] loading the prefix checkpoint from {CHECKPOINT_CFG["path"]}...')
PREFIX_MOD = deformers.models.prefix.CompositeBytePrefix.load_checkpoint(
    path=CHECKPOINT_CFG['path'],
    shape=(BATCH_CFG['batch_dim'], BATCH_CFG['sequence_dim'], BATCH_CFG['patch_dim']),
    device=CHECKPOINT_CFG['device'])
PREFIX_MOD.eval()

# UTILITIES ####################################################################

print('[init] creating specialized utilities...')

vectorize = functools.partial(
    deformers.pipelines.prefix.vectorize_indices,
    text_tok=TEXT_TOK,
    byte_tok=BYTE_TOK,
    **VECTORIZE_CFG)

def embed(
    indices_arr: torch.Tensor,
    model_obj: object=SOURCE_MOD,
) -> torch.Tensor:
    """Get teacher token embeddings (no grad)."""
    return model_obj.model.embed_tokens(indices_arr)

def forward(
    embeds_arr: torch.Tensor,
    mask_arr: torch.Tensor,
    model_obj: object=SOURCE_MOD,
) -> torch.Tensor:
    """Run the trunk and return the last hidden state (no grad)."""
    return model_obj.model(
        inputs_embeds=embeds_arr,
        attention_mask=mask_arr,
        use_cache=False).last_hidden_state

def logits(
    hidden_arr: torch.Tensor,
    model_obj: object=SOURCE_MOD,
) -> torch.Tensor:
    """Project hidden states to vocabulary logits."""
    return model_obj.lm_head(hidden_arr)

# TENSORBOARD ##################################################################

print(f'[init] opening TensorBoard writer at {LOGGING_CFG["log_dir"]}...')
LOG_TB = torch.utils.tensorboard.SummaryWriter(log_dir=LOGGING_CFG['log_dir'])

# REPORT ACCUMULATOR ###########################################################

REPORT = {
    'timestamp': _TIMESTAMP,
    'model': MAIN_CFG['model_str'],
    'checkpoint': CHECKPOINT_CFG['path'],
    'eval_split': DATASET_CFG['split'],
    'batch_num': EVAL_CFG['batch_num'],
    'topk_num': EVAL_CFG['topk_num'],
    'eval_loop': {},
    'sentence_probe': {},
    'vocab_probe': {},}

# EVALUATION LOOP ##############################################################

print('[eval] starting evaluation loop...')

__n_batches = 0
__acc = {
    'embed_mse': 0.0,
    'embed_cos': 0.0,
    'hidden_mse': 0.0,
    'hidden_cos': 0.0,
    'kl': 0.0,
    'top1': 0.0,
    'topk': 0.0,}

__dataset = DATASET_OBJ.iter(batch_size=BATCH_CFG['batch_dim'])

for __batch in __dataset:
    if __n_batches >= EVAL_CFG['batch_num']:
        break

    # mask (B, T), tokens (B, T), bytes (B, T, G)
    __mask_arr, __indices_arr, __bytes_arr = vectorize(__batch['indices'])

    with torch.no_grad():
        # teacher: embeddings + hidden states + logits
        __teacher_embeds  = embed(__indices_arr)
        __teacher_hidden  = forward(__teacher_embeds, __mask_arr)
        __teacher_logits  = logits(__teacher_hidden)

        # student: prefix bytes -> embeds -> same trunk -> logits
        with MIXED_CTX:
            __student_embeds = PREFIX_MOD(__bytes_arr).to(dtype=__teacher_embeds.dtype)
        __student_hidden = forward(__student_embeds, __mask_arr)
        __student_logits = logits(__student_hidden)

    # per-position (B, T) masked metrics
    __metrics = deformers.pipelines.eval.per_token_metrics(
        teacher_embeds=__teacher_embeds,
        student_embeds=__student_embeds,
        teacher_hidden=__teacher_hidden,
        student_hidden=__student_hidden,
        teacher_logits=__teacher_logits,
        student_logits=__student_logits,
        mask=__mask_arr,
        k_num=EVAL_CFG['topk_num'])

    # accumulate masked means for each metric
    __mask_cpu = __mask_arr.cpu()
    for __key in __acc:
        __acc[__key] += deformers.pipelines.eval.summary_stats(
            __metrics[__key], __mask_cpu)['mean']

    __n_batches += 1
    if __n_batches % 4 == 0:
        print(f'[eval] batch {__n_batches}/{EVAL_CFG["batch_num"]}')

# SUMMARY ######################################################################

if __n_batches > 0:
    print('\n[eval] === summary metrics ===')
    print(f'[eval] batches evaluated  : {__n_batches}')
    __avgs = {__k: __v / __n_batches for __k, __v in __acc.items()}
    for __key, __val in __avgs.items():
        print(f'[eval] {__key:<18}: {__val:.6f}')

    # log summary scalars to TensorBoard
    for __key, __val in __avgs.items():
        LOG_TB.add_scalar(f'benchmark/eval/{__key}', __val, global_step=0)

    REPORT['eval_loop'] = {'n_batches': __n_batches, **__avgs}

# FIXED SENTENCE PROBE #########################################################

if EVAL_CFG['probe_sentences']:
    print('\n[eval] === fixed sentence probe ===')

    # build probe tensors from raw text
    __probe_mask, __probe_ids, __probe_bytes = deformers.pipelines.prefix.vectorize_strings(
        text_arr=EVAL_CFG['probe_sentences'],
        text_tok=TEXT_TOK,
        byte_tok=BYTE_TOK,
        sequence_dim=BATCH_CFG['sequence_dim'],
        patch_dim=BATCH_CFG['patch_dim'],
        device_str=MAIN_CFG['device_str'],
        left_pad=True)

    with torch.no_grad():
        __p_teacher_embeds = embed(__probe_ids)
        with MIXED_CTX:
            __p_student_embeds = PREFIX_MOD(__probe_bytes).to(dtype=__p_teacher_embeds.dtype)

    __p_embed_mse = mlable.losses.mse_loss(
        __p_student_embeds.float(), __p_teacher_embeds.float(),
        mask_arr=__probe_mask, relative_opt=True, reduce_opt=False).cpu()
    __p_embed_cos = mlable.losses.cos_sim(
        __p_student_embeds.float(), __p_teacher_embeds.float(),
        mask_arr=__probe_mask, reduce_opt=False).cpu()

    __sentence_report = []
    for __i, __sent in enumerate(EVAL_CFG['probe_sentences']):
        __s_mask = __probe_mask[__i:__i + 1].cpu()
        __s_mse = deformers.pipelines.eval.summary_stats(__p_embed_mse[__i:__i + 1], __s_mask)
        __s_cos = deformers.pipelines.eval.summary_stats(__p_embed_cos[__i:__i + 1], __s_mask)
        print(f'[eval] sentence {__i}: "{__sent[:60]}"')
        print(f'[eval]   embed_mse={__s_mse["mean"]:.4f}  embed_cos={__s_cos["mean"]:.4f}')
        __sentence_report.append({
            'sentence': __sent,
            'embed_mse': __s_mse,
            'embed_cos': __s_cos})

    REPORT['sentence_probe'] = __sentence_report

# VOCAB PROBE ##################################################################

if EVAL_CFG['vocab_probe']:
    print('\n[eval] === vocab probe ===')

    # build tensors from the deterministic token ID grid
    __vocab_mask, __vocab_ids, __vocab_bytes = vectorize(
        deformers.pipelines.eval.indices_probe(
            vocab_dim=VOCAB_LEN,
            batch_dim=BATCH_CFG['batch_dim'],
            sequence_dim=BATCH_CFG['sequence_dim']))

    with torch.no_grad():
        __v_teacher_embeds = embed(__vocab_ids)
        with MIXED_CTX:
            __v_student_embeds = PREFIX_MOD(__vocab_bytes).to(dtype=__v_teacher_embeds.dtype)

    __v_embed_mse = mlable.losses.mse_loss(
        __v_student_embeds.float(), __v_teacher_embeds.float(),
        mask_arr=__vocab_mask, relative_opt=True, reduce_opt=False).cpu()
    __v_embed_cos = mlable.losses.cos_sim(
        __v_student_embeds.float(), __v_teacher_embeds.float(),
        mask_arr=__vocab_mask, reduce_opt=False).cpu()

    __vocab_mask_cpu = __vocab_mask.cpu()
    __vocab_summary = {
        'embed_mse': deformers.pipelines.eval.summary_stats(__v_embed_mse, __vocab_mask_cpu),
        'embed_cos': deformers.pipelines.eval.summary_stats(__v_embed_cos, __vocab_mask_cpu),}

    print('[eval] vocab probe summary:')
    for __key, __stats in __vocab_summary.items():
        print(f'[eval]   {__key:<18}: mean={__stats["mean"]:.6f}'
              f'  median={__stats["median"]:.6f}'
              f'  p95={__stats["p95"]:.6f}')

    # log vocab probe scalars to TensorBoard
    for __key, __stats in __vocab_summary.items():
        for __stat_name, __val in __stats.items():
            LOG_TB.add_scalar(
                f'benchmark/vocab_probe/{__key}/{__stat_name}', __val, global_step=0)

    # build per-token table: flatten (B, T) positions and decode token IDs
    __flat_ids = __vocab_ids.flatten().tolist()
    __flat_strings = [TEXT_TOK.decode([__i]) for __i in __flat_ids]
    __flat_metrics = {
        'embed_mse': __v_embed_mse.flatten().tolist(),
        'embed_cos': __v_embed_cos.flatten().tolist(),}
    __table = deformers.pipelines.eval.token_table(
        token_ids=__flat_ids,
        token_strings=__flat_strings,
        metrics=__flat_metrics)

    # hardest tokens: top-10 by embed_mse
    __sorted_hard = sorted(__table, key=lambda __r: __r['embed_mse'], reverse=True)
    __sorted_easy = sorted(__table, key=lambda __r: __r['embed_mse'])

    print(f'\n[eval] 10 hardest tokens (by embed_mse):')
    print(f'  {"id":>8}  {"token":<20}  {"bytes":>5}  {"embed_mse":>10}  {"embed_cos":>10}')
    for __row in __sorted_hard[:10]:
        print(f'  {__row["token_id"]:>8}  {repr(__row["token_string"]):<20}  '
              f'{__row["byte_length"]:>5}  {__row["embed_mse"]:>10.6f}  '
              f'{__row["embed_cos"]:>10.6f}')

    print(f'\n[eval] 10 easiest tokens (by embed_mse):')
    print(f'  {"id":>8}  {"token":<20}  {"bytes":>5}  {"embed_mse":>10}  {"embed_cos":>10}')
    for __row in __sorted_easy[:10]:
        print(f'  {__row["token_id"]:>8}  {repr(__row["token_string"]):<20}  '
              f'{__row["byte_length"]:>5}  {__row["embed_mse"]:>10.6f}  '
              f'{__row["embed_cos"]:>10.6f}')

    REPORT['vocab_probe'] = {
        'summary': __vocab_summary,
        'hardest_10': __sorted_hard[:10],
        'easiest_10': __sorted_easy[:10],
        'full_table': __table,}

# SAVE REPORT ##################################################################

print(f'\n[eval] saving JSON report to {LOGGING_CFG["report_path"]}...')
deformers.pipelines.eval.save_json_report(REPORT, LOGGING_CFG['report_path'])

print(f'[eval] closing TensorBoard writer...')
LOG_TB.close()

print('[eval] done.')

# COLAB USAGE ##################################################################
# Upload your checkpoint to /content/checkpoints/prefix.pt, then run:
#   !python scripts/benchmark.py
# or set a custom path:
#   BENCHMARK_CHECKPOINT=/path/to/prefix.pt python scripts/benchmark.py
# TensorBoard:
#   %load_ext tensorboard
#   %tensorboard --logdir .logs
