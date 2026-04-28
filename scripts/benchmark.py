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

Metrics computed (all masked):
- embedding MSE and cosine similarity
- hidden-state MSE and cosine similarity at the configured trunk depth
- logit KL divergence (teacher vs student)
- top-1 and top-k set match rate

Probes:
- fixed sentence probe: teacher vs student top-k tokens at last real position
- vocab probe: deterministic (B, T) token tensor; per-token table + rankings

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
    'model_str':    'qwen/qwen3.5-9b',
    'device_str':   'cuda' if torch.cuda.is_available() else 'cpu',
    'dtype_obj':    torch.bfloat16,
    'encoding_str': 'utf-8',
    'seed_num':     1337,
    'batch_dim':    4,
    'sequence_dim': 256,
    'patch_dim':    32,
    'depth_num':    1,
    'batch_num':    16,
    'topk_num':     10,}

# DATA CONFIG ##################################################################

DATASET_CFG = {
    'path':      'wikimedia/wikipedia',
    'name':      '20231101.en',
    'split':     'train[90%:]',
    'streaming': False,}

BATCH_CFG = {
    'batch_dim':    MAIN_CFG['batch_dim'],
    'sequence_dim': MAIN_CFG['sequence_dim'],
    'patch_dim':    MAIN_CFG['patch_dim'],}

# PREPROCESSING CONFIG #########################################################

PREPROC_CFG = {
    'truncation': 'longest_first',
    'padding':    'max_length',
    'max_length': BATCH_CFG['sequence_dim'],}

VECTORIZE_CFG = {
    'sequence_dim': BATCH_CFG['sequence_dim'],
    'patch_dim':    BATCH_CFG['patch_dim'],
    'device_str':   MAIN_CFG['device_str'],
    'dtype_obj':    torch.long,
    'left_pad':     True,}

TOKEN_CFG = {
    'pretrained_model_name_or_path': MAIN_CFG['model_str'],
    'use_fast':                      True,}

BYTE_CFG = {
    'encoding': MAIN_CFG['encoding_str'],}

# MODEL CONFIG #################################################################

DOWNLOAD_CFG = {
    'repo_id':         MAIN_CFG['model_str'],
    'repo_type':       'model',
    'local_dir':       os.path.abspath('downloads'),
    'ignore_patterns': ['*.onnx', '*.tflite', '*.msgpack'],}

CONFIG_CFG = {
    'pretrained_model_name_or_path': DOWNLOAD_CFG['local_dir'],
    'trust_remote_code':             False,}

MODEL_CFG = {
    'pretrained_model_name_or_path': DOWNLOAD_CFG['local_dir'],
    'trust_remote_code':             CONFIG_CFG['trust_remote_code'],
    'dtype':                         torch.bfloat16,
    'low_cpu_mem_usage':             True,
    'ignore_mismatched_sizes':       True,}

PREFIX_CFG = {
    'embed_dim':     128,   # dimension of each byte embedding
    'patch_dim':     -1,    # inferred from input shape
    'hidden_dim':    4096,  # intermediate MLP width
    'output_dim':    4096,  # teacher hidden_size (qwen3.5-9b)
    'vocab_dim':     256,   # byte vocabulary size
    'padding_idx':   128,   # ByteTokenizer pad value
    'block_num':     4,     # number of ByteTransformer blocks
    'head_num':      4,     # self-attention heads
    'dropout_rate':  0.001,}

# CHECKPOINT CONFIG ############################################################

REPOSITORY_CFG = {
    'repo_path': '',}  # optional HF repo to download checkpoint from

CHECKPOINT_CFG = {
    'path':   os.environ.get('BENCHMARK_CHECKPOINT', '/content/checkpoints/prefix.pt'),
    'device': MAIN_CFG['device_str'],}
# NOTE: default path /content/checkpoints/prefix.pt targets Colab.
# Override via: BENCHMARK_CHECKPOINT=/path/to/prefix.pt python scripts/benchmark.py

# EVAL CONFIG ##################################################################

EVAL_CFG = {
    'batch_num': MAIN_CFG['batch_num'],
    'topk_num':  MAIN_CFG['topk_num'],
    'probe_sentences': [
        'The quick brown fox jumps over the lazy dog.',
        'In the beginning was the Word, and the Word was with God.',],
    'vocab_probe': True,}

# OUTPUT CONFIG ################################################################

_TIMESTAMP = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H%M%SZ')

LOGGING_CFG = {
    'log_dir':     os.path.abspath('.logs'),
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
PREFIX_MOD = deformers.pipelines.eval.load_prefix_checkpoint(
    path=CHECKPOINT_CFG['path'],
    shape=(BATCH_CFG['batch_dim'], BATCH_CFG['sequence_dim'], BATCH_CFG['patch_dim']),
    device=CHECKPOINT_CFG['device'])
PREFIX_MOD.eval()

print('[init] prefix model summary...')
deformers.pipelines.eval.model_summary(
    model_obj=PREFIX_MOD,
    input_shape=(BATCH_CFG['batch_dim'], BATCH_CFG['sequence_dim'], BATCH_CFG['patch_dim']))

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
    'model':     MAIN_CFG['model_str'],
    'checkpoint': CHECKPOINT_CFG['path'],
    'eval_split': DATASET_CFG['split'],
    'batch_num':  EVAL_CFG['batch_num'],
    'topk_num':   EVAL_CFG['topk_num'],
    'eval_loop': {},
    'sentence_probe': {},
    'vocab_probe': {},}

# EVALUATION LOOP ##############################################################

print('[eval] starting evaluation loop...')

__n_batches = 0
__acc = {
    'embed_mse':  0.0,
    'embed_cos':  0.0,
    'hidden_mse': 0.0,
    'hidden_cos': 0.0,
    'kl':         0.0,
    'top1':       0.0,
    'topk':       0.0,}

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
        __p_teacher_embeds  = embed(__probe_ids)
        __p_teacher_hidden  = forward(__p_teacher_embeds, __probe_mask)
        __p_teacher_logits  = logits(__p_teacher_hidden)
        with MIXED_CTX:
            __p_student_embeds = PREFIX_MOD(__probe_bytes).to(dtype=__p_teacher_embeds.dtype)
        __p_student_hidden = forward(__p_student_embeds, __probe_mask)
        __p_student_logits = logits(__p_student_hidden)

    __probe_metrics = deformers.pipelines.eval.per_token_metrics(
        teacher_embeds=__p_teacher_embeds,
        student_embeds=__p_student_embeds,
        teacher_hidden=__p_teacher_hidden,
        student_hidden=__p_student_hidden,
        teacher_logits=__p_teacher_logits,
        student_logits=__p_student_logits,
        mask=__probe_mask,
        k_num=EVAL_CFG['topk_num'])

    __k = EVAL_CFG['topk_num']
    __sentence_report = []
    for __i, __sent in enumerate(EVAL_CFG['probe_sentences']):
        # position of the last real (non-padding) token
        __pos = int(__probe_mask[__i].sum().item()) - 1
        __t_top = __p_teacher_logits[__i, __pos].topk(__k).indices.tolist()
        __s_top = __p_student_logits[__i, __pos].topk(__k).indices.tolist()
        __t_toks = TEXT_TOK.convert_ids_to_tokens(__t_top)
        __s_toks = TEXT_TOK.convert_ids_to_tokens(__s_top)
        # per-sentence summary stats
        __s_mask = __probe_mask[__i:__i + 1].cpu()
        __s_stats = {
            __key: deformers.pipelines.eval.summary_stats(
                __probe_metrics[__key][__i:__i + 1], __s_mask)
            for __key in __probe_metrics}
        print(f'[eval] sentence {__i}: "{__sent[:60]}"')
        print(f'[eval]   teacher top-{__k}: {__t_toks}')
        print(f'[eval]   student top-{__k}: {__s_toks}')
        print(f'[eval]   embed_mse={__s_stats["embed_mse"]["mean"]:.4f}'
              f'  embed_cos={__s_stats["embed_cos"]["mean"]:.4f}'
              f'  kl={__s_stats["kl"]["mean"]:.4f}')
        __sentence_report.append({
            'sentence':   __sent,
            'last_pos':   __pos,
            'teacher_topk': __t_toks,
            'student_topk': __s_toks,
            **{__key: __s_stats[__key] for __key in __s_stats}})

    REPORT['sentence_probe'] = __sentence_report

# VOCAB PROBE ##################################################################

if EVAL_CFG['vocab_probe']:
    print('\n[eval] === vocab probe ===')

    # deterministic (B, T) token ID grid cycling over the vocabulary
    __vocab_ids_list = deformers.pipelines.eval.indices_probe(
        vocab_dim=VOCAB_LEN,
        batch_dim=BATCH_CFG['batch_dim'],
        sequence_dim=BATCH_CFG['sequence_dim'])

    # build byte patches from vocab probe IDs
    __vocab_bytes_list = deformers.pipelines.eval.vocab_probe_bytes(
        vocab_ids=__vocab_ids_list,
        text_tok=TEXT_TOK,
        byte_tok=BYTE_TOK,
        patch_dim=BATCH_CFG['patch_dim'])

    # convert to tensors
    __vocab_ids  = torch.tensor(__vocab_ids_list, dtype=torch.long, device=MAIN_CFG['device_str'])
    __vocab_bytes = torch.tensor(__vocab_bytes_list, dtype=torch.long, device=MAIN_CFG['device_str'])
    __vocab_mask  = torch.ones_like(__vocab_ids)

    with torch.no_grad():
        __v_teacher_embeds  = embed(__vocab_ids)
        __v_teacher_hidden  = forward(__v_teacher_embeds, __vocab_mask)
        __v_teacher_logits  = logits(__v_teacher_hidden)
        with MIXED_CTX:
            __v_student_embeds = PREFIX_MOD(__vocab_bytes).to(dtype=__v_teacher_embeds.dtype)
        __v_student_hidden = forward(__v_student_embeds, __vocab_mask)
        __v_student_logits = logits(__v_student_hidden)

    # per-position (B, T) masked metrics
    __vocab_metrics = deformers.pipelines.eval.per_token_metrics(
        teacher_embeds=__v_teacher_embeds,
        student_embeds=__v_student_embeds,
        teacher_hidden=__v_teacher_hidden,
        student_hidden=__v_student_hidden,
        teacher_logits=__v_teacher_logits,
        student_logits=__v_student_logits,
        mask=__vocab_mask,
        k_num=EVAL_CFG['topk_num'])

    # global summary over the full vocab probe
    __vocab_mask_cpu = __vocab_mask.cpu()
    __vocab_summary = {
        __key: deformers.pipelines.eval.summary_stats(__vocab_metrics[__key], __vocab_mask_cpu)
        for __key in __vocab_metrics}

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
    __flat_ids = [__i for __row in __vocab_ids_list for __i in __row]
    __flat_strings = [TEXT_TOK.decode([__i]) for __i in __flat_ids]
    __flat_metrics = {
        __key: __vocab_metrics[__key].flatten().tolist()
        for __key in __vocab_metrics}
    __table = deformers.pipelines.eval.token_table(
        token_ids=__flat_ids,
        token_strings=__flat_strings,
        metrics=__flat_metrics)

    # hardest tokens: top-10 by embed_mse
    __sorted_hard = sorted(__table, key=lambda __r: __r['embed_mse'], reverse=True)
    __sorted_easy = sorted(__table, key=lambda __r: __r['embed_mse'])

    print(f'\n[eval] 10 hardest tokens (by embed_mse):')
    print(f'  {"id":>8}  {"token":<20}  {"bytes":>5}  {"embed_mse":>10}  {"embed_cos":>10}  {"kl":>10}')
    for __row in __sorted_hard[:10]:
        print(f'  {__row["token_id"]:>8}  {repr(__row["token_string"]):<20}  '
              f'{__row["byte_length"]:>5}  {__row["embed_mse"]:>10.6f}  '
              f'{__row["embed_cos"]:>10.6f}  {__row["kl"]:>10.6f}')

    print(f'\n[eval] 10 easiest tokens (by embed_mse):')
    print(f'  {"id":>8}  {"token":<20}  {"bytes":>5}  {"embed_mse":>10}  {"embed_cos":>10}  {"kl":>10}')
    for __row in __sorted_easy[:10]:
        print(f'  {__row["token_id"]:>8}  {repr(__row["token_string"]):<20}  '
              f'{__row["byte_length"]:>5}  {__row["embed_mse"]:>10.6f}  '
              f'{__row["embed_cos"]:>10.6f}  {__row["kl"]:>10.6f}')

    REPORT['vocab_probe'] = {
        'summary':       __vocab_summary,
        'hardest_10':    __sorted_hard[:10],
        'easiest_10':    __sorted_easy[:10],
        'full_table':    __table,}

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
