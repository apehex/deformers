"""
Benchmark / evaluation script for prefix patch experiments.

Usage (Colab):
    edit CHECKPOINT_CFG['path'] if needed, then:
    !python scripts/benchmark.py

Assumptions:
- Base model is qwen/qwen3.5-9b with hidden_size=4096.
- Tokenizer boundaries are identical to the base model.
- Trunk is frozen; prefix checkpoint must be provided.
- Byte block size follows docs/roadmap.md (patch_dim=32), configurable.
- The byte tokenizer uses pad_id=128 (as implemented by ByteTokenizer).
- Memory-safe defaults: small batch, mixed precision on CUDA.

Metrics computed (masked, padding excluded):
- embedding MSE and cosine similarity
- hidden-state MSE and cosine similarity at configured trunk depth
- KL divergence (per-token, teacher vs student logits)
- top-1 match rate
- top-k set match rate
- top-k ordered match rate

Probes:
- fixed sentence probe: teacher vs student top-k tokens for fixed sentences
- vocab probe: deterministic (B, T) token tensor with per-token inspection table

Reports:
- summary printed to stdout
- machine-readable JSON saved under LOGGING_CFG['log_dir']
- TensorBoard scalars logged under benchmark/*
"""

import contextlib
import functools
import os

import datasets
import huggingface_hub
import torch
import torch.amp
import torch.nn
import torch.nn.functional
import torch.utils.tensorboard
import transformers

import mlable.losses
import mlable.metrics
import mlable.models

import deformers.models.generic
import deformers.pipelines.eval
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
    # trunk depth for hidden-state matching; keep aligned with scripts/prefix.py MAIN_CFG['depth_num']
    'depth_num': 1,
    'batch_num': 16,
    'topk_num': 10,}

# DATA CONFIG ##################################################################

DATASET_CFG = {
    'path': 'wikimedia/wikipedia',
    'name': '20231101.en',
    'split': 'train[90%:]',  # bounded, reproducible eval split
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
    'torch_dtype': MAIN_CFG['dtype_obj'],
    'low_cpu_mem_usage': True,
    'ignore_mismatched_sizes': True,}

# CHECKPOINT CONFIG ############################################################

CHECKPOINT_CFG = {
    'path': '/content/checkpoints/prefix.pt',
    'shape': (BATCH_CFG['batch_dim'], BATCH_CFG['sequence_dim'], BATCH_CFG['patch_dim']),
    'device': MAIN_CFG['device_str'],}

# EVAL CONFIG ##################################################################

EVAL_CFG = {
    'batch_num': MAIN_CFG['batch_num'],
    'topk_num': MAIN_CFG['topk_num'],
    'probe_sentences': [
        'The quick brown fox jumps over the lazy dog.',
        'In the beginning was the Word, and the Word was with God.',],
    'vocab_probe': True,
    'inspect_topn': 20,}  # hardest / easiest tokens to display in the table

# OUTPUT CONFIG ################################################################

LOGGING_CFG = {
    'log_dir': os.path.abspath('.logs'),}

# MIXED PRECISION ##############################################################

MIXED_CTX = (
    torch.amp.autocast(device_type='cuda', dtype=MAIN_CFG['dtype_obj'])
    if MAIN_CFG['device_str'] == 'cuda'
    else contextlib.nullcontext())

# DATASET ######################################################################

print('[init] downloading the dataset...')
DATASET_OBJ = datasets.load_dataset(**DATASET_CFG)

print('[init] shuffling the dataset...')
DATASET_OBJ = DATASET_OBJ.shuffle(seed=MAIN_CFG['seed_num'])

# TOKENIZERS ###################################################################

print('[init] loading the tokenizers...')
TEXT_TOK = transformers.AutoTokenizer.from_pretrained(**TOKEN_CFG)
BYTE_TOK = deformers.tokenizers.byte.ByteTokenizer(**BYTE_CFG)

# MODELS #######################################################################

print('[init] creating output directories...')
os.makedirs(DOWNLOAD_CFG['local_dir'], exist_ok=True)
os.makedirs(LOGGING_CFG['log_dir'], exist_ok=True)

print('[init] downloading the teacher...')
huggingface_hub.snapshot_download(**DOWNLOAD_CFG)

print('[init] loading the config...')
TRUNK_CFG = transformers.AutoConfig.from_pretrained(**CONFIG_CFG)

print('[init] truncating the config...')
TRUNK_CFG = deformers.models.generic.truncate_config(
    TRUNK_CFG, layer_num=MAIN_CFG['depth_num'], target_key='text_config')

print('[init] loading the teacher...')  # load only layers up to the chosen depth
SOURCE_MOD = transformers.AutoModelForCausalLM.from_pretrained(
    config=TRUNK_CFG, **MODEL_CFG).to(device=MAIN_CFG['device_str'])

print('[init] freezing the teacher...')
SOURCE_MOD.eval()
mlable.models.freeze(SOURCE_MOD)

print('[init] freeing unused memory...')
mlable.models.free_memory()

print(f'[init] loading the prefix from {CHECKPOINT_CFG["path"]}...')
PREFIX_MOD = deformers.pipelines.eval.load_prefix_checkpoint(**CHECKPOINT_CFG)
PREFIX_MOD.eval()

# UTILITIES ####################################################################

# partial for index-based preprocessing (aligned with training pipeline)
vectorize = functools.partial(
    deformers.pipelines.prefix.tensors_from_indices,
    text_tok=TEXT_TOK,
    byte_tok=BYTE_TOK,
    dtype_obj=torch.long,
    sequence_dim=BATCH_CFG['sequence_dim'],
    patch_dim=BATCH_CFG['patch_dim'],
    device_str=MAIN_CFG['device_str'],
    left_pad=True)

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

# LOGGING ######################################################################

print(f'[init] opening TensorBoard writer at {LOGGING_CFG["log_dir"]}...')
LOG_TB = torch.utils.tensorboard.SummaryWriter(log_dir=LOGGING_CFG['log_dir'])

# ACCUMULATORS #################################################################

__n_batches = 0
__sum_embed_mse = 0.0
__sum_embed_cos = 0.0
__sum_hidden_mse = 0.0
__sum_hidden_cos = 0.0
__sum_kl = 0.0
__sum_top1 = 0.0
__sum_topk_set = 0.0
__sum_topk_ord = 0.0

# EVALUATION LOOP ##############################################################

print(f'[eval] starting evaluation over {EVAL_CFG["batch_num"]} batches...')
__dataset = DATASET_OBJ.iter(batch_size=BATCH_CFG['batch_dim'])

for __batch in __dataset:
    if __n_batches >= EVAL_CFG['batch_num']:
        break

    # mask (B, T), tokens (B, T), bytes (B, T, G)
    __mask_arr, __indices_arr, __bytes_arr = deformers.pipelines.prefix.tensors_from_strings(
        text_arr=__batch['text'],
        text_tok=TEXT_TOK,
        byte_tok=BYTE_TOK,
        sequence_dim=BATCH_CFG['sequence_dim'],
        patch_dim=BATCH_CFG['patch_dim'],
        dtype_obj=torch.long,
        device_str=MAIN_CFG['device_str'],
        left_pad=True)

    with torch.no_grad():
        # teacher forward: original embeddings -> hidden states -> logits
        __teacher_embeds = embed(__indices_arr)
        __teacher_hidden = forward(__teacher_embeds, __mask_arr)
        __teacher_logits = SOURCE_MOD.lm_head(__teacher_hidden)

        # student forward: bytes -> prefix -> inputs_embeds -> trunk -> logits
        with MIXED_CTX:
            __student_embeds = PREFIX_MOD(__bytes_arr).to(dtype=__teacher_embeds.dtype)
        __student_hidden = forward(__student_embeds, __mask_arr)
        __student_logits = SOURCE_MOD.lm_head(__student_hidden)

    # accumulate masked metrics
    __sum_embed_mse += mlable.losses.mse_loss(
        predict_arr=__student_embeds.float(),
        target_arr=__teacher_embeds.float(),
        mask_arr=__mask_arr).item()
    __sum_embed_cos += deformers.pipelines.eval.masked_cosine(
        __student_embeds.float(), __teacher_embeds.float(), __mask_arr).item()
    __sum_hidden_mse += mlable.losses.mse_loss(
        predict_arr=__student_hidden.float(),
        target_arr=__teacher_hidden.float(),
        mask_arr=__mask_arr).item()
    __sum_hidden_cos += deformers.pipelines.eval.masked_cosine(
        __student_hidden.float(), __teacher_hidden.float(), __mask_arr).item()
    __sum_kl += mlable.losses.kl_div(
        predict_arr=__student_logits.float(),
        target_arr=__teacher_logits.float(),
        mask_arr=__mask_arr).item()
    __sum_top1 += mlable.metrics.topk_rate(
        predict_arr=__student_logits,
        target_arr=__teacher_logits,
        mask_arr=__mask_arr,
        k_num=1).item()
    __sum_topk_set += deformers.pipelines.eval.topk_set_rate(
        student_arr=__student_logits,
        teacher_arr=__teacher_logits,
        mask_arr=__mask_arr,
        k_num=EVAL_CFG['topk_num']).item()
    __sum_topk_ord += mlable.metrics.topk_rate(
        predict_arr=__student_logits,
        target_arr=__teacher_logits,
        mask_arr=__mask_arr,
        k_num=EVAL_CFG['topk_num']).item()

    __n_batches += 1
    if __n_batches % 4 == 0:
        print(f'[eval] batch {__n_batches}/{EVAL_CFG["batch_num"]}')

# SUMMARY ######################################################################

__report = {}
__k_str = str(EVAL_CFG['topk_num'])

if __n_batches > 0:
    __n = __n_batches
    __report['eval'] = {
        'batches': __n,
        'embed_mse': __sum_embed_mse / __n,
        'embed_cosine': __sum_embed_cos / __n,
        'hidden_mse': __sum_hidden_mse / __n,
        'hidden_cosine': __sum_hidden_cos / __n,
        'kl_divergence': __sum_kl / __n,
        'top1_rate': __sum_top1 / __n,
        f'top{__k_str}_set': __sum_topk_set / __n,
        f'top{__k_str}_ordered': __sum_topk_ord / __n,}

    print('\n[eval] === summary metrics ===')
    print(f'[eval] batches evaluated   : {__n}')
    print(f'[eval] embed MSE           : {__report["eval"]["embed_mse"]:.6f}')
    print(f'[eval] embed cosine        : {__report["eval"]["embed_cosine"]:.4f}')
    print(f'[eval] hidden MSE          : {__report["eval"]["hidden_mse"]:.6f}')
    print(f'[eval] hidden cosine       : {__report["eval"]["hidden_cosine"]:.4f}')
    print(f'[eval] KL divergence       : {__report["eval"]["kl_divergence"]:.6f}')
    print(f'[eval] top-1 match         : {__report["eval"]["top1_rate"]:.4f}')
    print(f'[eval] top-{__k_str} set match    : {__report["eval"][f"top{__k_str}_set"]:.4f}')
    print(f'[eval] top-{__k_str} order match  : {__report["eval"][f"top{__k_str}_ordered"]:.4f}')

    # TensorBoard scalars under benchmark/*
    LOG_TB.add_scalar('benchmark/embed_mse', __report['eval']['embed_mse'], 0)
    LOG_TB.add_scalar('benchmark/embed_cosine', __report['eval']['embed_cosine'], 0)
    LOG_TB.add_scalar('benchmark/hidden_mse', __report['eval']['hidden_mse'], 0)
    LOG_TB.add_scalar('benchmark/hidden_cosine', __report['eval']['hidden_cosine'], 0)
    LOG_TB.add_scalar('benchmark/kl_divergence', __report['eval']['kl_divergence'], 0)
    LOG_TB.add_scalar('benchmark/top1_rate', __report['eval']['top1_rate'], 0)
    LOG_TB.add_scalar(f'benchmark/top{__k_str}_set', __report['eval'][f'top{__k_str}_set'], 0)
    LOG_TB.add_scalar(f'benchmark/top{__k_str}_ordered', __report['eval'][f'top{__k_str}_ordered'], 0)

# FIXED SENTENCE PROBE #########################################################

if EVAL_CFG['probe_sentences']:
    print('\n[eval] === fixed sentence probe ===')
    # mask (B, T), tokens (B, T), bytes (B, T, G)
    __probe_mask, __probe_tokens, __probe_bytes = deformers.pipelines.prefix.tensors_from_strings(
        text_arr=EVAL_CFG['probe_sentences'],
        text_tok=TEXT_TOK,
        byte_tok=BYTE_TOK,
        sequence_dim=BATCH_CFG['sequence_dim'],
        patch_dim=BATCH_CFG['patch_dim'],
        dtype_obj=torch.long,
        device_str=MAIN_CFG['device_str'],
        left_pad=True)

    with torch.no_grad():
        __p_teacher_embeds = embed(__probe_tokens)
        __p_teacher_hidden = forward(__p_teacher_embeds, __probe_mask)
        __p_teacher_logits = SOURCE_MOD.lm_head(__p_teacher_hidden)
        with MIXED_CTX:
            __p_student_embeds = PREFIX_MOD(__probe_bytes).to(dtype=__p_teacher_embeds.dtype)
        __p_student_hidden = forward(__p_student_embeds, __probe_mask)
        __p_student_logits = SOURCE_MOD.lm_head(__p_student_hidden)

    __k = EVAL_CFG['topk_num']
    __probe_report = []
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
        __probe_report.append({
            'sentence': __sent,
            'teacher_top': __t_toks,
            'student_top': __s_toks,})

    __report['sentence_probe'] = __probe_report

# VOCAB PROBE ##################################################################

if EVAL_CFG['vocab_probe']:
    print('\n[eval] === vocab probe ===')
    # resolve vocab size from model config (handles multimodal wrappers)
    __vocab_size = (
        SOURCE_MOD.config.text_config.vocab_size
        if hasattr(SOURCE_MOD.config, 'text_config')
        else SOURCE_MOD.config.vocab_size)

    # build deterministic probe batch: (B, T) indices
    __probe_indices = deformers.pipelines.eval.indices_probe(
        vocab_dim=__vocab_size,
        batch_dim=BATCH_CFG['batch_dim'],
        sequence_dim=BATCH_CFG['sequence_dim'])

    # mask (B, T), tokens (B, T), bytes (B, T, G)
    __vocab_mask, __vocab_ids, __vocab_bytes = vectorize(__probe_indices)

    with torch.no_grad():
        __v_teacher_embeds = embed(__vocab_ids)
        __v_teacher_hidden = forward(__v_teacher_embeds, __vocab_mask)
        __v_teacher_logits = SOURCE_MOD.lm_head(__v_teacher_hidden)
        with MIXED_CTX:
            __v_student_embeds = PREFIX_MOD(__vocab_bytes).to(dtype=__v_teacher_embeds.dtype)
        __v_student_hidden = forward(__v_student_embeds, __vocab_mask)
        __v_student_logits = SOURCE_MOD.lm_head(__v_student_hidden)

    __v_embed_mse = mlable.losses.mse_loss(
        predict_arr=__v_student_embeds.float(),
        target_arr=__v_teacher_embeds.float(),
        mask_arr=__vocab_mask).item()
    __v_embed_cos = deformers.pipelines.eval.masked_cosine(
        __v_student_embeds.float(), __v_teacher_embeds.float(), __vocab_mask).item()
    __v_hidden_mse = mlable.losses.mse_loss(
        predict_arr=__v_student_hidden.float(),
        target_arr=__v_teacher_hidden.float(),
        mask_arr=__vocab_mask).item()
    __v_hidden_cos = deformers.pipelines.eval.masked_cosine(
        __v_student_hidden.float(), __v_teacher_hidden.float(), __vocab_mask).item()
    __v_kl = mlable.losses.kl_div(
        predict_arr=__v_student_logits.float(),
        target_arr=__v_teacher_logits.float(),
        mask_arr=__vocab_mask).item()
    __v_top1 = mlable.metrics.topk_rate(
        predict_arr=__v_student_logits,
        target_arr=__v_teacher_logits,
        mask_arr=__vocab_mask,
        k_num=1).item()

    print(f'[eval] vocab embed MSE     : {__v_embed_mse:.6f}')
    print(f'[eval] vocab embed cosine  : {__v_embed_cos:.4f}')
    print(f'[eval] vocab hidden MSE    : {__v_hidden_mse:.6f}')
    print(f'[eval] vocab hidden cosine : {__v_hidden_cos:.4f}')
    print(f'[eval] vocab KL            : {__v_kl:.6f}')
    print(f'[eval] vocab top-1         : {__v_top1:.4f}')

    # TensorBoard scalars
    LOG_TB.add_scalar('benchmark/vocab_embed_mse', __v_embed_mse, 0)
    LOG_TB.add_scalar('benchmark/vocab_embed_cosine', __v_embed_cos, 0)
    LOG_TB.add_scalar('benchmark/vocab_hidden_mse', __v_hidden_mse, 0)
    LOG_TB.add_scalar('benchmark/vocab_hidden_cosine', __v_hidden_cos, 0)
    LOG_TB.add_scalar('benchmark/vocab_kl', __v_kl, 0)
    LOG_TB.add_scalar('benchmark/vocab_top1', __v_top1, 0)

    __report['vocab_probe'] = {
        'embed_mse': __v_embed_mse,
        'embed_cosine': __v_embed_cos,
        'hidden_mse': __v_hidden_mse,
        'hidden_cosine': __v_hidden_cos,
        'kl_divergence': __v_kl,
        'top1_rate': __v_top1,}

    # PER-TOKEN INSPECTION TABLE ###############################################

    print('\n[eval] === per-token inspection table ===')
    __token_table = deformers.pipelines.eval.per_token_metrics(
        token_ids_arr=__vocab_ids,
        student_embeds_arr=__v_student_embeds.float(),
        teacher_embeds_arr=__v_teacher_embeds.float(),
        student_hidden_arr=__v_student_hidden.float(),
        teacher_hidden_arr=__v_teacher_hidden.float(),
        student_logits_arr=__v_student_logits.float(),
        teacher_logits_arr=__v_teacher_logits.float(),
        mask_arr=__vocab_mask)

    # aggregate statistics
    __embed_mse_vals = [__r['embed_mse'] for __r in __token_table]
    __hidden_mse_vals = [__r['hidden_mse'] for __r in __token_table]
    __kl_vals = [__r['kl'] for __r in __token_table]

    __embed_stats = deformers.pipelines.eval.aggregate_metrics(__embed_mse_vals)
    __hidden_stats = deformers.pipelines.eval.aggregate_metrics(__hidden_mse_vals)
    __kl_stats = deformers.pipelines.eval.aggregate_metrics(__kl_vals)

    print(f'[eval] embed MSE   mean={__embed_stats["mean"]:.6f}  median={__embed_stats["median"]:.6f}  p95={__embed_stats["p95"]:.6f}')
    print(f'[eval] hidden MSE  mean={__hidden_stats["mean"]:.6f}  median={__hidden_stats["median"]:.6f}  p95={__hidden_stats["p95"]:.6f}')
    print(f'[eval] KL div      mean={__kl_stats["mean"]:.6f}  median={__kl_stats["median"]:.6f}  p95={__kl_stats["p95"]:.6f}')

    # hardest tokens (highest embed MSE)
    __n_top = min(EVAL_CFG['inspect_topn'], len(__token_table))
    print(f'\n[eval] hardest {__n_top} tokens by embed MSE:')
    print(f'[eval] {"id":>8}  {"token":>24}  {"embed_mse":>12}  {"hidden_mse":>12}  {"kl":>10}  {"top1":>4}')
    for __r in __token_table[:__n_top]:
        __tok_str = (TEXT_TOK.convert_ids_to_tokens([__r['token_id']]) or [''])[0] or ''
        print(f'[eval] {__r["token_id"]:>8}  {__tok_str[:24]:>24}  {__r["embed_mse"]:>12.6f}  {__r["hidden_mse"]:>12.6f}  {__r["kl"]:>10.6f}  {__r["top1_match"]:>4}')

    # easiest tokens (lowest embed MSE)
    print(f'\n[eval] easiest {__n_top} tokens by embed MSE:')
    print(f'[eval] {"id":>8}  {"token":>24}  {"embed_mse":>12}  {"hidden_mse":>12}  {"kl":>10}  {"top1":>4}')
    for __r in reversed(__token_table[-__n_top:]):
        __tok_str = (TEXT_TOK.convert_ids_to_tokens([__r['token_id']]) or [''])[0] or ''
        print(f'[eval] {__r["token_id"]:>8}  {__tok_str[:24]:>24}  {__r["embed_mse"]:>12.6f}  {__r["hidden_mse"]:>12.6f}  {__r["kl"]:>10.6f}  {__r["top1_match"]:>4}')

    __report['token_table'] = __token_table[:__n_top]  # save hardest tokens in report
    __report['token_aggregates'] = {
        'embed_mse': __embed_stats,
        'hidden_mse': __hidden_stats,
        'kl_divergence': __kl_stats,}

    # TensorBoard histograms for distribution metrics
    if __embed_mse_vals:
        LOG_TB.add_histogram('benchmark/token_embed_mse', torch.tensor(__embed_mse_vals), 0)
    if __hidden_mse_vals:
        LOG_TB.add_histogram('benchmark/token_hidden_mse', torch.tensor(__hidden_mse_vals), 0)
    if __kl_vals:
        LOG_TB.add_histogram('benchmark/token_kl', torch.tensor(__kl_vals), 0)

# REPORT SAVE ##################################################################

__report['checkpoint'] = CHECKPOINT_CFG['path']
__json_path = deformers.pipelines.eval.save_report(
    report_dict=__report,
    log_dir=LOGGING_CFG['log_dir'],
    stem='benchmark')
print(f'\n[eval] report saved to {__json_path}')

# CLEANUP ######################################################################

LOG_TB.close()

# DATAVIZ ######################################################################

# !tensorboard --logdir=.logs
