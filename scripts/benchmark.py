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
    'streaming': True,}

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
DATASET_OBJ = DATASET_OBJ.shuffle(seed=MAIN_CFG['seed_num'], buffer_size=1000)

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

# LOGGING ######################################################################

print(f'[init] opening TensorBoard writer at {LOGGING_CFG["log_dir"]}...')
LOG_TB = torch.utils.tensorboard.SummaryWriter(log_dir=LOGGING_CFG['log_dir'])

# STATE ########################################################################

__k_str = str(EVAL_CFG['topk_num'])
__step = -1
__state = {
    'embed/mse': 0.0,
    'embed/cosine': 0.0,
    'hidden/mse': 0.0,
    'hidden/cosine': 0.0,
    'logit/kl': 0.0,
    'logit/top1': 0.0,
    f'logit/top{__k_str}/set': 0.0,
    f'logit/top{__k_str}/ordered': 0.0,}

# EVALUATION LOOP ##############################################################

print(f'[eval] starting evaluation over {EVAL_CFG["batch_num"]} batches...')

for __step, __batch in enumerate(
        DATASET_OBJ
        .take(EVAL_CFG['batch_num'] * BATCH_CFG['batch_dim'])
        .iter(batch_size=BATCH_CFG['batch_dim'])):

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
        # teacher: one call gives embeddings, hidden states, and logits
        __teacher_out = SOURCE_MOD(
            input_ids=__indices_arr,
            attention_mask=__mask_arr,
            output_hidden_states=True,
            use_cache=False)
        # student: prefix produces embeddings; trunk produces hidden states and logits
        with MIXED_CTX:
            __student_embeds = PREFIX_MOD(__bytes_arr)
        __student_out = SOURCE_MOD(
            inputs_embeds=__student_embeds.to(dtype=__teacher_out.hidden_states[0].dtype),
            attention_mask=__mask_arr,
            output_hidden_states=True,
            use_cache=False)

    # accumulate masked metrics
    __state['embed/mse'] += mlable.losses.mse_loss(
        predict_arr=__student_embeds.float(),
        target_arr=__teacher_out.hidden_states[0].float(),
        mask_arr=__mask_arr).item()
    __state['embed/cosine'] += deformers.pipelines.eval.masked_cosine(
        predict_arr=__student_embeds.float(),
        target_arr=__teacher_out.hidden_states[0].float(),
        mask_arr=__mask_arr).item()
    __state['hidden/mse'] += mlable.losses.mse_loss(
        predict_arr=__student_out.hidden_states[-1].float(),
        target_arr=__teacher_out.hidden_states[-1].float(),
        mask_arr=__mask_arr).item()
    __state['hidden/cosine'] += deformers.pipelines.eval.masked_cosine(
        predict_arr=__student_out.hidden_states[-1].float(),
        target_arr=__teacher_out.hidden_states[-1].float(),
        mask_arr=__mask_arr).item()
    __state['logit/kl'] += mlable.losses.kl_div(
        predict_arr=__student_out.logits.float(),
        target_arr=__teacher_out.logits.float(),
        mask_arr=__mask_arr).item()
    __state['logit/top1'] += mlable.metrics.topk_rate(
        predict_arr=__student_out.logits,
        target_arr=__teacher_out.logits,
        mask_arr=__mask_arr,
        k_num=1).item()
    __state[f'logit/top{__k_str}/set'] += deformers.pipelines.eval.topk_set_rate(
        student_arr=__student_out.logits,
        teacher_arr=__teacher_out.logits,
        mask_arr=__mask_arr,
        k_num=EVAL_CFG['topk_num']).item()
    __state[f'logit/top{__k_str}/ordered'] += mlable.metrics.topk_rate(
        predict_arr=__student_out.logits,
        target_arr=__teacher_out.logits,
        mask_arr=__mask_arr,
        k_num=EVAL_CFG['topk_num']).item()

    if (__step + 1) % 4 == 0:
        print(f'[eval] batch {__step + 1}/{EVAL_CFG["batch_num"]}')

# SUMMARY ######################################################################

__n = max(1, __step + 1)
__report = {
    'batches': __n,
    'embed/mse': __state['embed/mse'] / __n,
    'embed/cosine': __state['embed/cosine'] / __n,
    'hidden/mse': __state['hidden/mse'] / __n,
    'hidden/cosine': __state['hidden/cosine'] / __n,
    'logit/kl': __state['logit/kl'] / __n,
    'logit/top1': __state['logit/top1'] / __n,
    f'logit/top{__k_str}/set': __state[f'logit/top{__k_str}/set'] / __n,
    f'logit/top{__k_str}/ordered': __state[f'logit/top{__k_str}/ordered'] / __n,}

print('\n[eval] === summary metrics ===')
print(f'[eval] batches evaluated   : {__n}')
print(f'[eval] embed MSE           : {__report["embed/mse"]:.6f}')
print(f'[eval] embed cosine        : {__report["embed/cosine"]:.4f}')
print(f'[eval] hidden MSE          : {__report["hidden/mse"]:.6f}')
print(f'[eval] hidden cosine       : {__report["hidden/cosine"]:.4f}')
print(f'[eval] KL divergence       : {__report["logit/kl"]:.6f}')
print(f'[eval] top-1 match         : {__report["logit/top1"]:.4f}')
print(f'[eval] top-{__k_str} set match    : {__report[f"logit/top{__k_str}/set"]:.4f}')
print(f'[eval] top-{__k_str} order match  : {__report[f"logit/top{__k_str}/ordered"]:.4f}')

# TensorBoard scalars under benchmark/*
LOG_TB.add_scalar('benchmark/embed/mse', __report['embed/mse'], 0)
LOG_TB.add_scalar('benchmark/embed/cosine', __report['embed/cosine'], 0)
LOG_TB.add_scalar('benchmark/hidden/mse', __report['hidden/mse'], 0)
LOG_TB.add_scalar('benchmark/hidden/cosine', __report['hidden/cosine'], 0)
LOG_TB.add_scalar('benchmark/logit/kl', __report['logit/kl'], 0)
LOG_TB.add_scalar('benchmark/logit/top1', __report['logit/top1'], 0)
LOG_TB.add_scalar(f'benchmark/logit/top{__k_str}/set', __report[f'logit/top{__k_str}/set'], 0)
LOG_TB.add_scalar(f'benchmark/logit/top{__k_str}/ordered', __report[f'logit/top{__k_str}/ordered'], 0)

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
        __p_teacher_out = SOURCE_MOD(
            input_ids=__probe_tokens,
            attention_mask=__probe_mask,
            output_hidden_states=True,
            use_cache=False)
        with MIXED_CTX:
            __p_student_embeds = PREFIX_MOD(__probe_bytes)
        __p_student_out = SOURCE_MOD(
            inputs_embeds=__p_student_embeds.to(dtype=__p_teacher_out.hidden_states[0].dtype),
            attention_mask=__probe_mask,
            output_hidden_states=True,
            use_cache=False)

    __k = EVAL_CFG['topk_num']
    __probe_report = []
    for __i, __sent in enumerate(EVAL_CFG['probe_sentences']):
        # report at the last real token position
        __pos = int(__probe_mask[__i].sum().item()) - 1
        __t_top = __p_teacher_out.logits[__i, __pos].topk(__k).indices.tolist()
        __s_top = __p_student_out.logits[__i, __pos].topk(__k).indices.tolist()
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
    __vocab_mask, __vocab_ids, __vocab_bytes = deformers.pipelines.prefix.tensors_from_indices(
        indices_arr=__probe_indices,
        text_tok=TEXT_TOK,
        byte_tok=BYTE_TOK,
        dtype_obj=torch.long,
        sequence_dim=BATCH_CFG['sequence_dim'],
        patch_dim=BATCH_CFG['patch_dim'],
        device_str=MAIN_CFG['device_str'],
        left_pad=True)

    with torch.no_grad():
        __v_teacher_out = SOURCE_MOD(
            input_ids=__vocab_ids,
            attention_mask=__vocab_mask,
            output_hidden_states=True,
            use_cache=False)
        with MIXED_CTX:
            __v_student_embeds = PREFIX_MOD(__vocab_bytes)
        __v_student_out = SOURCE_MOD(
            inputs_embeds=__v_student_embeds.to(dtype=__v_teacher_out.hidden_states[0].dtype),
            attention_mask=__vocab_mask,
            output_hidden_states=True,
            use_cache=False)

    __v_embed_mse = mlable.losses.mse_loss(
        predict_arr=__v_student_embeds.float(),
        target_arr=__v_teacher_out.hidden_states[0].float(),
        mask_arr=__vocab_mask).item()
    __v_embed_cos = deformers.pipelines.eval.masked_cosine(
        predict_arr=__v_student_embeds.float(),
        target_arr=__v_teacher_out.hidden_states[0].float(),
        mask_arr=__vocab_mask).item()
    __v_hidden_mse = mlable.losses.mse_loss(
        predict_arr=__v_student_out.hidden_states[-1].float(),
        target_arr=__v_teacher_out.hidden_states[-1].float(),
        mask_arr=__vocab_mask).item()
    __v_hidden_cos = deformers.pipelines.eval.masked_cosine(
        predict_arr=__v_student_out.hidden_states[-1].float(),
        target_arr=__v_teacher_out.hidden_states[-1].float(),
        mask_arr=__vocab_mask).item()
    __v_kl = mlable.losses.kl_div(
        predict_arr=__v_student_out.logits.float(),
        target_arr=__v_teacher_out.logits.float(),
        mask_arr=__vocab_mask).item()
    __v_top1 = mlable.metrics.topk_rate(
        predict_arr=__v_student_out.logits,
        target_arr=__v_teacher_out.logits,
        mask_arr=__vocab_mask,
        k_num=1).item()

    print(f'[eval] vocab embed MSE     : {__v_embed_mse:.6f}')
    print(f'[eval] vocab embed cosine  : {__v_embed_cos:.4f}')
    print(f'[eval] vocab hidden MSE    : {__v_hidden_mse:.6f}')
    print(f'[eval] vocab hidden cosine : {__v_hidden_cos:.4f}')
    print(f'[eval] vocab KL            : {__v_kl:.6f}')
    print(f'[eval] vocab top-1         : {__v_top1:.4f}')

    # TensorBoard scalars
    LOG_TB.add_scalar('benchmark/vocab/embed/mse', __v_embed_mse, 0)
    LOG_TB.add_scalar('benchmark/vocab/embed/cosine', __v_embed_cos, 0)
    LOG_TB.add_scalar('benchmark/vocab/hidden/mse', __v_hidden_mse, 0)
    LOG_TB.add_scalar('benchmark/vocab/hidden/cosine', __v_hidden_cos, 0)
    LOG_TB.add_scalar('benchmark/vocab/logit/kl', __v_kl, 0)
    LOG_TB.add_scalar('benchmark/vocab/logit/top1', __v_top1, 0)

    __report['vocab_probe'] = {
        'embed/mse': __v_embed_mse,
        'embed/cosine': __v_embed_cos,
        'hidden/mse': __v_hidden_mse,
        'hidden/cosine': __v_hidden_cos,
        'logit/kl': __v_kl,
        'logit/top1': __v_top1,}

    # PER-TOKEN INSPECTION TABLE ###############################################

    print('\n[eval] === per-token inspection table ===')
    __token_table = deformers.pipelines.eval.per_token_metrics(
        token_ids_arr=__vocab_ids,
        student_embeds_arr=__v_student_embeds.float(),
        teacher_embeds_arr=__v_teacher_out.hidden_states[0].float(),
        student_hidden_arr=__v_student_out.hidden_states[-1].float(),
        teacher_hidden_arr=__v_teacher_out.hidden_states[-1].float(),
        student_logits_arr=__v_student_out.logits.float(),
        teacher_logits_arr=__v_teacher_out.logits.float(),
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
