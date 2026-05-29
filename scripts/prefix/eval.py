"""
Evaluation script for prefix patch experiments.

Loads a teacher model and a trained byte prefix, then evaluates the prefix with
`deformers.pipelines.prefix.runner.PrefixTester`.

Assumptions:
- Base model is qwen/qwen3.5-9b with hidden_size=4096.
- Tokenizer boundaries are identical to the base model.
- Trunk is frozen; prefix checkpoint is required (local path or HF repo).
- Byte block size default follows docs/roadmap.md (patch_dim=32), configurable.
- The byte tokenizer uses pad_id=128 (as implemented by ByteTokenizer).
"""

import os

import datasets
import huggingface_hub
import torch
import transformers

import mlable.models

import deformers.datasets.generic
import deformers.models.generic
import deformers.models.prefix
import deformers.pipelines.eval
import deformers.pipelines.prefix.runner
import deformers.tokenizers.byte

# COMMON CONFIG ################################################################

MAIN_CFG = {
    'teacher_str': 'nvidia/llama-3.1-nemotron-nano-8b-v1', # 'qwen/qwen3.5-9b',
    'student_str': '/content/drive/MyDrive/models/prefix.128x4.8192.pt',
    'device_str': 'cuda' if torch.cuda.is_available() else 'cpu',
    'dtype_obj': torch.bfloat16,
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
    'split': 'train[90%:]',
    'streaming': False,}

BATCH_CFG = {
    'batch_dim': MAIN_CFG['batch_dim'],
    'sequence_dim': MAIN_CFG['sequence_dim'],
    'patch_dim': MAIN_CFG['patch_dim'],
    'device_str': MAIN_CFG['device_str'],
    'dtype_obj': torch.long,
    'padding_str': '',
    'left_pad': True,}

# TOKENIZER CONFIG #############################################################

TOKEN_CFG = {
    'pretrained_model_name_or_path': MAIN_CFG['teacher_str'],
    'use_fast': True,}

BYTE_CFG = {
    'encoding': MAIN_CFG['encoding_str'],}

# MODEL CONFIG #################################################################

DOWNLOAD_CFG = {
    'repo_id': MAIN_CFG['teacher_str'],
    'repo_type': 'model',
    'local_dir': os.path.abspath('downloads'),
    'ignore_patterns': ['*.onnx', '*.tflite', '*.msgpack'],}

CONFIG_CFG = {
    'pretrained_model_name_or_path': DOWNLOAD_CFG['local_dir'],
    'trust_remote_code': False,}

MODEL_CFG = {
    'pretrained_model_name_or_path': DOWNLOAD_CFG['local_dir'],
    'trust_remote_code': CONFIG_CFG['trust_remote_code'],
    'dtype': MAIN_CFG['dtype_obj'],
    'low_cpu_mem_usage': True,
    'ignore_mismatched_sizes': True,}

# CHECKPOINT CONFIG ############################################################

CHECKPOINT_CFG = {
    'path': MAIN_CFG['student_str'],
    'shape': (
        MAIN_CFG['batch_dim'],
        MAIN_CFG['sequence_dim'],
        MAIN_CFG['patch_dim']),
    'device': MAIN_CFG['device_str'],}

# EVAL CONFIG ##################################################################

CONTEXT_CFG = {
    'dtype': MAIN_CFG['dtype_obj'],
    'device': MAIN_CFG['device_str'],}

PHASE_CFG = {
    'epoch_num': 1,
    'column_str': 'text',}

LOSS_CFG = {
    'mse_0_rate': 1.0,
    'mse_k_rate': 1.0,
    'cos_0_rate': 1.0,
    'cos_k_rate': 1.0,
    'relative_opt': True,}

GRADIENT_CFG = {
    'every_num': 1,
    'max_norm': 1.0,}

TESTING_CFG = {
    'every_num': 1,
    'topk_num': MAIN_CFG['topk_num'],}

EVAL_CFG = {
    'batch_num': MAIN_CFG['batch_num'],
    'probe_sentences': [
        'The quick brown fox jumps over the lazy dog.',
        'In the beginning was the Word, and the Word was with God.',],
    'vocab_probe': True,}

# TOKENIZERS ###################################################################

print('[init] loading the tokenizers...')
TEXT_TOK = transformers.AutoTokenizer.from_pretrained(**TOKEN_CFG)
BYTE_TOK = deformers.tokenizers.byte.ByteTokenizer(**BYTE_CFG)

print('[init] defining a padding token...')
TEXT_TOK.pad_token = TEXT_TOK.eos_token if not bool(TEXT_TOK.pad_token) else TEXT_TOK.pad_token

# DATASET ######################################################################

print('[init] downloading the dataset...')
DATASET_OBJ = datasets.load_dataset(**DATASET_CFG)

print('[init] preprocessing the dataset...')
DATASET_OBJ = DATASET_OBJ.shuffle(seed=MAIN_CFG['seed_num'])
DATASET_OBJ = deformers.datasets.generic.BatchedDataset(
    dataset_obj=DATASET_OBJ.select_columns(['text']),
    batch_dim=BATCH_CFG['batch_dim'],
    batch_num=EVAL_CFG['batch_num'])

# MODELS #######################################################################

print('[init] creating the output directories...')
os.makedirs(DOWNLOAD_CFG['local_dir'], exist_ok=True)

print('[init] downloading the teacher...')
huggingface_hub.snapshot_download(**DOWNLOAD_CFG)

print('[init] loading the config...')
TRUNK_CFG = transformers.AutoConfig.from_pretrained(**CONFIG_CFG)

print('[init] truncating the config...')
TRUNK_CFG = deformers.models.generic.truncate_config(
    TRUNK_CFG,
    layer_num=MAIN_CFG['depth_num'],
    target_key='text_config')

print('[init] loading the teacher...')
SOURCE_MOD = transformers.AutoModelForCausalLM.from_pretrained(
    config=TRUNK_CFG,
    **MODEL_CFG).to(device=MAIN_CFG['device_str'])

print('[init] freezing the teacher...')
SOURCE_MOD.eval()
mlable.models.freeze(SOURCE_MOD)

print('[init] freeing unused memory...')
mlable.models.free_memory()

print('[init] loading the prefix weights...')
PREFIX_MOD = deformers.models.prefix.CompositeBytePrefix.load_checkpoint(**CHECKPOINT_CFG)
PREFIX_MOD.eval()

print('[init] building the prefix...')
PREFIX_MOD.build(
    shape=CHECKPOINT_CFG['shape'],
    device=MAIN_CFG['device_str'],
    dtype=torch.float32)

# TESTER #######################################################################

print('[init] building tester...')
TESTER = deformers.pipelines.prefix.runner.PrefixTester(
    text_tok=TEXT_TOK,
    byte_tok=BYTE_TOK,
    teacher_mod=SOURCE_MOD,
    student_mod=PREFIX_MOD,)

print('[init] setting up evaluation context...')
TESTER.setup_global(context_cfg=CONTEXT_CFG)

# DATASET PROBE ################################################################

print('[eval] configuring evaluation phase...')
TESTER.setup_phase(
    dataset_obj=DATASET_OBJ,
    phase_cfg=PHASE_CFG,
    batch_cfg=BATCH_CFG,
    loss_cfg=LOSS_CFG,
    gradient_cfg=GRADIENT_CFG,
    testing_cfg=TESTING_CFG,)

print('[eval] starting evaluation...')
ACCUMULATOR = deformers.pipelines.eval.prepare_scalar_accumulator()
TESTER.add_callback(ACCUMULATOR)
TESTER.run_phase()
SUMMARY = deformers.pipelines.eval.scalar_means(ACCUMULATOR)

print('\n[eval] === summary metrics ===')
print(f'[eval] batches evaluated : {len(ACCUMULATOR["values"]["loss/total"])}')
print(f'[eval] total loss        : {SUMMARY["loss/total"]:.6f}')
print(f'[eval] embed MSE         : {SUMMARY["loss/mse/0"]:.6f}')
print(f'[eval] hidden MSE        : {SUMMARY["loss/mse/k"]:.6f}')
print(f'[eval] embed cosine      : {SUMMARY["loss/cos/0"]:.6f}')
print(f'[eval] hidden cosine     : {SUMMARY["loss/cos/k"]:.6f}')
print(f'[eval] KL divergence     : {SUMMARY["test/kld/k"]:.6f}')
print(f'[eval] top-k match       : {SUMMARY["test/topk/k"]:.4f} (k={TESTING_CFG["topk_num"]})')

# FIXED SENTENCE PROBE #########################################################

if EVAL_CFG['probe_sentences']:
    print('\n[eval] === fixed sentence probe ===')
    PROBE_STATE = deformers.pipelines.eval.run_probe(
        runner_obj=TESTER,
        batch_arr=deformers.pipelines.eval.text_probe(EVAL_CFG['probe_sentences']),
        column_str='text')
    PROBE_TOPK = deformers.pipelines.eval.topk_tokens(
        state=PROBE_STATE,
        model_obj=SOURCE_MOD,
        tokenizer_obj=TEXT_TOK,
        k_num=TESTING_CFG['topk_num'])
    for __idx, (__sent, __row) in enumerate(zip(EVAL_CFG['probe_sentences'], PROBE_TOPK)):
        print(f'[eval] sentence {__idx}: "{__sent[:60]}"')
        print(f'[eval] teacher top-{TESTING_CFG["topk_num"]}: {__row["teacher_tokens"]}')
        print(f'[eval] student top-{TESTING_CFG["topk_num"]}: {__row["student_tokens"]}')

# VOCAB PROBE ##################################################################

if EVAL_CFG['vocab_probe']:
    print('\n[eval] === vocab probe ===')
    VOCAB_SIZE = (
        SOURCE_MOD.config.text_config.vocab_size
        if hasattr(SOURCE_MOD.config, 'text_config')
        else SOURCE_MOD.config.vocab_size)
    PROBE_STATE = deformers.pipelines.eval.run_probe(
        runner_obj=TESTER,
        batch_arr=deformers.pipelines.eval.vocab_probe(
            vocab_dim=VOCAB_SIZE,
            batch_dim=BATCH_CFG['batch_dim'],
            sequence_dim=BATCH_CFG['sequence_dim']),
        column_str='indices')
    PROBE_SCALARS = PROBE_STATE['scalars']
    print(f'[eval] vocab total loss   : {PROBE_SCALARS["loss/total"]:.6f}')
    print(f'[eval] vocab embed MSE    : {PROBE_SCALARS["loss/mse/0"]:.6f}')
    print(f'[eval] vocab hidden MSE   : {PROBE_SCALARS["loss/mse/k"]:.6f}')
    print(f'[eval] vocab embed cosine : {PROBE_SCALARS["loss/cos/0"]:.6f}')
    print(f'[eval] vocab hidden cosine: {PROBE_SCALARS["loss/cos/k"]:.6f}')
    print(f'[eval] vocab KL           : {PROBE_SCALARS["test/kld/k"]:.6f}')
    print(f'[eval] vocab top-k        : {PROBE_SCALARS["test/topk/k"]:.4f}')

# CLEANUP ######################################################################

TESTER.close_callbacks()
mlable.models.free_memory()
