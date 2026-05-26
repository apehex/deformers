"""
Evaluation script for prefix patch experiments.

Runs evaluation through the prefix runner lifecycle:
    tester.setup_global(...)
    tester.setup_phase(...)
    tester.run_phase()
    tester.close_callbacks()
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
import deformers.pipelines.prefix.trainer
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

REPOSITORY_CFG = {
    'repo_path': '',}

CHECKPOINT_CFG = {
    'path': os.path.abspath('checkpoints/prefix.pt'),
    'device': MAIN_CFG['device_str'],}

# RUNNER CONFIG ################################################################

CONTEXT_CFG = {
    'dtype': MAIN_CFG['dtype_obj'],
    'device': MAIN_CFG['device_str'],}

PHASE_CFG = {
    'epoch_num': 1,
    'column_str': 'text',}

LOSS_CFG = {
    'mse_0_rate': 1.0,
    'mse_k_rate': 1.0,
    'cos_0_rate': 0.0,
    'cos_k_rate': 0.0,
    'relative_opt': True,}

TESTING_CFG = {
    'batch_num': MAIN_CFG['batch_num'],
    'topk_num': MAIN_CFG['topk_num'],}

# UTILS ########################################################################

def summarize_metrics(state: dict) -> None:
    """Print averaged evaluation metrics from `state['scalars']`."""
    __scalars = state['scalars']
    __count = int(state.get('count', 0))
    if __count < 1:
        print('[eval] no evaluation batches were processed.')
        return
    print('\n[eval] === summary metrics ===')
    print(f'[eval] batches evaluated : {__count}')
    print(f'[eval] embed MSE         : {__scalars["loss/mse/0"] / __count:.6f}')
    print(f'[eval] hidden MSE        : {__scalars["loss/mse/k"] / __count:.6f}')
    print(f'[eval] KL divergence     : {__scalars["metric/kld/k"] / __count:.6f}')
    print(f'[eval] top-k match       : {__scalars["metric/topk/k"] / __count:.4f} (k={TESTING_CFG["topk_num"]})')

def prepare_summary_callback(summary_obj: dict) -> dict:
    """Aggregate per-step evaluation scalars into a local summary dict."""
    def __operation(state: dict) -> None:
        summary_obj['count'] += 1
        summary_obj['scalars']['loss/mse/0'] += state['scalars']['loss/mse/0']
        summary_obj['scalars']['loss/mse/k'] += state['scalars']['loss/mse/k']
        summary_obj['scalars']['metric/kld/k'] += state['scalars']['metric/kld/k']
        summary_obj['scalars']['metric/topk/k'] += state['scalars']['metric/topk/k']
    return {
        'name': 'summary',
        'trigger': lambda state: True,
        'operation': __operation,
        'cleanup': lambda: None,}

# DATASET ######################################################################

print('[init] downloading the dataset...')
DATASET_OBJ = datasets.load_dataset(**DATASET_CFG)

print('[init] preprocessing the dataset...')
DATASET_OBJ = DATASET_OBJ.shuffle(seed=MAIN_CFG['seed_num']).select_columns(['text'])
DATASET_OBJ = deformers.datasets.generic.BatchedDataset(
    dataset_obj=DATASET_OBJ,
    batch_dim=BATCH_CFG['batch_dim'],
    batch_num=TESTING_CFG['batch_num'])

# TOKENIZERS ###################################################################

print('[init] loading the tokenizers...')
TEXT_TOK = transformers.AutoTokenizer.from_pretrained(**TOKEN_CFG)
TEXT_TOK.pad_token = TEXT_TOK.pad_token or TEXT_TOK.eos_token
BYTE_TOK = deformers.tokenizers.byte.ByteTokenizer(**BYTE_CFG)

# MODELS #######################################################################

print('[init] creating the output directories...')
os.makedirs(DOWNLOAD_CFG['local_dir'], exist_ok=True)

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

if REPOSITORY_CFG['repo_path']:
    print('[init] downloading the prefix checkpoint...')
    huggingface_hub.hf_hub_download(
        repo_id=REPOSITORY_CFG['repo_path'],
        filename=os.path.basename(CHECKPOINT_CFG['path']),
        local_dir=os.path.dirname(CHECKPOINT_CFG['path']),
        repo_type='model')

print('[init] loading the prefix weights...')
PREFIX_MOD = deformers.models.prefix.CompositeBytePrefix.load_checkpoint(
    shape=(BATCH_CFG['batch_dim'], BATCH_CFG['sequence_dim'], BATCH_CFG['patch_dim']),
    **CHECKPOINT_CFG)
PREFIX_MOD.eval()

# TESTER #######################################################################

print('[init] building tester...')
TESTER = deformers.pipelines.prefix.trainer.PrefixTester(
    text_tok=TEXT_TOK,
    byte_tok=BYTE_TOK,
    teacher_mod=SOURCE_MOD,
    student_mod=PREFIX_MOD,)

print('[init] setting up long-lived utilities...')
TESTER.setup_global(context_cfg=CONTEXT_CFG)

print('[eval] configuring phase...')
TESTER.setup_phase(
    dataset_obj=DATASET_OBJ,
    phase_cfg=PHASE_CFG,
    batch_cfg=BATCH_CFG,
    loss_cfg=LOSS_CFG,
    testing_cfg=TESTING_CFG,)

SUMMARY_STATE = {
    'count': 0,
    'scalars': {
        'loss/mse/0': 0.0,
        'loss/mse/k': 0.0,
        'metric/kld/k': 0.0,
        'metric/topk/k': 0.0,},}
TESTER._callbacks.append(prepare_summary_callback(summary_obj=SUMMARY_STATE))

print('[eval] running phase...')
TESTER.run_phase()

print('[eval] cleaning up callbacks...')
TESTER.close_callbacks()

summarize_metrics(SUMMARY_STATE)
