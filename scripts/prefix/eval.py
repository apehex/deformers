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

import mlable.losses
import mlable.metrics
import mlable.models

import deformers.datasets.generic
import deformers.models.generic
import deformers.models.prefix
import deformers.pipelines.eval
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
    'topk_num': MAIN_CFG['topk_num'],
    'hidden_opt': True,
    'logits_opt': True,}

PROBE_CFG = {
    'sentences': [
        'The quick brown fox jumps over the lazy dog.',
        'In the beginning was the Word, and the Word was with God.',],
    'vocab_opt': True,}

# WRAPPERS #####################################################################

class BoundedDataset:
    """Limit iteration count while preserving a stable __len__ for runner progress."""

    def __init__(self, dataset_obj: object, batch_num: int) -> None:
        self._dataset = dataset_obj
        self._batch = int(batch_num)

    def __len__(self) -> int:
        if self._batch < 1:
            return len(self._dataset)
        return min(len(self._dataset), self._batch)

    def __iter__(self) -> object:
        for __i, __row in enumerate(iter(self._dataset)):
            if (self._batch > 0) and (__i >= self._batch):
                break
            yield __row

# UTILS ########################################################################

def summarize_metrics(state: dict) -> None:
    """Print averaged evaluation metrics from `state['scalars']`."""
    __scalars = state['scalars']
    __count = int(__scalars.get('metric/count', 0))
    if __count < 1:
        print('[eval] no evaluation batches were processed.')
        return
    print('\n[eval] === summary metrics ===')
    print(f'[eval] batches evaluated : {__count}')
    print(f'[eval] embed MSE         : {__scalars["metrics/mse/0"] / __count:.6f}')
    print(f'[eval] hidden MSE        : {__scalars["metrics/mse/k"] / __count:.6f}')
    print(f'[eval] KL divergence     : {__scalars["metrics/kld/k"] / __count:.6f}')
    print(f'[eval] top-k match       : {__scalars["metrics/topk/k"] / __count:.4f} (k={TESTING_CFG["topk_num"]})')

def forward_probe(
    runner_obj: deformers.pipelines.prefix.trainer.PrefixTester,
    batch_arr: dict,
    column_str: str,
) -> dict:
    """Run one probe batch and return detached tensors, clearing runner tensor state afterwards."""
    runner_obj.step_batch(batch_arr=batch_arr, column_str=column_str)
    runner_obj.step_forward()
    __outputs = {
        __k: (__v.detach().clone() if torch.is_tensor(__v) else __v)
        for (__k, __v) in runner_obj._state['tensors'].items()}
    runner_obj._state['tensors'] = {}
    return __outputs

# DATASET ######################################################################

print('[init] downloading the dataset...')
DATASET_OBJ = datasets.load_dataset(**DATASET_CFG)

print('[init] preprocessing the dataset...')
DATASET_OBJ = DATASET_OBJ.shuffle(seed=MAIN_CFG['seed_num']).select_columns(['text'])
DATASET_OBJ = deformers.datasets.generic.BatchedDataset(
    dataset_obj=DATASET_OBJ,
    batch_dim=BATCH_CFG['batch_dim'])
DATASET_OBJ = BoundedDataset(
    dataset_obj=DATASET_OBJ,
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

print('[eval] running phase...')
TESTER.run_phase()

print('[eval] cleaning up callbacks...')
TESTER.close_callbacks()

summarize_metrics(TESTER._state)

# FIXED SENTENCE PROBE #########################################################

if PROBE_CFG['sentences']:
    print('\n[eval] === fixed sentence probe ===')
    __probe = forward_probe(
        runner_obj=TESTER,
        batch_arr={'text': PROBE_CFG['sentences']},
        column_str='text')
    __k = int(TESTING_CFG['topk_num'])
    __mask = __probe['inputs/mask']
    __teacher_logits = __probe['outputs/teacher/logits']
    __student_logits = __probe['outputs/student/logits']
    for __i, __sentence in enumerate(PROBE_CFG['sentences']):
        __len = int(__mask[__i].sum().item())
        if __len < 1:
            print(f'[eval] sentence {__i}: "{__sentence[:60]}"')
            print('[eval] skipped: empty tokenized sentence.')
            continue
        __pos = __len - 1
        __teacher_top = __teacher_logits[__i, __pos].topk(__k).indices.tolist()
        __student_top = __student_logits[__i, __pos].topk(__k).indices.tolist()
        print(f'[eval] sentence {__i}: "{__sentence[:60]}"')
        print(f'[eval] teacher top-{__k}: {TEXT_TOK.convert_ids_to_tokens(__teacher_top)}')
        print(f'[eval] student top-{__k}: {TEXT_TOK.convert_ids_to_tokens(__student_top)}')

# VOCAB PROBE ##################################################################

if PROBE_CFG['vocab_opt']:
    print('\n[eval] === vocab probe ===')
    __vocab_dim = (
        SOURCE_MOD.config.text_config.vocab_size
        if hasattr(SOURCE_MOD.config, 'text_config')
        else SOURCE_MOD.config.vocab_size)
    __indices = deformers.pipelines.eval.indices_probe(
        vocab_dim=__vocab_dim,
        batch_dim=BATCH_CFG['batch_dim'],
        sequence_dim=BATCH_CFG['sequence_dim'])
    __probe = forward_probe(
        runner_obj=TESTER,
        batch_arr={'indices': __indices},
        column_str='indices')
    __embed_mse = torch.nn.functional.mse_loss(
        __probe['outputs/teacher/0'].float(),
        __probe['outputs/student/0'].float())
    __hidden_mse = torch.nn.functional.mse_loss(
        __probe['outputs/teacher/k'].float(),
        __probe['outputs/student/k'].float())
    __kl = mlable.losses.kl_div(
        predict_arr=__probe['outputs/student/logits'].float(),
        target_arr=__probe['outputs/teacher/logits'].float(),
        reduce_opt=True)
    __topk = mlable.metrics.topk_rate(
        predict_arr=__probe['outputs/student/logits'],
        target_arr=__probe['outputs/teacher/logits'],
        reduce_opt=True,
        k_num=TESTING_CFG['topk_num'])
    print(f'[eval] vocab embed MSE   : {float(__embed_mse.item()):.6f}')
    print(f'[eval] vocab hidden MSE  : {float(__hidden_mse.item()):.6f}')
    print(f'[eval] vocab KL          : {float(__kl.item()):.6f}')
    print(f'[eval] vocab top-k       : {float(__topk.item()):.4f} (k={TESTING_CFG["topk_num"]})')
