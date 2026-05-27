import torch

import deformers.pipelines.prefix.callbacks as _callbacks

# PROBE ########################################################################

def indices_probe(
    vocab_dim: int,
    batch_dim: int,
    sequence_dim: int
) -> list[list[int]]:
    # first indices / tokens of the vocabulary
    __ids = torch.arange(batch_dim * sequence_dim, dtype=torch.long) % vocab_dim
    # (B, T) integers
    return __ids.reshape(batch_dim, sequence_dim).tolist()

def text_probe(texts_arr: list[str]) -> dict:
    return {'text': list(texts_arr)}

def vocab_probe(
    vocab_dim: int,
    batch_dim: int,
    sequence_dim: int,
) -> dict:
    return {
        'indices': indices_probe(
            vocab_dim=vocab_dim,
            batch_dim=batch_dim,
            sequence_dim=sequence_dim),}

# STATE ########################################################################

def clone_state(state: dict) -> dict:
    return {
        'tensors': {
            __key: (
                __value.detach().clone()
                if isinstance(__value, torch.Tensor)
                else __value)
            for (__key, __value) in state.get('tensors', {}).items()},
        'scalars': dict(state.get('scalars', {})),}

# SCALARS ######################################################################

DEFAULT_SCALAR_KEYS = (
    'loss/total',
    'loss/mse/0',
    'loss/mse/k',
    'loss/cos/0',
    'loss/cos/k',
    'test/kld/k',
    'test/topk/k',)

def prepare_scalar_accumulator(keys_arr: tuple[str]=DEFAULT_SCALAR_KEYS) -> dict:
    __values = {__key: [] for __key in keys_arr}

    def __trigger(state: dict) -> bool:
        return True

    def __operation(state: dict) -> None:
        __scalars = state['scalars']
        for __key in keys_arr:
            __values[__key].append(float(__scalars[__key]))

    return {
        'name': 'scalar_accumulator',
        'trigger': __trigger,
        'operation': __operation,
        'cleanup': _callbacks.noop,
        'values': __values,}

def scalar_means(callback: dict) -> dict:
    return {
        __key: (
            sum(__values) / len(__values)
            if len(__values) > 0
            else 0.0)
        for (__key, __values) in callback.get('values', {}).items()}

# RUNNER #######################################################################

def run_probe(
    runner_obj: object,
    batch_arr: dict,
    column_str: str,
    step_num: int=1,
    callbacks_opt: bool=False,
) -> dict:
    __callbacks = runner_obj._callbacks
    if not callbacks_opt:
        runner_obj._callbacks = []
    for __key in ['loss/total', 'loss/mse/0', 'loss/mse/k', 'loss/cos/0', 'loss/cos/k']:
        runner_obj._state['scalars'][__key] = 0.0
    try:
        runner_obj.init_step(step_num=step_num)
        runner_obj.run_step(
            step_num=step_num,
            batch_arr=batch_arr,
            column_str=column_str)
        return clone_state(runner_obj._state)
    finally:
        runner_obj.close_step(step_num=step_num)
        runner_obj._callbacks = __callbacks

def topk_tokens(
    state: dict,
    model_obj: object,
    tokenizer_obj: object,
    k_num: int=10,
) -> list[dict]:
    __mask = state['tensors']['inputs/mask']
    with torch.no_grad():
        __teacher_logits = model_obj.lm_head(state['tensors']['outputs/teacher/k'])
        __student_logits = model_obj.lm_head(state['tensors']['outputs/student/k'])
    __outputs = []
    for __idx in range(int(__mask.shape[0])):
        __pos = max(0, int(__mask[__idx].sum().item()) - 1)
        __teacher_ids = __teacher_logits[__idx, __pos].topk(k_num).indices.tolist()
        __student_ids = __student_logits[__idx, __pos].topk(k_num).indices.tolist()
        __outputs.append({
            'position': __pos,
            'teacher_ids': __teacher_ids,
            'student_ids': __student_ids,
            'teacher_tokens': tokenizer_obj.convert_ids_to_tokens(__teacher_ids),
            'student_tokens': tokenizer_obj.convert_ids_to_tokens(__student_ids),})
    return __outputs
