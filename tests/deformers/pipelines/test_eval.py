import torch

import deformers.pipelines.eval

# PROBE ########################################################################

def test_indices_probe_is_deterministic_2d_list():
    __probe = deformers.pipelines.eval.indices_probe(
        vocab_dim=5,
        batch_dim=2,
        sequence_dim=4)
    assert __probe == [[0, 1, 2, 3], [4, 0, 1, 2]]

def test_text_probe_returns_runner_batch():
    assert deformers.pipelines.eval.text_probe(['a', 'b']) == {'text': ['a', 'b']}

def test_vocab_probe_returns_runner_batch():
    assert deformers.pipelines.eval.vocab_probe(
        vocab_dim=3,
        batch_dim=2,
        sequence_dim=2) == {'indices': [[0, 1], [2, 0]]}

# STATE ########################################################################

def test_clone_state_detaches_and_clones_tensors():
    __tensor = torch.ones(2, requires_grad=True)
    __state = {
        'tensors': {'x': __tensor},
        'scalars': {'loss/total': 1.0},}
    __clone = deformers.pipelines.eval.clone_state(__state)
    assert __clone['tensors']['x'] is not __tensor
    assert not __clone['tensors']['x'].requires_grad
    assert __clone['scalars'] == {'loss/total': 1.0}

# SCALARS ######################################################################

def test_scalar_accumulator_collects_means():
    __callback = deformers.pipelines.eval.prepare_scalar_accumulator(
        keys_arr=('loss/total', 'test/topk/k'))
    __callback['operation']({
        'scalars': {
            'loss/total': 1.0,
            'test/topk/k': 0.25,}})
    __callback['operation']({
        'scalars': {
            'loss/total': 3.0,
            'test/topk/k': 0.75,}})
    assert deformers.pipelines.eval.scalar_means(__callback) == {
        'loss/total': 2.0,
        'test/topk/k': 0.5,}

def test_scalar_means_returns_zero_for_empty_values():
    __callback = deformers.pipelines.eval.prepare_scalar_accumulator(
        keys_arr=('loss/total',))
    assert deformers.pipelines.eval.scalar_means(__callback) == {'loss/total': 0.0}

# RUNNER #######################################################################

class _FakeRunner:

    def __init__(self):
        self._callbacks = [{'name': 'existing'}]
        self._state = {
            'tensors': {},
            'scalars': {
                'loss/total': 99.0,
                'loss/mse/0': 99.0,
                'loss/mse/k': 99.0,
                'loss/cos/0': 99.0,
                'loss/cos/k': 99.0,
                'test/topk/k': 0.0,},}
        self.calls = []

    def init_step(self, step_num: int) -> None:
        self.calls.append(('init', step_num))

    def run_step(self, step_num: int, batch_arr: dict, column_str: str) -> None:
        self.calls.append(('run', step_num, batch_arr, column_str, list(self._callbacks)))
        self._state['tensors']['x'] = torch.tensor([1.0])
        self._state['scalars']['loss/total'] = 2.0

    def close_step(self, step_num: int) -> None:
        self.calls.append(('close', step_num))
        self._state['tensors'] = {}

def test_run_probe_runs_one_isolated_step_and_restores_callbacks():
    __runner = _FakeRunner()
    __state = deformers.pipelines.eval.run_probe(
        runner_obj=__runner,
        batch_arr={'text': ['x']},
        column_str='text',
        step_num=7)
    assert __runner.calls[0] == ('init', 7)
    assert __runner.calls[1] == ('run', 7, {'text': ['x']}, 'text', [])
    assert __runner.calls[2] == ('close', 7)
    assert __runner._callbacks == [{'name': 'existing'}]
    assert torch.equal(__state['tensors']['x'], torch.tensor([1.0]))
    assert __state['scalars']['loss/total'] == 2.0

# TOP-K ########################################################################

class _FakeModel:

    def __init__(self):
        self.calls = []

    def lm_head(self, activations):
        self.calls.append(activations)
        return activations

class _FakeTokenizer:

    def convert_ids_to_tokens(self, ids):
        return [f'tok-{__id}' for __id in ids]

def test_topk_tokens_uses_last_unmasked_position():
    __teacher = torch.tensor([[
        [0.0, 1.0, 2.0],
        [3.0, 2.0, 1.0],
        [0.0, 0.0, 9.0],]])
    __student = torch.tensor([[
        [2.0, 1.0, 0.0],
        [1.0, 3.0, 2.0],
        [9.0, 0.0, 0.0],]])
    __state = {
        'tensors': {
            'inputs/mask': torch.tensor([[1, 1, 0]]),
            'outputs/teacher/k': __teacher,
            'outputs/student/k': __student,},}
    __rows = deformers.pipelines.eval.topk_tokens(
        state=__state,
        model_obj=_FakeModel(),
        tokenizer_obj=_FakeTokenizer(),
        k_num=2)
    assert __rows == [{
        'position': 1,
        'teacher_ids': [0, 1],
        'student_ids': [1, 2],
        'teacher_tokens': ['tok-0', 'tok-1'],
        'student_tokens': ['tok-1', 'tok-2'],}]
