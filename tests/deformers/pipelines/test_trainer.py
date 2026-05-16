import contextlib
import time
import unittest.mock

import pytest
import torch

import deformers.pipelines.prefix.callbacks as _callbacks
import deformers.pipelines.prefix.trainer as _trainer


GLOBAL_CFG = {
    'device': 'cpu',
    'dtype': torch.float32,}

PHASE_CFG = {
    'column_str': 'text',
    'epoch_num': 2,}

BATCH_CFG = {
    'batch_dim': 2,
    'sequence_dim': 4,
    'patch_dim': 2,
    'left_pad': True,}

LOSS_CFG = {
    'mse_0_rate': 1.0,
    'mse_k_rate': 0.0,
    'cos_0_rate': 0.0,
    'cos_k_rate': 0.0,
    'relative_opt': False,}

GRADIENT_CFG = {
    'every_num': 1,
    'max_norm': 1.0,}

SCHEDULER_CFG = {
    'start_rate': 1e-4,
    'end_rate': 1e-3,
    'total_num': 4,
    'warmup_num': 1,}


class _FakeDataset:

    def __init__(self, batches):
        self._batches = list(batches)

    def __len__(self):
        return len(self._batches)

    def __iter__(self):
        return iter(self._batches)


class _Student(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.float().sum(dim=-1, keepdim=True) * self.weight


class _FakeScaler:

    def __init__(self):
        self.unscale_ = unittest.mock.MagicMock()
        self.step = unittest.mock.MagicMock()
        self.update = unittest.mock.MagicMock()

    def scale(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


def _make_dataset() -> _FakeDataset:
    return _FakeDataset([
        {'text': ['alpha', 'beta']},
        {'text': ['gamma', 'delta']},
        {'text': ['epsilon', 'zeta']},])


def _make_inputs() -> dict[str, torch.Tensor]:
    return {
        'inputs/mask': torch.ones(2, 4, dtype=torch.long),
        'inputs/indices': torch.zeros(2, 4, dtype=torch.long),
        'inputs/bytes': torch.ones(2, 4, 2, dtype=torch.long),}


def _patch_forward(monkeypatch) -> None:
    monkeypatch.setattr(
        'deformers.pipelines.prefix.processors.embed',
        lambda **kwargs: torch.ones(2, 4, 1))
    monkeypatch.setattr(
        'deformers.pipelines.prefix.processors.forward',
        lambda **kwargs: kwargs['embeds_arr'] + 1.0)


def _configure_base_runner(runner: _trainer.BaseRunner, dataset_obj: _FakeDataset | None=None, testing_cfg: dict=None) -> _trainer.BaseRunner:
    runner.setup_global(context_cfg=GLOBAL_CFG)
    runner.setup_phase(
        dataset_obj=_make_dataset() if dataset_obj is None else dataset_obj,
        phase_cfg=PHASE_CFG,
        batch_cfg=BATCH_CFG,
        loss_cfg=LOSS_CFG,
        gradient_cfg=GRADIENT_CFG,
        testing_cfg={} if testing_cfg is None else testing_cfg)
    return runner


def _configure_trainer(trainer: _trainer.PrefixTrainer, dataset_obj: _FakeDataset | None=None) -> _trainer.PrefixTrainer:
    trainer.setup_global(
        context_cfg=GLOBAL_CFG,
        optimizer_cfg={'lr': 1e-3},
        scaler_cfg={'enabled': False})
    trainer.setup_phase(
        dataset_obj=_make_dataset() if dataset_obj is None else dataset_obj,
        phase_cfg=PHASE_CFG,
        batch_cfg=BATCH_CFG,
        loss_cfg=LOSS_CFG,
        gradient_cfg=GRADIENT_CFG,
        scheduler_cfg=SCHEDULER_CFG)
    return trainer


def _make_base_runner() -> _trainer.BaseRunner:
    return _trainer.BaseRunner(
        text_tok=unittest.mock.MagicMock(),
        byte_tok=unittest.mock.MagicMock(),
        teacher_mod=unittest.mock.MagicMock(),
        student_mod=_Student(),)


def _make_prefix_trainer() -> _trainer.PrefixTrainer:
    return _trainer.PrefixTrainer(
        text_tok=unittest.mock.MagicMock(),
        byte_tok=unittest.mock.MagicMock(),
        teacher_mod=unittest.mock.MagicMock(),
        student_mod=_Student(),)


def _make_prefix_tester() -> _trainer.PrefixTester:
    return _trainer.PrefixTester(
        text_tok=unittest.mock.MagicMock(),
        byte_tok=unittest.mock.MagicMock(),
        teacher_mod=unittest.mock.MagicMock(),
        student_mod=_Student(),)


class TestBaseRunnerState:

    def test_state_layout_uses_scalars_tensors_and_metadata(self):
        runner = _make_base_runner()
        assert tuple(runner._state.keys()) == ('scalars', 'tensors', 'metadata')
        assert runner._state['metadata']['runner/name'] == 'BaseRunner'

    def test_check_setup_requires_global_and_phase(self):
        runner = _make_base_runner()
        with pytest.raises(AssertionError):
            runner._check_setup()
        _configure_base_runner(runner)
        runner._check_setup()


class TestBaseRunnerLifecycle:

    def test_run_phase_preserves_shared_epoch_and_step_flow(self):
        runner = _configure_base_runner(_make_base_runner())
        calls = []
        runner.step_batch = lambda batch_arr, column_str: calls.append(('batch', column_str, batch_arr))
        runner.step_forward = lambda: calls.append(('forward', runner._state['scalars']['step/current']))
        runner.step_losses = lambda: calls.append(('loss', runner._state['scalars']['step/current']))
        runner.step_callbacks = lambda: calls.append(('callbacks', runner._state['scalars']['step/current']))
        runner.step_progress = lambda pbar_obj: None
        runner.close_step = lambda: None

        runner.run_phase()

        assert len([call for call in calls if call[0] == 'batch']) == 6
        assert runner._state['scalars']['step/global'] == 6
        assert runner._state['scalars']['epoch/current'] == 2
        assert runner._state['scalars']['step/total'] == 3

    def test_init_step_respects_testing_cadence(self):
        runner = _configure_base_runner(_make_base_runner(), testing_cfg={'every_num': 2})
        runner.init_step(step_num=1)
        assert runner._state['scalars']['switch/train'] == 0
        assert runner._state['scalars']['switch/grad'] == 1


class TestTrainerSetup:

    def test_setup_global_creates_optimizer_scaler_and_context(self):
        trainer = _make_prefix_trainer()
        trainer.setup_global(
            context_cfg=GLOBAL_CFG,
            optimizer_cfg={'lr': 1e-3},
            scaler_cfg={'enabled': False})
        assert trainer._optimizer is not None
        assert trainer._scaler is not None
        assert trainer._context is not None

    def test_setup_phase_creates_scheduler(self):
        trainer = _make_prefix_trainer()
        _configure_trainer(trainer)
        assert trainer._scheduler is not None
        trainer._check_setup()


class TestPrefixTrainerBehavior:

    def test_step_forward_keeps_student_gradients_enabled(self, monkeypatch):
        trainer = _configure_trainer(_make_prefix_trainer())
        trainer._config['loss']['mse_k_rate'] = 1.0
        trainer._state['tensors'].update(_make_inputs())
        _patch_forward(monkeypatch)

        trainer.step_forward()

        assert trainer._state['tensors']['outputs/student/0'].requires_grad
        assert trainer._state['tensors']['outputs/student/k'].shape == trainer._state['tensors']['outputs/student/0'].shape
        assert torch.equal(trainer._state['tensors']['outputs/student/k'], trainer._state['tensors']['outputs/student/0'] + 1.0)

    def test_step_update_runs_backward_and_optimizer(self):
        trainer = _configure_trainer(_make_prefix_trainer())
        trainer._optimizer = torch.optim.SGD(trainer._student.parameters(), lr=1e-3)
        trainer._scaler = _FakeScaler()
        trainer._scheduler = unittest.mock.MagicMock()
        trainer._state['tensors']['loss/total'] = torch.tensor(1.0, requires_grad=True)
        trainer._state['scalars']['switch/grad'] = 1

        trainer.step_update()

        trainer._scaler.unscale_.assert_called_once_with(trainer._optimizer)
        trainer._scaler.step.assert_called_once_with(trainer._optimizer)
        trainer._scaler.update.assert_called_once()
        trainer._scheduler.step.assert_called_once()

    def test_close_step_only_resets_on_grad_boundary(self):
        trainer = _configure_trainer(_make_prefix_trainer())
        trainer._state['tensors']['loss/total'] = torch.tensor(1.0)
        trainer._state['scalars']['loss/total'] = 3.0
        trainer._state['scalars']['switch/grad'] = 0

        trainer.close_step()

        assert 'loss/total' in trainer._state['tensors']
        trainer._state['scalars']['switch/grad'] = 1
        trainer.close_step()
        assert trainer._state['tensors'] == {}
        assert trainer._state['scalars']['loss/total'] == 0.0


class TestPrefixTesterBehavior:

    def test_tester_step_forward_disables_student_gradients(self, monkeypatch):
        tester = _configure_base_runner(_make_prefix_tester())
        tester._state['tensors'].update(_make_inputs())
        _patch_forward(monkeypatch)

        tester.step_forward()

        assert not tester._state['tensors']['outputs/student/0'].requires_grad
        assert not tester._state['tensors']['outputs/student/k'].requires_grad

    def test_tester_init_step_stays_in_test_mode_without_grad_updates(self):
        tester = _configure_base_runner(_make_prefix_tester(), testing_cfg={'every_num': 1})

        tester.init_step(step_num=0)

        assert tester._state['scalars']['switch/train'] == 0
        assert tester._state['scalars']['switch/grad'] == 0

    def test_tester_close_step_resets_without_grad_boundary(self):
        tester = _configure_base_runner(_make_prefix_tester())
        tester._state['tensors']['loss/total'] = torch.tensor(1.0)
        tester._state['scalars']['loss/total'] = 2.0
        tester._state['scalars']['switch/grad'] = 0

        tester.close_step()

        assert tester._state['tensors'] == {}
        assert tester._state['scalars']['loss/total'] == 0.0


class TestCallbackContract:

    def test_step_callbacks_receive_full_nested_state(self):
        runner = _configure_base_runner(_make_base_runner())
        seen = []
        runner._callbacks = [{
            'name': 'probe',
            'trigger': lambda state: seen.append(('trigger', sorted(state.keys()))) or True,
            'operation': lambda state: seen.append(('operation', state['metadata']['runner/name'])),
            'cleanup': lambda: None,}]

        runner.step_callbacks()

        assert seen == [
            ('trigger', ['metadata', 'scalars', 'tensors']),
            ('operation', 'BaseRunner'),]

    def test_speed_callback_accepts_full_state(self):
        callback = _callbacks.prepare_speed_callback(every_num=2, batch_len=8)
        state = _make_base_runner().init_state({
            'scalars': {
                'step/current': 2,
                'iter/start': time.monotonic() - 0.5,},})

        assert callback['trigger'](state)
        callback['operation'](state)

        assert state['scalars']['iter/time'] > 0.0
        assert state['scalars']['iter/tps'] > 0.0

    def test_format_state_accepts_full_state_layout(self):
        state = _make_base_runner().init_state({
            'scalars': {
                'switch/train': 0,
                'step/current': 1,
                'step/total': 2,
                'epoch/current': 1,
                'epoch/total': 1,},})

        formatted = _callbacks.format_state(state)

        assert formatted['switch'] == '[test]'
        assert formatted['step'] == '(1/2)'
