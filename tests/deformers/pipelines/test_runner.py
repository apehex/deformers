"""
Unit tests for deformers.pipelines.prefix.runner.PrefixTrainer.

Covers:
- init_state: populates nested tensors/scalars keys with expected defaults.
- init_step: updates step/current, step/global (monotonic), and operation switches.
- step_batch: routes to vectorize_strings vs vectorize_indices based on column_str.
- step_forward: populates teacher/student tensors; skips hidden forward when not needed.
- step_losses: stores tensor loss and accumulates detached scalar metrics.
- step_backward: raises AssertionError when tensor loss is missing.
- step_optimizer: only steps optimizer/scaler/scheduler when update trigger is active.
- close_step: resets transient tensors and scalars only on grad-update boundaries.
- run_epoch: iterates over dataset, calls run_step and close_step, closes pbar.
- run_phase: calls run_epoch for the correct number of epochs (uses stored phase config).
- setup_optimizer: creates optimizer from valid config and ignores empty config.
- setup_global: initializes optimizer, scaler, and context.
- setup_phase: updates active config, stores phase dataset/column/epoch, rebuilds scheduler and callbacks.
"""

import types
import unittest.mock

import pytest
import torch
import torch.nn

import deformers.pipelines.prefix.runner as _runner

# HELPERS ######################################################################

def _make_config(
    epoch_num: int = 2,
    grad_every: int = 1,
    log_every: int = 0,
    save_every: int = 0,
    test_every: int = 0,
) -> dict:
    return {
        'global': {
            'dtype': torch.float32,
            'device': 'cpu',},
        'phase': {
            'column_str': 'text',
            'epoch_num': epoch_num,},
        'batch': {
            'batch_dim': 2,
            'sequence_dim': 4,
            'patch_dim': 2,
            'padding_str': '',
            'left_pad': True,},
        'loss': {
            'mse_0_rate': 1.0,
            'mse_k_rate': 0.0,
            'cos_0_rate': 0.0,
            'cos_k_rate': 0.0,
            'relative_opt': False,},
        'gradient': {
            'every_num': grad_every,
            'max_norm': 1.0,},
        'logging': {'every_num': log_every,},
        'saving': {'every_num': save_every,},
        'testing': {'every_num': test_every,},
        'optimizer': {'lr': 1e-3,},
    }


def _make_fake_param() -> torch.nn.Parameter:
    """Single trainable parameter for optimizer stubs."""
    return torch.nn.Parameter(torch.zeros(1))


def _make_mock_scaler() -> unittest.mock.MagicMock:
    """Minimal stub for GradScaler."""
    __scaler = unittest.mock.MagicMock()
    __scaler.scale = lambda x: x
    __scaler.unscale_ = unittest.mock.MagicMock()
    __scaler.step = unittest.mock.MagicMock()
    __scaler.update = unittest.mock.MagicMock()
    return __scaler


def _make_runner(
    epoch_num: int = 2,
    grad_every: int = 1,
    log_every: int = 0,
    save_every: int = 0,
    test_every: int = 0,
    callbacks_arr: list = None,
) -> _runner.PrefixTrainer:
    """Build a PrefixTrainer with test-friendly stubs.

    The trainer is constructed without configs.
    _config, _optimizer, and _scaler are then set directly so that
    existing tests for step_*, close_step, run_epoch, and run_phase continue
    to work without calling setup_global().
    """
    cfg = _make_config(
        epoch_num=epoch_num,
        grad_every=grad_every,
        log_every=log_every,
        save_every=save_every,
        test_every=test_every)

    __param = _make_fake_param()
    __student = unittest.mock.MagicMock()
    # student.parameters() must return something iterable for clip_grad_norm_
    __student.parameters = lambda: [__param]

    __t = _runner.PrefixTrainer(
        text_tok=unittest.mock.MagicMock(),
        byte_tok=unittest.mock.MagicMock(),
        teacher_mod=unittest.mock.MagicMock(),
        student_mod=__student,)
    __t._config = {__k: dict(__v) for (__k, __v) in cfg.items()}

    # inject test-friendly utilities directly (bypass setup_global for existing tests)
    __t._optimizer = torch.optim.SGD([__param], lr=1e-3)
    __t._scaler = _make_mock_scaler()

    if callbacks_arr is not None:
        import deformers.pipelines.prefix.callbacks as _cb
        __t._callbacks = [__c for __c in callbacks_arr if _cb.is_callback(__c)]

    return __t


def _make_tester(
    epoch_num: int = 2,
    grad_every: int = 1,
    log_every: int = 0,
    save_every: int = 0,
    test_every: int = 0,
) -> _runner.PrefixTester:
    cfg = _make_config(
        epoch_num=epoch_num,
        grad_every=grad_every,
        log_every=log_every,
        save_every=save_every,
        test_every=test_every)

    __t = _runner.PrefixTester(
        text_tok=unittest.mock.MagicMock(),
        byte_tok=unittest.mock.MagicMock(),
        teacher_mod=unittest.mock.MagicMock(),
        student_mod=unittest.mock.MagicMock(),)
    __t._config = {__k: dict(__v) for (__k, __v) in cfg.items()}
    return __t


# INIT_STATE ###################################################################

class TestInitState:

    def test_returns_dict_with_tensors_and_scalars(self):
        __t = _make_runner()
        __state = __t.init_state()
        assert 'tensors' in __state
        assert 'scalars' in __state

    def test_tensors_empty_by_default(self):
        __t = _make_runner()
        __state = __t.init_state()
        assert __state['tensors'] == {}

    def test_scalars_has_switch_grad_zero(self):
        __t = _make_runner()
        __state = __t.init_state()
        assert __state['scalars']['switch/grad'] == 0

    def test_scalars_epoch_total_defaults_to_one(self):
        __t = _make_runner()
        __state = __t.init_state()
        assert __state['scalars']['epoch/total'] == 1

    def test_scalars_step_global_starts_at_zero(self):
        __t = _make_runner()
        __state = __t.init_state()
        assert __state['scalars']['step/global'] == 0

    def test_scalars_has_loss_keys(self):
        __t = _make_runner()
        __s = __t.init_state()['scalars']
        for __k in ['loss/total', 'loss/mse/0', 'loss/mse/k', 'loss/cos/0', 'loss/cos/k']:
            assert __k in __s

    def test_override_replaces_scalars(self):
        __t = _make_runner()
        __state = __t.init_state({'scalars': {'step/global': 99}})
        assert __state['scalars']['step/global'] == 99

    def test_override_tensors_dict(self):
        __t = _make_runner()
        __dummy = torch.zeros(2)
        __state = __t.init_state({'tensors': {'inputs/mask': __dummy}})
        assert 'inputs/mask' in __state['tensors']


# INIT_STEP ####################################################################

class TestInitStep:

    def _runner_with_steps(self, step_tot: int = 10, **kwargs) -> _runner.PrefixTrainer:
        __t = _make_runner(**kwargs)
        __t._state['scalars']['step/total'] = step_tot
        return __t

    def test_step_current_is_one_indexed(self):
        __t = self._runner_with_steps()
        __t.init_step(step_num=1)
        assert __t._state['scalars']['step/current'] == 1

    def test_step_current_increments(self):
        __t = self._runner_with_steps()
        __t.init_step(step_num=5)
        assert __t._state['scalars']['step/current'] == 5

    def test_step_global_increments_from_zero(self):
        """step/global starts at 0 and increases by 1 per init_step call."""
        __t = self._runner_with_steps(step_tot=10)
        assert __t._state['scalars']['step/global'] == 0
        __t.init_step(step_num=1)
        assert __t._state['scalars']['step/global'] == 1

    def test_step_global_monotonically_increases(self):
        """step/global increases by 1 on every call, regardless of epoch."""
        __t = self._runner_with_steps(step_tot=10)
        for __i in range(1, 6):
            __t.init_step(step_num=__i)
        assert __t._state['scalars']['step/global'] == 5

    def test_step_global_persists_across_epoch_resets(self):
        """step/global does not reset when epoch/current changes."""
        __t = self._runner_with_steps(step_tot=10)
        for __i in range(1, 11):
            __t.init_step(step_num=__i)
        # simulate epoch rollover
        __t._state['scalars']['epoch/current'] = 2
        __t.init_step(step_num=1)
        assert __t._state['scalars']['step/global'] == 11

    def test_switch_grad_true_at_multiple(self):
        __t = self._runner_with_steps(grad_every=2)
        __t.init_step(step_num=2)
        assert __t._state['scalars']['switch/grad'] == 1

    def test_switch_grad_false_at_non_multiple(self):
        __t = self._runner_with_steps(grad_every=2)
        __t.init_step(step_num=1)
        assert __t._state['scalars']['switch/grad'] == 0

    def test_switch_log_false_when_disabled(self):
        __t = self._runner_with_steps(log_every=0)
        __t.init_step(step_num=1)
        assert __t._state['scalars']['switch/log'] == 0

    def test_switch_log_true_at_cadence(self):
        __t = self._runner_with_steps(log_every=5)
        __t.init_step(step_num=5)
        assert __t._state['scalars']['switch/log'] == 1

    def test_switch_save_false_when_disabled(self):
        __t = self._runner_with_steps(save_every=0)
        __t.init_step(step_num=1)
        assert __t._state['scalars']['switch/save'] == 0


# STEP_BATCH ###################################################################

class TestStepBatch:

    def _runner_with_mock_processors(self, monkeypatch):
        __t = _make_runner()
        # stub returns (mask, indices, bytes) tensors
        __mask = torch.ones(2, 4, dtype=torch.long)
        __indices = torch.zeros(2, 4, dtype=torch.long)
        __bytes = torch.zeros(2, 4, 2, dtype=torch.long)
        __result = (__mask, __indices, __bytes)

        monkeypatch.setattr(
            'deformers.pipelines.prefix.processors.vectorize_strings',
            lambda **kwargs: __result)
        monkeypatch.setattr(
            'deformers.pipelines.prefix.processors.vectorize_indices',
            lambda **kwargs: __result)
        return __t, __result

    def test_routes_to_vectorize_strings_for_text_column(self, monkeypatch):
        __t, _ = self._runner_with_mock_processors(monkeypatch)
        __called = []

        def __stub_strings(**kwargs):
            __called.append(True)
            return (torch.ones(2, 4), torch.zeros(2, 4), torch.zeros(2, 4, 2))

        monkeypatch.setattr(
            'deformers.pipelines.prefix.processors.vectorize_strings',
            __stub_strings)
        __batch = {'text': ['hello', 'world']}
        __t.step_batch(step_num=1, batch_arr=__batch, column_str='text')
        assert __called

    def test_routes_to_vectorize_indices_for_indice_column(self, monkeypatch):
        __t, _ = self._runner_with_mock_processors(monkeypatch)
        __called = []

        def __stub_indices(**kwargs):
            __called.append(True)
            return (torch.ones(2, 4), torch.zeros(2, 4), torch.zeros(2, 4, 2))

        monkeypatch.setattr(
            'deformers.pipelines.prefix.processors.vectorize_indices',
            __stub_indices)
        __batch = {'indices': [[1, 2, 3, 4], [5, 6, 7, 8]]}
        __t.step_batch(step_num=1, batch_arr=__batch, column_str='indices')
        assert __called

    def test_stores_inputs_mask(self, monkeypatch):
        __t, __result = self._runner_with_mock_processors(monkeypatch)
        __batch = {'text': ['a', 'b']}
        __t.step_batch(step_num=1, batch_arr=__batch, column_str='text')
        assert 'inputs/mask' in __t._state['tensors']

    def test_stores_inputs_indices(self, monkeypatch):
        __t, __result = self._runner_with_mock_processors(monkeypatch)
        __batch = {'text': ['a', 'b']}
        __t.step_batch(step_num=1, batch_arr=__batch, column_str='text')
        assert 'inputs/indices' in __t._state['tensors']

    def test_stores_inputs_bytes(self, monkeypatch):
        __t, __result = self._runner_with_mock_processors(monkeypatch)
        __batch = {'text': ['a', 'b']}
        __t.step_batch(step_num=1, batch_arr=__batch, column_str='text')
        assert 'inputs/bytes' in __t._state['tensors']


# STEP_FORWARD #################################################################

class TestStepForward:

    def _setup(self, monkeypatch, hidden: bool = False):
        __cfg_loss = {
            'mse_k_rate': 1.0 if hidden else 0.0,
            'cos_k_rate': 0.0,}
        __t = _make_runner()
        __t._config['loss'].update(__cfg_loss)

        # pre-fill input tensors
        __B, __T, __H = 2, 4, 8
        __mask = torch.ones(__B, __T)
        __indices = torch.zeros(__B, __T, dtype=torch.long)
        __bytes = torch.zeros(__B, __T, 2, dtype=torch.long)
        __t._state['tensors']['inputs/mask'] = __mask
        __t._state['tensors']['inputs/indices'] = __indices
        __t._state['tensors']['inputs/bytes'] = __bytes

        # teacher embed stub returns (B, T, H)
        __embed_out = torch.randn(__B, __T, __H)
        monkeypatch.setattr(
            'deformers.pipelines.prefix.processors.embed',
            lambda **kwargs: __embed_out)
        # teacher forward stub returns same shape
        monkeypatch.setattr(
            'deformers.pipelines.prefix.processors.forward',
            lambda **kwargs: torch.randn(__B, __T, __H))

        # student returns float tensor of same shape
        __t._student.return_value = __embed_out.clone()

        return __t, __embed_out

    def test_populates_teacher_0(self, monkeypatch):
        __t, _ = self._setup(monkeypatch)
        __t.step_forward(step_num=1)
        assert 'outputs/teacher/0' in __t._state['tensors']

    def test_populates_student_0(self, monkeypatch):
        __t, _ = self._setup(monkeypatch)
        __t.step_forward(step_num=1)
        assert 'outputs/student/0' in __t._state['tensors']

    def test_skips_hidden_forward_when_disabled(self, monkeypatch):
        __t, _ = self._setup(monkeypatch, hidden=False)
        __forward_called = []
        monkeypatch.setattr(
            'deformers.pipelines.prefix.processors.forward',
            lambda **kwargs: (__forward_called.append(True) or torch.zeros(2, 4, 8)))
        __t.step_forward(step_num=1)
        assert not __forward_called

    def test_calls_hidden_forward_when_enabled(self, monkeypatch):
        __t, _ = self._setup(monkeypatch, hidden=True)
        __forward_called = []

        def __stub(**kwargs):
            __forward_called.append(True)
            return torch.zeros(2, 4, 8)

        monkeypatch.setattr('deformers.pipelines.prefix.processors.forward', __stub)
        __t.step_forward(step_num=1)
        assert len(__forward_called) >= 1

    def test_teacher_k_is_zeros_when_hidden_disabled(self, monkeypatch):
        __t, __embed = self._setup(monkeypatch, hidden=False)
        __t.step_forward(step_num=1)
        __tk = __t._state['tensors']['outputs/teacher/k']
        assert torch.all(__tk == 0)


class TestPrefixTesterStepForward:

    def test_runs_under_no_grad(self, monkeypatch):
        __t = _make_tester()
        __t._config['loss'].update({'mse_k_rate': 0.0, 'cos_k_rate': 0.0})
        __t._state['tensors']['inputs/mask'] = torch.ones(2, 4)
        __t._state['tensors']['inputs/indices'] = torch.zeros(2, 4, dtype=torch.long)
        __t._state['tensors']['inputs/bytes'] = torch.zeros(2, 4, 2, dtype=torch.long)

        monkeypatch.setattr(
            'deformers.pipelines.prefix.processors.embed',
            lambda **kwargs: torch.randn(2, 4, 8))
        monkeypatch.setattr(
            'deformers.pipelines.prefix.processors.forward',
            lambda **kwargs: torch.randn(2, 4, 8))

        __grad_enabled = []
        def __student(inputs):
            __grad_enabled.append(torch.is_grad_enabled())
            return torch.randn(2, 4, 8)
        __t._student = __student

        __t.step_forward(step_num=1)
        assert __grad_enabled == [False]


class TestPrefixTesterObjective:

    def test_calls_step_losses_only(self):
        __t = _make_tester()
        __t._step_losses = unittest.mock.MagicMock()
        __t._step_backward = unittest.mock.MagicMock()
        __t._step_optimizer = unittest.mock.MagicMock()
        __t.step_objective(step_num=1)
        __t._step_losses.assert_called_once()
        __t._step_backward.assert_not_called()
        __t._step_optimizer.assert_not_called()


class TestRunnerTriggers:

    def test_base_step_forward_is_abstract_contract(self):
        __runner = _runner.BaseRunner(
            text_tok=unittest.mock.MagicMock(),
            byte_tok=unittest.mock.MagicMock(),
            teacher_mod=unittest.mock.MagicMock(),
            student_mod=unittest.mock.MagicMock(),)
        with pytest.raises(NotImplementedError):
            __runner.step_forward(step_num=1)

    def test_prefix_tester_init_step_applies_trigger_switches(self):
        __t = _make_tester(grad_every=2, test_every=3)
        __t.init_step(step_num=1)
        assert __t._state['scalars']['switch/grad'] == 0
        assert __t._state['scalars']['switch/progress'] == 1
        assert __t._state['scalars']['switch/cleanup'] == 1

    def test_hidden_trigger_follows_test_trigger(self):
        __t = _make_runner(test_every=3)
        __t._config['loss'].update({'mse_k_rate': 0.0, 'cos_k_rate': 0.0})
        assert __t._trigger_hidden(step_num=3)

    def test_hidden_trigger_false_without_hidden_loss_or_test(self):
        __t = _make_runner(test_every=3)
        __t._config['loss'].update({'mse_k_rate': 0.0, 'cos_k_rate': 0.0})
        assert not __t._trigger_hidden(step_num=1)


class TestStepProgress:

    def test_uses_trigger_progress(self):
        __t = _make_runner()
        __pbar = unittest.mock.MagicMock()
        __calls = []

        def __trigger(step_num):
            __calls.append(step_num)
            return False

        __t._trigger_progress = __trigger
        __t._state['scalars']['step/current'] = 7
        __t._state['scalars']['switch/grad'] = 0
        __t.step_progress(step_num=7, pbar_obj=__pbar)
        assert __calls == [7]
        __pbar.set_postfix.assert_not_called()


# STEP_LOSSES ##################################################################

class TestStepLosses:

    def _setup(self, monkeypatch):
        __t = _make_runner()
        __B, __T, __H = 2, 4, 8
        # pre-fill teacher/student output tensors
        __ref = torch.ones(__B, __T, __H)
        __t._state['tensors']['inputs/mask'] = torch.ones(__B, __T)
        __t._state['tensors']['outputs/teacher/0'] = __ref.clone()
        __t._state['tensors']['outputs/teacher/k'] = __ref.clone()
        __t._state['tensors']['outputs/student/0'] = __ref.clone()
        __t._state['tensors']['outputs/student/k'] = __ref.clone()

        # stub compute_losses to return known values
        __fake_scalar = torch.tensor(0.25)
        __fake_loss = torch.tensor(1.0, requires_grad=True)
        monkeypatch.setattr(
            'deformers.pipelines.prefix.processors.compute_losses',
            lambda **kwargs: (
                __fake_scalar,
                __fake_scalar,
                __fake_scalar,
                __fake_scalar,
                __fake_loss,
                __fake_loss,))
        return __t

    def test_stores_tensor_loss(self, monkeypatch):
        __t = self._setup(monkeypatch)
        __t._step_losses()
        assert 'loss/total' in __t._state['tensors']
        assert hasattr(__t._state['tensors']['loss/total'], 'shape')

    def test_accumulates_scalar_total(self, monkeypatch):
        __t = self._setup(monkeypatch)
        __t._step_losses()
        assert __t._state['scalars']['loss/total'] > 0.0

    def test_accumulates_mse_0(self, monkeypatch):
        __t = self._setup(monkeypatch)
        __t._step_losses()
        assert __t._state['scalars']['loss/mse/0'] > 0.0

    def test_scalar_loss_is_detached(self, monkeypatch):
        __t = self._setup(monkeypatch)
        __t._step_losses()
        # the scalars should be plain floats, not tensors
        assert isinstance(__t._state['scalars']['loss/total'], float)


# STEP_BACKWARD ################################################################

class TestStepBackward:

    def test_raises_when_no_tensor_loss(self):
        __t = _make_runner()
        __t._state['tensors'] = {}
        with pytest.raises(AssertionError):
            __t._step_backward()

    def test_raises_when_tensor_loss_is_scalar(self):
        __t = _make_runner()
        # store a plain float, not a tensor
        __t._state['tensors']['loss/total'] = 1.23
        with pytest.raises(AssertionError):
            __t._step_backward()

    def test_calls_scaler_scale_backward(self):
        __t = _make_runner()
        __loss = torch.tensor(0.5, requires_grad=True)
        __t._state['tensors']['loss/total'] = __loss
        # backward on a leaf requires grad; scaler.scale returns the loss itself
        __t._scaler.scale = lambda x: x
        __t._step_backward()
        # no exception means backward was called; grad should be set on __loss
        # (since __loss is a leaf with grad fn after * 1)
        # Just verify no error was raised


# STEP_OPTIMIZER ###############################################################

class TestStepOptimizer:

    def test_does_not_step_when_switch_grad_off(self):
        __t = _make_runner(grad_every=2)
        __t._state['scalars']['switch/grad'] = 1
        __t._step_optimizer(step_num=1)
        __t._scaler.step.assert_not_called()

    def test_steps_optimizer_when_update_trigger_on(self):
        __t = _make_runner()
        __t._state['scalars']['switch/grad'] = 0
        __t._step_optimizer(step_num=1)
        __t._scaler.step.assert_called_once()

    def test_calls_scaler_update_when_update_trigger_on(self):
        __t = _make_runner()
        __t._state['scalars']['switch/grad'] = 0
        __t._step_optimizer(step_num=1)
        __t._scaler.update.assert_called_once()

    def test_steps_scheduler_when_present(self):
        __t = _make_runner()
        __scheduler = unittest.mock.MagicMock()
        __t._scheduler = __scheduler
        __t._step_optimizer(step_num=1)
        __scheduler.step.assert_called_once()

    def test_no_scheduler_step_when_none(self):
        __t = _make_runner()
        __t._scheduler = None
        # should not raise
        __t._step_optimizer(step_num=1)


class TestPrefixTrainerObjective:

    def test_runs_backward_and_optimizer_when_train_switch_on(self):
        __t = _make_runner()
        __t._step_losses = unittest.mock.MagicMock()
        __t._step_backward = unittest.mock.MagicMock()
        __t._step_optimizer = unittest.mock.MagicMock()
        __t.step_objective(step_num=1)
        __t._step_losses.assert_called_once()
        __t._step_backward.assert_called_once()
        __t._step_optimizer.assert_called_once_with(step_num=1)


# CLOSE_STEP ###################################################################

class TestCloseStep:

    def _runner_with_tensors(self, **kwargs) -> _runner.PrefixTrainer:
        __t = _make_runner(**kwargs)
        __t._state['tensors']['loss/total'] = torch.tensor(1.0)
        __t._state['scalars']['loss/total'] = 2.0
        __t._state['scalars']['loss/mse/0'] = 0.5
        return __t

    def test_resets_tensors_on_grad_boundary(self):
        __t = self._runner_with_tensors()
        __t._state['scalars']['switch/cleanup'] = 0
        __t.close_step(step_num=1)
        assert __t._state['tensors'] == {}

    def test_resets_loss_scalars_on_grad_boundary(self):
        __t = self._runner_with_tensors()
        __t._state['scalars']['switch/cleanup'] = 0
        __t.close_step(step_num=1)
        assert __t._state['scalars']['loss/total'] == 0.0
        assert __t._state['scalars']['loss/mse/0'] == 0.0

    def test_no_reset_when_switch_grad_off(self):
        __t = self._runner_with_tensors(grad_every=2)
        __t._state['scalars']['switch/cleanup'] = 1
        __t.close_step(step_num=1)
        # tensors should still be present
        assert 'loss/total' in __t._state['tensors']
        # scalar should still hold the old value
        assert __t._state['scalars']['loss/total'] == 2.0


# RUN_EPOCH ####################################################################

class _FakeDataset:
    """Minimal dataset stub that is iterable and has a len."""

    def __init__(self, batches):
        self._batches = list(batches)

    def __len__(self):
        return len(self._batches)

    def __iter__(self):
        return iter(self._batches)


class TestRunEpoch:

    def _make_run_step_stub(self, trainer: _runner.PrefixTrainer) -> list:
        """Patch run_step and close_step to be no-ops; return call log."""
        __calls = []

        def __fake_run_step(step_num, batch_arr, column_str):
            __calls.append(batch_arr)

        trainer.run_step = __fake_run_step
        trainer.close_step = lambda step_num: None
        trainer.step_progress = lambda step_num, pbar_obj: None
        return __calls

    def test_iterates_over_all_batches(self):
        __t = _make_runner()
        __batches = [{'text': [f'sample{i}']} for i in range(3)]
        __ds = _FakeDataset(__batches)
        __calls = self._make_run_step_stub(__t)
        __t.run_epoch(epoch_num=0, epoch_tot=1, dataset_obj=__ds, column_str='text')
        assert len(__calls) == 3

    def test_updates_epoch_current(self):
        __t = _make_runner()
        __ds = _FakeDataset([{'text': ['x']}])
        self._make_run_step_stub(__t)
        __t.run_epoch(epoch_num=2, epoch_tot=5, dataset_obj=__ds, column_str='text')
        assert __t._state['scalars']['epoch/current'] == 3

    def test_updates_step_total_from_dataset_len(self):
        __t = _make_runner()
        __batches = [{'text': ['a']}, {'text': ['b']}]
        __ds = _FakeDataset(__batches)
        self._make_run_step_stub(__t)
        __t.run_epoch(epoch_num=0, epoch_tot=1, dataset_obj=__ds, column_str='text')
        assert __t._state['scalars']['step/total'] == 2

    def test_passes_one_indexed_step_nums(self):
        __t = _make_runner()
        __seen = {'run': [], 'progress': [], 'close': []}
        __ds = _FakeDataset([{'text': ['a']}, {'text': ['b']}])

        __t.run_step = lambda step_num, batch_arr, column_str: __seen['run'].append(step_num)
        __t.step_progress = lambda step_num, pbar_obj: __seen['progress'].append(step_num)
        __t.close_step = lambda step_num: __seen['close'].append(step_num)
        __t.run_epoch(epoch_num=0, epoch_tot=1, dataset_obj=__ds, column_str='text')
        assert __seen == {'run': [1, 2], 'progress': [1, 2], 'close': [1, 2]}


# RUN_PHASE ####################################################################

class _FakeDataset:
    """Minimal dataset stub that is iterable and has a len."""

    def __init__(self, batches):
        self._batches = list(batches)

    def __len__(self):
        return len(self._batches)

    def __iter__(self):
        return iter(self._batches)


class TestRunPhase:

    def _setup_phase(self, trainer: _runner.PrefixTrainer, ds, epoch_num: int, column_str: str) -> None:
        """Set phase attributes directly to satisfy run_phase() readiness checks."""
        trainer._dataset = ds
        trainer._scheduler = unittest.mock.MagicMock()
        trainer._config['phase'] = {'column_str': column_str, 'epoch_num': epoch_num}
        trainer._state['scalars']['epoch/total'] = epoch_num

    def test_calls_run_epoch_for_each_epoch(self):
        __t = _make_runner()
        __epoch_calls = []

        def __fake_run_epoch(epoch_num, epoch_tot, dataset_obj, column_str):
            __epoch_calls.append(epoch_num)

        __t.run_epoch = __fake_run_epoch
        __ds = _FakeDataset([{'text': ['x']}])
        self._setup_phase(__t, __ds, epoch_num=3, column_str='text')
        __t.run_phase()
        assert __epoch_calls == [0, 1, 2]

    def test_passes_column_str_to_run_epoch(self):
        __t = _make_runner()
        __col_seen = []

        def __fake_run_epoch(epoch_num, epoch_tot, dataset_obj, column_str):
            __col_seen.append(column_str)

        __t.run_epoch = __fake_run_epoch
        __ds = _FakeDataset([{'text': ['x']}])
        self._setup_phase(__t, __ds, epoch_num=2, column_str='my_column')
        __t.run_phase()
        assert all(__c == 'my_column' for __c in __col_seen)

    def test_passes_correct_epoch_tot(self):
        __t = _make_runner()
        __epoch_tots = []

        def __fake_run_epoch(epoch_num, epoch_tot, dataset_obj, column_str):
            __epoch_tots.append(epoch_tot)

        __t.run_epoch = __fake_run_epoch
        __ds = _FakeDataset([])
        self._setup_phase(__t, __ds, epoch_num=4, column_str='text')
        __t.run_phase()
        assert all(__e == 4 for __e in __epoch_tots)

# SETUP_OPTIMIZER ##############################################################

class TestSetupOptimizer:

    def _make_base_runner(self) -> _runner.PrefixTrainer:
        """Trainer with a real student parameter; no utilities pre-set."""
        __student = torch.nn.Linear(1, 1, bias=False)
        __t = _runner.PrefixTrainer(
            text_tok=unittest.mock.MagicMock(),
            byte_tok=unittest.mock.MagicMock(),
            teacher_mod=unittest.mock.MagicMock(),
            student_mod=__student,)
        return __t

    def test_creates_optimizer_when_none(self):
        __t = self._make_base_runner()
        assert __t._optimizer is None
        __t.setup_optimizer(optimizer_cfg={'lr': 1e-3})
        assert __t._optimizer is not None

    def test_skips_when_cfg_missing(self):
        __t = self._make_base_runner()
        __t.setup_optimizer(optimizer_cfg={'lr': 1e-3})
        __first = __t._optimizer
        __t.setup_optimizer()
        assert __t._optimizer is __first

    def test_overwrites_when_valid_cfg_provided(self):
        __t = self._make_base_runner()
        __t.setup_optimizer(optimizer_cfg={'lr': 1e-3})
        __first = __t._optimizer
        __t.setup_optimizer(optimizer_cfg={'lr': 2e-3})
        assert __t._optimizer is not __first
        assert __t._optimizer.param_groups[0]['lr'] == 2e-3

    def test_uses_custom_cfg_override(self):
        __t = self._make_base_runner()
        __t.setup_optimizer(optimizer_cfg={'lr': 9e-9})
        __lr = __t._optimizer.param_groups[0]['lr']
        assert abs(__lr - 9e-9) < 1e-15


# SETUP_CALLBACKS ###############################################################

class TestSetupCallbacks:

    def _make_base_runner(self) -> _runner.PrefixTrainer:
        __student = unittest.mock.MagicMock()
        __t = _runner.PrefixTrainer(
            text_tok=unittest.mock.MagicMock(),
            byte_tok=unittest.mock.MagicMock(),
            teacher_mod=unittest.mock.MagicMock(),
            student_mod=__student,)
        return __t

    def test_empty_configs_do_not_reuse_internal_config(self):
        __t = self._make_base_runner()
        __t._config['speed'] = {'every_num': 1, 'batch_len': 1}
        __t.setup_callbacks()
        assert __t._callbacks == []

    def test_builds_callbacks_from_valid_configs(self):
        __t = self._make_base_runner()
        __t.setup_callbacks(speed_cfg={'every_num': 2, 'batch_len': 3})
        assert len(__t._callbacks) == 1
        assert __t._callbacks[0]['name'] == 'speed'
        assert not __t._callbacks[0]['trigger']({'scalars': {'step/current': 1}})
        assert __t._callbacks[0]['trigger']({'scalars': {'step/current': 2}})
        assert not __t._callbacks[0]['trigger']({'tensors': {}, 'scalars': {'step/current': 3}})
        __state = {'tensors': {}, 'scalars': {'step/current': 2, 'iter/start': 0.0, 'iter/time': 0.0, 'iter/tps': 0.0}}
        __t._callbacks[0]['operation'](__state)
        assert __state['scalars']['iter/time'] >= 0.0

    def test_trigger_raises_for_missing_scalars(self):
        __t = self._make_base_runner()
        __t.setup_callbacks(speed_cfg={'every_num': 2, 'batch_len': 3})
        with pytest.raises(KeyError):
            __t._callbacks[0]['trigger']({'tensors': {}})


class TestCallbackStateContract:

    def test_callbacks_receive_full_state(self):
        __t = _make_runner()
        __seen = []
        __t._callbacks = [{
            'name': 'test',
            'trigger': lambda state: (__seen.append(state) or True),
            'operation': lambda state: state['scalars'].__setitem__('loss/ema', 123.0),
            'cleanup': lambda: None,}]
        __t.step_callbacks(step_num=1)
        assert __seen[0] is __t._state
        assert __t._state['scalars']['loss/ema'] == 123.0


class TestRunnerLifecycle:

    def test_run_step_uses_shared_step_flow(self):
        __runner = _make_tester()
        __calls = []
        __step_args = {}

        def __step_batch(step_num, batch_arr, column_str):
            __calls.append('batch')
            __step_args['step_num'] = step_num
            __step_args['batch_arr'] = batch_arr
            __step_args['column_str'] = column_str

        def __step_forward(step_num):
            __calls.append('forward')

        def __step_objective(step_num):
            __calls.append('objective')

        def __step_callbacks(step_num):
            __calls.append('callbacks')

        __runner.step_batch = __step_batch
        __runner.step_forward = __step_forward
        __runner.step_objective = __step_objective
        __runner.step_callbacks = __step_callbacks
        __batch = {'text': ['x']}
        __runner.run_step(step_num=7, batch_arr=__batch, column_str='text')
        assert __calls == ['batch', 'forward', 'objective', 'callbacks']
        assert __step_args == {'step_num': 7, 'batch_arr': __batch, 'column_str': 'text'}



# SETUP_GLOBAL #################################################################

class TestSetupGlobal:

    def _setup_global(self, trainer: _runner.PrefixTrainer, overwrite_opt: bool = False) -> None:
        trainer.setup_global(
            context_cfg={'dtype': torch.float32, 'device': 'cpu'},
            optimizer_cfg={'lr': 1e-3},
            scaler_cfg={'enabled': False},
            overwrite_opt=overwrite_opt)

    def _make_base_runner(self) -> _runner.PrefixTrainer:
        __student = torch.nn.Linear(1, 1, bias=False)
        __t = _runner.PrefixTrainer(
            text_tok=unittest.mock.MagicMock(),
            byte_tok=unittest.mock.MagicMock(),
            teacher_mod=unittest.mock.MagicMock(),
            student_mod=__student,)
        return __t

    def test_creates_optimizer(self):
        __t = self._make_base_runner()
        self._setup_global(__t)
        assert __t._optimizer is not None

    def test_creates_scaler(self):
        __t = self._make_base_runner()
        self._setup_global(__t)
        assert __t._scaler is not None

    def test_creates_context(self):
        __t = self._make_base_runner()
        self._setup_global(__t)
        assert __t._config['context'] == {'dtype': torch.float32, 'device': 'cpu'}

    def test_preserves_optimizer_across_calls(self):
        """Calling setup_global() twice should not recreate the optimizer."""
        __t = self._make_base_runner()
        self._setup_global(__t)
        __first = __t._optimizer
        self._setup_global(__t)
        assert __t._optimizer is __first

    def test_overwrites_optimizer_when_flag_set(self):
        __t = self._make_base_runner()
        self._setup_global(__t)
        __first = __t._optimizer
        self._setup_global(__t, overwrite_opt=True)
        assert __t._optimizer is not __first


class TestPrefixTesterSetupGlobal:

    def _make_tester(self) -> _runner.PrefixTester:
        return _runner.PrefixTester(
            text_tok=unittest.mock.MagicMock(),
            byte_tok=unittest.mock.MagicMock(),
            teacher_mod=unittest.mock.MagicMock(),
            student_mod=unittest.mock.MagicMock(),)

    def test_stores_context_optimizer_scaler_configs(self):
        __t = self._make_tester()
        __t.setup_global(
            context_cfg={'dtype': torch.float32, 'device': 'cpu'},
            optimizer_cfg={'lr': 1e-3},
            scaler_cfg={'enabled': False},)
        assert __t._config['context'] == {'dtype': torch.float32, 'device': 'cpu'}
        assert __t._config['optimizer'] == {'lr': 1e-3}
        assert __t._config['scaler'] == {'enabled': False}

    def test_does_not_create_optimizer_or_scaler(self):
        __t = self._make_tester()
        __t.setup_global(
            context_cfg={'dtype': torch.float32, 'device': 'cpu'},
            optimizer_cfg={'lr': 1e-3},
            scaler_cfg={'enabled': False},)
        assert __t._optimizer is None
        assert __t._scaler is None


# SETUP_PHASE ##################################################################

class TestSetupPhase:

    def _setup_phase(
        self,
        trainer: _runner.PrefixTrainer,
        dataset_obj: object,
        epoch_num: int,
        column_str: str,
        **kwargs,
    ) -> None:
        __base_kwargs = {
            'phase_cfg': {'column_str': column_str, 'epoch_num': epoch_num},
            'batch_cfg': {'batch_dim': 2, 'sequence_dim': 4, 'patch_dim': 2, 'left_pad': True},
            'loss_cfg': {'mse_0_rate': 1.0, 'mse_k_rate': 0.0, 'cos_0_rate': 0.0, 'cos_k_rate': 0.0},
            'gradient_cfg': {'every_num': 1, 'max_norm': 1.0},
            'logging_cfg': {},
            'scheduler_cfg': {},
            'saving_cfg': {},
            'testing_cfg': {},
            'ema_cfg': {},
            'speed_cfg': {},
            'tboard_cfg': {},}
        __base_kwargs.update(kwargs)
        trainer.setup_phase(
            dataset_obj=dataset_obj,
            **__base_kwargs)

    def _make_ready_runner(self) -> _runner.PrefixTrainer:
        """Trainer with setup_global() already called."""
        __student = torch.nn.Linear(1, 1, bias=False)
        __t = _runner.PrefixTrainer(
            text_tok=unittest.mock.MagicMock(),
            byte_tok=unittest.mock.MagicMock(),
            teacher_mod=unittest.mock.MagicMock(),
            student_mod=__student,)
        __t.setup_global(
            context_cfg={'dtype': torch.float32, 'device': 'cpu'},
            optimizer_cfg={'lr': 1e-3},
            scaler_cfg={'enabled': False},)
        return __t

    def test_stores_dataset(self):
        __t = self._make_ready_runner()
        __ds = _FakeDataset([])
        self._setup_phase(__t, dataset_obj=__ds, epoch_num=3, column_str='text')
        assert __t._dataset is __ds

    def test_stores_column_str(self):
        __t = self._make_ready_runner()
        self._setup_phase(__t, dataset_obj=_FakeDataset([]), epoch_num=2, column_str='indices')
        assert __t._config['phase']['column_str'] == 'indices'

    def test_stores_epoch_num(self):
        __t = self._make_ready_runner()
        self._setup_phase(__t, dataset_obj=_FakeDataset([]), epoch_num=5, column_str='text')
        assert __t._config['phase']['epoch_num'] == 5

    def test_updates_state_epoch_total_from_phase_cfg(self):
        __t = self._make_ready_runner()
        self._setup_phase(__t, dataset_obj=_FakeDataset([]), epoch_num=5, column_str='text')
        assert __t._state['scalars']['epoch/total'] == 5

    def test_updates_current_config_with_phase_config(self):
        __t = self._make_ready_runner()
        self._setup_phase(
            __t,
            dataset_obj=_FakeDataset([]),
            epoch_num=2,
            column_str='text',
            loss_cfg={'mse_0_rate': 1.0, 'mse_k_rate': 0.5, 'cos_0_rate': 0.0, 'cos_k_rate': 0.0})
        assert __t._config['loss']['mse_k_rate'] == 0.5

    def test_uses_single_config_container(self):
        __t = self._make_ready_runner()
        assert hasattr(__t, '_config')
        assert not hasattr(__t, '_base_cfg')
        assert not hasattr(__t, '_active_cfg')

    def test_current_config_replaced_between_phases(self):
        """Second setup_phase replaces the current phase config."""
        __t = self._make_ready_runner()
        self._setup_phase(
            __t,
            dataset_obj=_FakeDataset([]),
            epoch_num=2,
            column_str='text',
            loss_cfg={'mse_0_rate': 1.0, 'mse_k_rate': 0.9, 'cos_0_rate': 0.0, 'cos_k_rate': 0.0})
        self._setup_phase(__t, dataset_obj=_FakeDataset([]), epoch_num=2, column_str='text')
        assert __t._config['loss']['mse_k_rate'] == 0.0

    def test_batch_cfg_missing_batch_dim_is_ignored(self):
        __t = self._make_ready_runner()
        __phase_cfg = {'column_str': 'text', 'epoch_num': 2}
        __batch_cfg = {'sequence_dim': 4, 'patch_dim': 2, 'left_pad': True}
        __t.setup_phase(
            dataset_obj=_FakeDataset([]),
            phase_cfg=__phase_cfg,
            batch_cfg=__batch_cfg,
            loss_cfg={'mse_0_rate': 1.0, 'mse_k_rate': 0.0, 'cos_0_rate': 0.0, 'cos_k_rate': 0.0},
            gradient_cfg={'every_num': 1, 'max_norm': 1.0},
        )
        assert __t._config['batch'] == {}

    def test_batch_cfg_with_batch_dim_is_stored(self):
        __t = self._make_ready_runner()
        __batch_cfg = {'batch_dim': 2, 'sequence_dim': 4, 'patch_dim': 2, 'left_pad': True}
        self._setup_phase(__t, dataset_obj=_FakeDataset([]), epoch_num=2, column_str='text', batch_cfg=__batch_cfg)
        assert __t._config['batch']['batch_dim'] == 2

    def test_creates_scheduler_when_cfg_provided(self):
        __t = self._make_ready_runner()
        self._setup_phase(
            __t,
            dataset_obj=_FakeDataset([]),
            epoch_num=2,
            column_str='text',
            scheduler_cfg={'start_rate': 1e-4, 'end_rate': 1e-3, 'total_num': 10, 'warmup_num': 2})
        assert __t._scheduler is not None

    def test_no_scheduler_when_cfg_empty(self):
        __t = self._make_ready_runner()
        self._setup_phase(__t, dataset_obj=_FakeDataset([]), epoch_num=2, column_str='text')
        assert __t._scheduler is None

    def test_scheduler_refreshed_across_phases(self):
        """Scheduler created in phase 1 is replaced in phase 2."""
        __t = self._make_ready_runner()
        __sched_cfg = {'start_rate': 1e-4, 'end_rate': 1e-3, 'total_num': 10, 'warmup_num': 2}
        self._setup_phase(__t, dataset_obj=_FakeDataset([]), epoch_num=2, column_str='text', scheduler_cfg=__sched_cfg)
        __first = __t._scheduler
        self._setup_phase(__t, dataset_obj=_FakeDataset([]), epoch_num=2, column_str='text', scheduler_cfg=__sched_cfg)
        assert __t._scheduler is not __first

    def test_optimizer_preserved_across_phases(self):
        """Optimizer set by setup_global is not touched by setup_phase."""
        __t = self._make_ready_runner()
        __opt = __t._optimizer
        self._setup_phase(__t, dataset_obj=_FakeDataset([]), epoch_num=2, column_str='text')
        assert __t._optimizer is __opt

    def test_callbacks_refreshed_across_phases(self):
        """Callbacks from phase 1 are replaced in phase 2 (list is different object)."""
        __t = self._make_ready_runner()
        self._setup_phase(__t, dataset_obj=_FakeDataset([]), epoch_num=2, column_str='text')
        __first_callbacks = __t._callbacks
        self._setup_phase(__t, dataset_obj=_FakeDataset([]), epoch_num=2, column_str='text')
        # both are empty lists here (no callback cfgs), but they must be different objects
        assert __t._callbacks is not __first_callbacks

    def test_check_setup_passes_after_setup(self):
        __t = self._make_ready_runner()
        self._setup_phase(
            __t,
            dataset_obj=_FakeDataset([]),
            epoch_num=2,
            column_str='text',
            scheduler_cfg={'start_rate': 1e-4, 'end_rate': 1e-3, 'total_num': 10, 'warmup_num': 2})
        # should not raise
        __t._check_setup()

    def test_check_setup_fails_before_phase(self):
        __t = self._make_ready_runner()
        with pytest.raises(AssertionError):
            __t._check_setup()
