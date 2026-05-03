"""
Unit tests for deformers.pipelines.prefix.trainer.PrefixTrainer.

Covers:
- init_state: populates nested tensors/scalars keys with expected defaults.
- init_step: updates step/current, step/global, and operation switches.
- step_batch: routes to vectorize_strings vs vectorize_indices based on column_str.
- step_forward: populates teacher/student tensors; skips hidden forward when not needed.
- step_losses: stores tensor loss and accumulates detached scalar metrics.
- step_backward: raises AssertionError when tensor loss is missing.
- step_optimizer: only steps optimizer/scaler/scheduler when switch/grad is active.
- close_step: resets transient tensors and scalars only on grad-update boundaries.
- run_epoch: iterates over dataset, calls run_step and close_step, closes pbar.
- run_phase: calls run_epoch for the correct number of epochs.
"""

import contextlib
import types
import unittest.mock

import pytest
import torch
import torch.nn

import deformers.pipelines.prefix.trainer as _trainer

# HELPERS ######################################################################

def _make_config(
    epoch_num: int = 2,
    grad_every: int = 1,
    log_every: int = 0,
    save_every: int = 0,
    test_every: int = 0,
) -> dict:
    return {
        'batch': {
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
        'training': {
            'epoch_num': epoch_num,
            'dtype': torch.float32,
            'device': 'cpu',},
        'logging': {'every_num': log_every,},
        'saving': {'every_num': save_every,},
        'testing': {'every_num': test_every,},
    }


def _make_fake_param() -> torch.nn.Parameter:
    """Single trainable parameter for optimizer stubs."""
    return torch.nn.Parameter(torch.zeros(1))


def _make_trainer(
    epoch_num: int = 2,
    grad_every: int = 1,
    log_every: int = 0,
    save_every: int = 0,
    test_every: int = 0,
    callbacks_arr: list = None,
) -> _trainer.PrefixTrainer:
    cfg = _make_config(
        epoch_num=epoch_num,
        grad_every=grad_every,
        log_every=log_every,
        save_every=save_every,
        test_every=test_every)

    __param = _make_fake_param()
    __optimizer = torch.optim.SGD([__param], lr=1e-3)

    # minimal stub for GradScaler
    __scaler = unittest.mock.MagicMock()
    __scaler.scale = lambda x: x
    __scaler.unscale_ = unittest.mock.MagicMock()
    __scaler.step = unittest.mock.MagicMock()
    __scaler.update = unittest.mock.MagicMock()

    # student model that returns a tensor of the right shape
    __student = unittest.mock.MagicMock()

    return _trainer.PrefixTrainer(
        text_tok=unittest.mock.MagicMock(),
        byte_tok=unittest.mock.MagicMock(),
        teacher_mod=unittest.mock.MagicMock(),
        student_mod=__student,
        optimizer_obj=__optimizer,
        scheduler_obj=None,
        scaler_obj=__scaler,
        context_obj=contextlib.nullcontext(),
        callbacks_arr=callbacks_arr or [],
        batch_cfg=cfg['batch'],
        loss_cfg=cfg['loss'],
        gradient_cfg=cfg['gradient'],
        training_cfg=cfg['training'],
        logging_cfg=cfg['logging'],
        saving_cfg=cfg['saving'],
        testing_cfg=cfg['testing'],
    )


# INIT_STATE ###################################################################

class TestInitState:

    def test_returns_dict_with_tensors_and_scalars(self):
        __t = _make_trainer()
        __state = __t.init_state()
        assert 'tensors' in __state
        assert 'scalars' in __state

    def test_tensors_empty_by_default(self):
        __t = _make_trainer()
        __state = __t.init_state()
        assert __state['tensors'] == {}

    def test_scalars_has_switch_train(self):
        __t = _make_trainer()
        __state = __t.init_state()
        assert __state['scalars']['switch/train'] == 1

    def test_scalars_has_switch_grad_zero(self):
        __t = _make_trainer()
        __state = __t.init_state()
        assert __state['scalars']['switch/grad'] == 0

    def test_scalars_has_epoch_total_from_config(self):
        __t = _make_trainer(epoch_num=7)
        __state = __t.init_state()
        assert __state['scalars']['epoch/total'] == 7

    def test_scalars_has_step_global_one(self):
        __t = _make_trainer()
        __state = __t.init_state()
        assert __state['scalars']['step/global'] == 1

    def test_scalars_has_loss_keys(self):
        __t = _make_trainer()
        __s = __t.init_state()['scalars']
        for __k in ['loss/total', 'loss/mse/0', 'loss/mse/k', 'loss/cos/0', 'loss/cos/k']:
            assert __k in __s

    def test_override_replaces_scalars(self):
        __t = _make_trainer()
        __state = __t.init_state({'scalars': {'step/global': 99}})
        assert __state['scalars']['step/global'] == 99

    def test_override_tensors_dict(self):
        __t = _make_trainer()
        __dummy = torch.zeros(2)
        __state = __t.init_state({'tensors': {'inputs/mask': __dummy}})
        assert 'inputs/mask' in __state['tensors']


# INIT_STEP ####################################################################

class TestInitStep:

    def _trainer_with_steps(self, step_tot: int = 10, **kwargs) -> _trainer.PrefixTrainer:
        __t = _make_trainer(**kwargs)
        __t._state['scalars']['step/total'] = step_tot
        return __t

    def test_step_current_is_one_indexed(self):
        __t = self._trainer_with_steps()
        __t.init_step(step_num=0)
        assert __t._state['scalars']['step/current'] == 1

    def test_step_current_increments(self):
        __t = self._trainer_with_steps()
        __t.init_step(step_num=4)
        assert __t._state['scalars']['step/current'] == 5

    def test_step_global_first_epoch(self):
        # epoch/current == 1 (epoch index 0)
        __t = self._trainer_with_steps(step_tot=10)
        __t._state['scalars']['epoch/current'] = 1
        __t.init_step(step_num=2)
        # global = (2+1) + 0 * 10 = 3
        assert __t._state['scalars']['step/global'] == 3

    def test_step_global_second_epoch(self):
        __t = self._trainer_with_steps(step_tot=10)
        __t._state['scalars']['epoch/current'] = 2
        __t.init_step(step_num=0)
        # global = 1 + 1 * 10 = 11
        assert __t._state['scalars']['step/global'] == 11

    def test_switch_grad_true_at_multiple(self):
        __t = self._trainer_with_steps(grad_every=2)
        __t.init_step(step_num=1)  # step/current = 2
        assert __t._state['scalars']['switch/grad'] == 1

    def test_switch_grad_false_at_non_multiple(self):
        __t = self._trainer_with_steps(grad_every=2)
        __t.init_step(step_num=0)  # step/current = 1
        assert __t._state['scalars']['switch/grad'] == 0

    def test_switch_log_false_when_disabled(self):
        __t = self._trainer_with_steps(log_every=0)
        __t.init_step(step_num=0)
        assert __t._state['scalars']['switch/log'] == 0

    def test_switch_log_true_at_cadence(self):
        __t = self._trainer_with_steps(log_every=5)
        __t.init_step(step_num=4)  # step/current = 5
        assert __t._state['scalars']['switch/log'] == 1

    def test_switch_save_false_when_disabled(self):
        __t = self._trainer_with_steps(save_every=0)
        __t.init_step(step_num=0)
        assert __t._state['scalars']['switch/save'] == 0

    def test_switch_train_true_when_test_disabled(self):
        __t = self._trainer_with_steps(test_every=0)
        __t.init_step(step_num=0)
        assert __t._state['scalars']['switch/train'] == 1

    def test_switch_train_false_at_test_cadence(self):
        __t = self._trainer_with_steps(test_every=3)
        __t.init_step(step_num=2)  # step/current = 3
        assert __t._state['scalars']['switch/train'] == 0


# STEP_BATCH ###################################################################

class TestStepBatch:

    def _trainer_with_mock_processors(self, monkeypatch):
        __t = _make_trainer()
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
        __t, _ = self._trainer_with_mock_processors(monkeypatch)
        __called = []

        def __stub_strings(**kwargs):
            __called.append(True)
            return (torch.ones(2, 4), torch.zeros(2, 4), torch.zeros(2, 4, 2))

        monkeypatch.setattr(
            'deformers.pipelines.prefix.processors.vectorize_strings',
            __stub_strings)
        __batch = {'text': ['hello', 'world']}
        __t.step_batch(batch_arr=__batch, column_str='text')
        assert __called

    def test_routes_to_vectorize_indices_for_indice_column(self, monkeypatch):
        __t, _ = self._trainer_with_mock_processors(monkeypatch)
        __called = []

        def __stub_indices(**kwargs):
            __called.append(True)
            return (torch.ones(2, 4), torch.zeros(2, 4), torch.zeros(2, 4, 2))

        monkeypatch.setattr(
            'deformers.pipelines.prefix.processors.vectorize_indices',
            __stub_indices)
        __batch = {'indices': [[1, 2, 3, 4], [5, 6, 7, 8]]}
        __t.step_batch(batch_arr=__batch, column_str='indices')
        assert __called

    def test_stores_inputs_mask(self, monkeypatch):
        __t, __result = self._trainer_with_mock_processors(monkeypatch)
        __batch = {'text': ['a', 'b']}
        __t.step_batch(batch_arr=__batch, column_str='text')
        assert 'inputs/mask' in __t._state['tensors']

    def test_stores_inputs_indices(self, monkeypatch):
        __t, __result = self._trainer_with_mock_processors(monkeypatch)
        __batch = {'text': ['a', 'b']}
        __t.step_batch(batch_arr=__batch, column_str='text')
        assert 'inputs/indices' in __t._state['tensors']

    def test_stores_inputs_bytes(self, monkeypatch):
        __t, __result = self._trainer_with_mock_processors(monkeypatch)
        __batch = {'text': ['a', 'b']}
        __t.step_batch(batch_arr=__batch, column_str='text')
        assert 'inputs/bytes' in __t._state['tensors']


# STEP_FORWARD #################################################################

class TestStepForward:

    def _setup(self, monkeypatch, hidden: bool = False):
        __cfg_loss = {
            'mse_k_rate': 1.0 if hidden else 0.0,
            'cos_k_rate': 0.0,}
        __t = _make_trainer()
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
        __t.step_forward()
        assert 'outputs/teacher/0' in __t._state['tensors']

    def test_populates_student_0(self, monkeypatch):
        __t, _ = self._setup(monkeypatch)
        __t.step_forward()
        assert 'outputs/student/0' in __t._state['tensors']

    def test_skips_hidden_forward_when_disabled(self, monkeypatch):
        __t, _ = self._setup(monkeypatch, hidden=False)
        __forward_called = []
        monkeypatch.setattr(
            'deformers.pipelines.prefix.processors.forward',
            lambda **kwargs: (__forward_called.append(True) or torch.zeros(2, 4, 8)))
        __t.step_forward()
        assert not __forward_called

    def test_calls_hidden_forward_when_enabled(self, monkeypatch):
        __t, _ = self._setup(monkeypatch, hidden=True)
        __forward_called = []

        def __stub(**kwargs):
            __forward_called.append(True)
            return torch.zeros(2, 4, 8)

        monkeypatch.setattr('deformers.pipelines.prefix.processors.forward', __stub)
        __t.step_forward()
        assert len(__forward_called) >= 1

    def test_teacher_k_is_zeros_when_hidden_disabled(self, monkeypatch):
        __t, __embed = self._setup(monkeypatch, hidden=False)
        __t.step_forward()
        __tk = __t._state['tensors']['outputs/teacher/k']
        assert torch.all(__tk == 0)


# STEP_LOSSES ##################################################################

class TestStepLosses:

    def _setup(self, monkeypatch):
        __t = _make_trainer()
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
        __t.step_losses()
        assert 'loss/total' in __t._state['tensors']
        assert hasattr(__t._state['tensors']['loss/total'], 'shape')

    def test_accumulates_scalar_total(self, monkeypatch):
        __t = self._setup(monkeypatch)
        __t.step_losses()
        assert __t._state['scalars']['loss/total'] > 0.0

    def test_accumulates_mse_0(self, monkeypatch):
        __t = self._setup(monkeypatch)
        __t.step_losses()
        assert __t._state['scalars']['loss/mse/0'] > 0.0

    def test_scalar_loss_is_detached(self, monkeypatch):
        __t = self._setup(monkeypatch)
        __t.step_losses()
        # the scalars should be plain floats, not tensors
        assert isinstance(__t._state['scalars']['loss/total'], float)


# STEP_BACKWARD ################################################################

class TestStepBackward:

    def test_raises_when_no_tensor_loss(self):
        __t = _make_trainer()
        __t._state['tensors'] = {}
        with pytest.raises(AssertionError):
            __t.step_backward()

    def test_raises_when_tensor_loss_is_scalar(self):
        __t = _make_trainer()
        # store a plain float, not a tensor
        __t._state['tensors']['loss/total'] = 1.23
        with pytest.raises(AssertionError):
            __t.step_backward()

    def test_calls_scaler_scale_backward(self):
        __t = _make_trainer()
        __loss = torch.tensor(0.5, requires_grad=True)
        __t._state['tensors']['loss/total'] = __loss
        # backward on a leaf requires grad; scaler.scale returns the loss itself
        __t._scaler.scale = lambda x: x
        __t.step_backward()
        # no exception means backward was called; grad should be set on __loss
        # (since __loss is a leaf with grad fn after * 1)
        # Just verify no error was raised


# STEP_OPTIMIZER ###############################################################

class TestStepOptimizer:

    def test_does_not_step_when_switch_grad_off(self):
        __t = _make_trainer(grad_every=2)
        __t._state['scalars']['switch/grad'] = 0
        __t.step_optimizer()
        __t._scaler.step.assert_not_called()

    def test_steps_optimizer_when_switch_grad_on(self):
        __t = _make_trainer()
        __t._state['scalars']['switch/grad'] = 1
        __t.step_optimizer()
        __t._scaler.step.assert_called_once()

    def test_calls_scaler_update_when_switch_grad_on(self):
        __t = _make_trainer()
        __t._state['scalars']['switch/grad'] = 1
        __t.step_optimizer()
        __t._scaler.update.assert_called_once()

    def test_steps_scheduler_when_present(self):
        __t = _make_trainer()
        __t._state['scalars']['switch/grad'] = 1
        __scheduler = unittest.mock.MagicMock()
        __t._scheduler = __scheduler
        __t.step_optimizer()
        __scheduler.step.assert_called_once()

    def test_no_scheduler_step_when_none(self):
        __t = _make_trainer()
        __t._state['scalars']['switch/grad'] = 1
        __t._scheduler = None
        # should not raise
        __t.step_optimizer()


# CLOSE_STEP ###################################################################

class TestCloseStep:

    def _trainer_with_tensors(self) -> _trainer.PrefixTrainer:
        __t = _make_trainer()
        __t._state['tensors']['loss/total'] = torch.tensor(1.0)
        __t._state['scalars']['loss/total'] = 2.0
        __t._state['scalars']['loss/mse/0'] = 0.5
        return __t

    def test_resets_tensors_on_grad_boundary(self):
        __t = self._trainer_with_tensors()
        __t._state['scalars']['switch/grad'] = 1
        __t.close_step()
        assert __t._state['tensors'] == {}

    def test_resets_loss_scalars_on_grad_boundary(self):
        __t = self._trainer_with_tensors()
        __t._state['scalars']['switch/grad'] = 1
        __t.close_step()
        assert __t._state['scalars']['loss/total'] == 0.0
        assert __t._state['scalars']['loss/mse/0'] == 0.0

    def test_no_reset_when_switch_grad_off(self):
        __t = self._trainer_with_tensors()
        __t._state['scalars']['switch/grad'] = 0
        __t.close_step()
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

    def _make_run_step_stub(self, trainer: _trainer.PrefixTrainer) -> list:
        """Patch run_step and close_step to be no-ops; return call log."""
        __calls = []

        def __fake_run_step(batch_arr, column_str):
            __calls.append(batch_arr)

        trainer.run_step = __fake_run_step
        trainer.close_step = lambda: None
        trainer.step_progress = lambda pbar: None
        return __calls

    def test_iterates_over_all_batches(self):
        __t = _make_trainer()
        __batches = [{'text': [f'sample{i}']} for i in range(3)]
        __ds = _FakeDataset(__batches)
        __calls = self._make_run_step_stub(__t)
        __t.run_epoch(epoch_num=0, epoch_tot=1, dataset_obj=__ds, column_str='text')
        assert len(__calls) == 3

    def test_updates_epoch_current(self):
        __t = _make_trainer()
        __ds = _FakeDataset([{'text': ['x']}])
        self._make_run_step_stub(__t)
        __t.run_epoch(epoch_num=2, epoch_tot=5, dataset_obj=__ds, column_str='text')
        assert __t._state['scalars']['epoch/current'] == 3

    def test_updates_step_total_from_dataset_len(self):
        __t = _make_trainer()
        __batches = [{'text': ['a']}, {'text': ['b']}]
        __ds = _FakeDataset(__batches)
        self._make_run_step_stub(__t)
        __t.run_epoch(epoch_num=0, epoch_tot=1, dataset_obj=__ds, column_str='text')
        assert __t._state['scalars']['step/total'] == 2


# RUN_PHASE ####################################################################

class TestRunPhase:

    def test_calls_run_epoch_for_each_epoch(self):
        __t = _make_trainer(epoch_num=3)
        __epoch_calls = []

        def __fake_run_epoch(epoch_num, epoch_tot, dataset_obj, column_str):
            __epoch_calls.append(epoch_num)

        __t.run_epoch = __fake_run_epoch
        __ds = _FakeDataset([{'text': ['x']}])
        __t.run_phase(epoch_tot=3, dataset_obj=__ds, column_str='text')
        assert __epoch_calls == [0, 1, 2]

    def test_passes_column_str_to_run_epoch(self):
        __t = _make_trainer()
        __col_seen = []

        def __fake_run_epoch(epoch_num, epoch_tot, dataset_obj, column_str):
            __col_seen.append(column_str)

        __t.run_epoch = __fake_run_epoch
        __ds = _FakeDataset([{'text': ['x']}])
        __t.run_phase(epoch_tot=2, dataset_obj=__ds, column_str='my_column')
        assert all(__c == 'my_column' for __c in __col_seen)

    def test_passes_correct_epoch_tot(self):
        __t = _make_trainer()
        __epoch_tots = []

        def __fake_run_epoch(epoch_num, epoch_tot, dataset_obj, column_str):
            __epoch_tots.append(epoch_tot)

        __t.run_epoch = __fake_run_epoch
        __ds = _FakeDataset([])
        __t.run_phase(epoch_tot=4, dataset_obj=__ds, column_str='text')
        assert all(__e == 4 for __e in __epoch_tots)
