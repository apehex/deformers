"""
Unit tests for deformers.pipelines.monitor utilities.

Covers:
- gpu_memory_mb: CPU environment returns zeros with correct keys.
- current_lr: reads LR from optimizer param groups.
- throughput: correct rate, edge cases (zero elapsed, negative elapsed).
- log_scalars: no-op when writer is None; calls add_scalar when writer provided.
"""

import pytest
import torch
import torch.optim

import deformers.pipelines.monitor

# GPU_MEMORY_MB ################################################################

class TestGpuMemoryMb:

    def test_returns_dict_with_required_keys(self):
        __result = deformers.pipelines.monitor.gpu_memory_mb()
        assert 'allocated_mb' in __result
        assert 'reserved_mb' in __result

    def test_cpu_environment_returns_zeros(self):
        # on CPU-only test runners, CUDA is not available
        if torch.cuda.is_available():
            pytest.skip('CUDA available: CPU-zero test not applicable')
        __result = deformers.pipelines.monitor.gpu_memory_mb()
        assert __result['allocated_mb'] == 0.0
        assert __result['reserved_mb'] == 0.0

    def test_values_are_floats(self):
        __result = deformers.pipelines.monitor.gpu_memory_mb()
        assert isinstance(__result['allocated_mb'], float)
        assert isinstance(__result['reserved_mb'], float)

    def test_values_are_non_negative(self):
        __result = deformers.pipelines.monitor.gpu_memory_mb()
        assert __result['allocated_mb'] >= 0.0
        assert __result['reserved_mb'] >= 0.0

# CURRENT_LR ###################################################################

class TestCurrentLr:

    def _make_optimizer(self, lr: float) -> torch.optim.Optimizer:
        __param = torch.nn.Parameter(torch.zeros(1))
        return torch.optim.SGD([__param], lr=lr)

    def test_reads_initial_lr(self):
        __opt = self._make_optimizer(1e-3)
        assert deformers.pipelines.monitor.current_lr(__opt) == pytest.approx(1e-3)

    def test_returns_float(self):
        __opt = self._make_optimizer(3e-4)
        assert isinstance(deformers.pipelines.monitor.current_lr(__opt), float)

    def test_reflects_lr_update(self):
        __opt = self._make_optimizer(1e-3)
        # manually change LR
        __opt.param_groups[0]['lr'] = 1e-4
        assert deformers.pipelines.monitor.current_lr(__opt) == pytest.approx(1e-4)

    def test_small_lr(self):
        __opt = self._make_optimizer(1e-9)
        assert deformers.pipelines.monitor.current_lr(__opt) == pytest.approx(1e-9)

# THROUGHPUT ###################################################################

class TestThroughput:

    def test_correct_rate(self):
        assert deformers.pipelines.monitor.throughput(1000, 1.0) == pytest.approx(1000.0)

    def test_returns_float(self):
        assert isinstance(deformers.pipelines.monitor.throughput(100, 0.5), float)

    def test_zero_elapsed_returns_zero(self):
        assert deformers.pipelines.monitor.throughput(1000, 0.0) == 0.0

    def test_negative_elapsed_returns_zero(self):
        assert deformers.pipelines.monitor.throughput(1000, -1.0) == 0.0

    def test_zero_count(self):
        assert deformers.pipelines.monitor.throughput(0, 1.0) == pytest.approx(0.0)

    def test_fractional_elapsed(self):
        assert deformers.pipelines.monitor.throughput(500, 0.5) == pytest.approx(1000.0)

# LOG_SCALARS ##################################################################

class _FakeWriter:
    """Minimal SummaryWriter stub that records add_scalar calls."""

    def __init__(self):
        self.calls = []

    def add_scalar(self, tag, value, step):
        self.calls.append((tag, value, step))

class TestLogScalars:

    def test_noop_when_writer_is_none(self):
        # should not raise
        deformers.pipelines.monitor.log_scalars(None, {'a': 1.0, 'b': 2.0}, step=0)

    def test_calls_add_scalar_for_each_entry(self):
        __writer = _FakeWriter()
        deformers.pipelines.monitor.log_scalars(
            __writer,
            {'train/loss': 0.5, 'train/lr': 1e-3},
            step=10)
        assert len(__writer.calls) == 2

    def test_correct_tags_logged(self):
        __writer = _FakeWriter()
        deformers.pipelines.monitor.log_scalars(__writer, {'tag_a': 1.0}, step=0)
        assert __writer.calls[0][0] == 'tag_a'

    def test_correct_step_logged(self):
        __writer = _FakeWriter()
        deformers.pipelines.monitor.log_scalars(__writer, {'x': 0.0}, step=42)
        assert __writer.calls[0][2] == 42

    def test_values_cast_to_float(self):
        __writer = _FakeWriter()
        # pass an int value; should be stored as float
        deformers.pipelines.monitor.log_scalars(__writer, {'n': 7}, step=0)
        assert isinstance(__writer.calls[0][1], float)

    def test_empty_dict_no_calls(self):
        __writer = _FakeWriter()
        deformers.pipelines.monitor.log_scalars(__writer, {}, step=0)
        assert len(__writer.calls) == 0
