"""
Unit tests for deformers.pipelines.eval metric helpers.

Covers:
- top1_match_rate: correct and incorrect prediction cases
- topk_rate: identical order, same set different order
- kl_divergence: identical logits (zero KL), finite output
- build_vocab_probe: shape, determinism, values in [0, vocab_dim)
"""

import math
import pytest
import torch

import deformers.pipelines.eval

# META #########################################################################

_B, _T, _V = 2, 8, 64  # small synthetic shapes
_K = 5

# FIXTURES #####################################################################

def _logits_from_ranks(ranks: list, vocab_dim: int, shape: tuple) -> torch.Tensor:
    """Build (B, T, V) logits where position 0 has a known top-k ordering."""
    __logits = torch.zeros(shape)
    # assign descending scores to the requested token ids at all positions
    for __rank, __tok in enumerate(ranks):
        __logits[:, :, __tok] = float(len(ranks) - __rank)
    return __logits

# KL_DIVERGENCE ################################################################

class TestKlDivergence:

    def test_zero_on_identical_logits(self):
        __x = torch.randn(_B, _T, _V)
        __kl = deformers.pipelines.eval.kl_divergence(__x, __x).item()
        assert __kl == pytest.approx(0.0, abs=1e-4)

    def test_positive_on_different_logits(self):
        __t = torch.zeros(_B, _T, _V)
        __s = torch.zeros(_B, _T, _V)
        __t[:, :, 0] = 10.0  # teacher strongly prefers token 0
        __s[:, :, 1] = 10.0  # student strongly prefers token 1
        assert deformers.pipelines.eval.kl_divergence(__t, __s).item() > 0.0

    def test_returns_float(self):
        __x = torch.randn(_B, _T, _V)
        assert isinstance(deformers.pipelines.eval.kl_divergence(__x, __x), torch.Tensor)

    def test_finite_output(self):
        __t = torch.randn(_B, _T, _V)
        __s = torch.randn(_B, _T, _V)
        __kl = deformers.pipelines.eval.kl_divergence(__t, __s).item()
        assert math.isfinite(__kl)

# TOP1_MATCH_RATE ##############################################################

class TestTop1MatchRate:

    def test_rate_one_on_identical_logits(self):
        __x = torch.randn(_B, _T, _V)
        assert deformers.pipelines.eval.topk_rate(__x, __x, 1) == pytest.approx(1.0)

    def test_rate_zero_when_all_mismatch(self):
        # teacher always picks token 0, student always picks token 1
        __t = torch.zeros(_B, _T, _V)
        __t[:, :, 0] = 1.0
        __s = torch.zeros(_B, _T, _V)
        __s[:, :, 1] = 1.0
        assert deformers.pipelines.eval.topk_rate(__t, __s, 1) == pytest.approx(0.0)

    def test_partial_match(self):
        # B=1, T=4: first 2 positions match, last 2 do not
        __t = torch.zeros(1, 4, _V)
        __s = torch.zeros(1, 4, _V)
        __t[:, :, 0] = 1.0          # teacher always picks 0
        __s[:, :2, 0] = 1.0         # student picks 0 for first 2
        __s[:, 2:, 1] = 1.0         # student picks 1 for last 2
        assert deformers.pipelines.eval.topk_rate(__t, __s, 1) == pytest.approx(0.5)

    def test_returns_float(self):
        __x = torch.randn(_B, _T, _V)
        assert isinstance(deformers.pipelines.eval.topk_rate(__x, __x, 1), float)

    def test_value_in_unit_interval(self):
        __t = torch.randn(_B, _T, _V)
        __s = torch.randn(_B, _T, _V)
        __r = deformers.pipelines.eval.topk_rate(__t, __s, 1)
        assert 0.0 <= __r <= 1.0

# TOPK MATCH RATE ##############################################################

class TestTopkOrderMatchRate:

    def test_rate_one_on_identical_logits(self):
        __x = torch.randn(_B, _T, _V)
        assert deformers.pipelines.eval.topk_rate(__x, __x, k_num=_K) == pytest.approx(1.0)

    def test_rate_zero_when_sets_disjoint(self):
        __t = torch.zeros(_B, _T, _V)
        __s = torch.zeros(_B, _T, _V)
        for __i in range(_K):
            __t[:, :, __i] = float(_K - __i)
            __s[:, :, _K + __i] = float(_K - __i)
        assert deformers.pipelines.eval.topk_rate(__t, __s, k_num=_K) == pytest.approx(0.0)

    def test_same_set_different_order_is_zero(self):
        # same top-k tokens but in reverse order -> exact order match fails
        __tokens = list(range(_K))
        __t = _logits_from_ranks(__tokens, _V, (_B, _T, _V))
        __s = _logits_from_ranks(__tokens[::-1], _V, (_B, _T, _V))
        # only matches if the reversed order happens to be the same (k_num=1 edge case excluded)
        if _K > 1:
            assert deformers.pipelines.eval.topk_rate(__t, __s, k_num=_K) == pytest.approx(0.0)

    def test_returns_float(self):
        __x = torch.randn(_B, _T, _V)
        assert isinstance(deformers.pipelines.eval.topk_rate(__x, __x, k_num=_K), float)

    def test_value_in_unit_interval(self):
        __t = torch.randn(_B, _T, _V)
        __s = torch.randn(_B, _T, _V)
        __r = deformers.pipelines.eval.topk_rate(__t, __s, k_num=_K)
        assert 0.0 <= __r <= 1.0

# BUILD_VOCAB_PROBE ############################################################

class TestBuildVocabProbe:

    def test_output_shape(self):
        __ids = deformers.pipelines.eval.build_vocab_probe(vocab_dim=100, batch_dim=_B, seq_dim=_T)
        assert __ids.shape == (_B, _T)

    def test_values_in_range(self):
        __vocab = 50
        __ids = deformers.pipelines.eval.build_vocab_probe(vocab_dim=__vocab, batch_dim=_B, seq_dim=_T)
        assert __ids.min().item() >= 0
        assert __ids.max().item() < __vocab

    def test_deterministic(self):
        __a = deformers.pipelines.eval.build_vocab_probe(vocab_dim=100, batch_dim=_B, seq_dim=_T)
        __b = deformers.pipelines.eval.build_vocab_probe(vocab_dim=100, batch_dim=_B, seq_dim=_T)
        assert torch.equal(__a, __b)

    def test_dtype_is_long(self):
        __ids = deformers.pipelines.eval.build_vocab_probe(vocab_dim=100, batch_dim=_B, seq_dim=_T)
        assert __ids.dtype == torch.long

    def test_sequential_fill(self):
        # first B*T values should be 0, 1, 2, ... mod vocab_dim
        __vocab = 1000
        __ids = deformers.pipelines.eval.build_vocab_probe(vocab_dim=__vocab, batch_dim=_B, seq_dim=_T)
        __expected = torch.arange(_B * _T, dtype=torch.long).reshape(_B, _T)
        assert torch.equal(__ids, __expected)
