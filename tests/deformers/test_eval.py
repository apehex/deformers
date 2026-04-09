"""
Unit tests for deformers.eval metric helpers.

Covers:
- top1_match_rate: correct and incorrect prediction cases
- topk_set_match_rate: identical sets, disjoint sets, partial overlap
- topk_order_match_rate: identical order, same set different order
- kl_divergence: identical logits (zero KL), shape mismatch assertion,
  reduction behavior (batchmean vs sum), finite output
- embed_mse / hidden_mse: zero error on identical tensors, positive on different
- build_vocab_probe: shape, determinism, values in [0, vocab_size)
"""

import math
import pytest
import torch

import deformers.eval


# FIXTURES #####################################################################

_B, _T, _V = 2, 8, 64  # small synthetic shapes
_K = 5


def _logits_from_ranks(ranks: list, vocab_size: int, shape: tuple) -> torch.Tensor:
    """
    Build (B, T, V) logits where position 0 has a known top-k ordering.

    All other positions get uniform logits.
    """
    __logits = torch.zeros(shape)
    # assign descending scores to the requested token ids at all positions
    for __rank, __tok in enumerate(ranks):
        __logits[:, :, __tok] = float(len(ranks) - __rank)
    return __logits


# EMBED_MSE / HIDDEN_MSE #######################################################

class TestEmbedMse:

    def test_zero_error_on_identical_tensors(self):
        __x = torch.randn(_B, _T, 32)
        assert deformers.eval.embed_mse(__x, __x) == pytest.approx(0.0, abs=1e-6)

    def test_positive_on_different_tensors(self):
        __x = torch.zeros(_B, _T, 32)
        __y = torch.ones(_B, _T, 32)
        assert deformers.eval.embed_mse(__x, __y) > 0.0

    def test_returns_float(self):
        __x = torch.randn(_B, _T, 32)
        __y = torch.randn(_B, _T, 32)
        assert isinstance(deformers.eval.embed_mse(__x, __y), float)


class TestHiddenMse:

    def test_zero_error_on_identical_tensors(self):
        __x = torch.randn(_B, _T, 32)
        assert deformers.eval.hidden_mse(__x, __x) == pytest.approx(0.0, abs=1e-6)

    def test_positive_on_different_tensors(self):
        __x = torch.zeros(_B, _T, 32)
        __y = torch.ones(_B, _T, 32)
        assert deformers.eval.hidden_mse(__x, __y) > 0.0


# KL_DIVERGENCE ################################################################

class TestKlDivergence:

    def test_zero_on_identical_logits(self):
        __x = torch.randn(_B, _T, _V)
        __kl = deformers.eval.kl_divergence(__x, __x)
        assert __kl == pytest.approx(0.0, abs=1e-4)

    def test_positive_on_different_logits(self):
        __t = torch.zeros(_B, _T, _V)
        __s = torch.zeros(_B, _T, _V)
        __t[:, :, 0] = 10.0  # teacher strongly prefers token 0
        __s[:, :, 1] = 10.0  # student strongly prefers token 1
        assert deformers.eval.kl_divergence(__t, __s) > 0.0

    def test_returns_float(self):
        __x = torch.randn(_B, _T, _V)
        assert isinstance(deformers.eval.kl_divergence(__x, __x), float)

    def test_shape_mismatch_raises(self):
        __t = torch.randn(_B, _T, _V)
        __s = torch.randn(_B, _T, _V + 1)
        with pytest.raises(AssertionError):
            deformers.eval.kl_divergence(__t, __s)

    def test_finite_output(self):
        __t = torch.randn(_B, _T, _V)
        __s = torch.randn(_B, _T, _V)
        __kl = deformers.eval.kl_divergence(__t, __s)
        assert math.isfinite(__kl)

    def test_reduction_sum_differs_from_batchmean(self):
        __t = torch.randn(_B, _T, _V)
        __s = torch.randn(_B, _T, _V)
        __kl_bm = deformers.eval.kl_divergence(__t, __s, reduction='batchmean')
        __kl_sum = deformers.eval.kl_divergence(__t, __s, reduction='sum')
        # sum >= batchmean when batch > 1
        assert __kl_sum >= __kl_bm


# TOP1_MATCH_RATE ##############################################################

class TestTop1MatchRate:

    def test_rate_one_on_identical_logits(self):
        __x = torch.randn(_B, _T, _V)
        assert deformers.eval.top1_match_rate(__x, __x) == pytest.approx(1.0)

    def test_rate_zero_when_all_mismatch(self):
        # teacher always picks token 0, student always picks token 1
        __t = torch.zeros(_B, _T, _V)
        __t[:, :, 0] = 1.0
        __s = torch.zeros(_B, _T, _V)
        __s[:, :, 1] = 1.0
        assert deformers.eval.top1_match_rate(__t, __s) == pytest.approx(0.0)

    def test_partial_match(self):
        # B=1, T=4: first 2 positions match, last 2 do not
        __t = torch.zeros(1, 4, _V)
        __s = torch.zeros(1, 4, _V)
        __t[:, :, 0] = 1.0          # teacher always picks 0
        __s[:, :2, 0] = 1.0         # student picks 0 for first 2
        __s[:, 2:, 1] = 1.0         # student picks 1 for last 2
        assert deformers.eval.top1_match_rate(__t, __s) == pytest.approx(0.5)

    def test_returns_float(self):
        __x = torch.randn(_B, _T, _V)
        assert isinstance(deformers.eval.top1_match_rate(__x, __x), float)

    def test_value_in_unit_interval(self):
        __t = torch.randn(_B, _T, _V)
        __s = torch.randn(_B, _T, _V)
        __r = deformers.eval.top1_match_rate(__t, __s)
        assert 0.0 <= __r <= 1.0


# TOPK_SET_MATCH_RATE ##########################################################

class TestTopkSetMatchRate:

    def test_rate_one_on_identical_logits(self):
        __x = torch.randn(_B, _T, _V)
        assert deformers.eval.topk_set_match_rate(__x, __x, k=_K) == pytest.approx(1.0)

    def test_rate_zero_when_sets_disjoint(self):
        # teacher top-k = [0..k-1], student top-k = [k..2k-1]
        __t = torch.zeros(_B, _T, _V)
        __s = torch.zeros(_B, _T, _V)
        for __i in range(_K):
            __t[:, :, __i] = float(_K - __i)
            __s[:, :, _K + __i] = float(_K - __i)
        assert deformers.eval.topk_set_match_rate(__t, __s, k=_K) == pytest.approx(0.0)

    def test_ignores_order_within_set(self):
        # same top-k tokens but in reverse order -> should still be 1.0
        __tokens = list(range(_K))
        __t = _logits_from_ranks(__tokens, _V, (_B, _T, _V))
        __s = _logits_from_ranks(__tokens[::-1], _V, (_B, _T, _V))
        assert deformers.eval.topk_set_match_rate(__t, __s, k=_K) == pytest.approx(1.0)

    def test_returns_float(self):
        __x = torch.randn(_B, _T, _V)
        assert isinstance(deformers.eval.topk_set_match_rate(__x, __x, k=_K), float)

    def test_value_in_unit_interval(self):
        __t = torch.randn(_B, _T, _V)
        __s = torch.randn(_B, _T, _V)
        __r = deformers.eval.topk_set_match_rate(__t, __s, k=_K)
        assert 0.0 <= __r <= 1.0

    def test_k_equals_one_matches_top1_rate(self):
        __t = torch.randn(_B, _T, _V)
        __s = torch.randn(_B, _T, _V)
        __set_rate = deformers.eval.topk_set_match_rate(__t, __s, k=1)
        __top1_rate = deformers.eval.top1_match_rate(__t, __s)
        assert __set_rate == pytest.approx(__top1_rate, abs=1e-6)


# TOPK_ORDER_MATCH_RATE ########################################################

class TestTopkOrderMatchRate:

    def test_rate_one_on_identical_logits(self):
        __x = torch.randn(_B, _T, _V)
        assert deformers.eval.topk_order_match_rate(__x, __x, k=_K) == pytest.approx(1.0)

    def test_rate_zero_when_sets_disjoint(self):
        __t = torch.zeros(_B, _T, _V)
        __s = torch.zeros(_B, _T, _V)
        for __i in range(_K):
            __t[:, :, __i] = float(_K - __i)
            __s[:, :, _K + __i] = float(_K - __i)
        assert deformers.eval.topk_order_match_rate(__t, __s, k=_K) == pytest.approx(0.0)

    def test_same_set_different_order_is_zero(self):
        # same top-k tokens but in reverse order -> exact order match fails
        __tokens = list(range(_K))
        __t = _logits_from_ranks(__tokens, _V, (_B, _T, _V))
        __s = _logits_from_ranks(__tokens[::-1], _V, (_B, _T, _V))
        # only matches if the reversed order happens to be the same (k=1 edge case excluded)
        if _K > 1:
            assert deformers.eval.topk_order_match_rate(__t, __s, k=_K) == pytest.approx(0.0)

    def test_returns_float(self):
        __x = torch.randn(_B, _T, _V)
        assert isinstance(deformers.eval.topk_order_match_rate(__x, __x, k=_K), float)

    def test_value_in_unit_interval(self):
        __t = torch.randn(_B, _T, _V)
        __s = torch.randn(_B, _T, _V)
        __r = deformers.eval.topk_order_match_rate(__t, __s, k=_K)
        assert 0.0 <= __r <= 1.0

    def test_order_rate_leq_set_rate(self):
        # exact order is at most as frequent as set match
        __t = torch.randn(_B, _T, _V)
        __s = torch.randn(_B, _T, _V)
        __order = deformers.eval.topk_order_match_rate(__t, __s, k=_K)
        __set = deformers.eval.topk_set_match_rate(__t, __s, k=_K)
        assert __order <= __set + 1e-6

    def test_k_equals_one_matches_top1_rate(self):
        __t = torch.randn(_B, _T, _V)
        __s = torch.randn(_B, _T, _V)
        __order_rate = deformers.eval.topk_order_match_rate(__t, __s, k=1)
        __top1_rate = deformers.eval.top1_match_rate(__t, __s)
        assert __order_rate == pytest.approx(__top1_rate, abs=1e-6)


# BUILD_VOCAB_PROBE ############################################################

class TestBuildVocabProbe:

    def test_output_shape(self):
        __ids = deformers.eval.build_vocab_probe(vocab_size=100, batch_dim=_B, seq_dim=_T)
        assert __ids.shape == (_B, _T)

    def test_values_in_range(self):
        __vocab = 50
        __ids = deformers.eval.build_vocab_probe(vocab_size=__vocab, batch_dim=_B, seq_dim=_T)
        assert __ids.min().item() >= 0
        assert __ids.max().item() < __vocab

    def test_deterministic(self):
        __a = deformers.eval.build_vocab_probe(vocab_size=100, batch_dim=_B, seq_dim=_T)
        __b = deformers.eval.build_vocab_probe(vocab_size=100, batch_dim=_B, seq_dim=_T)
        assert torch.equal(__a, __b)

    def test_dtype_is_long(self):
        __ids = deformers.eval.build_vocab_probe(vocab_size=100, batch_dim=_B, seq_dim=_T)
        assert __ids.dtype == torch.long

    def test_sequential_fill(self):
        # first B*T values should be 0, 1, 2, ... mod vocab_size
        __vocab = 1000
        __ids = deformers.eval.build_vocab_probe(vocab_size=__vocab, batch_dim=_B, seq_dim=_T)
        __expected = torch.arange(_B * _T, dtype=torch.long).reshape(_B, _T)
        assert torch.equal(__ids, __expected)
