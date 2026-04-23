"""
Unit tests for deformers.pipelines.eval utilities.

Covers:
- indices_probe: deterministic construction, shape, range.
- masked_mse: correct masked reduction, mask edge cases.
- masked_cosine: correct masked reduction, mask edge cases.
- kl_divergence: non-negativity, mask edge cases, no-mask fallback.
- topk_rate: simple tensor cases, set and ordered modes.
- top1_rate: agreement on trivial inputs.
- per_token_metrics: shape, keys, sort order.
- aggregate_metrics: mean/median/p95 correctness, empty list.
- save_report: file created, valid JSON, correct stem in filename.
"""

import json
import os
import pytest
import torch

import deformers.pipelines.eval

# INDICES_PROBE ################################################################

class TestIndicesProbe:

    def test_shape_matches_batch_and_sequence(self):
        __result = deformers.pipelines.eval.indices_probe(
            vocab_dim=100, batch_dim=3, sequence_dim=8)
        assert len(__result) == 3
        assert all(len(__r) == 8 for __r in __result)

    def test_values_within_vocab_range(self):
        __vocab = 50
        __result = deformers.pipelines.eval.indices_probe(
            vocab_dim=__vocab, batch_dim=4, sequence_dim=16)
        for __row in __result:
            for __val in __row:
                assert 0 <= __val < __vocab

    def test_deterministic_across_calls(self):
        __a = deformers.pipelines.eval.indices_probe(vocab_dim=200, batch_dim=2, sequence_dim=4)
        __b = deformers.pipelines.eval.indices_probe(vocab_dim=200, batch_dim=2, sequence_dim=4)
        assert __a == __b

    def test_consecutive_ids_wrap_at_vocab_dim(self):
        # with vocab_dim=3, B=1, T=6: IDs should be [0,1,2,0,1,2]
        __result = deformers.pipelines.eval.indices_probe(vocab_dim=3, batch_dim=1, sequence_dim=6)
        assert __result[0] == [0, 1, 2, 0, 1, 2]

    def test_returns_list_of_lists(self):
        __result = deformers.pipelines.eval.indices_probe(vocab_dim=10, batch_dim=2, sequence_dim=3)
        assert isinstance(__result, list)
        assert all(isinstance(__r, list) for __r in __result)

# MASKED_MSE ###################################################################

class TestMaskedMse:

    def _tensors(self, B=2, T=4, H=8):
        __pred = torch.randn(B, T, H)
        __target = torch.randn(B, T, H)
        __mask = torch.ones(B, T, dtype=torch.long)
        return __pred, __target, __mask

    def test_returns_scalar(self):
        __pred, __target, __mask = self._tensors()
        __result = deformers.pipelines.eval.masked_mse(__pred, __target, __mask)
        assert __result.ndim == 0

    def test_zero_on_identical_tensors(self):
        __pred, _, __mask = self._tensors()
        __result = deformers.pipelines.eval.masked_mse(__pred, __pred, __mask)
        assert __result.item() == pytest.approx(0.0)

    def test_non_negative(self):
        __pred, __target, __mask = self._tensors()
        assert deformers.pipelines.eval.masked_mse(__pred, __target, __mask).item() >= 0.0

    def test_all_masked_out_returns_zero(self):
        __pred = torch.randn(2, 4, 8)
        __target = torch.randn(2, 4, 8)
        __mask = torch.zeros(2, 4, dtype=torch.long)
        __result = deformers.pipelines.eval.masked_mse(__pred, __target, __mask)
        # denominator clamped to 1, numerator is 0 -> result is 0
        assert __result.item() == pytest.approx(0.0)

    def test_partial_mask_ignores_masked_positions(self):
        # build two tensors that differ only at position (0, 2)
        __pred = torch.zeros(1, 4, 8)
        __target = torch.zeros(1, 4, 8)
        __target[0, 2, :] = 10.0  # large error at position 2
        # mask out position 2 -> MSE should be 0
        __mask = torch.ones(1, 4, dtype=torch.long)
        __mask[0, 2] = 0
        __result = deformers.pipelines.eval.masked_mse(__pred, __target, __mask)
        assert __result.item() == pytest.approx(0.0)

    def test_full_mask_includes_all_positions(self):
        __pred = torch.zeros(1, 4, 1)
        __target = torch.ones(1, 4, 1)   # MSE = 1 at every position
        __mask = torch.ones(1, 4, dtype=torch.long)
        __result = deformers.pipelines.eval.masked_mse(__pred, __target, __mask)
        assert __result.item() == pytest.approx(1.0)

# MASKED_COSINE ################################################################

class TestMaskedCosine:

    def test_returns_scalar(self):
        __pred = torch.randn(2, 4, 8)
        __target = torch.randn(2, 4, 8)
        __mask = torch.ones(2, 4, dtype=torch.long)
        __result = deformers.pipelines.eval.masked_cosine(__pred, __target, __mask)
        assert __result.ndim == 0

    def test_one_on_identical_tensors(self):
        __x = torch.randn(2, 4, 8)
        __mask = torch.ones(2, 4, dtype=torch.long)
        __result = deformers.pipelines.eval.masked_cosine(__x, __x, __mask)
        assert __result.item() == pytest.approx(1.0, abs=1e-5)

    def test_bounded_in_minus_one_to_one(self):
        __pred = torch.randn(2, 4, 8)
        __target = torch.randn(2, 4, 8)
        __mask = torch.ones(2, 4, dtype=torch.long)
        __val = deformers.pipelines.eval.masked_cosine(__pred, __target, __mask).item()
        assert -1.0 - 1e-5 <= __val <= 1.0 + 1e-5

    def test_all_masked_out_returns_zero(self):
        __pred = torch.randn(2, 4, 8)
        __target = torch.randn(2, 4, 8)
        __mask = torch.zeros(2, 4, dtype=torch.long)
        __result = deformers.pipelines.eval.masked_cosine(__pred, __target, __mask)
        assert __result.item() == pytest.approx(0.0)

# KL_DIVERGENCE ################################################################

class TestKlDivergence:

    def _logits(self, B=2, T=4, V=16):
        return torch.randn(B, T, V), torch.randn(B, T, V)

    def test_returns_scalar(self):
        __s, __t = self._logits()
        __mask = torch.ones(2, 4, dtype=torch.long)
        assert deformers.pipelines.eval.kl_divergence(__s, __t, __mask).ndim == 0

    def test_non_negative(self):
        __s, __t = self._logits()
        __mask = torch.ones(2, 4, dtype=torch.long)
        assert deformers.pipelines.eval.kl_divergence(__s, __t, __mask).item() >= 0.0

    def test_zero_when_distributions_match(self):
        __s = torch.randn(2, 4, 16)
        # student == teacher -> KL should be ~0
        __result = deformers.pipelines.eval.kl_divergence(__s, __s)
        assert __result.item() == pytest.approx(0.0, abs=1e-5)

    def test_no_mask_argument(self):
        __s, __t = self._logits()
        # should not raise and returns a non-negative scalar
        __result = deformers.pipelines.eval.kl_divergence(__s, __t)
        assert __result.item() >= 0.0

    def test_all_masked_out_returns_zero(self):
        __s, __t = self._logits()
        __mask = torch.zeros(2, 4, dtype=torch.long)
        __result = deformers.pipelines.eval.kl_divergence(__s, __t, __mask)
        assert __result.item() == pytest.approx(0.0)

    def test_partial_mask_reduces_valid_tokens(self):
        # use identical logits at all positions -> KL is 0 regardless of mask
        __x = torch.randn(2, 4, 8)
        __mask_full = torch.ones(2, 4, dtype=torch.long)
        __mask_half = __mask_full.clone()
        __mask_half[0, :] = 0
        __full = deformers.pipelines.eval.kl_divergence(__x, __x, __mask_full).item()
        __half = deformers.pipelines.eval.kl_divergence(__x, __x, __mask_half).item()
        assert __full == pytest.approx(0.0, abs=1e-5)
        assert __half == pytest.approx(0.0, abs=1e-5)

# TOPK_RATE ####################################################################

class TestTopkRate:

    def _identical_logits(self, B=1, T=4, V=8):
        __x = torch.randn(B, T, V)
        return __x, __x.clone()

    def test_returns_scalar(self):
        __s, __t = self._identical_logits()
        assert deformers.pipelines.eval.topk_rate(__s, __t).ndim == 0

    def test_one_on_identical_logits_set_mode(self):
        __s, __t = self._identical_logits()
        __result = deformers.pipelines.eval.topk_rate(__s, __t, k_num=3, ordered=False)
        assert __result.item() == pytest.approx(1.0)

    def test_one_on_identical_logits_ordered_mode(self):
        __s, __t = self._identical_logits()
        __result = deformers.pipelines.eval.topk_rate(__s, __t, k_num=3, ordered=True)
        assert __result.item() == pytest.approx(1.0)

    def test_bounded_in_zero_to_one(self):
        __s = torch.randn(2, 4, 16)
        __t = torch.randn(2, 4, 16)
        __mask = torch.ones(2, 4, dtype=torch.long)
        __val = deformers.pipelines.eval.topk_rate(__s, __t, __mask, k_num=5).item()
        assert 0.0 <= __val <= 1.0

    def test_no_mask_argument(self):
        __s, __t = self._identical_logits()
        # should not raise
        deformers.pipelines.eval.topk_rate(__s, __t, k_num=2)

    def test_all_masked_returns_zero(self):
        __s = torch.randn(1, 4, 8)
        __t = torch.randn(1, 4, 8)
        __mask = torch.zeros(1, 4, dtype=torch.long)
        __result = deformers.pipelines.eval.topk_rate(__s, __t, __mask, k_num=3)
        assert __result.item() == pytest.approx(0.0)

    def test_k1_set_equals_ordered(self):
        # for k=1, set mode and ordered mode should give the same result
        __s = torch.randn(2, 5, 10)
        __t = torch.randn(2, 5, 10)
        __mask = torch.ones(2, 5, dtype=torch.long)
        __set_r = deformers.pipelines.eval.topk_rate(__s, __t, __mask, k_num=1, ordered=False)
        __ord_r = deformers.pipelines.eval.topk_rate(__s, __t, __mask, k_num=1, ordered=True)
        assert __set_r.item() == pytest.approx(__ord_r.item())

    def test_ordered_stricter_than_set(self):
        # ordered mode can only be <= set mode
        __s = torch.randn(2, 6, 12)
        __t = torch.randn(2, 6, 12)
        __mask = torch.ones(2, 6, dtype=torch.long)
        __set_r = deformers.pipelines.eval.topk_rate(__s, __t, __mask, k_num=4, ordered=False)
        __ord_r = deformers.pipelines.eval.topk_rate(__s, __t, __mask, k_num=4, ordered=True)
        assert __ord_r.item() <= __set_r.item() + 1e-6

# TOP1_RATE ####################################################################

class TestTop1Rate:

    def test_one_on_identical_logits(self):
        __x = torch.randn(2, 4, 8)
        __mask = torch.ones(2, 4, dtype=torch.long)
        assert deformers.pipelines.eval.top1_rate(__x, __x, __mask).item() == pytest.approx(1.0)

    def test_matches_topk_rate_k1_ordered(self):
        __s = torch.randn(2, 5, 10)
        __t = torch.randn(2, 5, 10)
        __mask = torch.ones(2, 5, dtype=torch.long)
        __top1 = deformers.pipelines.eval.top1_rate(__s, __t, __mask).item()
        __topk = deformers.pipelines.eval.topk_rate(__s, __t, __mask, k_num=1, ordered=True).item()
        assert __top1 == pytest.approx(__topk)

    def test_known_mismatch_returns_zero(self):
        # student always argmax at index 0, teacher always at index 1
        __s = torch.zeros(1, 1, 4)
        __s[0, 0, 0] = 10.0   # student top-1 = 0
        __t = torch.zeros(1, 1, 4)
        __t[0, 0, 1] = 10.0   # teacher top-1 = 1
        __mask = torch.ones(1, 1, dtype=torch.long)
        assert deformers.pipelines.eval.top1_rate(__s, __t, __mask).item() == pytest.approx(0.0)

# PER_TOKEN_METRICS ############################################################

class TestPerTokenMetrics:

    def _build_inputs(self, B=2, T=4, H=8, V=16):
        __mask = torch.ones(B, T, dtype=torch.long)
        __ids = torch.arange(B * T).reshape(B, T)
        __se = torch.randn(B, T, H)
        __te = torch.randn(B, T, H)
        __sh = torch.randn(B, T, H)
        __th = torch.randn(B, T, H)
        __sl = torch.randn(B, T, V)
        __tl = torch.randn(B, T, V)
        return __ids, __se, __te, __sh, __th, __sl, __tl, __mask

    def test_length_matches_valid_tokens(self):
        __ids, __se, __te, __sh, __th, __sl, __tl, __mask = self._build_inputs(B=2, T=4)
        __result = deformers.pipelines.eval.per_token_metrics(
            __ids, __se, __te, __sh, __th, __sl, __tl, __mask)
        assert len(__result) == 8  # 2*4 all valid

    def test_masked_positions_excluded(self):
        __ids, __se, __te, __sh, __th, __sl, __tl, __mask = self._build_inputs(B=1, T=4)
        __mask[0, 2] = 0  # mask one position
        __result = deformers.pipelines.eval.per_token_metrics(
            __ids, __se, __te, __sh, __th, __sl, __tl, __mask)
        assert len(__result) == 3

    def test_required_keys_present(self):
        __ids, __se, __te, __sh, __th, __sl, __tl, __mask = self._build_inputs()
        __result = deformers.pipelines.eval.per_token_metrics(
            __ids, __se, __te, __sh, __th, __sl, __tl, __mask)
        for __r in __result:
            assert 'token_id' in __r
            assert 'embed_mse' in __r
            assert 'embed_cosine' in __r
            assert 'hidden_mse' in __r
            assert 'kl' in __r
            assert 'top1_match' in __r

    def test_sorted_by_embed_mse_descending(self):
        __ids, __se, __te, __sh, __th, __sl, __tl, __mask = self._build_inputs(B=2, T=6)
        __result = deformers.pipelines.eval.per_token_metrics(
            __ids, __se, __te, __sh, __th, __sl, __tl, __mask)
        __mses = [__r['embed_mse'] for __r in __result]
        assert __mses == sorted(__mses, reverse=True)

    def test_none_hidden_omits_hidden_keys(self):
        __ids, __se, __te, _, _, __sl, __tl, __mask = self._build_inputs(B=1, T=3)
        __result = deformers.pipelines.eval.per_token_metrics(
            __ids, __se, __te, None, None, __sl, __tl, __mask)
        for __r in __result:
            assert 'hidden_mse' not in __r

    def test_none_logits_omits_kl_and_top1(self):
        __ids, __se, __te, __sh, __th, _, _, __mask = self._build_inputs(B=1, T=3)
        __result = deformers.pipelines.eval.per_token_metrics(
            __ids, __se, __te, __sh, __th, None, None, __mask)
        for __r in __result:
            assert 'kl' not in __r
            assert 'top1_match' not in __r

    def test_identical_embeds_give_zero_embed_mse(self):
        __ids, _, _, _, _, _, _, __mask = self._build_inputs(B=1, T=4)
        __e = torch.randn(1, 4, 8)
        __result = deformers.pipelines.eval.per_token_metrics(
            __ids, __e, __e, None, None, None, None, __mask)
        for __r in __result:
            assert __r['embed_mse'] == pytest.approx(0.0, abs=1e-6)

# AGGREGATE_METRICS ############################################################

class TestAggregateMetrics:

    def test_empty_list_returns_zeros(self):
        __result = deformers.pipelines.eval.aggregate_metrics([])
        assert __result == {'mean': 0.0, 'median': 0.0, 'p95': 0.0}

    def test_single_element(self):
        __result = deformers.pipelines.eval.aggregate_metrics([3.0])
        assert __result['mean'] == pytest.approx(3.0)
        assert __result['median'] == pytest.approx(3.0)
        assert __result['p95'] == pytest.approx(3.0)

    def test_known_values(self):
        # [1, 2, 3, 4, 5]: mean=3, median=3, p95=4.8
        __result = deformers.pipelines.eval.aggregate_metrics([1.0, 2.0, 3.0, 4.0, 5.0])
        assert __result['mean'] == pytest.approx(3.0, abs=1e-5)
        assert __result['median'] == pytest.approx(3.0, abs=1e-5)
        assert __result['p95'] == pytest.approx(4.8, abs=1e-4)

    def test_all_same_values(self):
        __result = deformers.pipelines.eval.aggregate_metrics([7.0, 7.0, 7.0])
        assert __result['mean'] == pytest.approx(7.0)
        assert __result['median'] == pytest.approx(7.0)
        assert __result['p95'] == pytest.approx(7.0)

    def test_returns_python_floats(self):
        __result = deformers.pipelines.eval.aggregate_metrics([1.0, 2.0])
        assert isinstance(__result['mean'], float)
        assert isinstance(__result['median'], float)
        assert isinstance(__result['p95'], float)

    def test_required_keys_present(self):
        __result = deformers.pipelines.eval.aggregate_metrics([0.0, 1.0])
        assert 'mean' in __result
        assert 'median' in __result
        assert 'p95' in __result

# SAVE_REPORT ##################################################################

class TestSaveReport:

    def test_creates_json_file(self, tmp_path):
        __report = {'metric': 1.0, 'count': 5}
        __path = deformers.pipelines.eval.save_report(__report, str(tmp_path), stem='test')
        assert os.path.isfile(__path)
        assert __path.endswith('.json')

    def test_valid_json_content(self, tmp_path):
        __report = {'a': 1.0, 'b': [1, 2, 3]}
        __path = deformers.pipelines.eval.save_report(__report, str(tmp_path), stem='test')
        with open(__path) as __f:
            __loaded = json.load(__f)
        assert __loaded['a'] == pytest.approx(1.0)
        assert __loaded['b'] == [1, 2, 3]

    def test_stem_in_filename(self, tmp_path):
        __path = deformers.pipelines.eval.save_report({}, str(tmp_path), stem='mystem')
        assert 'mystem' in os.path.basename(__path)

    def test_default_stem_is_benchmark(self, tmp_path):
        __path = deformers.pipelines.eval.save_report({}, str(tmp_path))
        assert 'benchmark' in os.path.basename(__path)

    def test_creates_log_dir_if_missing(self, tmp_path):
        __new_dir = str(tmp_path / 'newdir' / 'sub')
        deformers.pipelines.eval.save_report({}, __new_dir, stem='test')
        assert os.path.isdir(__new_dir)

    def test_returns_path_string(self, tmp_path):
        __result = deformers.pipelines.eval.save_report({}, str(tmp_path), stem='test')
        assert isinstance(__result, str)
