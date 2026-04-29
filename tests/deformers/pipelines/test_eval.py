import json
import math
import os
import tempfile

import pytest
import torch

import mlable.losses
import mlable.metrics

import deformers.pipelines.eval

# META #########################################################################

# INDICES_PROBE ################################################################

class TestIndicesProbe:

    def test_returns_list_of_lists(self):
        __result = deformers.pipelines.eval.indices_probe(
            vocab_dim=100, batch_dim=2, sequence_dim=4)
        assert isinstance(__result, list)
        assert all(isinstance(__r, list) for __r in __result)

    def test_shape(self):
        __result = deformers.pipelines.eval.indices_probe(
            vocab_dim=100, batch_dim=3, sequence_dim=8)
        assert len(__result) == 3
        assert all(len(__r) == 8 for __r in __result)

    def test_ids_within_vocab(self):
        __vocab = 50
        __result = deformers.pipelines.eval.indices_probe(
            vocab_dim=__vocab, batch_dim=4, sequence_dim=16)
        for __row in __result:
            assert all(0 <= __i < __vocab for __i in __row)

    def test_deterministic(self):
        __a = deformers.pipelines.eval.indices_probe(10, 2, 5)
        __b = deformers.pipelines.eval.indices_probe(10, 2, 5)
        assert __a == __b

    def test_cycles_over_vocab(self):
        __result = deformers.pipelines.eval.indices_probe(
            vocab_dim=3, batch_dim=1, sequence_dim=6)
        assert __result[0] == [0, 1, 2, 0, 1, 2]

    def test_all_integers(self):
        __result = deformers.pipelines.eval.indices_probe(
            vocab_dim=256, batch_dim=2, sequence_dim=8)
        for __row in __result:
            assert all(isinstance(__i, int) for __i in __row)

# PER_TOKEN_METRICS ############################################################

class TestPerTokenMetrics:

    def _make_tensors(self, B=2, T=4, H=8, V=16):
        __teacher_embeds  = torch.randn(B, T, H)
        __student_embeds  = torch.randn(B, T, H)
        __teacher_hidden  = torch.randn(B, T, H)
        __student_hidden  = torch.randn(B, T, H)
        __teacher_logits  = torch.randn(B, T, V)
        __student_logits  = torch.randn(B, T, V)
        __mask = torch.ones(B, T, dtype=torch.long)
        return (__teacher_embeds, __student_embeds,
                __teacher_hidden, __student_hidden,
                __teacher_logits, __student_logits,
                __mask)

    def test_returns_dict_with_expected_keys(self):
        __args = self._make_tensors()
        __result = deformers.pipelines.eval.per_token_metrics(*__args)
        for __key in ('embed_mse', 'embed_cos', 'hidden_mse', 'hidden_cos', 'kl', 'top1', 'topk'):
            assert __key in __result

    def test_output_shapes(self):
        __B, __T = 2, 4
        __args = self._make_tensors(B=__B, T=__T)
        __result = deformers.pipelines.eval.per_token_metrics(*__args)
        for __key, __val in __result.items():
            assert __val.shape == (__B, __T), f'{__key}: expected ({__B},{__T}), got {__val.shape}'

    def test_all_zero_mask_produces_zeros(self):
        __B, __T = 2, 4
        __te = torch.randn(__B, __T, 8)
        __se = torch.randn(__B, __T, 8)
        __th = torch.randn(__B, __T, 8)
        __sh = torch.randn(__B, __T, 8)
        __tl = torch.randn(__B, __T, 16)
        __sl = torch.randn(__B, __T, 16)
        __mask = torch.zeros(__B, __T, dtype=torch.long)
        __result = deformers.pipelines.eval.per_token_metrics(
            __te, __se, __th, __sh, __tl, __sl, __mask)
        for __key, __val in __result.items():
            assert __val.abs().sum().item() == pytest.approx(0.0), \
                f'{__key} should be all zeros with zero mask'

    def test_partial_mask(self):
        __B, __T = 1, 4
        __te = torch.randn(__B, __T, 8)
        __se = torch.randn(__B, __T, 8)
        __th = torch.randn(__B, __T, 8)
        __sh = torch.randn(__B, __T, 8)
        __tl = torch.randn(__B, __T, 16)
        __sl = torch.randn(__B, __T, 16)
        __mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)
        __result = deformers.pipelines.eval.per_token_metrics(
            __te, __se, __th, __sh, __tl, __sl, __mask)
        # masked-out positions must be zero
        for __key, __val in __result.items():
            assert __val[0, 2].item() == pytest.approx(0.0), \
                f'{__key}[0,2] should be zero (masked out)'
            assert __val[0, 3].item() == pytest.approx(0.0), \
                f'{__key}[0,3] should be zero (masked out)'

    def test_cpu_output(self):
        __args = self._make_tensors()
        __result = deformers.pipelines.eval.per_token_metrics(*__args)
        for __val in __result.values():
            assert __val.device.type == 'cpu'

# SUMMARY_STATS ################################################################

class TestSummaryStats:

    def test_returns_expected_keys(self):
        __vals = torch.tensor([1.0, 2.0, 3.0])
        __result = deformers.pipelines.eval.summary_stats(__vals)
        assert 'mean' in __result
        assert 'median' in __result
        assert 'p95' in __result

    def test_returns_floats(self):
        __vals = torch.tensor([1.0, 2.0, 3.0])
        __result = deformers.pipelines.eval.summary_stats(__vals)
        for __v in __result.values():
            assert isinstance(__v, float)

    def test_mean_correct(self):
        __vals = torch.tensor([1.0, 2.0, 3.0, 4.0])
        __result = deformers.pipelines.eval.summary_stats(__vals)
        assert __result['mean'] == pytest.approx(2.5)

    def test_median_correct(self):
        __vals = torch.tensor([1.0, 2.0, 3.0])
        __result = deformers.pipelines.eval.summary_stats(__vals)
        assert __result['median'] == pytest.approx(2.0)

    def test_p95_correct(self):
        __vals = torch.arange(1, 101, dtype=torch.float)
        __result = deformers.pipelines.eval.summary_stats(__vals)
        # torch.quantile uses linear interpolation between sorted positions:
        # index = 0.95 * (n-1) = 0.95 * 99 = 94.05 -> lerp(95, 96, 0.05) = 95.05
        assert __result['p95'] == pytest.approx(95.05, abs=0.1)

    def test_mask_filters_values(self):
        __vals = torch.tensor([1.0, 2.0, 3.0, 4.0])
        __mask = torch.tensor([1, 1, 0, 0])
        __result = deformers.pipelines.eval.summary_stats(__vals, __mask)
        assert __result['mean'] == pytest.approx(1.5)

    def test_empty_mask_returns_zeros(self):
        __vals = torch.tensor([1.0, 2.0, 3.0])
        __mask = torch.zeros(3, dtype=torch.long)
        __result = deformers.pipelines.eval.summary_stats(__vals, __mask)
        assert __result == {'mean': 0.0, 'median': 0.0, 'p95': 0.0}

    def test_2d_input(self):
        __vals = torch.ones(3, 4)
        __result = deformers.pipelines.eval.summary_stats(__vals)
        assert __result['mean'] == pytest.approx(1.0)

# TOKEN_TABLE ##################################################################

class TestTokenTable:

    def test_returns_list_of_dicts(self):
        __result = deformers.pipelines.eval.token_table(
            token_ids=[0, 1],
            token_strings=['a', 'b'],
            metrics={'mse': [0.1, 0.2]})
        assert isinstance(__result, list)
        assert all(isinstance(__r, dict) for __r in __result)

    def test_length_matches_token_count(self):
        __n = 5
        __result = deformers.pipelines.eval.token_table(
            token_ids=list(range(__n)),
            token_strings=[str(__i) for __i in range(__n)],
            metrics={'x': [float(__i) for __i in range(__n)]})
        assert len(__result) == __n

    def test_required_fields_present(self):
        __result = deformers.pipelines.eval.token_table(
            token_ids=[42],
            token_strings=['hello'],
            metrics={'embed_mse': [0.5]})
        assert __result[0]['token_id'] == 42
        assert __result[0]['token_string'] == 'hello'
        assert 'byte_length' in __result[0]
        assert 'embed_mse' in __result[0]

    def test_byte_length_correct(self):
        __result = deformers.pipelines.eval.token_table(
            token_ids=[0],
            token_strings=['\u00e9'],  # 2-byte UTF-8 character
            metrics={})
        assert __result[0]['byte_length'] == 2

    def test_metric_values_are_float(self):
        __result = deformers.pipelines.eval.token_table(
            token_ids=[0, 1],
            token_strings=['x', 'y'],
            metrics={'score': [1, 2]})
        for __row in __result:
            assert isinstance(__row['score'], float)

# SAVE_JSON_REPORT #############################################################

class TestSaveJsonReport:

    def test_creates_file(self):
        with tempfile.TemporaryDirectory() as __d:
            __path = os.path.join(__d, 'sub', 'report.json')
            deformers.pipelines.eval.save_json_report({'x': 1}, __path)
            assert os.path.isfile(__path)

    def test_valid_json(self):
        with tempfile.TemporaryDirectory() as __d:
            __path = os.path.join(__d, 'report.json')
            __data = {'a': 1, 'b': [1, 2, 3]}
            deformers.pipelines.eval.save_json_report(__data, __path)
            with open(__path) as __f:
                __loaded = json.load(__f)
            assert __loaded == __data

    def test_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as __d:
            __path = os.path.join(__d, 'deep', 'nested', 'report.json')
            deformers.pipelines.eval.save_json_report({}, __path)
            assert os.path.isfile(__path)

# MLABLE METRIC EDGE CASES #####################################################

class TestMlableMetricEdgeCases:
    """Integration smoke tests: mlable.losses / mlable.metrics with edge-case masks."""

    def test_mse_loss_all_zero_mask(self):
        __pred = torch.randn(2, 4, 8)
        __targ = torch.randn(2, 4, 8)
        __mask = torch.zeros(2, 4)
        # should return 0 (no valid positions)
        __result = mlable.losses.mse_loss(__pred, __targ, mask_arr=__mask)
        assert __result.item() == pytest.approx(0.0)

    def test_cos_sim_identical_inputs(self):
        __x = torch.randn(2, 4, 8)
        __mask = torch.ones(2, 4)
        __result = mlable.losses.cos_sim(__x, __x, mask_arr=__mask)
        assert __result.item() == pytest.approx(1.0, abs=1e-5)

    def test_kl_div_identical_logits(self):
        __logits = torch.randn(2, 4, 16)
        __mask = torch.ones(2, 4)
        __result = mlable.losses.kl_div(__logits, __logits, mask_arr=__mask)
        assert __result.item() == pytest.approx(0.0, abs=1e-5)

    def test_topk_rate_identical_logits(self):
        __logits = torch.randn(2, 4, 16)
        __mask = torch.ones(2, 4)
        __result = mlable.metrics.topk_rate(__logits, __logits, mask_arr=__mask, k_num=5)
        assert __result.item() == pytest.approx(1.0, abs=1e-5)

    def test_topk_rate_all_zero_mask(self):
        __logits = torch.randn(2, 4, 16)
        __mask = torch.zeros(2, 4)
        __result = mlable.metrics.topk_rate(__logits, __logits, mask_arr=__mask, k_num=5)
        assert __result.item() == pytest.approx(0.0)
