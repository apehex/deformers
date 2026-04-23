"""Unit tests for deformers.pipelines.eval."""

import json
import os

import pytest
import torch

import deformers.pipelines.eval


class TestIndicesProbe:

    def test_shape_and_range(self):
        __out = deformers.pipelines.eval.indices_probe(vocab_dim=7, batch_dim=2, sequence_dim=5)
        assert len(__out) == 2
        assert all(len(__row) == 5 for __row in __out)
        assert all(0 <= __id < 7 for __row in __out for __id in __row)

    def test_deterministic(self):
        __a = deformers.pipelines.eval.indices_probe(vocab_dim=11, batch_dim=3, sequence_dim=4)
        __b = deformers.pipelines.eval.indices_probe(vocab_dim=11, batch_dim=3, sequence_dim=4)
        assert __a == __b


class TestMaskedCosine:

    def test_identical_is_one(self):
        __x = torch.randn(2, 3, 8)
        __m = torch.ones(2, 3, dtype=torch.long)
        __val = deformers.pipelines.eval.masked_cosine(__x, __x, __m)
        assert __val.item() == pytest.approx(1.0, abs=1e-5)

    def test_all_masked_is_zero(self):
        __a = torch.randn(2, 3, 8)
        __b = torch.randn(2, 3, 8)
        __m = torch.zeros(2, 3, dtype=torch.long)
        __val = deformers.pipelines.eval.masked_cosine(__a, __b, __m)
        assert __val.item() == pytest.approx(0.0)

    def test_bounded(self):
        __a = torch.randn(2, 3, 8)
        __b = torch.randn(2, 3, 8)
        __m = torch.ones(2, 3, dtype=torch.long)
        __val = deformers.pipelines.eval.masked_cosine(__a, __b, __m).item()
        assert -1.0 - 1e-5 <= __val <= 1.0 + 1e-5


class TestTopkSetRate:

    def test_identical_logits_is_one(self):
        __x = torch.randn(2, 4, 9)
        __m = torch.ones(2, 4, dtype=torch.long)
        __val = deformers.pipelines.eval.topk_set_rate(__x, __x, __m, k_num=3)
        assert __val.item() == pytest.approx(1.0)

    def test_all_masked_is_zero(self):
        __a = torch.randn(1, 4, 8)
        __b = torch.randn(1, 4, 8)
        __m = torch.zeros(1, 4, dtype=torch.long)
        __val = deformers.pipelines.eval.topk_set_rate(__a, __b, __m, k_num=3)
        assert __val.item() == pytest.approx(0.0)

    def test_order_insensitive(self):
        # same top-2 set but different order
        __student = torch.tensor([[[4.0, 3.0, 0.0]]])
        __teacher = torch.tensor([[[3.0, 4.0, 0.0]]])
        __mask = torch.ones(1, 1, dtype=torch.long)
        __val = deformers.pipelines.eval.topk_set_rate(__student, __teacher, __mask, k_num=2)
        assert __val.item() == pytest.approx(1.0)


class TestPerTokenMetrics:

    def _inputs(self, B=2, T=3, H=5, V=7):
        __ids = torch.arange(B * T).reshape(B, T)
        __mask = torch.ones(B, T, dtype=torch.long)
        __se = torch.randn(B, T, H)
        __te = torch.randn(B, T, H)
        __sh = torch.randn(B, T, H)
        __th = torch.randn(B, T, H)
        __sl = torch.randn(B, T, V)
        __tl = torch.randn(B, T, V)
        return __ids, __se, __te, __sh, __th, __sl, __tl, __mask

    def test_valid_length_matches_mask(self):
        __ids, __se, __te, __sh, __th, __sl, __tl, __mask = self._inputs(B=1, T=4)
        __mask[0, 2] = 0
        __rows = deformers.pipelines.eval.per_token_metrics(
            token_ids_arr=__ids,
            student_embeds_arr=__se,
            teacher_embeds_arr=__te,
            student_hidden_arr=__sh,
            teacher_hidden_arr=__th,
            student_logits_arr=__sl,
            teacher_logits_arr=__tl,
            mask_arr=__mask)
        assert len(__rows) == 3

    def test_required_keys(self):
        __ids, __se, __te, __sh, __th, __sl, __tl, __mask = self._inputs()
        __rows = deformers.pipelines.eval.per_token_metrics(
            token_ids_arr=__ids,
            student_embeds_arr=__se,
            teacher_embeds_arr=__te,
            student_hidden_arr=__sh,
            teacher_hidden_arr=__th,
            student_logits_arr=__sl,
            teacher_logits_arr=__tl,
            mask_arr=__mask)
        for __r in __rows:
            assert 'token_id' in __r
            assert 'embed_mse' in __r
            assert 'embed_cosine' in __r
            assert 'hidden_mse' in __r
            assert 'hidden_cosine' in __r
            assert 'kl' in __r
            assert 'top1_match' in __r

    def test_sorted_desc_by_embed_mse(self):
        __ids, __se, __te, __sh, __th, __sl, __tl, __mask = self._inputs(B=2, T=5)
        __rows = deformers.pipelines.eval.per_token_metrics(
            token_ids_arr=__ids,
            student_embeds_arr=__se,
            teacher_embeds_arr=__te,
            student_hidden_arr=__sh,
            teacher_hidden_arr=__th,
            student_logits_arr=__sl,
            teacher_logits_arr=__tl,
            mask_arr=__mask)
        __vals = [__r['embed_mse'] for __r in __rows]
        assert __vals == sorted(__vals, reverse=True)


class TestAggregateMetrics:

    def test_empty(self):
        assert deformers.pipelines.eval.aggregate_metrics([]) == {
            'mean': 0.0,
            'median': 0.0,
            'p95': 0.0,
        }

    def test_known_values(self):
        __res = deformers.pipelines.eval.aggregate_metrics([1.0, 2.0, 3.0, 4.0, 5.0])
        assert __res['mean'] == pytest.approx(3.0)
        assert __res['median'] == pytest.approx(3.0)
        assert __res['p95'] == pytest.approx(4.8, abs=1e-4)


class TestSaveReport:

    def test_writes_json(self, tmp_path):
        __path = deformers.pipelines.eval.save_report({'x': 1.0}, str(tmp_path), stem='test')
        assert os.path.isfile(__path)
        with open(__path) as __f:
            __obj = json.load(__f)
        assert __obj == {'x': 1.0}
