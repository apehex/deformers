import pytest
import torch

import mlable.layers.transformer

import deformers.layers.prefix


def test_byte_encoder_shape_and_build_rank3():
    torch.manual_seed(0)
    __layer = deformers.layers.prefix.ByteEncoder(embed_dim=6, patch_dim=-1, vocab_dim=256)
    assert __layer._built is False
    __inputs = torch.randint(0, 256, (2, 5, 4), dtype=torch.long)
    __outputs = __layer(__inputs)
    assert __outputs.shape == (2, 5, 4, 6)
    assert __outputs.dtype == torch.float32
    assert __layer.output_shape((2, 5, 4)) == (2, 5, 4, 6)
    assert __layer._built is True
    assert __layer._value is not None
    assert __layer._position is not None


@pytest.mark.parametrize(
    ('patch_dim', 'input_shape', 'expected_shape'),
    [(4, (2, 20), (2, 5, 4, 3)), (-1, (2, 5, 4), (2, 5, 4, 3))])
def test_byte_encoder_constructor_arguments_control_shapes(patch_dim, input_shape, expected_shape):
    torch.manual_seed(0)
    __layer = deformers.layers.prefix.ByteEncoder(embed_dim=3, patch_dim=patch_dim, vocab_dim=32)
    __inputs = torch.randint(0, 32, input_shape, dtype=torch.long)
    __outputs = __layer(__inputs)
    assert __outputs.shape == expected_shape
    assert __layer.get_config()['patch_dim'] == patch_dim
    assert __layer.get_config()['embed_dim'] == 3


def test_byte_transformer_shape_build_and_constructor_values():
    torch.manual_seed(0)
    __layer = deformers.layers.prefix.ByteTransformer(head_num=2, dropout_rate=0.25)
    assert __layer._built is False
    __inputs = torch.randn(2, 3, 4, 8)
    __outputs = __layer(__inputs)
    assert __outputs.shape == __inputs.shape
    assert __layer.output_shape(tuple(__inputs.shape)) == tuple(__inputs.shape)
    assert __layer._built is True
    assert __layer._attend.get_config()['head_num'] == 2
    assert __layer._attend.get_config()['dropout_rate'] == 0.25
    assert __layer._attend.get_config()['attention_idx'] == -2


def test_byte_transformer_byte_padding_influences_attention(monkeypatch):
    __received = []

    def _spy(self, inputs, paddings=None, is_causal=False):
        __received.append(None if (paddings is None) else paddings.clone())
        if paddings is None:
            return torch.zeros_like(inputs)
        return (~paddings.to(dtype=torch.bool)).to(dtype=inputs.dtype).unsqueeze(-1).expand_as(inputs)

    monkeypatch.setattr(mlable.layers.transformer.SelfAttention, 'forward', _spy)
    torch.manual_seed(0)
    __layer = deformers.layers.prefix.ByteTransformer(head_num=1, dropout_rate=0.0)
    __inputs = torch.zeros(1, 1, 4, 3)
    __paddings = torch.tensor([[[False, True, True, True]]], dtype=torch.bool)
    __masked = __layer(__inputs, paddings=__paddings)
    __unmasked = __layer(__inputs, paddings=None)
    assert __received[0] is not None
    assert torch.equal(__received[0], __paddings)
    assert __received[1] is None
    assert not torch.allclose(__masked, __unmasked)


def test_byte_mixer_shape_build_and_length_embedding_effect():
    torch.manual_seed(0)
    __layer = deformers.layers.prefix.ByteMixer()
    assert __layer._built is False
    __inputs = torch.zeros(1, 2, 4, 3)
    __paddings_a = torch.tensor([[[False, False, False, True], [False, True, True, True]]], dtype=torch.bool)
    __a = __layer(__inputs, __paddings_a)
    assert __a.shape == (1, 2, 12)
    assert __layer.output_shape((1, 2, 4, 3)) == (1, 2, 12)
    assert __layer._built is True
    with torch.no_grad():
        __layer._measure.weight.zero_()
        __layer._measure.weight[1].fill_(1.0)
        __layer._measure.weight[3].fill_(3.0)
    __paddings_b = torch.tensor([[[False, True, True, True], [False, False, False, True]]], dtype=torch.bool)
    __b = __layer(__inputs, __paddings_b)
    assert torch.all(__b[0, 0] == 1.0)
    assert torch.all(__b[0, 1] == 3.0)


@pytest.mark.parametrize(
    ('hidden_dim', 'output_dim'),
    [(7, 5), (11, 9)])
def test_token_projector_shape_and_constructor_behavior(hidden_dim, output_dim):
    torch.manual_seed(0)
    __layer = deformers.layers.prefix.TokenProjector(hidden_dim=hidden_dim, output_dim=output_dim)
    assert __layer._built is False
    __inputs = torch.randn(2, 4, 12)
    __outputs = __layer(__inputs)
    assert __outputs.shape == (2, 4, output_dim)
    assert __layer.output_shape((2, 4, 12)) == (2, 4, output_dim)
    assert __layer._built is True
    assert __layer.get_config()['hidden_dim'] == hidden_dim
    assert __layer.get_config()['output_dim'] == output_dim
