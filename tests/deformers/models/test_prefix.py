import inspect

import torch

import deformers.layers.prefix
import deformers.models.prefix


def _build_prefix(**kwargs):
    __config = {
        'embed_dim': 4,
        'output_dim': 12,
        'patch_dim': -1,
        'hidden_dim': -1,
        'padding_idx': 128,
        'block_num': 3,
        'head_num': 2,
        'dropout_rate': 0.0,
        **kwargs}
    return deformers.models.prefix.CompositeBytePrefix(
        **__config)


def test_config_keys_match_constructor_signature():
    __signature = inspect.signature(deformers.models.prefix.CompositeBytePrefix.__init__)
    __expected = {
        __name
        for __name, __param in __signature.parameters.items()
        if (__name != 'self') and (__param.kind is not inspect.Parameter.VAR_KEYWORD)}
    __model = _build_prefix()
    assert set(__model.get_config().keys()) == __expected


def test_block_composition_matches_model_configuration():
    torch.manual_seed(0)
    __model = _build_prefix(block_num=4, head_num=2, dropout_rate=0.1)
    __inputs = torch.randint(0, 256, (2, 5, 8), dtype=torch.long)
    __outputs = __model(__inputs)
    assert __outputs.shape == (2, 5, 12)
    assert isinstance(__model._embed, deformers.layers.prefix.ByteEncoder)
    assert isinstance(__model._blocks, torch.nn.ModuleList)
    assert len(__model._blocks) == 4
    assert all(isinstance(__b, deformers.layers.prefix.ByteTransformer) for __b in __model._blocks)
    assert isinstance(__model._combine, deformers.layers.prefix.ByteMixer)
    assert isinstance(__model._project, deformers.layers.prefix.TokenProjector)
    for __block in __model._blocks:
        assert __block._config['head_num'] == 2
        assert __block._config['dropout_rate'] == 0.1


def test_attention_is_on_byte_axis():
    torch.manual_seed(0)
    __model = _build_prefix(block_num=2)
    __inputs = torch.randint(0, 256, (2, 4, 6), dtype=torch.long)
    __model(__inputs)
    for __block in __model._blocks:
        assert __block._attend.get_config()['attention_idx'] == -2


def test_forward_shapes_rank3_and_rank2_inputs():
    torch.manual_seed(0)
    __rank3 = _build_prefix(patch_dim=-1, output_dim=10)
    __rank3_inputs = torch.randint(0, 256, (2, 7, 6), dtype=torch.long)
    __rank3_outputs = __rank3(__rank3_inputs)
    assert __rank3_outputs.shape == (2, 7, 10)
    assert __rank3_outputs.is_floating_point()
    assert torch.isfinite(__rank3_outputs).all()
    assert __rank3.output_shape((2, 7, 6)) == (2, 7, 10)

    __rank2 = _build_prefix(patch_dim=6, output_dim=10)
    __rank2_inputs = torch.randint(0, 256, (2, 42), dtype=torch.long)
    __rank2_outputs = __rank2(__rank2_inputs)
    assert __rank2_outputs.shape == (2, 7, 10)
    assert __rank2.output_shape((2, 42)) == (2, 7, 10)


def test_first_block_uses_padding_mask_then_unmasked_blocks(monkeypatch):
    __calls = []
    __original = deformers.layers.prefix.ByteTransformer.forward

    def _spy(self, inputs, paddings=None):
        __calls.append(paddings is None)
        return __original(self, inputs=inputs, paddings=paddings)

    monkeypatch.setattr(deformers.layers.prefix.ByteTransformer, 'forward', _spy)

    torch.manual_seed(0)
    __model = _build_prefix(block_num=3, padding_idx=128)
    __inputs = torch.randint(0, 256, (2, 4, 5), dtype=torch.long)
    __inputs[:, :, 0] = 128
    __model(__inputs)
    assert __calls == [False, True, True]


def test_token_outputs_are_independent_across_token_axis():
    torch.manual_seed(0)
    __model = _build_prefix(block_num=2, patch_dim=-1)
    __model.eval()
    __a = torch.randint(0, 256, (2, 3, 5), dtype=torch.long)
    __b = __a.clone()
    __b[:, 1, :] = torch.randint(0, 256, (2, 5), dtype=torch.long)
    __out_a = __model(__a)
    __out_b = __model(__b)
    assert torch.equal(__out_a[:, 0, :], __out_b[:, 0, :])
    assert torch.equal(__out_a[:, 2, :], __out_b[:, 2, :])
