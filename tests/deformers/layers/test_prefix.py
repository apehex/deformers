"""
Unit tests for CompositeBytePrefix.

Covers:
- output tensor shape (B, T, H) from (B, T, G) input
- output dtype is float
- device propagation via .to()
- rank-2 flat input mode (group_dim > 0)
- parameter registration (submodules are discoverable)
"""

import pytest
import torch

import deformers.layers.prefix


# FIXTURES #####################################################################

B, T, G, E, H = 2, 31, 16, 8, 64


@pytest.fixture
def prefix_rank3():
    """Prefix with group_dim=-1 (expects rank-3 inputs)."""
    return deformers.layers.prefix.CompositeBytePrefix(
        embed_dim=E,
        vocab_dim=256,
        latent_dim=H,
        group_dim=-1,)


@pytest.fixture
def prefix_rank2():
    """Prefix with group_dim=G (expects rank-2 flat inputs)."""
    return deformers.layers.prefix.CompositeBytePrefix(
        embed_dim=E,
        vocab_dim=256,
        latent_dim=H,
        group_dim=G,)


# SHAPE TESTS ##################################################################

class TestCompositeBytePrefix:

    def test_rank3_output_shape(self, prefix_rank3):
        __x = torch.randint(0, 256, (B, T, G), dtype=torch.long)
        __y = prefix_rank3(__x)
        assert __y.shape == (B, T, H), f'expected ({B}, {T}, {H}), got {tuple(__y.shape)}'

    def test_rank2_output_shape(self, prefix_rank2):
        __x = torch.randint(0, 256, (B, T * G), dtype=torch.long)
        __y = prefix_rank2(__x)
        assert __y.shape == (B, T, H), f'expected ({B}, {T}, {H}), got {tuple(__y.shape)}'

    def test_output_dtype_is_float(self, prefix_rank3):
        __x = torch.randint(0, 256, (B, T, G), dtype=torch.long)
        __y = prefix_rank3(__x)
        assert __y.is_floating_point(), f'expected float output, got dtype={__y.dtype}'

    def test_rank3_shape_assertion(self, prefix_rank3):
        # rank-3 mode rejects rank-2 input
        __x = torch.randint(0, 256, (B, T * G), dtype=torch.long)
        with pytest.raises(AssertionError):
            prefix_rank3(__x)

    def test_rank2_shape_assertion(self, prefix_rank2):
        # rank-2 mode rejects rank-3 input
        __x = torch.randint(0, 256, (B, T, G), dtype=torch.long)
        with pytest.raises(AssertionError):
            prefix_rank2(__x)

    def test_latent_dim_defaults_to_embed_dim_times_group(self):
        # when latent_dim=-1, it defaults to group_dim * embed_dim
        __prefix = deformers.layers.prefix.CompositeBytePrefix(
            embed_dim=E,
            vocab_dim=256,
            latent_dim=-1,
            group_dim=-1,)
        __x = torch.randint(0, 256, (B, T, G), dtype=torch.long)
        __y = __prefix(__x)
        # expected latent_dim = G * E
        assert __y.shape == (B, T, G * E)

    def test_parameters_are_registered(self, prefix_rank3):
        # trigger build
        __x = torch.randint(0, 256, (B, T, G), dtype=torch.long)
        prefix_rank3(__x)
        # after build, _layers must be a registered submodule
        __params = list(prefix_rank3.parameters())
        assert len(__params) > 0, 'no parameters found after build'

    def test_to_device_propagation(self, prefix_rank3):
        # trigger build on CPU
        __x = torch.randint(0, 256, (B, T, G), dtype=torch.long)
        prefix_rank3(__x)
        # all parameters should be on CPU after build
        for __p in prefix_rank3.parameters():
            assert __p.device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
    def test_cuda_device_propagation(self, prefix_rank3):
        __prefix = prefix_rank3.cuda()
        __x = torch.randint(0, 256, (B, T, G), dtype=torch.long).cuda()
        __y = __prefix(__x)
        assert __y.device.type == 'cuda'
        assert __y.shape == (B, T, H)

    def test_config_stored(self, prefix_rank3):
        assert prefix_rank3._config['embed_dim'] == E
        assert prefix_rank3._config['vocab_dim'] == 256
        assert prefix_rank3._config['latent_dim'] == H
        assert prefix_rank3._config['group_dim'] == -1

    def test_batch_size_1(self, prefix_rank3):
        __x = torch.randint(0, 256, (1, T, G), dtype=torch.long)
        __y = prefix_rank3(__x)
        assert __y.shape == (1, T, H)

    def test_single_token(self, prefix_rank3):
        __x = torch.randint(0, 256, (B, 1, G), dtype=torch.long)
        __y = prefix_rank3(__x)
        assert __y.shape == (B, 1, H)

    def test_accepts_pad_byte_id_128(self, prefix_rank3):
        # pad_id=128 (ByteTokenizer default) should not cause errors
        __x = torch.full((B, T, G), fill_value=128, dtype=torch.long)
        __y = prefix_rank3(__x)
        assert __y.shape == (B, T, H)
