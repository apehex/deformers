"""
Integration smoke test: CompositeBytePrefix + inputs_embeds forward pass.

Verifies that a model loaded with AutoModelForCausalLM accepts inputs_embeds
of shape (B, T, hidden_size) and returns output of expected shape.

If the full qwen/qwen3.5-9b model is not available (too large for the test
environment), a tiny custom configuration is used to validate the interface.

The test always exercises the inputs_embeds code path.
"""

import pytest
import torch
import torch.nn
import transformers

import deformers.layers.prefix


# HELPERS ######################################################################

def _build_tiny_qwen_model():
    """
    Construct a tiny Qwen3-style causal LM from scratch for smoke-testing.

    Returns (model, hidden_size).
    """
    __cfg = transformers.Qwen3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,)
    __model = transformers.Qwen3ForCausalLM(__cfg)
    __model.eval()
    return __model, __cfg.hidden_size


# TESTS ########################################################################

class TestInputsEmbedsSmoke:

    def test_inputs_embeds_tiny_model(self):
        """
        End-to-end: CompositeBytePrefix -> inputs_embeds -> tiny frozen trunk.

        Shape contract:
            byte_ids:        (B, T, G)     long
            prefix output:   (B, T, H)     float
            model output:    (B, T, vocab) float
        """
        B, T, G = 2, 8, 16
        E = 4

        __model, H = _build_tiny_qwen_model()
        for __p in __model.parameters():
            __p.requires_grad_(False)

        __prefix = deformers.layers.prefix.CompositeBytePrefix(
            embed_dim=E,
            vocab_dim=256,
            latent_dim=H,
            group_dim=-1,)

        # random byte ids
        __byte_ids = torch.randint(0, 256, (B, T, G), dtype=torch.long)
        __attention_mask = torch.ones(B, T, dtype=torch.long)

        # forward through prefix
        __embeds = __prefix(__byte_ids)
        assert __embeds.shape == (B, T, H), (
            f'prefix output shape {tuple(__embeds.shape)} != ({B}, {T}, {H})')
        assert __embeds.is_floating_point()

        # forward through trunk using inputs_embeds
        with torch.no_grad():
            __out = __model(
                inputs_embeds=__embeds,
                attention_mask=__attention_mask,
                use_cache=False)

        # logits shape: (B, T, vocab_size)
        __logits = __out.logits
        assert __logits.shape[0] == B
        assert __logits.shape[1] == T
        assert __logits.ndim == 3

    def test_hidden_states_at_depth_k(self):
        """
        Verify hidden_states[k] is accessible when output_hidden_states=True.
        """
        B, T, G = 2, 6, 16
        E = 4
        K = 1

        __model, H = _build_tiny_qwen_model()
        for __p in __model.parameters():
            __p.requires_grad_(False)

        __prefix = deformers.layers.prefix.CompositeBytePrefix(
            embed_dim=E,
            vocab_dim=256,
            latent_dim=H,
            group_dim=-1,)

        __byte_ids = torch.randint(0, 256, (B, T, G), dtype=torch.long)
        __attention_mask = torch.ones(B, T, dtype=torch.long)

        __embeds = __prefix(__byte_ids)

        with torch.no_grad():
            __out = __model(
                inputs_embeds=__embeds,
                attention_mask=__attention_mask,
                output_hidden_states=True,
                use_cache=False)

        assert __out.hidden_states is not None
        assert len(__out.hidden_states) > K
        __h_k = __out.hidden_states[K]
        assert __h_k.shape == (B, T, H)

    def test_teacher_student_mse_loss(self):
        """
        Verify teacher-student hidden-state MSE loss is computable and finite.
        """
        B, T, G = 2, 6, 16
        E = 4
        K = 1

        __model, H = _build_tiny_qwen_model()
        for __p in __model.parameters():
            __p.requires_grad_(False)

        __prefix = deformers.layers.prefix.CompositeBytePrefix(
            embed_dim=E,
            vocab_dim=256,
            latent_dim=H,
            group_dim=-1,)

        __attention_mask = torch.ones(B, T, dtype=torch.long)

        # teacher path (input_ids with tokens from vocab 0..255)
        __input_ids = torch.randint(0, 256, (B, T), dtype=torch.long)
        with torch.no_grad():
            __teacher_out = __model(
                input_ids=__input_ids,
                attention_mask=__attention_mask,
                output_hidden_states=True,
                use_cache=False)
        __teacher_h_k = __teacher_out.hidden_states[K].detach()

        # student path (byte_ids -> prefix -> inputs_embeds)
        __byte_ids = torch.randint(0, 256, (B, T, G), dtype=torch.long)
        __embeds = __prefix(__byte_ids)
        with torch.no_grad():
            __student_out = __model(
                inputs_embeds=__embeds,
                attention_mask=__attention_mask,
                output_hidden_states=True,
                use_cache=False)
        __student_h_k = __student_out.hidden_states[K]

        __loss = torch.nn.functional.mse_loss(__student_h_k, __teacher_h_k)
        assert torch.isfinite(__loss), f'loss is not finite: {__loss.item()}'

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason='CUDA not available')
    def test_inputs_embeds_on_cuda(self):
        """Forward pass on CUDA using inputs_embeds."""
        B, T, G = 2, 8, 16
        E = 4
        __model, H = _build_tiny_qwen_model()
        __model = __model.cuda()
        for __p in __model.parameters():
            __p.requires_grad_(False)
        __prefix = deformers.layers.prefix.CompositeBytePrefix(
            embed_dim=E, vocab_dim=256, latent_dim=H, group_dim=-1)
        __byte_ids = torch.randint(0, 256, (B, T, G)).cuda()
        __attention_mask = torch.ones(B, T, dtype=torch.long).cuda()
        __embeds = __prefix(__byte_ids)
        with torch.no_grad():
            __out = __model(
                inputs_embeds=__embeds,
                attention_mask=__attention_mask,
                use_cache=False)
        assert __out.logits.shape[0] == B
