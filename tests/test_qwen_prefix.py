import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from deformers.models.qwen.prefix import Qwen3PrefixPatchForCausalLM

def _tiny_qwen3_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        layer_types=["full_attention", "full_attention"],
        pad_token_id=0,)

def test_prefix_patch_forward_shapes() -> None:
    __model = Qwen3PrefixPatchForCausalLM(_tiny_qwen3_config())
    __inputs = torch.randint(
        0,
        __model.prefix_patch_config.byte_vocab_size,
        (2, 6 * __model.prefix_patch_config.patch_bytes),)
    __embeds = __model._prefix_patch_forward(input_ids=__inputs)
    assert tuple(__embeds.shape) == (2, 6, __model.config.hidden_size)

def test_prefix_patch_trainable_params_are_scoped() -> None:
    __model = Qwen3PrefixPatchForCausalLM(_tiny_qwen3_config())
    __trainable = [__name for (__name, __p) in __model.named_parameters() if __p.requires_grad]
    assert any(__name.startswith("model.embed_tokens.") for __name in __trainable)
    assert any(__name.startswith("prefix_encoder.") for __name in __trainable)
    assert all(not __name.startswith("lm_head.") for __name in __trainable)

def test_prefix_patch_rejects_invalid_flattened_input_shape() -> None:
    __model = Qwen3PrefixPatchForCausalLM(_tiny_qwen3_config())
    __inputs = torch.randint(0, __model.prefix_patch_config.byte_vocab_size, (1, 33))
    try:
        __model._prefix_patch_forward(input_ids=__inputs)
        raise AssertionError("Expected ValueError was not raised")
    except ValueError as __err:
        assert "divisible by patch size" in str(__err)
