import dataclasses
from typing import Optional

import torch
import torch.nn
import transformers
import transformers.modeling_outputs
import transformers.processing_utils
import transformers.utils.generic
from mlable.layers.embedding import CompositeEmbedding
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, create_causal_mask

# MODEL ########################################################################

@dataclasses.dataclass
class PrefixPatchConfig:
    byte_vocab_size: int = 256
    patch_bytes: int = 32
    byte_embedding_dim: int = 0
    add_prefix_encoder_layer: bool = True

class Qwen3PrefixPatchForCausalLM(transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        __patch = 32
        __byte_dim = config.hidden_size // __patch
        if __patch * __byte_dim != config.hidden_size:
            raise ValueError("hidden_size must be divisible by patch size")
        self.prefix_patch_config = PrefixPatchConfig(
            patch_bytes=__patch,
            byte_embedding_dim=__byte_dim,)
        self.model.embed_tokens = CompositeEmbedding(
            input_dim=self.prefix_patch_config.byte_vocab_size,
            output_dim=self.prefix_patch_config.byte_embedding_dim,
            group_dim=self.prefix_patch_config.patch_bytes,
            merge_axes=True,)
        self.prefix_encoder = Qwen3DecoderLayer(config, layer_idx=0) if self.prefix_patch_config.add_prefix_encoder_layer else torch.nn.Identity()
        self._freeze_trunk()

    def _freeze_trunk(self) -> None:
        for __name, __param in self.model.named_parameters():
            __param.requires_grad = False
            if __name.startswith("embed_tokens."):
                __param.requires_grad = True
        for __param in self.prefix_encoder.parameters():
            __param.requires_grad = True
        for __param in self.lm_head.parameters():
            __param.requires_grad = False

    def _prefix_patch_forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor]=None,
        position_ids: Optional[torch.LongTensor]=None,
    ) -> torch.Tensor:
        if input_ids.shape[-1] % self.prefix_patch_config.patch_bytes != 0:
            raise ValueError("input_ids last dimension must be divisible by patch size")
        __inputs_embeds = self.model.embed_tokens(input_ids)
        if attention_mask is not None and attention_mask.ndim == 2 and attention_mask.shape[-1] == input_ids.shape[-1]:
            attention_mask = attention_mask.reshape(
                attention_mask.shape[0],
                -1,
                self.prefix_patch_config.patch_bytes).amax(dim=-1)
        if position_ids is None:
            position_ids = torch.arange(__inputs_embeds.shape[1], device=__inputs_embeds.device).unsqueeze(0).expand(__inputs_embeds.shape[0], -1)
        __attention_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=__inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,)
        __position_embeddings = self.model.rotary_emb(__inputs_embeds, position_ids)
        if isinstance(self.prefix_encoder, torch.nn.Identity):
            return __inputs_embeds
        return self.prefix_encoder(
            hidden_states=__inputs_embeds,
            attention_mask=__attention_mask,
            position_ids=position_ids,
            position_embeddings=__position_embeddings,)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[transformers.cache_utils.Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: transformers.processing_utils.Unpack[transformers.utils.generic.TransformersKwargs],
    ) -> transformers.modeling_outputs.CausalLMOutputWithPast:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds must be provided")
            inputs_embeds = self._prefix_patch_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,)
            input_ids = None
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            **kwargs,)
