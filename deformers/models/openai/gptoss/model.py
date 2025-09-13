from typing import List, Dict, Any, Optional, Union

import torch
import transformers.cache_utils
import transformers.modeling_outputs
import transformers.models.gpt_oss.modeling_gpt_oss
import transformers.processing_utils
import transformers.utils.generic

# PATCH ########################################################################

transformers.models.gpt_oss.modeling_gpt_oss.GptOssPreTrainedModel._can_record_outputs = {
    "router_logits": transformers.utils.generic.OutputRecorder(transformers.models.gpt_oss.modeling_gpt_oss.GptOssMLP, index=1),
    "hidden_states": transformers.models.gpt_oss.modeling_gpt_oss.GptOssDecoderLayer,
    "attentions": transformers.models.gpt_oss.modeling_gpt_oss.GptOssAttention,}

# INFERENCE ####################################################################

class GptOssForCausalInference(transformers.models.gpt_oss.modeling_gpt_oss.GptOssForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        # self.model = GptOssModel(config)  # Use debug model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[transformers.cache_utils.Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: transformers.processing_utils.Unpack[transformers.utils.generic.TransformersKwargs],
    ) -> transformers.modeling_outputs.MoeCausalLMOutputWithPast:
        # toggle router logits
        __router = output_router_logits if output_router_logits is not None else self.config.output_router_logits
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        __outputs: transformers.modeling_outputs.MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=__router,
            cache_position=cache_position,
            **kwargs,)
        # compute the loss on a restricted token range
        __slice = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        __logits = self.lm_head(__outputs.last_hidden_state[:, __slice, :])
        # compute the loss
        __loss = None
        if labels is not None:
            __loss = self.loss_function(__logits, labels, self.vocab_size, **kwargs)
        # pack everything
        return transformers.modeling_outputs.MoeCausalLMOutputWithPast(
            loss=__loss,
            aux_loss=None, #was computing load_balancing_loss here but we are NOT training
            logits=__logits,
            past_key_values=__outputs.past_key_values,
            hidden_states=__outputs.hidden_states,
            attentions=__outputs.attentions,
            router_logits=__outputs.router_logits,)
