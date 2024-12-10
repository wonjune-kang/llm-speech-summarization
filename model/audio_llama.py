from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss

from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import KwargsForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import logging
from transformers import LlamaForCausalLM


logger = logging.get_logger(__name__)


class AudioLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Follows the forward() function of HuggingFace's LlamaForCausalLM with
        the exception of the loss computation as of transformers version 4.47.0.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we
        # are not computing the loss.
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        # TODO: Currently assumes a batch size of 1. Change to incorporate
        # sizes > 1.
        if labels is not None:
            loss = 0.0
            for sample_logits, sample_labels in zip(logits, labels):
                # # Shift so that tokens < n predict n
                # shift_logits = logits[..., :-1, :].contiguous()
                # shift_labels = labels[..., 1:].contiguous()

                sample_logits = sample_logits.unsqueeze(0)
                sample_labels = sample_labels.unsqueeze(0).to(self.device)

                # Shift so that tokens < n predict n, but only for tokens
                # corresponding to the response portion of the LLM.
                response_len = sample_labels.shape[1]
                shift_logits = sample_logits[..., -response_len:-1, :].contiguous()

                # Labels are only provided for the response portion of the LLM in
                # the first place.
                shift_labels = sample_labels[..., 1:].contiguous()

                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)

                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss += loss_fct(shift_logits, shift_labels)

            # Manually perform mean reduction for cross entropy loss.
            loss /= logits.shape[0]

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
