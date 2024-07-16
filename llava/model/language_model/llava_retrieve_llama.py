#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaRetrieveMetaModel, LlavaRetrieveMetaForCausalLM #LlavaPOIMetaForCausalLM
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, USER_TOKEN_INDEX, POI_TOKEN_INDEX
from llava.model.retriever.builder import build_retriever

class LlavaRetrievalConfig(LlamaConfig):
    model_type = "llava_retrieval"
    
class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"
    
class LlavaPOIConfig(LlamaConfig):
    model_type = "llava_poi_llama"


class LlavaRetrieveModel(LlavaRetrieveMetaModel, LlamaModel):
    config_class = LlavaRetrievalConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaRetrieveModel, self).__init__(config)


class LlavaRetrieveForCausalLM(LlamaForCausalLM, LlavaRetrieveMetaForCausalLM):
    config_class = LlavaRetrievalConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        #print('llama model config', config)
        self.model = LlavaRetrieveModel(config)
        #print('model', self.model)
        # for k,v in self.model.named_parameters():
        #     print(k, v)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # self.retriever = build_retriever()
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model
    
    def make_retriever(self):
        self.retriever = build_retriever()
        

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        triplets: Optional[List[List[tuple]]] =  None,
        prompts: Optional[List[List]] = None,
        tasks: Optional[List[List]] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        #print('input_embeds', inputs_embeds, input_ids)
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
                triplets,
                prompts,
                tasks
            )

        if inputs_embeds is None and input_ids is not None:
            input_ids[input_ids>=self.vocab_size] = self.vocab_size - 1
            input_ids[input_ids<0] = self.vocab_size - 1

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            #print('inputs', inputs.shape)
            inputs_embeds = self.get_model().embed_tokens(inputs)
        #print('input_embeds.', inputs_embeds.shape)
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        
    @torch.no_grad()
    def generate_rec(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            num_beams=10,
            num_return_sequences=5,
            no_repeat_ngram_size=2,
            early_stopping=True
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_retrieval", LlavaRetrievalConfig)
AutoModelForCausalLM.register(LlavaRetrievalConfig, LlavaRetrieveForCausalLM)

#AutoConfig.register("llava_poi_llama", LlavaPOIConfig)
#AutoModelForCausalLM.register(LlavaPOIConfig, LlavaPOILlamaForCausalLM)