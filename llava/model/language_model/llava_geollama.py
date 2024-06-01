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
LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""
_CONFIG_FOR_DOC = "LlamaConfig"


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, LlamaForCausalLM, \
                         LlamaModel, LlamaConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from llava.model.llava_arch import LlavaGeoMetaModel, LlavaGeoMetaForCausalLM
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings
from transformers.generation.utils import GenerateOutput
import torch.nn.functional as F

class LlavaGeoLlamaConfig(LlamaConfig):
    model_type = "llava_geollama"


class LlavaGeoLlamaModel(LlavaGeoMetaModel, LlamaModel):
    config_class = LlavaGeoLlamaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaGeoLlamaModel, self).__init__(config)

def cosine_similarity(a, b):
    return F.cosine_similarity(a, b, dim=-1)

def compute_triplet_loss(anchor, positive, negative, margin=1.0):
    pos_sim = cosine_similarity(anchor, positive)
    neg_sim = cosine_similarity(anchor, negative)
    # マージンを加えたトリプレットロスを計算
    loss = F.relu(margin - pos_sim + neg_sim).mean()
    return loss

def compute_loss(hidden_states, geoent_labels, review_labels, pref_labels, margin=1.0):
    batch_size, id_length, hidden_dim = hidden_states.shape
    total_loss = 0
    valid_triplets = 0
    
    for i in range(batch_size):
        for j in range(id_length):
            if geoent_labels[i][j] <= -10000:  # geo_tokenをアンカーとして選択
                anchor = hidden_states[i][j].unsqueeze(0)  # アンカーの状態ベクトル
                
                # ポジティブサンプルを同じレビュー内のentityから選択
                pos_mask = (review_labels[i] == review_labels[i][j]) & (geoent_labels[i] > 0)
                if pos_mask.any():
                    positive_features = hidden_states[i][pos_mask]
                    positive = positive_features.mean(dim=0, keepdim=True)  # entityの平均
                    
                    # ネガティブサンプルを異なる都道府県の異なるレビューから選択
                    neg_mask = (review_labels[i] != review_labels[i][j])
                    for k, review_idx in enumerate(review_labels[i]):
                        if neg_mask[k] and (review_idx > 0) and (pref_labels[review_labels[i][j]-1][0] != pref_labels[review_idx-1][0]):
                            neg_mask[k] = True
                        else:
                            neg_mask[k] = False
                            
                    if neg_mask.any():
                        negative_features = hidden_states[i][neg_mask]
                        negative = negative_features.mean(dim=0, keepdim=True)  # 選択されたネガティブサンプルの平均
                        
                        # トリプレット損失を計算
                        loss = compute_triplet_loss(anchor, positive, negative, margin)
                        total_loss += loss
                        valid_triplets += 1
    
    return total_loss / valid_triplets if valid_triplets > 0 else torch.tensor(0.0)

class LlavaGeoLlamaForCausalLM(LlamaForCausalLM, LlavaGeoMetaForCausalLM):
    config_class = LlavaGeoLlamaConfig
    base_model = "llama"

    def __init__(self, config):
        super(LlavaGeoLlamaForCausalLM, self).__init__(config)
        self.model = LlavaGeoLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        print('initialize geo llama')
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model
    
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        aspect_labels: Optional[torch.LongTensor] = None,
        geoent_labels: Optional[torch.LongTensor] = None,
        pref_labels: Optional[torch.LongTensor] = None,
        review_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        #print('input embeds', inputs_embeds)
        #print('labels', labels, aspect_labels)
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                aspect_labels,
                geoent_labels,
                review_labels, 
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                aspect_labels,
                geoent_labels,
                review_labels,
            )
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
        )

        hidden_states = outputs[0]
        #print('hidden states', hidden_states.shape)
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        #print('geoent_labels', geoent_labels, geoent_labels.nonzero(), geoent_labels[geoent_labels<-10000])
        #print('review_labels', review_labels, review_labels.nonzero())
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            #print('shift', shift_logits, shift_labels)
            loss = loss_fct(shift_logits, shift_labels)

        if aspect_labels is not None:
            # Shift so that tokens < n predict n
            shift_aspect_logits = logits[..., :-1, :].contiguous()
            shift_aspect_labels = aspect_labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_aspect_logits = shift_aspect_logits.view(-1, self.config.vocab_size)
            shift_aspect_labels = shift_aspect_labels.view(-1)
            # Enable model parallelism
            shift_aspect_labels = shift_aspect_labels.to(shift_aspect_logits.device)
            loss_aspect = loss_fct(shift_aspect_logits, shift_aspect_labels)
            loss = (loss + loss_aspect)/2
            #print(loss, loss_aspect)
        if geoent_labels is not None and pref_labels is not None and review_labels is not None:
            contra_loss = compute_loss(hidden_states, geoent_labels, review_labels, pref_labels, margin=1.0)
            loss = loss + contra_loss * 0.3
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
        #print('inputs', inputs, 'images', images)
        #if images is not None:
        (
            inputs,
            position_ids,
            attention_mask,
            _,
            inputs_embeds,
            _,
            _
        ) = self.prepare_inputs_labels_for_multimodal(
            inputs,
            position_ids,
            attention_mask,
            None,
            None,
            images,
            aspect_labels=None,
            image_sizes=image_sizes
        )
        #else:
        #    inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        
    def forward_(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        aspect_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                aspect_labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                aspect_labels=None
            )
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

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs


AutoConfig.register("llava_geollama", LlavaGeoLlamaConfig)
AutoModelForCausalLM.register(LlavaGeoLlamaConfig, LlavaGeoLlamaForCausalLM)