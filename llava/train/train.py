# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch
import time

import transformers
import tokenizers

from llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer
import numpy as np

from llava import conversation as conversation_lib
from llava.model import *
from llava.model.language_model.llava_geollama import LlavaGeoLlamaForCausalLM
from llava.mm_utils import tokenizer_image_token
from llava.model.builder import (
    load_pretrained_model,
    load_checkpoint_model,
    load_pretrained_model2,
)
from safetensors import safe_open
from safetensors.torch import save_file
from PIL import Image
from PIL import ImageFile
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import warnings
from typing import Optional
import random
import copy

import torch
from huggingface_hub import file_exists, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from safetensors.torch import load_file as safe_load_file


local_rank = None

prefectures = [
    "北海道", "青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県",
    "茨城県", "栃木県", "群馬県", "埼玉県", "千葉県", "東京都", "神奈川県",
    "新潟県", "富山県", "石川県", "福井県", "山梨県", "長野県",
    "岐阜県", "静岡県", "愛知県", "三重県",
    "滋賀県", "京都府", "大阪府", "兵庫県", "奈良県", "和歌山県",
    "鳥取県", "島根県", "岡山県", "広島県", "山口県",
    "徳島県", "香川県", "愛媛県", "高知県",
    "福岡県", "佐賀県", "長崎県", "熊本県", "大分県", "宮崎県", "鹿児島県", "沖縄県"
]

# 都道府県名をキー、1から始まるインデックスを値とする辞書を作成
prefecture_to_index = {pref: i + 1 for i, pref in enumerate(prefectures)}

def set_peft_model_state_dict(model, peft_model_state_dict, adapter_name="default"):
    """
    Set the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model.
        peft_model_state_dict (`dict`): The state dict of the Peft model.
    """
    config = model.peft_config[adapter_name]
    state_dict = {}
    if getattr(model, "modules_to_save", None) is not None:
        for key, value in peft_model_state_dict.items():
            if any(module_name in key for module_name in model.modules_to_save):
                for module_name in model.modules_to_save:
                    if module_name in key:
                        key = key.replace(
                            module_name, f"{module_name}.modules_to_save.{adapter_name}"
                        )
                        break
            state_dict[key] = value
    else:
        state_dict = peft_model_state_dict

    peft_model_state_dict = {}

    parameter_prefix = "lora"
    for k, v in state_dict.items():
        if "vision" in k:
            continue
        if parameter_prefix in k:
            suffix = k.split(parameter_prefix)[1]
            if "." in suffix:
                suffix_to_replace = ".".join(suffix.split(".")[1:])
                k = k.replace(suffix_to_replace, f"{adapter_name}.{suffix_to_replace}")
            else:
                k = f"{k}.{adapter_name}"
            peft_model_state_dict[k] = v
        else:
            peft_model_state_dict[k] = v

    load_result = model.load_state_dict(peft_model_state_dict, strict=False)
    if config.is_prompt_learning:
        model.prompt_encoder[adapter_name].embedding.load_state_dict(
            {"weight": peft_model_state_dict["prompt_embeddings"]}, strict=True
        )
    return load_result


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse(
    "0.14"
)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    resume_from_ckpt: bool = field(default=False)
    model_path: Optional[str] = field(default=None)
    model_base: Optional[str] = field(default=None)
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(
        default=-1
    )  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    down_sample: bool = field(default=False)
    use_poi_token: bool = field(default=False)
    load_4bit: bool = field(default=False)
    load_8bit: bool = field(default=False)
    geo_tower: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    group_by_poi: bool = field(default=False)
    multiple_image: bool = False
    llavatour: bool = False


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(
                    f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}"
                )
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {
        k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()
    }
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {
        k: t
        for k, t in named_params
        if any(key_match in k for key_match in keys_to_match)
    }
    to_return = {
        k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()
    }
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["mm_projector", "vision_tower", "vision_resampler"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ["mm_projector"]
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(["embed_tokens", "embed_in"])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(
            trainer.model.named_parameters(), keys_to_match
        )
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split("/")[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith("checkpoint-"):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(
                    weight_to_save,
                    os.path.join(mm_projector_folder, f"{current_folder}.bin"),
                )
            else:
                torch.save(
                    weight_to_save, os.path.join(output_dir, f"mm_projector.bin")
                )
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2 : cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = (
            BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        )
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources
    # sources: [{'id': '磊々峡_29', 'image': '磊々峡_29.jpg', 'conversations': [{'from': 'human', 'value': '<image>\nこれは磊々峡の写真です与えられた画像を見て内容を教えてください.'},
    # {'from': 'gpt', 'value': '雪や紅葉とのマリアージュもフォト ジェニックですね'},
    # {'from': 'human', 'value': 'この画像を見て訪問したつもりでレビューを手短に生成してください.'},
    # {'from': 'gpt', 'value': '秋保温泉の入り口あたりにある名勝地です'},
    # {'from': 'human', 'value': '入力された画像について訪問したつもりでレビューを生成してください.'}, {'from': 'gpt', 'value': '秋保温泉の入り口あたりにある名勝地です。秋保温泉に宿泊して、近場の散策として訪れると効率的に観光できると思います。'}]}]
    for source in sources:
        for sentence in source:
            # ここは結局何もしていない
            if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                sentence["value"] = (
                    sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                )
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(
                        DEFAULT_IMAGE_TOKEN,
                        "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>",
                    )
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = (
                    DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                )
            sentence["value"] = sentence["value"].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
    # sources:
    # [[{'from': 'human', 'value': '<image>\nこれは磊々峡の写真です与えられた画像を見て内容を教えてください.'},
    #   {'from': 'gpt', 'value': '雪や紅葉とのマリアージュもフォトジェニックですね'},
    #   {'from': 'human', 'value': 'この画像を見て訪問したつもりでレビューを手短に生成してください.'},
    #   {'from': 'gpt', 'value': '秋保温泉の入り口あたりにある名勝地です'}, {'from': 'human', 'value': '入力された画像について訪問したつもりでレビューを生成してください.'},
    #   {'from': 'gpt', 'value': '秋保温泉の入り口あたりにある名勝地です。秋保温泉に宿泊して、近場の散策として訪れると効率的に観光できると思います。'}]]
    return sources


def preprocess_llama_2(
    sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [
                tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {
        "human": conv.roles[0],
        "gpt": conv.roles[1],
    }  # {'human': 'USER', 'gpt': 'ASSISTANT'}

    # Apply prompt templates
    # sources例: [[{'from': 'human', 'value': '<image>\nこれは磊々峡の写真です与えられた画像を見て内容を教えてください.'},
    # {'from': 'gpt', 'value': '雪や紅葉とのマリアージュもフォトジェニックですね'},
    # {'from': 'human', 'value': 'この画像を見て訪問したつもりでレビューを手短に生成してください.'},
    # {'from': 'gpt', 'value': '秋保温泉の入り口あたりにある名勝地です'},
    # {'from': 'human', 'value': '入力された画像について訪問したつもりでレビューを生成してください.'}, {'from': 'gpt', 'value': '秋保温泉の入り口あた
    # りにある名勝地です。秋保温泉に宿泊して、近場の散策として訪れると効率的に観光できると思います。'}]]
    conversations = []
    # print('sources', sources)
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        #
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    # print('get_prompt', conv.get_prompt())
    # conversationの例
    # "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
    # USER: <image>\nこれは磊々峡の写真です与えられた画像を見て内容を教えてください.
    # ASSISTANT: 雪や紅葉とのマリアージュもフォトジェニックですね</s>
    # USER: この画像を見て訪問したつもりでレビューを手短に生成してください.
    # ASSISTANT: 秋保温泉の入り口あたりにある名勝地です</s>
    # USER: 入力された画像について訪問したつもりでレビューを生成してください.
    # ASSISTANT: 秋保温泉の入り口あたりにある名勝地です。秋保温泉に宿泊して、近場の散策として訪れると効率的に観光できると思います。</s>"
    # Tokenize conversations

    if has_image:
        # <image>トークンを-200に置き換えている
        input_ids = torch.stack(
            [
                tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
    # Mask targets
    # Assisitant: 以降のところだけを残して応答部分だけを生成する
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(
            conv.sep2
        )  # 一回のやり取りごとに</s>で区切られている
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            # ASSISTANT: の前後で区切る
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1
            # instruction部分をmask
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_llavatourrec(
    sources, 
    #matches,
    tokenizer: transformers.PreTrainedTokenizer, 
    has_image: bool = False
) -> Dict:
    #print('preprocess llavatour')
    conv = conversation_lib.default_conversation.copy()
    roles = {
        "human": conv.roles[0],
        "gpt": conv.roles[1],
    }  # {'human': 'USER', 'gpt': 'ASSISTANT'}

    # Apply prompt templates
    # sources例: [[{'from': 'human', 'value': '<image>\nこれは磊々峡の写真です与えられた画像を見て内容を教えてください.'},
    # {'from': 'gpt', 'value': '雪や紅葉とのマリアージュもフォトジェニックですね'},
    # {'from': 'human', 'value': 'この画像を見て訪問したつもりでレビューを手短に生成してください.'},
    # {'from': 'gpt', 'value': '秋保温泉の入り口あたりにある名勝地です'},
    # {'from': 'human', 'value': '入力された画像について訪問したつもりでレビューを生成してください.'}, {'from': 'gpt', 'value': '秋保温泉の入り口あた
    # りにある名勝地です。秋保温泉に宿泊して、近場の散策として訪れると効率的に観光できると思います。'}]]
    conversations = []
    # print('sources', sources)
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        #
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    # print('get_prompt', conv.get_prompt())
    # conversationの例
    # "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
    # USER: <image>\nこれは磊々峡の写真です与えられた画像を見て内容を教えてください.
    # ASSISTANT: 雪や紅葉とのマリアージュもフォトジェニックですね</s>
    # USER: この画像を見て訪問したつもりでレビューを手短に生成してください.
    # ASSISTANT: 秋保温泉の入り口あたりにある名勝地です</s>
    # USER: 入力された画像について訪問したつもりでレビューを生成してください.
    # ASSISTANT: 秋保温泉の入り口あたりにある名勝地です。秋保温泉に宿泊して、近場の散策として訪れると効率的に観光できると思います。</s>"
    # Tokenize conversations

    #if has_image:
        # <image>トークンを-200に置き換えている
    input_ids = torch.stack(
        [
            tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
            for prompt in conversations
        ],
        dim=0,
    )
    # else:
    #     input_ids = tokenizer(
    #         conversations,
    #         return_tensors="pt",
    #         padding="longest",
    #         max_length=tokenizer.model_max_length,
    #         truncation=True,
    #     ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
    # Mask targets
    # Assisitant: 以降のところだけを残して応答部分だけを生成する
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(
            conv.sep2
        )  # 一回のやり取りごとに</s>で区切られている
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            # ASSISTANT: の前後で区切る
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            #if has_image:
            round_len = len(tokenizer_image_token(rou, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            #else:
                #round_len = len(tokenizer(rou).input_ids)
                #instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1
            # instruction部分をmask
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [
                tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(
                conv.sep.join(rounds[conv_idx : conv_idx + 2])
            )  # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if (
                i != 0
                and getattr(tokenizer, "legacy", False)
                and IS_TOKENIZER_GREATER_THAN_0_14
            ):
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]["value"]
        source[0]["value"] = DEFAULT_IMAGE_TOKEN
        conversation = (
            source[0]["value"]
            + source[1]["value"]
            + conversation_lib.default_conversation.sep
        )
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversations
    ]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]["value"], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if (
        conversation_lib.default_conversation.sep_style
        == conversation_lib.SeparatorStyle.PLAIN
    ):
        # print('preprocess plain')
        return preprocess_plain(sources, tokenizer)
    if (
        conversation_lib.default_conversation.sep_style
        == conversation_lib.SeparatorStyle.LLAMA_2
    ):
        # print('preprocess llama2')
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        # print('preprocess v1')
        # どうやらこれを使ってるぽい
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        # print('preprocess mpt')
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [
            tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
            for prompt in conversations
        ]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn(
                [header] + [s["value"] for s in source], tokenizer
            )["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

def get_coordinates(data_args):
    data_path = data_args.data_path
    data_dir = os.path.dirname(data_path)
    return np.load(os.path.join(data_dir, 'coords.npy'))


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
    ):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def spot_names(self):
        spot_names = []
        for sample in self.list_data_dict:
            spot_names.append(sample["id"].split("_")[0])
        _, spot_ids = np.unique(np.array(spot_names), return_inverse=True)
        return spot_ids

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            # print('sample', sample)
            # print(sample['conversations'])
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        # print('sources init', sources)
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
            if self.data_args.image_aspect_ratio == "pad":

                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(
                            pil_img.mode, (width, width), background_color
                        )
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(
                            pil_img.mode, (height, height), background_color
                        )
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                image = expand2square(
                    image, tuple(int(x * 255) for x in processor.image_mean)
                )
                # processor CLIPImageProcessor "crop_size": {"height": 336,"width": 336},
                # "do_center_crop": true, "do_convert_rgb": true"do_normalize": true,"do_rescale": true,"do_resize": true,
                # image_mean": [0.48145466, 0.4578275,0.4082107],"image_std": [0.26862954,0.26130258,0.27577711],
                # "image_processor_type": "CLIPImageProcessor",
                # "resample": 3,"rescale_factor": 0.00392156862745098, # "size": "shortest_edge": 336}}
                image = processor.preprocess(image, return_tensors="pt")[
                    "pixel_values"
                ][0]
            else:
                image = processor.preprocess(image, return_tensors="pt")[
                    "pixel_values"
                ][0]
            # print('before multi modal', copy.deepcopy([e["conversations"] for e in sources]))
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]), self.data_args
            )
            # print('after multimodal', sources)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources, self.tokenizer, has_image=("image" in self.list_data_dict[i])
        )
        # {'input_ids': input_ids, 'labels': labels}
        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0]
            )

        # image exist in the data
        # print(image.shape)
        if "image" in self.list_data_dict[i]:
            data_dict["image"] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict["image"] = torch.zeros(3, crop_size["height"], crop_size["width"])
        if "image" not in self.list_data_dict[i]:
            print('image not in data')
            data_dict["image"] = torch.zeros(3, crop_size["height"], crop_size["width"])
        return data_dict


class MultiSupervisedDataset(LazySupervisedDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
    ):
        super().__init__(data_path, tokenizer, data_args)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        # print('sources init', sources)
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if "image" in sources[0]:
            image_files = self.list_data_dict[i]["image"]
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            if isinstance(image_files, str):
                images = [
                    Image.open(os.path.join(image_folder, image_files)).convert("RGB")
                ]
            elif isinstance(image_files, list):
                images = [
                    Image.open(os.path.join(image_folder, image_file)).convert("RGB")
                    if os.path.exists(os.path.join(image_folder, image_file))
                    else Image.open(os.path.join(image_folder, "函館山_0.jpg")).convert(
                        "RGB"
                    )
                    for image_file in image_files
                ]

            if len(images) == 0:
                images = [
                    Image.open(os.path.join(image_folder, "函館山_0.jpg")).convert(
                        "RGB"
                    )
                ]
            if self.data_args.image_aspect_ratio == "pad":

                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(
                            pil_img.mode, (width, width), background_color
                        )
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(
                            pil_img.mode, (height, height), background_color
                        )
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                images = [
                    expand2square(
                        image, tuple(int(x * 255) for x in processor.image_mean)
                    )
                    for image in images
                ]
                # processor CLIPImageProcessor "crop_size": {"height": 336,"width": 336},
                # "do_center_crop": true, "do_convert_rgb": true"do_normalize": true,"do_rescale": true,"do_resize": true,
                # image_mean": [0.48145466, 0.4578275,0.4082107],"image_std": [0.26862954,0.26130258,0.27577711],
                # "image_processor_type": "CLIPImageProcessor",
                # "resample": 3,"rescale_factor": 0.00392156862745098, # "size": "shortest_edge": 336}}
                image = processor.preprocess(images, return_tensors="pt")[
                    "pixel_values"
                ]
            else:
                image = processor.preprocess(images, return_tensors="pt")[
                    "pixel_values"
                ]
            # print('before multi modal', copy.deepcopy([e["conversations"] for e in sources]))
            sources = copy.deepcopy([e["conversations"] for e in sources])
            # sources = preprocess_multimodal(
            #     copy.deepcopy([e["conversations"] for e in sources]),
            #     self.data_args)
            # print('after multimodal', sources)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources, self.tokenizer, has_image=("image" in self.list_data_dict[i])
        )
        # {'input_ids': input_ids, 'labels': labels}

        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0]
            )

        # image exist in the data
        # print(image.shape)
        if "image" in self.list_data_dict[i]:
            if image.dim() == 3:
                image = image.unsqueeze(0)
                data_dict["image"] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict["image"] = torch.zeros(
                1, 3, crop_size["height"], crop_size["width"]
            )
        return data_dict


class LLaVATourRecDataset(LazySupervisedDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
    ):
        print('llavatour dataset')
        super().__init__(data_path, tokenizer, data_args)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        # print('sources init', sources)
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        #print('sources', sources)
        if 'match' in sources[0]:
            matches = copy.deepcopy([e.get('match', -1) for e in sources])
        if 'geoents' in sources[0]:
            geo_ents = copy.deepcopy([e.get('geoents', -1) for e in sources])
        if 'prefs' in sources[0]:
            prefs = copy.deepcopy([e.get('prefs', -1) for e in sources])
        if 'reviews' in sources[0]:
            reviews = copy.deepcopy([e.get('reviews', -1) for e in sources])
        #print('matches', matches)
        if "image" in sources[0]:
            image_files = self.list_data_dict[i]["image"]
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            if isinstance(image_files, str):
                images = [
                    Image.open(os.path.join(image_folder, image_files)).convert("RGB")
                ]
            elif isinstance(image_files, list):
                images = [
                    Image.open(os.path.join(image_folder, image_file)).convert("RGB")
                    if os.path.exists(os.path.join(image_folder, image_file))
                    else Image.open(os.path.join(image_folder, "函館山_0.jpg")).convert(
                        "RGB"
                    )
                    for image_file in image_files
                ]

            if len(images) == 0:
                images = [
                    Image.open(os.path.join(image_folder, "函館山_0.jpg")).convert(
                        "RGB"
                    )
                ]
            if self.data_args.image_aspect_ratio == "pad":

                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(
                            pil_img.mode, (width, width), background_color
                        )
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(
                            pil_img.mode, (height, height), background_color
                        )
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                images = [
                    expand2square(
                        image, tuple(int(x * 255) for x in processor.image_mean)
                    )
                    for image in images
                ]
                #print('begin', images, len(images))
                # processor CLIPImageProcessor "crop_size": {"height": 336,"width": 336},
                # "do_center_crop": true, "do_convert_rgb": true"do_normalize": true,"do_rescale": true,"do_resize": true,
                # image_mean": [0.48145466, 0.4578275,0.4082107],"image_std": [0.26862954,0.26130258,0.27577711],
                # "image_processor_type": "CLIPImageProcessor",
                # "resample": 3,"rescale_factor": 0.00392156862745098, # "size": "shortest_edge": 336}}
                image = processor.preprocess(images, return_tensors="pt")[
                    "pixel_values"
                ]
            else:
                image = processor.preprocess(images, return_tensors="pt")[
                    "pixel_values"
                ]
            sources = copy.deepcopy([e["conversations"] for e in sources])
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess_llavatourrec(
            sources,  self.tokenizer, has_image=("image" in self.list_data_dict[i])
        )

        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0]
            )

        # image exist in the data
        # print(image.shape)
        if 'match' in self.list_data_dict[i]:
            data_dict['match'] = matches
        if 'geoents' in self.list_data_dict[i]:
            data_dict['geoents'] = geo_ents
        if 'prefs' in self.list_data_dict[i]:
            data_dict['prefs'] = prefs
        if 'reviews' in self.list_data_dict[i]:
            data_dict['reviews'] = reviews
        
        if "image" in self.list_data_dict[i]:
            if image.dim() == 3:
                image = image.unsqueeze(0)
                data_dict["image"] = image
            else:
                data_dict['image'] = image
        #elif self.data_args.is_multimodal:
        else:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict["image"] = torch.zeros(
                1, 3, crop_size["height"], crop_size["width"]
            )
        #if 'image' not in data_dict:
        #    print('image not in data dict')
        #print('final', len(data_dict['image']))
        #print('data_dict', data_dict)
        return data_dict

def find_sublist_indices(sublist, main_list):
    # 主リストの長さとサブリストの長さを取得
    main_len, sublist_len = len(main_list), len(sublist)
    # 主リストをスライドさせながらサブリストと比較
    for start in range(main_len - sublist_len + 1):
        end = start + sublist_len
        if main_list[start:end].equal(sublist):
            return start, end - 1
    return None

def update_labels_for_matches(input_ids, labels, matches_ids,):
    batch_size = input_ids.size(0)
    labels_new = copy.deepcopy(labels)
    for i in range(batch_size):
        for match_ids in matches_ids[i]:
            # トークン列のインデックスを見つける
            match_indices = find_sublist_indices(torch.tensor(match_ids), input_ids[i])
            if match_indices:
                start, end = match_indices
                # 対応するlabelsのマスクを解除または更新
                labels_new[i, start:end + 1] = torch.tensor(match_ids).to(labels.device)

    return labels_new

def label_geo_entities(input_ids, geo_ents_ids):
    batch_size = input_ids.size(0)
    label_ids = torch.zeros_like(input_ids)

    for i in range(batch_size):  # バッチごとに処理
        label_counter = 1
        last_position = {}  # 各geo_entの最後のマッチ位置を記録
        #print(f'{i}', input_ids[i], geo_ents_ids[i])
        for review in geo_ents_ids[i]:  # 各レビューごとに処理
            for geo_ent in review:
                if len(geo_ent)==0:
                    continue
                geo_ent = torch.tensor(geo_ent).to(input_ids.device)
                match_indices = find_sublist_indices(torch.tensor(geo_ent), input_ids[i])
                if match_indices:
                    start, end = match_indices
                    label_ids[i, start:end + 1] = torch.tensor(label_counter).to(label_ids.device)
                label_counter+=1
                
    return label_ids

def label_reviews(input_ids, prefs, reviews):
    #print('prefs', prefs)
    #print('reviws', reviews)
    batch_size = input_ids.size(0)
    label_size = input_ids.size(1)
    
    # 出力テンソルの初期化
    prefs = [[prefecture_to_index[p] for p in pref] for pref in prefs]
    pref_labels = torch.zeros_like(input_ids)
    review_labels = torch.zeros_like(input_ids)
    
    # 各バッチアイテムに対してラベルを割り当てる
    for i in range(batch_size):
        #current_prefs = prefs[i]
        for j,review in enumerate(reviews[i]):
            match_indices = find_sublist_indices(torch.tensor(review), input_ids[i])
            if match_indices:
                start, end = match_indices
                review_labels[i, start:end + 1] = torch.tensor(j+1).to(review_labels.device)
    #print('prefs', prefs)
    #print('review_labels', review_labels.nonzero())
    return prefs, review_labels

    

@dataclass
class DataCollatorForLLaVATourRecDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        #print('input_ids', input_ids)
        labels = labels[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                ##batch["images"] = torch.stack(images)
                batch["images"] = torch.cat(images)
            else:
                batch["images"] = torch.cat(images)#images
            #print(batch['images'].shape)
        else:
            batch_size = input_ids.shape[0]
            batch['images'] = torch.zeros(batch_size, 3, 336, 336)
            #print('image not in batch')
        #print('instance', instances[0])
        if 'match' in instances[0]:
            #print('instances', instances)
            #print(instances[0]['match'])
            matches = [d['match'][0] for d in instances]
            matches_flatten = [sum(match, []) for match in matches]
            # キーフレーズを最大5個選択する
            matches_selected = [random.sample(match_flatten, min(3, len(match_flatten))) for match_flatten in matches_flatten]
            # キーフレーズをエンコードする。
            match_ids = [[self.tokenizer.encode(match, add_special_tokens=False)[1:] for match in matches if len(match)] for matches in matches_selected]
            aspect_labels = update_labels_for_matches(input_ids, labels, match_ids)
            batch['aspect_labels'] = aspect_labels
        else:
            batch['aspect_labels'] = None
        if 'geoents' in instances[0]:
            geoents = [d['geoents'][0] for d in instances]
            #print('geoents', geoents)
            geoents_ids = [[[self.tokenizer.encode(geo_ent, add_special_tokens=False)[1:] for geo_ent in geo_ents] for geo_ents in geoent] for geoent in geoents]
            geoent_labels = label_geo_entities(input_ids, geoents_ids)
            batch['geoent_labels'] = geoent_labels
        if 'prefs' in instances[0] and 'reviews' in instances[0]:
            reviews = [d['reviews'][0] for d in instances]
            reviews = [[self.tokenizer.encode(review, add_special_tokens=False)[1:] for review in reviews_] for reviews_ in reviews]
            #print('reviews', reviews)
            #print('prefs', batch['prefs'])
            prefs = [d['prefs'][0] for d in instances]
            pref_labels, review_labels = label_reviews(input_ids, prefs, reviews)
            batch['pref_labels'] = pref_labels
            batch['review_labels'] = review_labels
        return batch


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]

            batch["images"] = torch.cat(images)
        else:
            batch_size = input_ids.shape[0]
            batch['images'] = torch.zeros(batch_size, 3, 336, 336)

        return batch


@dataclass
class DataCollatorForMultiSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]

            batch["images"] = torch.cat(images)

        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


def make_llavatour_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    train_dataset = LLaVATourRecDataset(
        tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args
    )
    data_collator = DataCollatorForLLaVATourRecDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


def make_multi_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = MultiSupervisedDataset(
        tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args
    )
    data_collator = DataCollatorForMultiSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


def prepare_model_from_checkpoint(model_args, training_args):
    model_name = get_model_name_from_path(model_args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model2(
        model_args.model_path, model_args.model_base, model_name, False, False
    )
    # model =load_checkpoint_model(model_args.model_path, model_args.model_base, model_name, False, False)
    model.config.use_cache = False  # True #False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training

        model.config.torch_dtype = (
            torch.float32
            if training_args.fp16
            else (torch.bfloat16 if training_args.bf16 else torch.float32)
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    return model


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["mm_projector"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )

    print("model_args", model_args)
    if model_args.vision_tower is not None:
        if "mpt" in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(
                model_args.model_name_or_path, trust_remote_code=True
            )
            config.attn_config["attn_impl"] = training_args.mpt_attn_impl
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args,
            )
        elif (
            "qwen" in model_args.model_name_or_path.lower()
            or "qurasu" in model_args.model_name_or_path.lower()
        ):
            config = transformers.AutoConfig.from_pretrained(
                model_args.model_name_or_path, trust_remote_code=True
            )
            # config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaQwenForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args,
            )

        else:
            if model_args.geo_tower is not None:
                #print('llavageollamaforcausallm')
                model = LlavaGeoLlamaForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    **bnb_model_from_pretrained_args,
                )
            else:
                #print('llavallamaforcausallm')
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    **bnb_model_from_pretrained_args,
                )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args,
        )
    model.config.use_cache = False  # True #False
    

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training

        model.config.torch_dtype = (
            torch.float32
            if training_args.fp16
            else (torch.bfloat16 if training_args.bf16 else torch.float32)
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if "mpt" in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                model_args.version
            ]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                "vicuna_v1"
            ]

    if model_args.vision_tower is not None:
        # print(model_args.vision_tower)
        #if True:  # not model_args.resume_from_ckpt:
        model.get_model().initialize_vision_modules(
            model_args=model_args, fsdp=training_args.fsdp
        )

        vision_tower = model.get_vision_tower()
        vision_tower.to(
            dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
            device=training_args.device,
        )

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = (
            training_args.tune_mm_mlp_adapter
        ) = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(
                dtype=compute_dtype, device=training_args.device
            )

        model.config.mm_use_im_start_end = (
            data_args.mm_use_im_start_end
        ) = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
        # for k,v in model.named_parameters():
        #     print(k, v.shape)
        #pass
    
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    if training_args.llavatour:
        data_module = make_llavatour_data_module(
            tokenizer=tokenizer, data_args=data_args
        )

    elif training_args.multiple_image:
        data_module = make_multi_supervised_data_module(
            tokenizer=tokenizer, data_args=data_args
        )
    else:
        data_module = make_supervised_data_module(
            tokenizer=tokenizer, data_args=data_args
        )
    if model_args.geo_tower is not None:
        coordinates = get_coordinates(data_args)
        model.get_model().initialize_geo_modules(
            model_args=model_args,
            coordinates=coordinates
        )

    if model_args.resume_from_ckpt:
        non_lora_trainables = torch.load(
            os.path.join(model_args.model_path, "non_lora_trainables.bin"),
            map_location="cpu",
        )
        non_lora_trainables = {
            (k[11:] if k.startswith("base_model.") else k): v
            for k, v in non_lora_trainables.items()
        }
        if any(k.startswith("model.model.") for k in non_lora_trainables):
            non_lora_trainables = {
                (k[6:] if k.startswith("model.") else k): v
                for k, v in non_lora_trainables.items()
            }
        model.load_state_dict(non_lora_trainables, strict=False)

        # for k,v in model.named_parameters():
        #    print(k, v.shape)
        from peft import PeftModel


        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True
        tensors = {}
        with safe_open(
            os.path.join(model_args.model_path, "adapter_model.safetensors"),
            framework="pt",
            device="cpu",
        ) as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        # parameter_keys = model.named_parameters.keys()
        set_peft_model_state_dict(model, tensors)

    trainer = LLaVATrainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)

            torch.save(
                non_lora_state_dict,
                os.path.join(training_args.output_dir, "non_lora_trainables.bin"),
            )
    else:
        safe_save_model_for_hf_trainer(
            trainer=trainer, output_dir=training_args.output_dir
        )


if __name__ == "__main__":
    train()
