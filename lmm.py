import argparse
import torch
import copy
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import numpy as np
from PIL import Image
import pickle
from easydict import EasyDict
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
from vllm import LLM as vllm_LLM
from vllm import SamplingParams
from tqdm import tqdm
import requests
from PIL import Image
from io import BytesIO
from rapidfuzz.distance import Levenshtein
import os
from transformers import TextStreamer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaTokenizer, AutoModelForVision2Seq, BlipImageProcessor
from transformers import AutoTokenizer, AutoModelForVision2Seq, AutoImageProcessor
import torch
#from heron.models.video_blip import VideoBlipForConditionalGeneration, VideoBlipProcessor
from transformers import LlamaTokenizer
import pandas as pd
import spacy
import json
import random
from rapidfuzz.distance import Levenshtein
import sys
from vllm import LLM, SamplingParams
from openai import OpenAI
import ray
import api_key
# sys.path.append('..')
# from PEPLER.module import ContinuousPromptLearning
# from PEPLER.utils import DataLoader, ids2tokens

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

class LLaVAInference:
    def __init__(self, args):
            # Model
        disable_torch_init()
        self.model_name = get_model_name_from_path(args.model_path)
        print('inference input model args', args.model_path, args.model_base, self.model_name, args.load_8bit, args.load_4bit, args.device)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(args.model_path, args.model_base, self.model_name, args.load_8bit, args.load_4bit, device=args.device)

        if 'llama-2' in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"
        self.conv_mode = conv_mode
        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode
    
        self.args = args
        
    def ask_single_question(self, inp, conv_mode, model_name, image, image_processor, model, tokenizer, japanese=False):
        conv = conv_templates[conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles

        #image = load_image(args.image_file)
        # Similar operation in model_worker.py
        image_tensor = process_images([image], self.image_processor, self.args)
        if type(image_tensor) is list:
            image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

                
        #print(f"{roles[1]}: ", end="")

        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                if japanese:
                    inp+='ただし、日本語で回答してください'
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                if japanese:
                    inp+='ただし、日本語で回答してください'
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        #print(prompt)

        input_ids = tokenizer.encode(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=self.args.temperature,
                max_new_tokens=self.args.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        #outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        outputs = tokenizer.decode(output_ids[0, :]).strip()
        conv.messages[-1][-1] = outputs

        if self.args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
        return outputs
    
    def ask_multiple_question(self, inp, conv_mode, model_name, images, image_processor, model, tokenizer):
        conv = conv_templates[conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles

        #image = load_image(args.image_file)
        # Similar operation in model_worker.py
        image_tensor = process_images(images, self.image_processor, self.args)
        if type(image_tensor) is list:
            image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

                
        #print(f"{roles[1]}: ", end="")

        if images is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            images = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=self.args.temperature,
                max_new_tokens=self.args.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        outputs = tokenizer.decode(output_ids).strip()
        #outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        conv.messages[-1][-1] = outputs

        if self.args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
        return outputs
    
    def _process_images(self, image_paths,prompt, japanese=False, task=None, context=None):
        image_dir = '/home/yamanishi/project/trip_recommend/data/jalan_image_with_caption'
        # image_paths = [os.path.join(image_dir, '五稜郭公園_3.jpg'), os.path.join(image_dir, '銀山温泉_4.jpg'),
        #                os.path.join(image_dir, '函館山_3.jpg'),os.path.join(image_dir, 'すみだ水族館_5.jpg')]
        outputs = []
        for image_path in tqdm(image_paths):
            spot_name = image_path.split('_')[0]
            if task=='review_generation':
                prompt = f'この観光地は{spot_name}です。観光客になったつもりで画像にあったレビューを生成してください'
            elif task=='context_review_generation':
                prompt = f'あなたは{spot_name}を訪れた{context}の観光客です。画像からレビューを生成してください。'
            image = load_image(os.path.join(image_dir, image_path))
            output = self.ask_single_question(prompt, self.conv_mode, self.model_name, image, self.image_processor, self.model, self.tokenizer, japanese=japanese)
            print('output', output)
            outputs.append(output)
            
        return outputs
    
    def inference_spot_names(self, image_paths, japanese=False):
        outputs = self._process_images(image_paths, 'この観光地の名前を教えてください', japanese=japanese)
        return outputs
        
    def generate_reviews(self, image_paths, japanese=False):
        outputs = self._process_images(image_paths, '観光客になったつもりで画像にあったレビューを生成してください', japanese=japanese, task='review_generation')
        return outputs
    
    def generate_review_context(self, image_paths, japanese=False, context='男性'):
        outputs = self._process_images(image_paths, f'あなたは{context}の観光客です。写真からレビューを生成してください', japanese=japanese, task='context_review_generation', context=context)
        return outputs
        
class BLIPInference:
    def __init__(self):
        self.model = AutoModelForVision2Seq.from_pretrained("stabilityai/japanese-instructblip-alpha", load_in_8bit=True, trust_remote_code=True)
        self.processor = BlipImageProcessor.from_pretrained("stabilityai/japanese-instructblip-alpha")
        self.tokenizer = LlamaTokenizer.from_pretrained("novelai/nerdstash-tokenizer-v1", additional_special_tokens=['▁▁'])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #self.model.to(self.device)
        
    @staticmethod
    def build_prompt(prompt="", sep="\n\n### "):
        sys_msg = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。"
        p = sys_msg
        roles = ["指示", "応答"]
        user_query = "与えられた画像について、詳細に述べてください。"
        msgs = [": \n" + user_query, ": "]
        if prompt:
            roles.insert(1, "入力")
            msgs.insert(1, ": \n" + prompt)
        for role, msg in zip(roles, msgs):
            p += sep + role + msg
        return p
    
    def inference(self, image_path, prompt):        
        image = Image.open(image_path).convert("RGB")
        prompt = "" # input empty string for image captioning. You can also input questions as prompts 
        prompt = BLIPInference.build_prompt(prompt)
        inputs = self.processor(images=image, return_tensors="pt")
        text_encoding = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        text_encoding["qformer_input_ids"] = text_encoding["input_ids"].clone()
        text_encoding["qformer_attention_mask"] = text_encoding["attention_mask"].clone()
        inputs.update(text_encoding)

        # generate
        outputs = self.model.generate(
            **inputs.to(self.device, dtype=self.model.dtype),
            num_beams=5,
            max_new_tokens=32,
            min_length=1,
        )
        generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        #print(generated_text)
        return generated_text
    
    def _process_images(self, image_paths, prompt, task=None):
        image_dir = '/home/yamanishi/project/trip_recommend/data/jalan_image_with_caption'
        outputs = []
        for image_path in tqdm(image_paths):
            if task=='review_generation':
                spot_name = image_path.split('_')[0]
                if task=='review_generation':
                    prompt = f'この観光地は{spot_name}です。画像にあったレビューを生成してください'
            output = self.inference(os.path.join(image_dir, image_path), prompt)
            outputs.append(output)
        return outputs
    
    def inference_spot_names(self, image_paths):
        outputs = self._process_images(image_paths, prompt='この観光地の名前を教えて')
        return outputs

    def generate_reviews(self, image_paths):
        outputs = self._process_images(image_paths, prompt='', task='review_generation')
        return outputs
            
class StableVLMInference:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForVision2Seq.from_pretrained("stabilityai/japanese-stable-vlm", load_in_8bit=True, trust_remote_code=True)
        self.processor = AutoImageProcessor.from_pretrained("stabilityai/japanese-stable-vlm")
        self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/japanese-stable-vlm")
        #self.model.to(self.device)
        
    @staticmethod
    def build_prompt(task="caption", input=None, sep="\n\n### ", spot_name=None):
        TASK2INSTRUCTION = {
            "caption": "画像を詳細に述べてください。",
            "tag": "与えられた単語を使って、画像を詳細に述べてください。",
            "vqa": "与えられた画像を下に、質問に答えてください。",
            "spot_name": '画像の観光地の名前を教えて',
            'review_generation': f'この観光地は{spot_name}です。画像に合うレビューを生成して'
        }
        assert (
            task in TASK2INSTRUCTION
        ), f"Please choose from {list(TASK2INSTRUCTION.keys())}"
        if task in ["tag", "vqa"]:
            assert input is not None, "Please fill in `input`!"
            if task == "tag" and isinstance(input, list):
                input = "、".join(input)
        else:
            assert input is None, f"`{task}` mode doesn't support to input questions"
        sys_msg = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。"
        p = sys_msg
        roles = ["指示", "応答"]
        instruction = TASK2INSTRUCTION[task]
        msgs = [": \n" + instruction, ": \n"]
        if input:
            roles.insert(1, "入力")
            msgs.insert(1, ": \n" + input)
        for role, msg in zip(roles, msgs):
            p += sep + role + msg
        return p
    
    def inference(self, task, image_path, spot_name=None):
        image = Image.open(image_path).convert("RGB")
        prompt = StableVLMInference.build_prompt(task=task, spot_name=spot_name)

        inputs = self.processor(images=image, return_tensors="pt")
        text_encoding = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        inputs.update(text_encoding)

        # generate
        outputs = self.model.generate(
            **inputs.to(self.device, dtype=self.model.dtype),
            do_sample=False,
            num_beams=5,
            max_new_tokens=128,
            min_length=1,
            repetition_penalty=1.5,
        )
        generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        return generated_text
    
    def inference_multi(self, task, image_paths, spot_name=None):
        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        prompt = StableVLMInference.build_prompt(task=task, spot_name=spot_name)

        inputs = self.processor(images=images, return_tensors="pt")
        text_encoding = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        inputs.update(text_encoding)

        # generate
        outputs = self.model.generate(
            **inputs.to(self.device, dtype=self.model.dtype),
            do_sample=False,
            num_beams=5,
            max_new_tokens=128,
            min_length=1,
            repetition_penalty=1.5,
        )
        generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        return generated_text
    
    def _process_images(self, image_paths, task):
        image_dir = '/home/yamanishi/project/trip_recommend/data/jalan_image_with_caption'
        outputs = []
        for image_path in tqdm(image_paths):
            spot_name = image_path.split('_')[0]
            output = self.inference(task, os.path.join(image_dir, image_path), spot_name)
            outputs.append(output)
        return outputs
    
    def inference_spot_names(self, image_paths):
        outputs = self._process_images(image_paths, task='spot_name')
        return outputs

    def generate_reviews(self, image_paths):
        outputs = self._process_images(image_paths, task='review_generation')
        return outputs
        # 桜越しの東京スカイツリー
        
class HeronInference:
    def __init__(self):
        device='cuda'
        MODEL_NAME = "turing-motors/heron-chat-blip-ja-stablelm-base-7b-v1"
    
        self.model = VideoBlipForConditionalGeneration.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, ignore_mismatched_sizes=True
        )

        self.model = self.model.half()
        self.model.eval()
        self.model.to(device)

        # prepare a processor
        self.processor = VideoBlipProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.tokenizer = LlamaTokenizer.from_pretrained("novelai/nerdstash-tokenizer-v1", additional_special_tokens=['▁▁'])
        self.processor.tokenizer = self.tokenizer
        
    def inference(self, image_path, prompt):
        image = Image.open(image_path)

        text = f"##human: {prompt}\n##gpt: "

        # do preprocessing
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            truncation=True,
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(self.device, torch.float16)

        # set eos token
        eos_token_id_list = [
            self.processor.tokenizer.pad_token_id,
            self.processor.tokenizer.eos_token_id,
            int(self.tokenizer.convert_tokens_to_ids("##"))
        ]

        # do inference
        with torch.no_grad():
            out = self.model.generate(**inputs, max_length=256, do_sample=False, temperature=0., eos_token_id=eos_token_id_list, no_repeat_ngram_size=2)

        # print result
        print(self.processor.tokenizer.batch_decode(out))
        
class PEPLER:
    def __init__(self, args):
        tokenizer = AutoTokenizer.from_pretrained('rinna/japanese-gpt2-medium')
        corpus = DataLoader('../PEPLER/data/jalan', '../PEPLER/data/jalan', tokenizer, 50)
        nuser, nitem, ntoken = len(corpus.user_dict), len(corpus.item_dict), len(tokenizer)
        self.model = ContinuousPromptLearning.from_pretrained('rinna/japanese-gpt2-medium', nuser, nitem)
        self.device='cuda'
        
    def generate(self, data):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()
        bos,eos,pad = '<bos>','<eos>', '<pad>'
        idss_predict = []
        with torch.no_grad():
            while True:
                user, item, _, seq, _ = data.next_batch()  # data.step += 1
                user = user.to(self.device)  # (batch_size,)
                item = item.to(self.device)
                text = seq[:, :1].to(self.device)  # bos, (batch_size, 1)
                for idx in range(seq.size(1)):
                    # produce a word at each step
                    outputs = self.model(user, item, text, None)
                    last_token = outputs.logits[:, -1, :]  # the last token, (batch_size, ntoken)
                    word_prob = torch.softmax(last_token, dim=-1)
                    token = torch.argmax(word_prob, dim=1, keepdim=True)  # (batch_size, 1), pick the one with the largest probability
                    text = torch.cat([text, token], 1)  # (batch_size, len++)
                ids = text[:, 1:].tolist()  # remove bos, (batch_size, seq_len)
                idss_predict.extend(ids)

                if data.step == data.total_step:
                    break
        tokens_predict = [ids2tokens(ids, self.tokenizer, eos) for ids in idss_predict]
        return tokens_predict


            
class ELYZALLlama:
    def __init__(self):
        text = "クマが海辺に行ってアザラシと友達になり、最終的には家に帰るというプロットの短編小説を書いてください。"

        self.model_name = "elyza/ELYZA-japanese-Llama-2-7b-instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def generate(self, prompts):
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。"
        prompts = ["{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
            bos_token=self.tokenizer.bos_token,
            b_inst=B_INST,
            system=f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}",
            prompt=prompt,
            e_inst=E_INST,
            ) for prompt in prompts]
        model = AutoModelForCausalLM.from_pretrained(self.model_name,)# torch_dtype="auto")
        model.cuda()
        with torch.no_grad():
            token_ids = self.tokenizer.encode(prompts, add_special_tokens=False, return_tensors="pt")

            output_ids = model.generate(
                token_ids.to(model.device),
                max_new_tokens=256,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        output = self.tokenizer.batch_decode(output_ids.tolist()[0][token_ids.size(1) :], skip_special_tokens=True)
        print(output)

class AoKarasuTransformer:
    def __init__(self):
        model_name = 'lightblue/ao-karasu-72B-AWQ-4bit'
        model_name = "lightblue/ao-karasu-72B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

        self.pipe = pipeline("text-generation", model=model, tokenizer=self.tokenizer)
        
    def generate(self, prompts):
        prompt_all = []
        for prompt in prompts:
            messages = [{"role": "system", "content": "あなたはAIアシスタントです。"}]
            messages.append({"role": "user", "content": prompt})
            prompt = self.tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True, tokenize=False)
            prompt_all.append(prompt)
        result = self.pipe(prompt_all, max_new_tokens=500, do_sample=False, temperature=0.1, return_full_text=False)
        return result
    
class AoKarasu:
    def __init__(self, is_4bit=False):
        ray.init(ignore_reinit_error=True)
        if is_4bit:
            self.model_name = 'lightblue/ao-karasu-72B-AWQ-4bit'
        else:
            self.model_name = "lightblue/aokarasu-72B"
        self.is_4bit = is_4bit
        pass
    
    def generate(self, input_prompts, print=False):
        sampling_params = SamplingParams(temperature=0.0, max_tokens=1000)
        model_name = self.model_name
        if self.is_4bit:
            LLM(model=model_name)
        else:
            tensor_parallel_size=4
            llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size, max_model_len=10144)  # noqa: E999
        prompts = []
        for prompt in input_prompts:
            messages = [{"role": "system", "content": "あなたはAIアシスタントです。"}]
            messages.append({"role": "user", "content": prompt})
            prompt = llm.llm_engine.tokenizer.tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True, tokenize=False)
            prompts.append(prompt)
        outputs = llm.generate(prompts, sampling_params)
        #outputs = [output.outputs[0].text for output in outputs]
        if print:
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        return [output.outputs[0].text for output in outputs]
    
class Qarasu:
    def __init__(self, model_name="lightblue/qarasu-14B-chat-plus-unleashed"):
        destroy_model_parallel()
        self.llm = vllm_LLM(model=model_name, dtype='bfloat16', trust_remote_code=True, gpu_memory_utilization=0.9)
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=100)
        
    def generate(self, prompts):
        prompts_llm = []
        for prompt in prompts:
            messages = [{"role": "system", "content": "あなたはAIアシスタントです。"}]
            messages.append({"role": "user", "content": prompt})
            prompt = self.llm.llm_engine.tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True, tokenize=False)
            prompts_llm.append(prompt)
        outputs = self.llm.generate(prompts_llm, self.sampling_params)
        generated_texts = []
        print('outputs', len(outputs))
        with open('./output.pkl', 'wb') as f:
            pickle.dump(outputs, f)
        for k,output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            generated_texts.append(generated_text)
        #ray.shutdown()
        return generated_texts
    
class QarasuTransformer:
    def __init__(self):
        #model_name = 'lightblue/ao-karasu-72B-AWQ-4bit'
        model_name = "lightblue/qarasu-14B-chat-plus-unleashed"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",trust_remote_code=True)

        self.pipe = pipeline("text-generation", model=model, tokenizer=self.tokenizer, trust_remote_code=True)
        
    def generate(self, prompts):
        prompt_all = []
        for prompt in prompts:
            messages = [{"role": "system", "content": "あなたはAIアシスタントです。"}]
            messages.append({"role": "user", "content": prompt})
            prompt = self.tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True, tokenize=False)
            prompt_all.append(prompt)
        result = self.pipe(prompt_all, max_new_tokens=500, do_sample=False, temperature=0.0, return_full_text=False)
        return result
    
class GPT3:
    def __init__(self):
        self.client = OpenAI(
            api_key=api_key.OPENAI_KEY
            )

        # 質問を定義
    def generate(self, prompts):
        # OpenAI GPT-3 APIを呼び出し
        responses = []
        for prompt in tqdm(prompts):

            completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="gpt-3.5-turbo",
            )
            responses.append(completion.choices[0].message.content)
        return responses
    
    def generate_and_save(self, prompts, spots, save_path,):
        # OpenAI GPT-3 APIを呼び出し
        responses = []
        for spot, prompt in tqdm(zip(spots, prompts)):

            completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="gpt-3.5-turbo",
            )
            response = completion.choices[0].message.content
            responses.append(response)
            pd.DataFrame({'user': [spot], 'response': [response]}).to_csv(save_path, 
                                                                          mode='a', header=False, index=False) 
        return responses
    
    
class LLaMA:
    def __init__(self, model_version='2', parameter_num='13', instruct=False, tensor_parallel_size=4):
        # https://huggingface.co/meta-llama
        if model_version == "2":
            self.model_name = f"meta-llama/Llama-{model_version}-{parameter_num}b-hf"
        elif model_version == "3":
            self.model_name = f"meta-llama/Meta-Llama-{model_version}-{parameter_num}B"
        if instruct:
            self.model_name += '-Instruct'
        self.tensor_parallel_size = tensor_parallel_size
            
    def generate(self, input_prompts,):
        sampling_params = SamplingParams(temperature=0.1, max_tokens=2000, repetition_penalty=1.1)
        model_name = self.model_name
        llm = LLM(model=model_name, tensor_parallel_size=self.tensor_parallel_size,
                  gpu_memory_utilization=0.9,dtype="bfloat16")  # noqa: E999

        outputs = llm.generate(input_prompts, sampling_params)

        return [output.outputs[0].text for output in outputs]
    