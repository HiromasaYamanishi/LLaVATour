import os
import pickle
from io import BytesIO

#import api_key
import openai
import copy
# from heron.models.video_blip import VideoBlipForConditionalGeneration, VideoBlipProcessor
import pandas as pd
import requests
import torch
from dotenv import load_dotenv
from tqdm import tqdm
import base64
import time
from openai import OpenAI
from PIL import Image
from tqdm import tqdm
import csv
from transformers import (
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoTokenizer,
    BlipImageProcessor,
    LlamaTokenizer,
    TextStreamer,
    pipeline,
    BertJapaneseTokenizer, 
    BertModel
)
from vllm import LLM
from vllm import LLM as vllm_LLM
from vllm import SamplingParams
#from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
import importlib.metadata
import pickle
from preprocess.clip import CLIP
from collections import defaultdict
# パッケージのバージョンを取得
try:
    version = importlib.metadata.version('transformers')
except importlib.metadata.PackageNotFoundError:
    version = None
class GPT3:
    def __init__(self,):
        load_dotenv('.env')
        openai.api_key = os.environ.get("OPENAI_KEY")
        self.api_key = os.environ.get("OPENAI_KEY")
        self.gpt_model = "gpt-3.5-turbo"
        self.client = OpenAI(
            # This is the default and can be omitted
            api_key=os.environ.get("OPENAI_KEY"),
        )
        
    def get_openai_response(self, prompt, max_token=100, ):

        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.gpt_model,
        )
        return chat_completion.choices[0].message.content
        
    def process_images(
        self, image_paths, prompts, max_token=100):
        results = []
        for prompt in tqdm(prompts):
            result = self.get_openai_response(prompt, max_token)
            print('result', result)
            results.append(result)
            time.sleep(1)
        return results
        
class PEPLER:
    def __init__(self, args):
        tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-medium")
        corpus = DataLoader(
            "../PEPLER/data/jalan", "../PEPLER/data/jalan", tokenizer, 50
        )
        nuser, nitem, ntoken = (
            len(corpus.user_dict),
            len(corpus.item_dict),
            len(tokenizer),
        )
        self.model = ContinuousPromptLearning.from_pretrained(
            "rinna/japanese-gpt2-medium", nuser, nitem
        )
        self.device = "cuda"

    def generate(self, data):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()
        bos, eos, pad = "<bos>", "<eos>", "<pad>"
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
                    last_token = outputs.logits[
                        :, -1, :
                    ]  # the last token, (batch_size, ntoken)
                    word_prob = torch.softmax(last_token, dim=-1)
                    token = torch.argmax(
                        word_prob, dim=1, keepdim=True
                    )  # (batch_size, 1), pick the one with the largest probability
                    text = torch.cat([text, token], 1)  # (batch_size, len++)
                ids = text[:, 1:].tolist()  # remove bos, (batch_size, seq_len)
                idss_predict.extend(ids)

                if data.step == data.total_step:
                    break
        tokens_predict = [ids2tokens(ids, self.tokenizer, eos) for ids in idss_predict]
        return tokens_predict

class ElyzaLLama3:
    def __init__(self, tensor_parallel_size):
        self.tensor_parallel_size = tensor_parallel_size
        pass
    
    def generate(self, prompts):
        llm = LLM(model="elyza/Llama-3-ELYZA-JP-8B", tensor_parallel_size = self.tensor_parallel_size)
        tokenizer = llm.get_tokenizer()

        DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。特に指示が無い場合は、常に日本語で回答してください。"
        sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=1000)
        prompts_new = []
        for prompt in prompts:
            prompts_new.append([
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],)
        prompts = [
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in prompts_new
        ]

        outputs = llm.generate(prompts, sampling_params)
        outputs = [output.outputs[0].text for output in outputs]
        return outputs
    
class ElyzaLLama3AWQ:
    def __init__(self, tensor_parallel_size):
        self.tensor_parallel_size = tensor_parallel_size
        pass
    
    def generate(self, prompts):
        llm = LLM(model="elyza/Llama-3-ELYZA-JP-8B-AWQ", quantization="awq", tensor_parallel_size = self.tensor_parallel_size)
        tokenizer = llm.get_tokenizer()

        DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。特に指示が無い場合は、常に日本語で回答してください。"
        sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=1000)
        prompts_new = []
        for prompt in prompts:
            prompts_new.append([
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],)
        prompts = [
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in prompts_new
        ]

        outputs = llm.generate(prompts, sampling_params)
        outputs = [output.outputs[0].text for output in outputs]
        return outputs

class ELYZALLlama:
    def __init__(self):
        text = "クマが海辺に行ってアザラシと友達になり、最終的には家に帰るというプロットの短編小説を書いてください。"

        self.model_name = "elyza/ELYZA-japanese-Llama-2-7b-instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def generate(self, prompts):
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。"
        prompts = [
            "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
                bos_token=self.tokenizer.bos_token,
                b_inst=B_INST,
                system=f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}",
                prompt=prompt,
                e_inst=E_INST,
            )
            for prompt in prompts
        ]
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
        )  # torch_dtype="auto")
        model.cuda()
        with torch.no_grad():
            token_ids = self.tokenizer.encode(
                prompts, add_special_tokens=False, return_tensors="pt"
            )

            output_ids = model.generate(
                token_ids.to(model.device),
                max_new_tokens=256,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        output = self.tokenizer.batch_decode(
            output_ids.tolist()[0][token_ids.size(1) :], skip_special_tokens=True
        )
        print(output)


class AoKarasuTransformer:
    def __init__(self):
        # model_name = "lightblue/ao-karasu-72B-AWQ-4bit"
        model_name = "lightblue/ao-karasu-72B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

        self.pipe = pipeline("text-generation", model=model, tokenizer=self.tokenizer)

    def generate(self, prompts):
        prompt_all = []
        for prompt in prompts:
            messages = [{"role": "system", "content": "あなたはAIアシスタントです。"}]
            messages.append({"role": "user", "content": prompt})
            prompt = self.tokenizer.apply_chat_template(
                conversation=messages, add_generation_prompt=True, tokenize=False
            )
            prompt_all.append(prompt)
        result = self.pipe(
            prompt_all,
            max_new_tokens=500,
            do_sample=False,
            temperature=0.1,
            return_full_text=False,
        )
        return result

class Gemma2:
    def __init__(self, is_4bit=False, tensor_parallel_size=4):
        # ray.init(ignore_reinit_error=True)
        self.model_name = "UCLA-AGI/Gemma-2-9B-It-SPPO-Iter3"
        #self.model_name = "google/gemma-2-2b-it"

        self.is_4bit = is_4bit
        self.tensor_parallel_size = tensor_parallel_size
        pass

    def generate(self, input_prompts, print=False):
        sampling_params = SamplingParams(temperature=0.0, max_tokens=1200, repetition_penalty=1.05)
        model_name = self.model_name
        #self.tensor_parallel_size = 8
        if self.is_4bit:
            llm = LLM(model=model_name,
                tensor_parallel_size=self.tensor_parallel_size,)
        else:
            llm = LLM(
                model=model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                #max_model_len=8192,
            )  # noqa: E999
        prompts = []
        for prompt in input_prompts:
            #messages = [{"role": "system", "content": "あなたはAIアシスタントです。"}]
            messages = [{"role": "user", "content": prompt}]
            # messages.append({"role": "user", "content": prompt})
            prompt = llm.llm_engine.tokenizer.tokenizer.apply_chat_template(
                conversation=messages, add_generation_prompt=True, tokenize=False
            )
            prompts.append(prompt)
        outputs = llm.generate(prompts, sampling_params)
        # outputs = [output.outputs[0].text for output in outputs]
        if print:
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        return [output.outputs[0].text for output in outputs]
    
class Phi3:
    def __init__(self, size='medium', context='128k', is_4bit=False, tensor_parallel_size=4):
        # ray.init(ignore_reinit_error=True)
        self.model_name = f'microsoft/Phi-3-{size}-{context}-instruct'

        self.is_4bit = is_4bit
        self.tensor_parallel_size = tensor_parallel_size
        pass

    def generate(self, input_prompts, print=False):
        sampling_params = SamplingParams(temperature=0.0, max_tokens=1200, repetition_penalty=1.1)
        model_name = self.model_name
        #self.tensor_parallel_size = 8
        if self.is_4bit:
            llm = LLM(model=model_name,
                tensor_parallel_size=self.tensor_parallel_size,)
        else:
            llm = LLM(
                model=model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                trust_remote_code=True,
                max_model_len=65536,
            )  # noqa: E999
        prompts = []
        for prompt in input_prompts:
            messages = [{"role": "system", "content": "あなたはAIアシスタントです。"}]
            messages.append({"role": "user", "content": prompt})
            prompt = llm.llm_engine.tokenizer.tokenizer.apply_chat_template(
                conversation=messages, add_generation_prompt=True, tokenize=False
            )
            prompts.append(prompt)
        outputs = llm.generate(prompts, sampling_params)
        # outputs = [output.outputs[0].text for output in outputs]
        if print:
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        return [output.outputs[0].text for output in outputs]

class Calm3:
    def __init__(self, is_4bit=False, tensor_parallel_size=4):
        # ray.init(ignore_reinit_error=True)
        self.model_name = "cyberagent/calm3-22b-chat"

        self.is_4bit = is_4bit
        self.tensor_parallel_size = tensor_parallel_size
        pass

    def generate(self, input_prompts, print=False):
        sampling_params = SamplingParams(temperature=0.0, max_tokens=1000)
        model_name = self.model_name
        #self.tensor_parallel_size = 8
        if self.is_4bit:
            llm = LLM(model=model_name,
                tensor_parallel_size=self.tensor_parallel_size,)
        else:
            llm = LLM(
                model=model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                max_model_len=10144,
            )  # noqa: E999
        prompts = []
        for prompt in input_prompts:
            messages = [{"role": "system", "content": "あなたはAIアシスタントです。"}]
            messages.append({"role": "user", "content": prompt})
            prompt = llm.llm_engine.tokenizer.tokenizer.apply_chat_template(
                conversation=messages, add_generation_prompt=True, tokenize=False
            )
            prompts.append(prompt)
        outputs = llm.generate(prompts, sampling_params)
        # outputs = [output.outputs[0].text for output in outputs]
        if print:
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        return [output.outputs[0].text for output in outputs]




class AoKarasu:
    def __init__(self, is_4bit=False, tensor_parallel_size=4):
        # ray.init(ignore_reinit_error=True)
        if is_4bit:
            self.model_name = "lightblue/ao-karasu-72B-AWQ-4bit"
        else:
            self.model_name = "lightblue/aokarasu-72B"
        self.is_4bit = is_4bit
        self.tensor_parallel_size = tensor_parallel_size
        pass

    def generate(self, input_prompts, print=False):
        sampling_params = SamplingParams(temperature=0.0, max_tokens=1000)
        model_name = self.model_name
        self.tensor_parallel_size = 8
        if self.is_4bit:
            llm = LLM(model=model_name,
                tensor_parallel_size=self.tensor_parallel_size,)
        else:
            llm = LLM(
                model=model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                max_model_len=10144,
            )  # noqa: E999
        prompts = []
        for prompt in input_prompts:
            messages = [{"role": "system", "content": "あなたはAIアシスタントです。"}]
            messages.append({"role": "user", "content": prompt})
            prompt = llm.llm_engine.tokenizer.tokenizer.apply_chat_template(
                conversation=messages, add_generation_prompt=True, tokenize=False
            )
            prompts.append(prompt)
        outputs = llm.generate(prompts, sampling_params)
        # outputs = [output.outputs[0].text for output in outputs]
        if print:
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        return [output.outputs[0].text for output in outputs]


class Qarasu:
    def __init__(self, model_name="lightblue/qarasu-14B-chat-plus-unleashed"):
        #destroy_model_parallel()
        self.llm = vllm_LLM(
            model=model_name,
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
        )
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=100)

    def generate(self, prompts):
        prompts_llm = []
        for prompt in prompts:
            messages = [{"role": "system", "content": "あなたはAIアシスタントです。"}]
            messages.append({"role": "user", "content": prompt})
            prompt = self.llm.llm_engine.tokenizer.apply_chat_template(
                conversation=messages, add_generation_prompt=True, tokenize=False
            )
            prompts_llm.append(prompt)
        outputs = self.llm.generate(prompts_llm, self.sampling_params)
        generated_texts = []
        print("outputs", len(outputs))
        with open("./output.pkl", "wb") as f:
            pickle.dump(outputs, f)
        for k, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            generated_texts.append(generated_text)
        # ray.shutdown()
        return generated_texts


class QarasuTransformer:
    def __init__(self):
        # model_name = 'lightblue/ao-karasu-72B-AWQ-4bit'
        model_name = "lightblue/qarasu-14B-chat-plus-unleashed"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", trust_remote_code=True
        )

        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            trust_remote_code=True,
        )

    def generate(self, prompts):
        prompt_all = []
        for prompt in prompts:
            messages = [{"role": "system", "content": "あなたはAIアシスタントです。"}]
            messages.append({"role": "user", "content": prompt})
            prompt = self.tokenizer.apply_chat_template(
                conversation=messages, add_generation_prompt=True, tokenize=False
            )
            prompt_all.append(prompt)
        result = self.pipe(
            prompt_all,
            max_new_tokens=500,
            do_sample=False,
            temperature=0.0,
            return_full_text=False,
        )
        return result




class LLaMA:
    def __init__(
        self,
        model_version="2",
        parameter_num="13",
        instruct=False,
        tensor_parallel_size=4,
    ):
        # https://huggingface.co/meta-llama
        if model_version == "2":
            self.model_name = f"meta-llama/Llama-{model_version}-{parameter_num}b-hf"
        elif model_version == "3":
            self.model_name = f"meta-llama/Meta-Llama-{model_version}-{parameter_num}B"
        if instruct:
            self.model_name += "-Instruct"
        self.tensor_parallel_size = tensor_parallel_size

    def generate(
        self,
        input_prompts,
    ):
        sampling_params = SamplingParams(
            temperature=0.1, max_tokens=2000, repetition_penalty=1.1
        )
        model_name = self.model_name
        llm = LLM(
            model=model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=0.9,
            dtype="bfloat16",
        )  # noqa: E999

        outputs = llm.generate(input_prompts, sampling_params)

        return [output.outputs[0].text for output in outputs]

    
class SentenceBertJapanese:
    def __init__(self, model_name_or_path="sonoisa/sentence-bert-base-ja-mean-tokens-v2", device=None):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
        self.model = BertModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode(self, sentences, batch_size=100):
        all_embeddings = []
        iterator = tqdm(range(0, len(sentences), batch_size))
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest", 
                                           truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        # return torch.stack(all_embeddings).numpy()
        return torch.stack(all_embeddings)
    