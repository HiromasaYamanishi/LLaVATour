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
import re
import time
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
import math
from torchvision.transforms.functional import InterpolationMode
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from transformers import AutoModel, AutoTokenizer
from torchvision.transforms.functional import InterpolationMode
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
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
from utils import overlay_attention
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
    
    
if version and version>="4.40":    
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
# sys.path.append('..'
# )
# from PEPLER.module import ContinuousPromptLearning
# from PEPLER.utils import DataLoader, ids2tokens


def load_image(image_file):
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

class LLaVANext:
    def __init__(self, args):
        self.args = args
        self.model_name = args.model_name
        self.model = LlavaNextForConditionalGeneration.from_pretrained(args.model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True) 
        self.model.to("cuda")
        self.processor = LlavaNextProcessor.from_pretrained(args.model_name)
        
    def inferece(self, image_path, prompt):
        image = Image.open(image_path).convert('RGB')
        if '13' in self.args.model_name:
            prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{prompt} ASSISTANT:"
        elif '34' in self.args.model_name:
            prompt = f"<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\n{prompt}<|im_end|><|im_start|>assistant\n"
        
        inputs = self.processor(prompt, image, return_tensors="pt").to("cuda:0")
        # autoregressively complete prompt
        output = self.model.generate(**inputs, max_new_tokens=200)
        return output
    
    def process_images(self, image_paths, prompts):
        results = []
        for image_path, prompt in tqdm(zip(image_paths, prompts)):
            result = self.inference(image_path, prompt)
            results.append(result)
        return results

class QwenVLChat:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()
        self.model.cuda()
        
    def inference(self, image_path, prompt):
        query = self.tokenizer.from_list_format([
                {'image': image_path},  # Either a local path or an url
                {'text': prompt},
        ])

        with torch.inference_mode():
            outputs, history = self.model.chat(self.tokenizer, query=query, history=None)
        return outputs
    
    def process_images(self, image_paths, prompts):
        results = []
        for image_path, prompt in tqdm(zip(image_paths, prompts)):
            result = self.inference(image_path, prompt)
            #print(result)
            results.append(result)
        return results
    
    
class LLaVANextLarge:
    def __init__(self, args):
        self.args = args
        #pretrained = "lmms-lab/llama3-llava-next-8b"
        pretrained = args.model_name
        model_name, self.device, device_map = "llava_llama3", "cuda", "auto"
        print('conv temp', conv_templates)
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map) 
        self.model.eval()
        self.model.tie_weights()
        
    def _inference_one(self, image_path, prompt):
        image = Image.open(image_path)
        image_tensor = process_images([image], self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
        #conv_template = "llava_llama_3" # Make sure you use correct chat template for different models
        conv_template = "qwen_1_5"
        question = DEFAULT_IMAGE_TOKEN + prompt
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        #print('prompt', prompt_question)
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to('cuda')
        print('input ids', input_ids)
        image_sizes = [image.size]
        cont = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=32,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
        print(text_outputs)
        return text_outputs
        
    def inference(self, image_paths, prompts):
        results = []
        for image_path, prompt in tqdm(zip(image_paths, prompts)):
            result = self._inference_one(image_path, prompt)
            results.append(result)
        return result
    
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


    
class GPT4:
    def __init__(self, model_name="gpt-4-vision-preview"):
        load_dotenv('.env')
        openai.api_key = os.environ.get("OPENAI_KEY")
        self.api_key = os.environ.get("OPENAI_KEY")
        self.gpt_model = model_name
        self.image_dir = "/home/yamanishi/project/trip_recommend/data/jalan_image_with_caption"
        self.client = OpenAI(
            # This is the default and can be omitted
            api_key=os.environ.get("OPENAI_KEY"),
        )
        
        
    def process_image(self, image_path, prompt, max_token=100):
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

        # Getting the base64 string

        base64_image = encode_image(os.path.join(self.image_dir, image_path))

        response = self.client.chat.completions.create(
            model=self.gpt_model,
            messages=[
                {"role": "system", "content": "あなたは優秀なAIアシスタントです"},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]}
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content #response.choices[0].message.content
    
    def process_image_with_retrieval(self, image_path, prompt, retrieve_num=5, max_token=100):
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

        # Getting the base64 string
        spot_name = image_path.split('_')[0]
        reviews = self.spot2reviews[spot_name]
        if len(reviews):
            retrieved_review = self.clip.retrieve_text_from_image_topk([os.path.join('/home/yamanishi/project/trip_recommend/data/jalan_image_with_caption/', image_path)], 
                                                                        reviews,
                                                                        topk=retrieve_num)
            review_sents = '\n'.join(retrieved_review)
            prompt+=f'ただし, 次のレビューはこの観光地に対するあるレビューです.\n {review_sents}\n'
            prompt+='日本語100文字くらいで生成してください'
        print('retrieved prompt', prompt)
        base64_image = encode_image(os.path.join(self.image_dir, image_path))

        response = self.client.chat.completions.create(
            model=self.gpt_model,
            messages=[
                {"role": "system", "content": "あなたは優秀なAIアシスタントです"},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]}
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content #response.choices[0].message.content

    def process_prompt_text_only(self, prompt):
        response = self.client.chat.completions.create(
            model=self.gpt_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
   
    def process_images(
        self, image_paths, prompts, max_token=100, test_df=None, save_path=None, retrieval=False, retrieve_num=1):
        results = []
        if os.path.exists(save_path):
            df = pd.read_csv(save_path)
            start_count = len(df)
        else:
            start_count = 0
        
        if retrieval:
            #df_review = pd.read_pickle('/home/yamanishi/project/trip_recommend/data/jalan/review/review_all_period_.pkl')
            df_review = pd.read_csv('/home/yamanishi/project/airport/src/analysis/LLaVA/data/dataset_personalize/review_train.csv')
            df_test = pd.read_csv('./data/df_review_feature_eval.csv')
            test_reviews = df_test['conversations'].values
            self.spot2reviews = defaultdict(list)
            for spot, review in zip(df_review['spot'], df_review['review']):
                if review in test_reviews:continue
                self.spot2reviews[spot].append(review)   
            self.clip = CLIP()
             
        # already_prompts = df['prompt'].values
        for i, (image_path, prompt) in tqdm(enumerate(zip(image_paths, prompts))):
            if i<start_count:continue
            #if prompt in already_prompts:continue
            if image_path is not None and retrieval:
                result = self.process_image_with_retrieval(image_path, prompt,retrieve_num=retrieve_num, max_token=max_token)
            elif image_path is not None and not retrieval:
                result = self.process_image(image_path, prompt, max_token)
            else:
                result = self.process_prompt_text_only(prompt)
            test_df.loc[i, 'predicted'] = result
            
            with open(save_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if i==0:writer.writerow(test_df.columns)
                writer.writerow(test_df.loc[i].values)
                
            print('result', result)
            results.append(result)
            time.sleep(1)
        return results


        
class LLaVAInference:
    def __init__(self, args):
        # Model
        disable_torch_init()
        self.model_name = get_model_name_from_path(args.model_path)
        print('model name', self.model_name)
        (
            self.tokenizer,
            self.model,
            self.image_processor,
            self.context_len,
        ) = load_pretrained_model(
            args.model_path,
            args.model_base,
            self.model_name,
            args.load_8bit,
            args.load_4bit,
            device=args.device,
        )
        print('load model')
        if "llama-2" in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"
        self.conv_mode = conv_mode
        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print(
                "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                    conv_mode, args.conv_mode, args.conv_mode
                )
            )
        else:
            args.conv_mode = conv_mode

        self.args = args
        self.image_dir = (
            "/home/yamanishi/project/trip_recommend/data/jalan_image_with_caption"
        )

    def ask_single_question(
        self,
        inp,
        conv_mode,
        model_name,
        image,
        image_processor,
        model,
        tokenizer,
        topk=False,
        k=1,
        japanese=False,
        output_attention=False,
        image_path=False,
        task=None,
    ):
        conv = conv_templates[conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ("user", "assistant")
        else:
            roles = conv.roles

        # image = load_image(args.image_file)
        # Similar operation in model_worker.py
        if image is not None:
            image_tensor = process_images([image], self.image_processor, self.args)
            if type(image_tensor) is list:
                image_tensor = [
                    image.to(self.model.device, dtype=torch.float16)
                    for image in image_tensor
                ]
            else:
                image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        # print(f"{roles[1]}: ", end="")
        print('output attention', output_attention)
        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + inp
                )
                if japanese:
                    inp += "ただし、日本語で回答してください"
            else:
                inp = DEFAULT_IMAGE_TOKEN + "\n" + inp
                if japanese:
                    inp += "ただし、日本語で回答してください"
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
            image_tensor=None
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = (
            #tokenizer.encode(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        do_sample = not (self.args.temperature==0)
        with torch.inference_mode():
            print('task', task)
            if not topk and not output_attention:
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=do_sample,
                    repetition_penalty=1.1,
                    temperature=self.args.temperature,
                    max_new_tokens=self.args.max_new_tokens,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )
                outputs = tokenizer.decode(output_ids[0, :]).strip()
            elif output_attention:
                print('attention')
                output_ids, attention, labels = model.generate_with_attention(
                    input_ids,
                    images=image_tensor,
                    do_sample=do_sample,
                    repetition_penalty=1.1,
                    temperature=self.args.temperature,
                    max_new_tokens=self.args.max_new_tokens,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )
                #print('labels', labels)
                #print([i.shape for i in attention[0]], [i.shape for i in attention[-1]])
                index = (labels[0] == -100).nonzero(as_tuple=True)[0][0]
                #print('index', index)
                for i in range(len(attention)):
                    attention_image = torch.mean(attention[i][-1], dim=1).reshape(-1)
                    #print('attention_image.shape', attention_image.shape, len(labels[0]), index)
                    # print('image index', index)
                    # max_index = torch.argmax(attention_image)
                    # print('attention max index', torch.argmax(attention_image), torch.max(attention_image))
                    # print('attention high region', sum(attention_image[:max_index+5]))
                    # print('sum attention', sum(attention_image))
                    attention_image = attention_image[index:index+576]
                    attention_image = attention_image/sum(attention_image)
                    #print('sum attention image', sum(attention_image))
                    attention_image = attention_image.reshape(24, 24)
                    #print('attention_image', attention_image)
                    image_suffix = image_path.split('.')[0]
                    image_suffix = image_path.split('.')[0]
                    if not os.path.exists(f'./result/attention/{task}/{self.model_name}/{image_suffix}'):
                        os.makedirs(f'./result/attention/{task}/{self.model_name}/{image_suffix}')
                    overlay_attention(os.path.join(self.image_dir, image_path), attention_image, f'./result/attention/{task}/{self.model_name}/{image_suffix}/{i}.jpg')
                #print('attention_image', attention_image)
                #print('outputs with attention', output_ids,)
                outputs = tokenizer.decode(output_ids[0, :]).strip()
                print('outputs', outputs)
            elif topk:
                if k<=5:
                    num_beams = num_beam_groups = num_return_sequences = k
                else:
                    #num_beam_groups = k//2
                    num_beam_groups = num_beams = num_return_sequences = k
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=False,  # サンプリングを有効に
                    temperature=1.1,  # 例えば0.7や1.0など
                    top_k=50,  # top_kサンプリング
                    top_p=0.8,
                    num_beams=num_beams,
                    num_beam_groups=num_beam_groups,
                    num_return_sequences=num_return_sequences,
                    max_new_tokens=self.args.max_new_tokens,
                    use_cache=True,
                    diversity_penalty=0.9,
                    stopping_criteria=[stopping_criteria],
                )
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                outputs = ','.join(outputs)
                print('outputs', outputs)
        # outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        #outputs = tokenizer.decode(output_ids[0, :]).strip()
        conv.messages[-1][-1] = outputs

        if self.args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
        return outputs

    def ask_multiple_question(
        self, inp, conv_mode, model_name, images, image_processor, model, tokenizer
    ):
        conv = conv_templates[conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ("user", "assistant")
        else:
            roles = conv.roles

        # image = load_image(args.image_file)
        # Similar operation in model_worker.py
        image_tensor = process_images(images, self.image_processor, self.args)
        if type(image_tensor) is list:
            image_tensor = [
                image.to(self.model.device, dtype=torch.float16)
                for image in image_tensor
            ]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        # print(f"{roles[1]}: ", end="")

        if images is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + inp
                )
            else:
                inp = DEFAULT_IMAGE_TOKEN + "\n" + inp
            conv.append_message(conv.roles[0], inp)
            images = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )
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
                stopping_criteria=[stopping_criteria],
            )
        outputs = tokenizer.decode(output_ids).strip()
        # outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        conv.messages[-1][-1] = outputs

        if self.args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
        return outputs

    def process_images(
        self, image_paths, prompts, japanese=False, task=None, context=None, debug_prompt=False, output_attention=False
    ):
        image_dir = (
            "/home/yamanishi/project/trip_recommend/data/jalan_image_with_caption"
        )
        # image_dir = (
        #     "/home/yamanishi/project/trip_recommend/data/jalan_image/kumamoto"
        # )
        # image_paths = [os.path.join(image_dir, '五稜郭公園_3.jpg'), os.path.join(image_dir, '銀山温泉_4.jpg'),
        #                os.path.join(image_dir, '函館山_3.jpg'),os.path.join(image_dir, 'すみだ水族館_5.jpg')]
        outputs = []
        for i,image_path in tqdm(enumerate(image_paths)):
            
            if task == "review_generation":
                spot_name = image_path.split("_")[0]
                prompt = f"この観光地は{spot_name}です。観光客になったつもりで画像にあったレビューを生成してください"
            elif task == "context_review_generation":
                spot_name = image_path.split("_")[0]
                prompt = f"あなたは{spot_name}を訪れた{context}の観光客です。画像からレビューを生成してください。"
            if type(prompts) is list:
                prompt = prompts[i]
            if debug_prompt:
                print('prompt', prompt)
            #print(os.path.join(image_dir, image_path))
            if image_path is not None:
                image = load_image(os.path.join(image_dir, image_path))
            else:
                image = None
            #print('image', image)
            if task in ['spot_name_topk', 'sequential_topk']:
                output = self.ask_single_question(
                    prompt,
                    self.conv_mode,
                    self.model_name,
                    image,
                    self.image_processor,
                    self.model,
                    self.tokenizer,
                    topk=True,
                    k=6,
                    japanese=japanese,
                    output_attention=output_attention,
                    image_path=image_path,
                    task=task
                )
            else:
                output = self.ask_single_question(
                    prompt,
                    self.conv_mode,
                    self.model_name,
                    image,
                    self.image_processor,
                    self.model,
                    self.tokenizer,
                    japanese=japanese,
                    output_attention=output_attention,
                    image_path=image_path,
                    task=task
                )
            #print("output", output)
            outputs.append(output)

        return outputs

    def inference_spot_names(self, image_paths, japanese=False):
        outputs = self.process_images(
            image_paths, ["この日本の観光地の名前を教えてください。ただし地名のみ答えて." for _ in range(len(image_paths))], japanese=japanese, task='spot_name', output_attention=self.args.output_attention
        )
        return outputs
    
    def inference_spot_names_topk(self, image_paths, japanese=False):
        outputs = self.process_images(
            image_paths, ["この観光地の名前を教えてください" for _ in range(len(image_paths))], japanese=japanese, task='spot_name_topk'
        )
        return outputs
    
    def inference_tag_count(self, image_paths, prompts, japanese=False):
        outputs = self.process_images(
            image_paths, 
            prompts
        )
        return outputs

    def generate_reviews(self, image_paths, prompts, japanese=False):
        outputs = self.process_images(
            image_paths,
            prompts,
            japanese=japanese,
            task="review_generation",
            debug_prompt=True,
        )
        return outputs

    def generate_review_context(self, image_paths, japanese=False, context="男性"):
        outputs = self.process_images(
            image_paths,
            f"あなたは{context}の観光客です。写真からレビューを生成してください",
            japanese=japanese,
            task="context_review_generation",
            context=context,
        )
        return outputs

class LLaVARetrieveInfernce(LLaVAInference):
    def __init__(self, args):
        super().__init__(args)
        if args.retrieve_method in ['triplet', 'entity']:
            with open('./data/kg/sakg_adj.pkl', 'rb') as f:
                self.sakg = pickle.load(f)
        elif args.retrieve_method == 'posneg':
            self.pos_summary = pd.read_csv('./data/pos_spot_summary_calm.csv').groupby('spot').head(1)
            self.neg_summary = pd.read_csv('./data/neg_spot_summary_calm.csv').groupby('spot').head(1)
            self.spot2pos = dict(zip(self.pos_summary['spot'], self.pos_summary['summary']))
            self.spot2neg = dict(zip(self.neg_summary['spot'], self.neg_summary['summary']))
        elif args.retrieve_method == 'summary':
            df_summary_pre = pd.read_csv('./data/spot_review_summary_diverse_re_pre.csv')
            df_summary_middle = pd.read_csv('./data/spot_review_summary_diverse_re_middle.csv')
            df_summary_post = pd.read_csv('./data/spot_review_summary_diverse_re_post.csv')
            df_summary = pd.concat([df_summary_pre, df_summary_middle, df_summary_post]).reset_index()
            spot2summary = {}
            for ind in tqdm(range(len(df_summary))):
                spot = df_summary.loc[ind, 'spot']
                summary = df_summary.loc[ind, 'output'].replace('*', '').replace('**', '').replace('\n\n', '\n')
                spot2summary[spot] = summary
            self.spot2summary = spot2summary

        elif args.retrieve_method == 'review':
            self.spot2reviews = defaultdict(list)
            df_review = pd.read_csv('./data/dataset_personalize/review_train.csv')
            for spot, review in zip(df_review['spot'], df_review['review']):
                self.spot2reviews[spot].append(review)   
            self.clip = CLIP()

        self.image_dir = (
            "/home/yamanishi/project/trip_recommend/data/jalan_image_with_caption"
        )

        self.retrieve_method = args.retrieve_method


    
    def retrieve_triplet(self, start_entity, prompt, top_entity_num=3, top_relation_num=3):
        if start_entity is None:return []
        chosen_triplets = []
        if start_entity not in self.sakg:return []

        if '名前' in prompt or '名称' in prompt:return []
        
        neighbors = list(self.sakg.neighbors(start_entity))
        if not neighbors:return []
        # print('neighbors', neighbors)
        # entityから出るrelationのcountの総和を計算し、上位3つを選択
        neighbor_weights = []
        for neighbor in neighbors:
            total_count = sum(self.sakg[start_entity][neighbor]['relations'].values())
            neighbor_weights.append((neighbor, total_count))
        
        # 重みの降順でソートし、上位3つを選択
        top_neighbors = sorted(neighbor_weights, key=lambda x: x[1], reverse=True)[:top_entity_num]
        
        for neighbor, _ in top_neighbors:
            edge_data = self.sakg[start_entity][neighbor]['relations']
            
            if not edge_data:
                continue
            
            # relationを重みの降順でソートし、上位3つを選択
            top_relations = sorted(edge_data.items(), key=lambda x: x[1], reverse=True)[:top_relation_num]
            
            for relation, count in top_relations:
                chosen_triplets.append(str((neighbor, relation, count)))
                
       #  print('triplets', chosen_triplets)
        return chosen_triplets

    def retrieve_entity(self, start_entity, prompt, top_entity_num=3, top_relation_num=3):
        if start_entity is None:return []
        chosen_triplets = []
        if start_entity not in self.sakg:return []

        if '名前' in prompt or '名称' in prompt:return []
        
        neighbors = list(self.sakg.neighbors(start_entity))
        if not neighbors:return []
        # print('neighbors', neighbors)
        # entityから出るrelationのcountの総和を計算し、上位3つを選択
        neighbor_weights = []
        for neighbor in neighbors:
            total_count = sum(self.sakg[start_entity][neighbor]['relations'].values())
            neighbor_weights.append((neighbor, total_count))
        
        # 重みの降順でソートし、上位3つを選択
        top_neighbors = sorted(neighbor_weights, key=lambda x: x[1], reverse=True)[:top_entity_num]
        top_neighbors = [n[0] for n in top_neighbors]
        return top_neighbors

    def retrieve_posneg(self, entity):
        def extract_first_section(text):
            # 行ごとに分割
            lines = text.strip().split('\n')
            section = []
            
            for line in lines:
                if line.strip() == '':  # 空行が見つかったら終了
                    break
                section.append(line)
            
            # リストを文字列に戻す
            return '\n'.join(section) if section else None

        def remove_duplicates(text):
            # 文章を分割する（例: 、で区切る）
            parts = text.split('、')
            
            # 重複部分を検出し、除去する
            seen = set()
            unique_parts = []
            for part in parts:
                if part not in seen:
                    unique_parts.append(part)
                    seen.add(part)
            
            # 再構築する
            return '、'.join(unique_parts)

        sent = ''
        if entity in self.spot2pos:
            pos = self.spot2pos[entity]
            pos = remove_duplicates(pos)
            sent += 'ただし, 観光地のポジティブな意見は次です\n' + self.spot2pos[entity]
 
        if entity in self.spot2pos:
            neg = extract_first_section(self.spot2neg[entity])
            neg = remove_duplicates(neg)
            if neg is not None:
                sent += '観光地のネガティブな意見は次です\n' + neg
        
        return sent

    def convert_to_natural_language(self, data):
        # 名詞と形容詞の組み合わせを保持する辞書
        aspects = {}
        
        for item in data:
            # 文字列をタプルに変換
            item_tuple = eval(item)
            noun, adj, count = item_tuple
            
            # 辞書に追加
            if noun not in aspects:
                aspects[noun] = []
            aspects[noun].append((adj, count))
        
        sentences = []
        for noun, adj_counts in aspects.items():
            desc = []
            for adj, count in adj_counts:
                desc.append(f"{adj}（{count}回）")
            
            # 自然言語の文章に変換
            if len(desc) > 1:
                description = "、".join(desc[:-1]) + "、そして" + desc[-1]
            else:
                description = desc[0]
            
            sentence = f"{noun}については、{description}と評価されています。"
            sentences.append(sentence)
        return ''.join(sentences)
            

    def ask_single_question(
        self,
        inp,
        conv_mode,
        model_name,
        image,
        image_processor,
        model,
        tokenizer,
        topk=False,
        k=1,
        entity=None,
        task=None,
        japanese=False,
        output_attention=False,
        image_path=None,
        num_entity=3,
        num_relation=3,

    ):
        conv = conv_templates[conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ("user", "assistant")
        else:
            roles = conv.roles

        # image = load_image(args.image_file)
        # Similar operation in model_worker.py
        if image is not None:
            image_tensor = process_images([image], self.image_processor, self.args)
            if type(image_tensor) is list:
                image_tensor = [
                    image.to(self.model.device, dtype=torch.float16)
                    for image in image_tensor
                ]
            else:
                image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        # print(f"{roles[1]}: ", end="")
        # print('entity, inp', entity, inp)
        print('num_entity', num_entity, 'num_relation', num_relation)
        print('retriee method', self.retrieve_method)
        if '名前' not in inp and '名称' not in inp:
            if self.retrieve_method=='triplet':
                triplets = self.retrieve_triplet(entity, inp, top_entity_num=num_entity, top_relation_num=num_relation)
                #sequence = self.convert_to_natural_language(triplets)
                sent = ''
                if len(triplets):
                    sent = ''
                    for t in triplets:
                        sent+=str(t)
                    inp += 'ただし, 補助情報は以下です'+ sent
                print('inp', inp)
            elif self.retrieve_method == 'posneg':
                doc = self.retrieve_posneg(entity)
                #print('doc', doc)
                if doc is not None:
                    inp += doc

            elif self.retrieve_method == 'summary':
                if entity in self.spot2summary:
                    summary = self.spot2summary[entity]
                    inp += 'ただし, 次が観光地に対する情報です' + summary

            elif self.retrieve_method == 'entity':
                retrieved_entities = self.retrieve_entity(entity, inp, top_entity_num=num_entity)
                if retrieved_entities:
                    inp += 'ただし, 補助情報は以下です。「' + '、'.join(retrieved_entities) + '」'
                # inp += 'ただし, 補助情報は以下です'+'、'.join(triplets)
            elif self.retrieve_method == 'review':
                if image_path:
                    spot_name = image_path.split('_')[0]
                    reviews = self.spot2reviews.get(spot_name, [])
                    if reviews:
                        retrieved_reviews = self.clip.retrieve_text_from_image_topk(
                            [os.path.join('/home/yamanishi/project/trip_recommend/data/jalan_image_with_caption/', image_path)],
                            reviews,
                            topk=7
                        )
                        if len(retrieved_reviews):
                            review_sents = '\n'.join(retrieved_reviews)
                            inp += f'ただし、次が観光地について書かれたレビューの一部です\n {review_sents}\n'
                

        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + inp
                )
                if japanese:
                    inp += "ただし、日本語で回答してください"
            else:
                inp = DEFAULT_IMAGE_TOKEN + "\n" + inp
                if japanese:
                    inp += "ただし、日本語で回答してください"
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
            image_tensor=None
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        print('prompt',  prompt)
        input_ids = (
            #tokenizer.encode(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        do_sample = not (self.args.temperature==0)

        with torch.inference_mode():
            if not topk and not output_attention:
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=do_sample,
                    repetition_penalty=1.1,
                    temperature=self.args.temperature,
                    max_new_tokens=200, #self.args.max_new_tokens,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )
                outputs = tokenizer.decode(output_ids[0, :]).strip()
                print('outputs', outputs)
            elif output_attention:
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=do_sample,
                    repetition_penalty=1.1,
                    temperature=self.args.temperature,
                    max_new_tokens=self.args.max_new_tokens,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )
                outputs = tokenizer.decode(output_ids[0, :]).strip()
                print('outputs', outputs)
            elif topk:
                if k<=5:
                    num_beams = num_beam_groups = num_return_sequences = k
                else:
                    #num_beam_groups = k//2
                    num_beam_groups = num_beams = num_return_sequences = k
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=False,  # サンプリングを有効に
                    temperature=1.1,  # 例えば0.7や1.0など
                    top_k=50,  # top_kサンプリング
                    top_p=0.8,
                    num_beams=num_beams,
                    num_beam_groups=num_beam_groups,
                    num_return_sequences=num_return_sequences,
                    max_new_tokens=self.args.max_new_tokens,
                    use_cache=True,
                    diversity_penalty=0.9,
                    stopping_criteria=[stopping_criteria],
                )
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                outputs = ','.join(outputs)
                print('outputs', outputs)
        # outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        #outputs = tokenizer.decode(output_ids[0, :]).strip()
        conv.messages[-1][-1] = outputs

        if self.args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
        return outputs

    def process_images(
        self, image_paths, prompts, japanese=False, task=None, context=None, debug_prompt=False, output_attention=False, num_entity=3, num_relation=3
    ):
        image_dir = (
            "/home/yamanishi/project/trip_recommend/data/jalan_image_with_caption"
        )
        # image_paths = [os.path.join(image_dir, '五稜郭公園_3.jpg'), os.path.join(image_dir, '銀山温泉_4.jpg'),
        #                os.path.join(image_dir, '函館山_3.jpg'),os.path.join(image_dir, 'すみだ水族館_5.jpg')]
        outputs = []
        for i,image_path in tqdm(enumerate(image_paths)):
            
            if task == "review_generation":
                spot_name = image_path.split("_")[0]
                #prompt = f"この観光地は{spot_name}です。観光客になったつもりで画像にあったレビューを生成してください"
                prompt = f"あなたは{spot_name}を訪れた観光客です。画像からレビューを生成してください。"
            elif task == "context_review_generation":
                spot_name = image_path.split("_")[0]
                prompt = f"あなたは{spot_name}を訪れた{context}の観光客です。画像からレビューを生成してください。"
            else:
                spot_name = None
            if type(prompts) is list:
                prompt = prompts[i]
            if debug_prompt:
                print('prompt', prompt)
            #print(os.path.join(image_dir, image_path))
            if image_path is not None:
                image = load_image(os.path.join(image_dir, image_path))
            else:
                image = None

            if image_path is not None:
                spot_name = image_path.split('_')[0]
            #print('image', image)
            if task in ['spot_name_topk', 'sequential_topk']:
                output = self.ask_single_question(
                    prompt,
                    self.conv_mode,
                    self.model_name,
                    image,
                    self.image_processor,
                    self.model,
                    self.tokenizer,
                    topk=True,
                    k=6,
                    japanese=japanese,
                    output_attention=output_attention,
                    task=task
                )
            else:
                output = self.ask_single_question(
                    prompt,
                    self.conv_mode,
                    self.model_name,
                    image,
                    self.image_processor,
                    self.model,
                    self.tokenizer,
                    entity=spot_name,
                    japanese=japanese,
                    output_attention=output_attention,
                    task=task,
                    image_path=image_path,
                    num_entity=num_entity,
                    num_relation=num_relation
                )
            #print("output", output)
            outputs.append(output)

        return outputs

class BLIPInference:
    def __init__(self):
        self.model = AutoModelForVision2Seq.from_pretrained(
            "stabilityai/japanese-instructblip-alpha",
            load_in_8bit=True,
            trust_remote_code=True,
        )
        self.processor = BlipImageProcessor.from_pretrained(
            "stabilityai/japanese-instructblip-alpha"
        )
        self.tokenizer = LlamaTokenizer.from_pretrained(
            "novelai/nerdstash-tokenizer-v1", additional_special_tokens=["▁▁"]
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.model.to(self.device)

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
        prompt = ""  # input empty string for image captioning. You can also input questions as prompts
        prompt = BLIPInference.build_prompt(prompt)
        inputs = self.processor(images=image, return_tensors="pt")
        text_encoding = self.tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        )
        text_encoding["qformer_input_ids"] = text_encoding["input_ids"].clone()
        text_encoding["qformer_attention_mask"] = text_encoding[
            "attention_mask"
        ].clone()
        inputs.update(text_encoding)

        # generate
        outputs = self.model.generate(
            **inputs.to(self.device, dtype=self.model.dtype),
            num_beams=5,
            max_new_tokens=64,
            min_length=5,
        )
        generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[
            0
        ].strip()
        # print(generated_text)
        return generated_text

    def process_images(self, image_paths, prompt, task=None):
        image_dir = (
            "/home/yamanishi/project/trip_recommend/data/jalan_image_with_caption"
        )
        outputs = []
        for image_path in tqdm(image_paths):
            if task == "review_generation":
                spot_name = image_path.split("_")[0]
                if task == "review_generation":
                    prompt = f"この観光地は{spot_name}です。画像にあったレビューを生成してください"
            output = self.inference(os.path.join(image_dir, image_path), prompt)
            outputs.append(output)
        return outputs

    def inference_spot_names(self, image_paths):
        outputs = self.process_images(image_paths, prompt="写真の観光地の名前を教えて",)
        return outputs

    def generate_reviews(self, image_paths):
        outputs = self.process_images(image_paths, prompt="", task="review_generation")
        return outputs


class StableVLMInference:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForVision2Seq.from_pretrained(
            "stabilityai/japanese-stable-vlm", load_in_8bit=True, trust_remote_code=True
        )
        self.processor = AutoImageProcessor.from_pretrained(
            "stabilityai/japanese-stable-vlm"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "stabilityai/japanese-stable-vlm"
        )
        # self.model.to(self.device)

    @staticmethod
    def build_prompt(task="caption", input=None, sep="\n\n### ", spot_name=None):
        TASK2INSTRUCTION = {
            "caption": "画像を詳細に述べてください。",
            "tag": "与えられた単語を使って、画像を詳細に述べてください。",
            "vqa": "与えられた画像を下に、質問に答えてください。",
            "spot_name": "画像の観光地の名前を教えて",
            "review_generation": f"この観光地は{spot_name}です。画像に合うレビューを生成して",
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
        text_encoding = self.tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        )
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
        generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[
            0
        ].strip()
        return generated_text

    def inference_multi(self, task, image_paths, spot_name=None):
        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        prompt = StableVLMInference.build_prompt(task=task, spot_name=spot_name)

        inputs = self.processor(images=images, return_tensors="pt")
        text_encoding = self.tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        )
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
        generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[
            0
        ].strip()
        return generated_text

    def process_images(self, image_paths, task):
        image_dir = (
            "/home/yamanishi/project/trip_recommend/data/jalan_image_with_caption"
        )
        outputs = []
        for image_path in tqdm(image_paths):
            spot_name = image_path.split("_")[0]
            output = self.inference(
                task, os.path.join(image_dir, image_path), spot_name
            )
            outputs.append(output)
        return outputs

    def inference_spot_names(self, image_paths):
        outputs = self.process_images(image_paths, task="spot_name")
        return outputs

    def generate_reviews(self, image_paths):
        outputs = self.process_images(image_paths, task="review_generation")
        return outputs
        # 桜越しの東京スカイツリー


class HeronInference:
    def __init__(self):
        device = "cuda"
        MODEL_NAME = "turing-motors/heron-chat-blip-ja-stablelm-base-7b-v1"

        self.model = VideoBlipForConditionalGeneration.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, ignore_mismatched_sizes=True
        )

        self.model = self.model.half()
        self.model.eval()
        self.model.to(device)

        # prepare a processor
        self.processor = VideoBlipProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.tokenizer = LlamaTokenizer.from_pretrained(
            "novelai/nerdstash-tokenizer-v1", additional_special_tokens=["▁▁"]
        )
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
            int(self.tokenizer.convert_tokens_to_ids("##")),
        ]

        # do inference
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_length=256,
                do_sample=False,
                temperature=0.0,
                eos_token_id=eos_token_id_list,
                no_repeat_ngram_size=2,
            )

        # print result
        print(self.processor.tokenizer.batch_decode(out))


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
    
class InternVLInference:
    #def __init__(self, model_name='OpenGVLab/InternVL2-40B', tensor_parallel_size=1):
    def __init__(self, model_name='OpenGVLab/InternVL2-Llama3-76B', tensor_parallel_size=1):
        self.tensor_parallel_size = tensor_parallel_size
        self.device_map = self.split_model(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            load_in_8bit=True,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=self.device_map,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        self.generation_config = dict(max_new_tokens=1024, do_sample=False)
        self.image_dir = "/home/yamanishi/project/trip_recommend/data/jalan_image_with_caption"

    def split_model(self, model_name,):
        device_map = {}
        world_size = min(self.tensor_parallel_size, torch.cuda.device_count())
        num_layers = {
            'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
            'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80
        }[model_name.split('/')[-1]]
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f'language_model.model.layers.{layer_cnt}'] = i % world_size
                layer_cnt += 1
        device_map['vision_model'] = 0
        device_map['mlp1'] = 0
        device_map['language_model.model.tok_embeddings'] = 0
        device_map['language_model.model.embed_tokens'] = 0
        device_map['language_model.output'] = 0
        device_map['language_model.model.norm'] = 0
        device_map['language_model.lm_head'] = 0
        device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
        return device_map

    def inference(self, image_path, prompt):
        pixel_values = self.load_image(image_path)
        question = f'<image>\n{prompt}'
        with torch.inference_mode():
            response = self.model.chat(self.tokenizer, pixel_values, question, self.generation_config)
        return response

    def process_images(self, image_paths, prompts):
        results = []
        for image_path, prompt in tqdm(zip(image_paths, prompts)):
            result = self.inference(os.path.join(self.image_dir, image_path), prompt)
            print(result)
            results.append(result)
        return results

    def load_image(self, image_file, input_size=448, max_num=12):
        image = Image.open(image_file).convert('RGB')
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values.to(torch.bfloat16).cuda()

    def inference_spot_names(self, image_paths, japanese=False):
        outputs = self.process_images(
            image_paths, ["この日本の観光地の名前を教えてください。ただし地名のみ答えて." for _ in range(len(image_paths))]
        )
        return outputs

    def build_transform(self, input_size):
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        return transform

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio