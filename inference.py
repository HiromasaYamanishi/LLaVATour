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

from tqdm import tqdm
import requests
from PIL import Image
from io import BytesIO
from rapidfuzz.distance import Levenshtein
import os
from transformers import TextStreamer
from transformers import LlamaTokenizer, AutoModelForVision2Seq, BlipImageProcessor
from transformers import AutoTokenizer, AutoModelForVision2Seq, AutoImageProcessor
import torch
#from heron.models.video_blip import VideoBlipForConditionalGeneration, VideoBlipProcessor
from transformers import LlamaTokenizer
import pandas as pd
import json
import random
#from sumeval.metrics.bleu import BLEUCalculator
from rapidfuzz.distance import Levenshtein
from lmm import LLaVAInference, BLIPInference, StableVLMInference
#from sumeval.metrics.rouge import RougeCalculator

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def load_json(json_path):
    with open(json_path) as f:
        d = json.load(f)
    return d

class AverageMeter:
    def __init__(self):
        self.list = []
        
    def update(self, value):
        self.list.append(value)
        
    def get_average(self, ):
        return np.mean(self.list)

        
class Inferencer:
    def __init__(self,):
        self.experience_df = pd.read_csv('/home/yamanishi/project/airport/src/data/experience_light.csv')
        self.popular_spots = set(self.experience_df.sort_values('review_count', ascending=False)['spot_name'].values[:500])
        self.over_100_spots = set(self.experience_df.query('review_count >= 100')['spot_name'].values)
        self.test_data = load_json('/home/yamanishi/project/airport/src/analysis/LLaVA/playground/data/v4/test_conv2.json')
        print(len(self.test_data))
        
    @staticmethod
    def inference(args, model_name, function_name, image_paths):
        if model_name == 'stablevlm':
            model = StableVLMInference()
        elif model_name == 'llava':
            args.model_path = 'liuhaotian/llava-v1.5-13b'
            args.model_base = None
            args.load_4bit = True
            model = LLaVAInference(args)
        elif model_name == 'blip':
            model = BLIPInference()
        elif model_name == 'llavatour':
            model = LLaVAInference(args)
        
        if function_name == 'inference_spot_name':
            result = model.inference_spot_names(image_paths)
            pd.DataFrame({'image_path': image_paths, 'predicted': result}).to_csv(f'./result/spot_name/{model_name}.csv')
        elif function_name == 'generate_reviews':
            result = model.generate_reviews(image_paths)
            pd.DataFrame({'image_path': image_paths, 'predicted': result}).to_csv(f'./result/reviews/{model_name}.csv')
            print('result len:', len(result))
        del model
        return result
    
    @staticmethod
    def inference_and_save(args, model_name, function_name, image_paths, conversations=None,*args_param):
        result = Inferencer.inference(args, model_name, function_name, image_paths, *args_param)
        if function_name=='inference_spot_name':
            pd.DataFrame({'image_path': image_paths, 'predicted': result}).to_csv(f'./result/spot_name/{model_name}.csv')
        else:
            pd.DataFrame({'image_path': image_paths, 'predicted': result, 'conversations': conversations}).to_csv(f'./result/reviews/{model_name}_conv2.csv')
        print('len result: ', len(result))
        return result
    
    def _prepare_image_paths_for_spot_names(self):
        #image_paths = [d.get('image') for d in self.test_data if d['id'].split('_')[0] in self.popular_spots]
        #image_paths = random.sample(image_paths, 1000)
        return pd.read_csv('/home/yamanishi/project/airport/src/analysis/LLaVA/result/spot_name/llava.csv')['image_path'].values
        #return image_paths
            
        
    def inference_spot_names(self, args):
        image_paths = self._prepare_image_paths_for_spot_names()
        #print(len(image_paths))
        result = Inferencer.inference_and_save(args, args.model_name, 'inference_spot_name', image_paths)
        
    def inference_spot_names_all(self, args):
        for method in ['llavatour', 'llava', 'blip', 'stablevlm']:
            args.model_name = method
            self.inference_spot_names(args)
        
    def _prepare_image_paths_for_review_generation(self):
        #image_paths = [d.get('image') for d in self.test_data if ('retrieved_from_image' in d['id']) and (d['id'].split('_')[0] in self.over_100_spots)]
        #conversations = [d.get('conversations')[1]['value'] for d in self.test_data if ('retrieved_from_image' in d['id']) and (d['id'].split('_')[0] in self.over_100_spots)]
        inds_target = np.load('/home/yamanishi/project/airport/src/analysis/LLaVA/playground/data/v4/review_test_inds_conv2.npy')[:1000]
        image_paths = [d.get('image') for d in self.test_data]
        conversations = [d.get('conversations')[1]['value'] for d in self.test_data]
        #rand_ind = random.sample(inds_target, 2000)
        image_paths = [image_paths[i] for i in inds_target]
        conversations = [conversations[i] for i in inds_target]
        return image_paths, conversations
        
    def inference_review_generation(self, args):
        image_paths, conversations = self._prepare_image_paths_for_review_generation()
        result = Inferencer.inference_and_save(args, args.model_name, 'generate_reviews', image_paths, conversations=conversations)
        
    def inference_review_generation_all(self, args):
        for method in ['llavatour', 'llava', 'blip', 'stablevlm']:
            args.model_name = method
            self.inference_review_generation(args)
    
    def llavatour_inference(self, args):
        '''
        inference and evaluation with trained model
        '''
        llava_inferencer = LLaVAInference(args)
        spot_image_paths = pd.read_csv('/home/yamanishi/project/airport/src/analysis/LLaVA/result/spot_name/llava.csv')['image_path'].values
        review_df_llava = pd.read_csv('/home/yamanishi/project/airport/src/analysis/LLaVA/result/reviews/llava.csv')
        review_image_paths = review_df_llava['image_path'].values
        result_spot_names = llava_inferencer.inference_spot_names(spot_image_paths)
        result_reviews = llava_inferencer.generate_reviews(review_image_paths)
        result_reviews_male = llava_inferencer.generate_review_context(review_image_paths, context='男性')
        result_reviews_female = llava_inferencer.generate_review_context(review_image_paths, context='女性')
        pd.DataFrame({'image_path': spot_image_paths, 'predicted': result_spot_names}).to_csv('./result/spot_name/llavatour.csv')   
        pd.DataFrame({'image_path': review_image_paths, 'predicted': result_reviews, 'conversations': review_df_llava['conversations']}).to_csv('./result/reviews/llavatour.csv')
        pd.DataFrame({'image_path': review_image_paths, 'predicted': result_reviews_male, 'conversations': review_df_llava['conversations']}).to_csv('./result/reviews/llavatour_male.csv')
        pd.DataFrame({'image_path': review_image_paths, 'predicted': result_reviews_female, 'conversations': review_df_llava['conversations']}).to_csv('./result/reviews/llavatour_female.csv')

    def llavatour_inference_tag(self, args):
        '''
        inference and evaluation with trained model
        '''
        llava_inferencer = LLaVAInference(args)
        review_df_llava = pd.read_csv('/home/yamanishi/project/airport/src/analysis/LLaVA/result/reviews/llava.csv')
        review_image_paths = review_df_llava['image_path'].values
        result_reviews_male = llava_inferencer.generate_review_context(review_image_paths, context='男性')
        result_reviews_female = llava_inferencer.generate_review_context(review_image_paths, context='女性')
        result_reviews_couple = llava_inferencer.generate_review_context(review_image_paths, context='カップル・夫婦')
        result_reviews_family = llava_inferencer.generate_review_context(review_image_paths, context='家族')
        result_reviews_friend = llava_inferencer.generate_review_context(review_image_paths, context='友達同士')
        result_reviews_solo = llava_inferencer.generate_review_context(review_image_paths, context='一人')
        pd.DataFrame({'image_path': review_image_paths, 'predicted': result_reviews_male, 'conversations': review_df_llava['conversations']}).to_csv('./result/reviews/llava_male.csv')
        pd.DataFrame({'image_path': review_image_paths, 'predicted': result_reviews_female, 'conversations': review_df_llava['conversations']}).to_csv('./result/reviews/llava_female.csv')
        pd.DataFrame({'image_path': review_image_paths, 'predicted': result_reviews_couple, 'conversations': review_df_llava['conversations']}).to_csv('./result/reviews/llava_couple.csv')
        pd.DataFrame({'image_path': review_image_paths, 'predicted': result_reviews_family, 'conversations': review_df_llava['conversations']}).to_csv('./result/reviews/llava_family.csv')
        pd.DataFrame({'image_path': review_image_paths, 'predicted': result_reviews_friend, 'conversations': review_df_llava['conversations']}).to_csv('./result/reviews/llava_friend.csv')
        pd.DataFrame({'image_path': review_image_paths, 'predicted': result_reviews_solo, 'conversations': review_df_llava['conversations']}).to_csv('./result/reviews/llava_solo.csv')
    
    def llavatour_inference_one_tag(self, args):
        '''
        inference and evaluation with trained model
        '''
        llava_inferencer = LLaVAInference(args)
        review_df_llava = pd.read_csv('/home/yamanishi/project/airport/src/analysis/LLaVA/result/reviews/llavatour_epoch2.csv')
        review_image_paths = review_df_llava['image_path'].values
        result= llava_inferencer.generate_review_context(review_image_paths, context=args.tag)
        pd.DataFrame({'image_path': review_image_paths, 'predicted': result, 'conversations': review_df_llava['conversations']}).to_csv(f'./result/reviews/llava_{args.tag}.csv')
        
    def llavatour_inference_tag(self, args):
        '''
        inference and evaluation with trained model
        '''
        llava_inferencer = LLaVAInference(args)
        review_df_llava = pd.read_csv('/home/yamanishi/project/airport/src/analysis/LLaVA/result/reviews/llava.csv')
        review_image_paths = review_df_llava['image_path'].values
        result_reviews_male = llava_inferencer.generate_review_context(review_image_paths, context='男性')
        result_reviews_female = llava_inferencer.generate_review_context(review_image_paths, context='女性')
        result_reviews_couple = llava_inferencer.generate_review_context(review_image_paths, context='カップル・夫婦')
        result_reviews_family = llava_inferencer.generate_review_context(review_image_paths, context='家族')
        result_reviews_friend = llava_inferencer.generate_review_context(review_image_paths, context='友達同士')
        result_reviews_solo = llava_inferencer.generate_review_context(review_image_paths, context='一人')
        pd.DataFrame({'image_path': review_image_paths, 'predicted': result_reviews_male, 'conversations': review_df_llava['conversations']}).to_csv('./result/reviews/llava_male.csv')
        pd.DataFrame({'image_path': review_image_paths, 'predicted': result_reviews_female, 'conversations': review_df_llava['conversations']}).to_csv('./result/reviews/llava_female.csv')
        pd.DataFrame({'image_path': review_image_paths, 'predicted': result_reviews_couple, 'conversations': review_df_llava['conversations']}).to_csv('./result/reviews/llava_couple.csv')
        pd.DataFrame({'image_path': review_image_paths, 'predicted': result_reviews_family, 'conversations': review_df_llava['conversations']}).to_csv('./result/reviews/llava_family.csv')
        pd.DataFrame({'image_path': review_image_paths, 'predicted': result_reviews_friend, 'conversations': review_df_llava['conversations']}).to_csv('./result/reviews/llava_friend.csv')
        pd.DataFrame({'image_path': review_image_paths, 'predicted': result_reviews_solo, 'conversations': review_df_llava['conversations']}).to_csv('./result/reviews/llava_solo.csv')
        
        
    def llavatour_eval(self, args):
        spot_name_metric = Inferencer._calc_spot_name_metric(f'./result/spot_name/llavatour.csv')
        review_metric = Inferencer._calc_review_generation_metric(f'./result/reviews/llavatour.csv')
        print('spot_name_metric', spot_name_metric)
        print('review_generation_metric', review_metric)
        review_male_metric = Inferencer._calc_review_generation_metric(f'./result/reviews/llavatour_male.csv')
        review_female_metric = Inferencer._calc_review_generation_metric(f'./result/reviews/llavatour_female.csv')
        print('review_male_metric', review_male_metric)
        print('review_female_metric', review_female_metric)
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", type=str, default="spot_name")
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    #parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    parser.add_argument("--model_name", type=str, default='llava')
    parser.add_argument("--tag", type=str, default='llava')
    args = parser.parse_args()

    # inference = LLaVAInference(args)
    # inference.inference_spot_names()
    evaluator = Inferencer()
    #evaluator.evaluate_spot_names(args)
    if args.f == 'spot_name':
        evaluator.inference_spot_names(args)
    elif args.f == 'review_generation':
        evaluator.inference_review_generation(args)
    elif args.f == 'eval_spot_name':
        evaluator.evaluate_spot_names()
    elif args.f == 'eval_review_generation':
        evaluator.evaluate_review_generation()
    elif args.f == 'llavatour_inference':
        evaluator.llavatour_inference(args)
    elif args.f == 'llavatour_inference_one_tag':
        evaluator.llavatour_inference_one_tag(args)
    elif args.f == 'llavatour_inference_tag':
        evaluator.llavatour_inference_tag(args)
    elif args.f == 'llavatour_eval':
        evaluator.llavatour_eval(args)
    elif args.f == 'inference_spot_names':
        evaluator.inference_spot_names(args)
    elif args.f == 'inference_spot_names_all':
        evaluator.inference_spot_names_all(args)
    elif args.f == 'inference_review_generation':
        evaluator.inference_review_generation(args)
    elif args.f == 'inference_name_one':
        evaluator.inference_review_generation_all(args)
    # elif args.f == 'llavatour_inference_and_eval':
    #     evaluator.llavatour_inference_and_eval()
        pass