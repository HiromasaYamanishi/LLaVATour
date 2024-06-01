import argparse
import torch
import copy
import numpy as np
from PIL import Image
import pickle
from easydict import EasyDict

from tqdm import tqdm
import requests
from PIL import Image
from io import BytesIO
from rapidfuzz.distance import Levenshtein
import pandas as pd
import json
import random
from sumeval.metrics.bleu import BLEUCalculator
from rapidfuzz.distance import Levenshtein
from text_processing import GinzaTokenizer
from sumeval.metrics.rouge import RougeCalculator
import itertools
import base64
import requests

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
        return round(np.mean(self.list), 4)

        
class Evaluator:
    def __init__(self,):
        self.experience_df = pd.read_csv('/home/yamanishi/project/airport/src/data/experience_light.csv')
        self.popular_spots = set(self.experience_df.sort_values('review_count', ascending=False)['spot_name'].values[:500])
        self.over_100_spots = set(self.experience_df.query('review_count >= 100')['spot_name'].values)
        self.test_data = load_json('/home/yamanishi/project/airport/src/analysis/LLaVA/playground/data/v4/test_conv2.json')
        #print(len(self.test_data))
        
    @staticmethod
    def _calc_spot_name_metric(df_path):
        df = pd.read_csv(df_path)
        accuracy = AverageMeter()
        similarity = AverageMeter()
        for image_path, predicted in zip(df['image_path'], df['predicted']):
            spot_name = image_path.split('_')[0]
            if pd.isna(predicted) or type(predicted) is not str:accuracy.update(0)
            elif spot_name in predicted:
                accuracy.update(1.0)
            else:accuracy.update(0.0)
            similarity.update(Levenshtein.normalized_similarity(spot_name, predicted))
        
        metric = {'similarity': similarity.get_average(),
                  'accuracy': accuracy.get_average()}
        return metric
    
    @staticmethod
    def _calc_DIV(all_features):
        div = 0
        eps = 1e-6
        for feature1, feature2 in itertools.combinations(all_features, 2):
            div += 1-len(set(feature1).intersection(set(feature2)))/(len(set(feature1).union(feature2))+eps)
            
        div/=(len(all_features)*(len(all_features)-1)/2)
        return div
    
    @staticmethod
    def _total_feature_unique_num(all_features):
        unique_features = set({})
        for features in all_features:
            unique_features.update(features)
        return len(unique_features)
    
    @staticmethod
    def _average_sentence_length(df):
        return df['predicted'].str.len().mean()
    
    def _replace_spot_names(df):
        for i,(image_path, predicted) in enumerate(zip(df['image_path'], df['predicted'])):
            spot_name = image_path.split('_')[0]
            if type(predicted) == str:
                predicted = predicted.replace('</s>', '')
                predicted = predicted.replace('<s>', '')
                predicted = predicted.replace(spot_name, '')
                df.loc[i, 'predicted'] = predicted
            else:
                df.loc[i, 'predicted'] = ''
        return df
    
    @staticmethod
    def _tfidf_include_ratio(df, features):
        experience_df = pd.read_csv('/home/yamanishi/project/airport/src/data/experience_light.csv')
        tfidf_top_words = np.load('/home/yamanishi/project/trip_recommend/data/jalan/graph/tfidf_top_words.npy')
        spot2topwords = {spot:top_words for spot, top_words in zip(experience_df['spot_name'], tfidf_top_words)}
        tfidf_ratio = 0
        for image_path, feature in zip(df['image_path'], features):
            spot_name = image_path.split('_')[0]
            tfidf_words = spot2topwords[spot_name]
            tfidf_ratio+=len(set(tfidf_words).intersection(feature))/len(tfidf_words)
        return tfidf_ratio/len(features)
            
    
    @staticmethod
    def _calc_review_diversity_metric(df_path):
        df = pd.read_csv(df_path)
        ginza = GinzaTokenizer()
        # 観光地名は取り除く
        df = Evaluator._replace_spot_names(df)
        features, features_propn = ginza.tokenize(list(df['predicted'].values))
        DIV = Evaluator._calc_DIV(features)
        ave_length = Evaluator._average_sentence_length(df)
        unique_propn_num = Evaluator._total_feature_unique_num(features_propn)
        unique_feature_num = Evaluator._total_feature_unique_num(features)
        tfidf_include_ratio = Evaluator._tfidf_include_ratio(df, features)
        metrics = {'DIV': DIV,
                   'ave_length': ave_length,
                   'unique_propn_num': unique_propn_num,
                   'unique_feature_num': unique_feature_num,
                   'tfidf_include_ratio': tfidf_include_ratio}
        return metrics 
    
    @staticmethod
    def _calc_review_quality_metric(df_path, row1='predicted', row2='conversations', max_user=-1):
        df = pd.read_csv(df_path)
        if max_user!=-1:df = df[:max_user]
        bleu_ja = BLEUCalculator(lang="ja")
        rouge = RougeCalculator(lang="ja")
        bleu, rouge_1, rouge_2, rouge_3, rouge_4, rouge_l, rouge_be = [AverageMeter() for _ in range(7)]
        for predicted, original in zip(df[row1], df[row2]):
            if pd.isna(predicted):
                bleu.update(0)
                rouge_1.update(0)
                rouge_2.update(0)
                rouge_3.update(0)
                rouge_4.update(0)
                rouge_l.update(0)
                rouge_be.update(0)
            else:
                bleu.update(bleu_ja.bleu(predicted, original))
                rouge_1.update(rouge.rouge_n(summary=predicted,references=original,n=1))
                rouge_2.update(rouge.rouge_n(summary=predicted,references=original,n=2))
                rouge_3.update(rouge.rouge_n(summary=predicted,references=original,n=3))
                rouge_4.update(rouge.rouge_n(summary=predicted,references=original,n=4))
                rouge_l.update(rouge.rouge_l(summary=predicted,references=original))
                rouge_be.update(rouge.rouge_be(summary=predicted,references=original))

        metric = {'bleu': bleu.get_average(),
                  'rouge_1': rouge_1.get_average(),
                  'rouge_2': rouge_2.get_average(),
                  'rouge_3': rouge_3.get_average(),
                  'rouge_4': rouge_4.get_average(),
                  'rouge_l': rouge_l.get_average(),
                  'rouge_be': rouge_be.get_average()}
        return metric
        
    def evaluate_spot_names(self):
        print ('evaluate spot names')
        metrics = []
        for method in ['llava', 'llavatour', 'blip', 'stablevlm', 'chatgpt']:
            metric = Evaluator._calc_spot_name_metric(f'./result/spot_name/{method}.csv')
            metrics.append((method, metric))
            print(method, metric)
        
    def evaluate_review_quality(self):
        print ('evaluate review metrics')
        metrics = []
        for method in ['llava', 'llavatour_epoch2', 'blip', 'pepler_pre']:
            metric = Evaluator._calc_review_quality_metric(f'./result/reviews/{method}.csv')
            metrics.append((method, metric))
            print(method, metric)
        print(metrics)
        
    def evaluate_review_diversity(self):
        print('evaluate review diversity')
        metrics = []
        for method in ['llava', 'llavatour', 'blip', 'pepler']:
            metrics = Evaluator._calc_review_diversity_metric(f'./result/reviews/{method}.csv')
            print(metrics)
            
    def evaluate_review_metric(self, ):
        metrics = []
        for method in ['llavatour_epoch2']:#['llavatour_conv2', 'llavatour_epoch2', 'blip', 'pepler_pre']:
            print(method)
            df_path = f'./result/reviews/{method}.csv'
            generation_metric = Evaluator._calc_review_quality_metric(df_path)
            diversity_metric = Evaluator._calc_review_diversity_metric(df_path)
            metrics.append((method, generation_metric, diversity_metric))
            print('generation', generation_metric)
            print('diversity', diversity_metric)
        print(metrics)
        
    def llavatour_eval(self, args):
        spot_name_metric = Evaluator._calc_spot_name_metric(f'./result/spot_name/llavatour.csv')
        review_metric = Evaluator._calc_review_quality_metric(f'./result/reviews/llavatour.csv')
        review_male_metric = Evaluator._calc_review_quality_metric(f'./result/reviews/llavatour_male.csv')
        review_female_metric = Evaluator._calc_review_quality_metric(f'./result/reviews/llavatour_female.csv')
        print('review_male_metric', review_male_metric)
        print('review_female_metric', review_female_metric)
        print('review_generation_metric', review_metric)
        print('spot_name_metric', spot_name_metric)
        
        
    def chatgpt_ask_image(self, image_path, prompt):
        api_key = ""

        # Function to encode the image
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

        # Getting the base64 string
        base64_image = encode_image(image_path)

        headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
        }

        payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": prompt,
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
                }
            ]
            }
        ],
        "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        return response.json()
    
    def chatgpt_spot_names(self):
        result = []
        df = pd.read_csv('/home/yamanishi/project/airport/src/analysis/LLaVA/result/spot_name/llavatour.csv')
        limit=1000
        for i in range(1000):
            image_path = f"./data/spot_name/{i}.jpg"
            response = self.chatgpt_ask_image(image_path, 'この観光地の名前を30文字以内で答えて')
            #print(response)
            #print(response['choices'][0])
            result.append(response['choices'][0]['message']['content'])
            if i==limit:break
        pd.DataFrame({'image_path': df['image_path'][:limit+1], 'predicted': result}).to_csv('./result/spot_name/chatgpt.csv')
        
        
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
    parser.add_argument("--df_path", type=str, default=None)
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    args = parser.parse_args()

    # inference = LLaVAInference(args)
    # inference.inference_spot_names()
    evaluator = Evaluator()
    #evaluator.evaluate_spot_names(args)
    if args.f == 'spot_name':
        evaluator.inference_spot_names(args)
    elif args.f == 'review_generation':
        evaluator.inference_review_generation(args)
    elif args.f == 'eval_spot_name':
        evaluator.evaluate_spot_names()
    elif args.f == 'eval_review_quality':
        evaluator.evaluate_review_quality()
    elif args.f == 'eval_review_diversity':
        evaluator.evaluate_review_diversity()
    elif args.f == 'llavatour_inference':
        evaluator.llavatour_inference(args)
    elif args.f == 'llavatour_eval':
        evaluator.llavatour_eval(args)
    elif args.f == 'chatgpt_spot_names':
        evaluator.chatgpt_spot_names()
    elif args.f == 'eval_review_metric':
        evaluator.evaluate_review_metric()
    # elif args.f == 'llavatour_inference_and_eval':
    #     evaluator.llavatour_inference_and_eval()
    else:
        pass