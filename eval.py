import argparse
import torch
import copy
import numpy as np
from PIL import Image
import pickle
import spacy
from easydict import EasyDict
from bert_score import score
from tqdm import tqdm
import requests
from PIL import Image
from io import BytesIO
#import google.generativeai as genai
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
from collections import Counter
import fire
from utils import *
import requests
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import re



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
        self.nlp = spacy.load('ja_ginza_electra')
        #self.test_data = load_json('/home/yamanishi/project/airport/src/analysis/LLaVA/playground/data/v4/test_conv2.json')
        #print(len(self.test_data))
        
    def calc_spot_metric_all(self, df_path):
        pop_thr, mid_thr, total_thr = 1000, 5000, 10000
        spots_pop = set(self.experience_df.sort_values('review_count', ascending=False)['spot_name'].values[:pop_thr])
        spots_mid = set(self.experience_df.sort_values('review_count', ascending=False)['spot_name'].values[pop_thr:mid_thr])
        spots_tail = set(self.experience_df.sort_values('review_count', ascending=False)['spot_name'].values[mid_thr:total_thr])
        result_df = pd.read_csv(df_path)
        pop_df = result_df[result_df['spot'].isin(spots_pop)]
        mid_df = result_df[result_df['spot'].isin(spots_mid)]
        tail_df = result_df[result_df['spot'].isin(spots_tail)]
        
        
    @staticmethod
    def _calc_spot_name_metric(df_path, thr=0.7):
        df = pd.read_csv(df_path)
        accuracy = AverageMeter()
        similarity = AverageMeter()
        for image_path, predicted in zip(df['image_path'], df['predicted']):
            spot_name = image_path.split('_')[0]
            if pd.isna(predicted) or type(predicted) is not str:
                accuracy.update(0)
                continue
            predicted = predicted.replace('<s> ', '')
            predicted = predicted.replace('</s>', '')
            if '(' in spot_name:spot_name = spot_name[:spot_name.find('(')]
            if '(' in predicted:predicted = predicted[:predicted.find('(')]
            match = re.search(r'「([^」]+)」', predicted)
            if match:
                predicted = match.group(1)

            elif (spot_name in predicted) or (predicted in spot_name) or Levenshtein.normalized_similarity(spot_name, predicted)>thr:
                accuracy.update(1.0)
            else:accuracy.update(0.0)
            similarity.update(Levenshtein.normalized_similarity(spot_name, predicted))
        
        metric = {'similarity': similarity.get_average(),
                  'accuracy': accuracy.get_average()}
        return metric
    
    @staticmethod
    def _calc_spot_name_metric_topk(df_path):
        df = pd.read_csv(df_path)
        df = pd.read_csv(df_path)
        accuracy1 = AverageMeter()
        accuracy3 = AverageMeter()
        accuracy5 = AverageMeter()
        similarity = AverageMeter()
        for image_path, predicted in zip(df['image_path'], df['predicted']):
            if pd.isna(predicted) or type(predicted) is not str:
                print('None')
                accuracy1.update(0)
                accuracy3.update(0)
                accuracy5.update(0)
                
            predicted = list(set(predicted.split(',')))
            predicted = unique_preserve_order(predicted)
            spot_name = image_path.split('_')[0]
            print(len(predicted))
            if spot_name in predicted[:1]:
                accuracy1.update(1.0)
            else:
                accuracy1.update(0)
            if spot_name in predicted[:3]:
                accuracy3.update(1.0)
            else:
                accuracy3.update(0)
            if spot_name in predicted[:5]:
                accuracy5.update(1.0)
            else:
                accuracy5.update(0)

            #similarity.update(Levenshtein.normalized_similarity(spot_name, predicted))
        
        metric = {'accuracy@1': accuracy1.get_average(),
                  'accuracy@3': accuracy3.get_average(),
                  'accuracy@5': accuracy5.get_average(),}
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
    def _calc_ipp_metric(df_path):
        metrics = {}
        df = pd.read_csv(df_path)
        df['task'] = df['task_id'].apply(lambda x:x.split('_')[1])
        df['predicted'] = df['predicted'].apply(lambda x:x.replace('<s> ', ''))
        df['predicted'] = df['predicted'].apply(lambda x:x.replace('</s>', ''))
        df_task_1 = df[df['task']=="1"]
        df_task_2 = df[df['task']=="2"]
        df_task_1['predicted'] = df_task_1['predicted'].map({'いいえ': 0, 'はい': 1})
        df_task_1['answer'] = df_task_1['answer'].map({'いいえ': 0, 'はい': 1})
        df_task_2['predicted'] = df_task_2['predicted'].map({'いいえ': 0, 'はい': 1})
        df_task_2['answer'] = df_task_2['answer'].map({'いいえ': 0, 'はい': 1})
        df_task_3 = df[df['task']=="3"]
        metrics['task_1_accuracy'] = accuracy_score(df_task_1['predicted'].fillna(0), df_task_1['answer'])
        metrics['task_1_f1'] =f1_score(df_task_1['predicted'].fillna(0), df_task_1['answer'])
        metrics['task_1_roc_auc'] = roc_auc_score(df_task_1['predicted'].fillna(0), df_task_1['answer'])
        metrics['task_2_accuracy'] = accuracy_score(df_task_2['predicted'].fillna(0), df_task_2['answer'])
        metrics['task_2_f1'] =f1_score(df_task_2['predicted'].fillna(0), df_task_2['answer'])
        metrics['task_2_roc_auc'] = roc_auc_score(df_task_2['predicted'].fillna(0), df_task_2['answer'])
        df_task_3['answer'] = df_task_3['answer'].astype(float)
        df_task_3['answer'] = df_task_3['answer'].fillna(0)
        df_task_3['predicted'] = df_task_3['predicted'].apply(lambda x:x[:3])
        df_task_3['predicted'] = df_task_3['predicted'].astype(float)#.fillna(df_task_3['predicted'].mean())
        df_task_3['predicted'] = df_task_3['predicted'].fillna(df_task_3['predicted'].mean())
        metrics['task_3_mae'] = mean_absolute_error(df_task_3['answer'], df_task_3['predicted'])
        metrics['task_3_mse'] = mean_squared_error(df_task_3['answer'], df_task_3['predicted'])
        metrics['task_3_r2'] = r2_score(df_task_3['answer'], df_task_3['predicted'])
        metrics['task_3_coef'] = np.corrcoef(df_task_3['answer'], df_task_3['predicted'])[0][1]
        return metrics 

    @staticmethod
    def calc_bert_score(cands, refs):
        """ BERTスコアの算出

        Args:
            cands ([List[str]]): [比較元の文]
            refs ([List[str]]): [比較対象の文]

        Returns:
            [(List[float], List[float], List[float])]: [(Precision, Recall, F1スコア)]
        """
        Precision, Recall, F1 = score(cands, refs, lang="ja", verbose=True)
        return np.mean(Precision.numpy()), np.mean(Recall.numpy().tolist()), np.mean(F1.numpy().tolist())
    
    @staticmethod
    def extract_entities(nlp, text):
        """
        GiNZAを使用してテキストから名詞、固有名詞、形容詞のエンティティを抽出する。
        """
        doc = nlp(text)
        entities = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN", "ADJ"]]
        return entities

    @staticmethod
    def calculate_entity_f1(nlp, gt, answer):
        """
        与えられた正解(gt)と回答(answer)のエンティティF1スコアを計算する。
        """
        gt_entities = Evaluator.extract_entities(nlp, gt)
        answer_entities = Evaluator.extract_entities(nlp, answer)

        gt_counter = Counter(gt_entities)
        answer_counter = Counter(answer_entities)

        true_positives = sum((gt_counter & answer_counter).values())
        false_positives = sum((answer_counter - gt_counter).values())
        false_negatives = sum((gt_counter - answer_counter).values())

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }
        
    @staticmethod
    def _calc_review_quality_metric(df_path, row1='predicted', row2='conversations', max_user=-1, ):
        df = pd.read_csv(df_path)
        if max_user!=-1:df = df[:max_user]
        bert_p, bert_r, bert_f = Evaluator.calc_bert_score(list(df[row1].values), list(df[row2].values))
        print(bert_p, bert_r, bert_f)
        nlp = spacy.load('ja_ginza_electra')
        bleu_ja = BLEUCalculator(lang="ja")
        rouge = RougeCalculator(lang="ja")
        bleu, rouge_1, rouge_2, rouge_3, rouge_4, rouge_l, rouge_be, entity_f1 = [AverageMeter() for _ in range(8)]
        for predicted, original in zip(df[row1], df[row2]):
            predicted = predicted.replace('<s> ', '')
            predicted = predicted.replace('</s>', '')
            if pd.isna(predicted):
                bleu.update(0)
                rouge_1.update(0)
                rouge_2.update(0)
                rouge_3.update(0)
                rouge_4.update(0)
                rouge_l.update(0)
                rouge_be.update(0)
                entity_f1.update(0)
            else:
                bleu.update(bleu_ja.bleu(predicted, original))
                rouge_1.update(rouge.rouge_n(summary=predicted,references=original,n=1))
                rouge_2.update(rouge.rouge_n(summary=predicted,references=original,n=2))
                rouge_3.update(rouge.rouge_n(summary=predicted,references=original,n=3))
                rouge_4.update(rouge.rouge_n(summary=predicted,references=original,n=4))
                rouge_l.update(rouge.rouge_l(summary=predicted,references=original))
                rouge_be.update(rouge.rouge_be(summary=predicted,references=original))
                #entity_f1.update(Evaluator.calculate_entity_f1(nlp, gt=original, answer=predicted))

        metric = {'bleu': bleu.get_average(),
                  'rouge_1': rouge_1.get_average(),
                  'rouge_2': rouge_2.get_average(),
                  'rouge_3': rouge_3.get_average(),
                  'rouge_4': rouge_4.get_average(),
                  'rouge_l': rouge_l.get_average(),
                  'rouge_be': rouge_be.get_average(),
                  #'entity_f1': entity_f1.get_average(),
                  'bert_p': bert_p,
                  'bert_r': bert_r,
                  'bert_f': bert_f}
        return metric
    
    @staticmethod
    def _calc_sequential_metric(df_path, pred_row='', gt_row=''):
        pass
    def evaluate_spot_names(self, model_names=['llavatour'], checkpoint=None):
        print ('evaluate spot names')

        metrics = []
        #for method in ['gpt-4o', 'llava', 'llavatour', 'blip', 'stablevlm', 'chatgpt']:
        for method in model_names:
            if method == 'llavatour':
                df_path = f'./result/spot_name/{method}/{checkpoint}.csv'
            else:
                df_path = f'./result/spot_name/{method}.csv'
            metric = Evaluator._calc_spot_name_metric(df_path)
            metrics.append((method, metric))
            print(method, metric)
            
    def evaluate_spot_names_old(self):
        print ('evaluate spot names')
        metrics = []
        for method in ['gpt-4o', 'llava', 'llavatour_conv0', 'blip', 'stablevlm', 'chatgpt']:
        #for method in ['llavatour', 'gpt-4o']:
            metric = Evaluator._calc_spot_name_metric(f'./result/spot_name_old/{method}.csv')
            metrics.append((method, metric))
            print(method, metric)
        
    def evaluate_spot_names_topk(self):
        print ('evaluate spot names')
        metrics = []
        for method in ['llavatour']:
            metric = Evaluator._calc_spot_name_metric_topk(f'./result/spot_name/{method}.csv')
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
            
    def evaluate_ipp_metric(self, model_names=['llavatour'], checkpoint=None):
        print('evaluating ipp metrics')
        metrics = []
        for method in model_names:
            print(method)
            if method == 'llavatour':
                df_path = f'./result/ipp/{method}/{checkpoint}.csv'
            else:
                df_path = f'./result/ipp/{method}.csv'
            metric = Evaluator._calc_ipp_metric(df_path)
            metrics.append((method, metric))
            print(df_path, metric)
            
    def evaluate_sequential_metric(self, model_names=['llavatour'], checkpoint=None, topk=False):
        metrics = []
        print('evaluating sequential')
        for method in model_names:
            if method == 'llavatour':
                if topk:
                    df_path = f'./result/sequential/{method}/{checkpoint}_topk.csv'
                else:
                    df_path = f'./result/sequential/{method}/{checkpoint}.csv'
            else:
                if topk:
                    df_path = f'./result/sequential/{method}_topk.csv'
                else:
                    df_path = f'./result/sequential/{method}.csv'
                    
            metric = Evaluator._calc_sequential_metric(df_path, pred_row='predicted', gt_row='answer')
            metrics.append((method, metric))
        print(metrics)
        
        
    def evaluate_qa_metric(self, model_names=['llavatour'], checkpoint=None, mode='qa'):
        '''
        Evaluating qa task
            mode: qa or pvqa
        '''
        metrics = []
        print(f'evaluating {mode}')
        print(f'evaluating {str(model_names)}')
        for method in model_names:#['llavatour_context', 'gpt-4-vision-preview']:#['llavatour_conv2', 'llavatour_epoch2', 'blip', 'pepler_pre']:
            
            print(method)
            if method == 'llavatour':
                df_path = f'./result/{mode}/{method}/{checkpoint}.csv'
            else:
                df_path = f'./result/{mode}/{method}.csv'
            if mode=='pvqa':
                generation_metric = Evaluator._calc_review_quality_metric(df_path, row1='predicted', row2='answer', max_user=1000)
            else:
                generation_metric = Evaluator._calc_review_quality_metric(df_path, row1='predicted', row2='answer')
            metrics.append((method, generation_metric))
            print('generation', generation_metric)
            if method == 'llavatour':save_path = f'./result/metric/{checkpoint}.txt'
            else:save_path = f'./result/metric/{method}.txt'
            with open(save_path, 'a') as f:
                f.write('evaluating '+ mode + '\n')
                f.write(str(checkpoint)+'\n')
                f.write(str(generation_metric)+'\n')
        print(metrics)
            
    def evaluate_review_metric(self, model_names=['llavatour'], checkpoint=None, quality=True, diversity=True):
        metrics = []
        for method in model_names:#['llavatour_context', 'gpt-4-vision-preview']:#['llavatour_conv2', 'llavatour_epoch2', 'blip', 'pepler_pre']:
            print(method)
            if method == 'llavatour':
                df_path = f'./result/reviews/{method}/{checkpoint}.csv'
            else:
                df_path = f'./result/reviews/{method}.csv'
            if quality:
                generation_metric = Evaluator._calc_review_quality_metric(df_path)
            if diversity:
                diversity_metric = Evaluator._calc_review_diversity_metric(df_path)
            metrics.append((method, generation_metric, diversity_metric))
            print('generation', generation_metric)
            print('diversity', diversity_metric)
            if checkpoint.endswith('feature') or checkpoint.endswith('context'):
                checkpoint = '_'.join(checkpoint.split('_')[:-1])
            if method == 'llavatour':save_path = f'./result/metric/{checkpoint}.txt'
            else:save_path = f'./result/metric/{method}.csv'
            with open(save_path, 'a') as f:
                f.write(str(checkpoint)+'\n')
                f.write(str(generation_metric)+'\n')
                f.write(str(diversity_metric)+'\n')
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
        
        
    def chatgpt_ask_image(self, image_path, prompt, gpt_model):

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
        "model": gpt_model,
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
    
    def gemini_spot_names(self, model_name=''):
        result = []
        df = pd.read_csv('/home/yamanishi/project/airport/src/analysis/LLaVA/result/spot_name/llavatour.csv')
        #limit=1000
        df_gemini = pd.read_csv(f'./result/spot_name/{model_name}.csv', 
                             names=['image_path', 'predicted'])
        already_image_paths = df_gemini['image_path'].values
        genai.configure(api_key=key)
        
        for i in tqdm(range(len(df))):
            image_path = f"./data/spot_name/{i}.jpg"
            true_image_path = df.loc[i, 'image_path']
            if true_image_path in already_image_paths:
                continue
            img = Image.open(image_path)
            print(img.size)
            prompt = 'この観光地の名前を30文字以内で答えて。地名のみ答えて。'
            model = genai.GenerativeModel(model_name)
            response = model.generate_content([prompt, img])
            print(true_image_path, response)
            result.append(response.text)
            pd.DataFrame({'image_path': [true_image_path], 'result': [response.text]}).to_csv(f'./result/spot_name/{model_name}.csv', mode='a', index=False, header=False)
    
    def chatgpt_spot_names(self, gpt_model="gpt-4-vision-preview"):
        result = []
        df = pd.read_csv('/home/yamanishi/project/airport/src/analysis/LLaVA/result/spot_name/llavatour.csv')
        #limit=1000
        df_gpt = pd.read_csv(f'./result/spot_name/{gpt_model}.csv', 
                             names=['image_path', 'predicted'])
        already_image_paths = df_gpt['image_path'].values
        for i in tqdm(range(len(df))):
            image_path = f"./data/spot_name/{i}.jpg"
            true_image_path = df.loc[i, 'image_path']
            if true_image_path in already_image_paths:
                continue
            img = Image.open(image_path)
            print(img.size)
            response = self.chatgpt_ask_image(image_path, 'この観光地の名前を30文字以内で答えて。地名のみ答えて。', gpt_model=gpt_model)
            #print(response['choices'][0])
            print(true_image_path, response)
            result.append(response['choices'][0]['message']['content'])
            #if i==limit:break
            result_tmp = response['choices'][0]['message']['content']
            pd.DataFrame({'image_path': [true_image_path], 'result': [result_tmp]}).to_csv(f'./result/spot_name/{gpt_model}.csv', mode='a', index=False, header=False)
        #pd.DataFrame({'image_path': df['image_path'][:limit+1], 'predicted': result}).to_csv(f'./result/spot_name/{gpt_model}.csv')
        
    def chatgpt_review_generation(self, gpt_model='gpt-4-vision-preview', feature=False):
        result = []
        #df = pd.read_csv('./result/reviews/llavatour.csv')
        df = pd.read_csv('./data/review_generation_eval.csv')
        print(len(df))
        df_already = pd.read_csv(f'./result/reviews/{gpt_model}.csv')
        image_already = df_already['image_path'].values
        for i in range(288,1000):
            print(i)
            image_path_tmp = df.loc[i, 'image_path']
            if image_path_tmp in image_already:continue
            gt = df.loc[i, 'conversations']
            spot_name = image_path_tmp.split('_')[0]
            image_path = f"./data/spot_review/{i}.jpg"
            if feature:
                response = self.chatgpt_ask_image(image_path, f'この観光地は{spot_name}です. 観光客になったつもりで観光地の特徴も踏まえながら画像に従ってレビューを100文字前後で出力してください', gpt_model=gpt_model)
            else:
                response = self.chatgpt_ask_image(image_path, f'この観光地は{spot_name}です. 観光客になったつもりで画像に従ってレビューを100文字前後で出力してください', gpt_model=gpt_model)
            print(response)
            result_tmp = response['choices'][0]['message']['content']
            if feature:
                pd.DataFrame({'image_path': [image_path_tmp], 'result': [result_tmp], 'conversations':[gt]}).to_csv(f'./result/reviews/{gpt_model}_feature.csv', mode='a', index=True, header=False)
            else:
                pd.DataFrame({'image_path': [image_path_tmp], 'result': [result_tmp], 'conversations':[gt]}).to_csv(f'./result/reviews/{gpt_model}.csv', mode='a', index=True, header=False)
            #print(response['choices'][0])
            #result.append(response['choices'][0]['message']['content'])
        # if feature:
        #     pd.DataFrame({'image_path': df['image_path'], 'predicted': result}).to_csv(f'./result/reviews/{gpt_model}_feature.csv')
        # else:
        #     pd.DataFrame({'image_path': df['image_path'], 'predicted': result}).to_csv(f'./result/reviews/{gpt_model}.csv')
            
        
if __name__ == '__main__':
    fire.Fire(Evaluator)
    exit()
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
        evaluator.chatgpt_spot_names(gpt_model='gpt-4o')
    elif args.f == 'eval_review_metric':
        evaluator.evaluate_review_metric()
    # elif args.f == 'llavatour_inference_and_eval':
    #     evaluator.llavatour_inference_and_eval()
    else:
        pass