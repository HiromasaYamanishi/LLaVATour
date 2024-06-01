from lmm import ELYZALLlama, AoKarasu, AoKarasuTransformer, Qarasu,QarasuTransformer, GPT3, LLaMA
import json
import pandas as pd
from utils import *
from prompt import *
import copy
from tqdm import tqdm
from typing import List
import ray
import os
import math
from collections import defaultdict
import argparse

class Summarizer:
    def __init__(self, args):
        self.args = args
        model_name = args.model_name
        if model_name == 'AoKarasu':
            self.model = AoKarasu()
        elif model_name == 'AoKarasu-4B':
            self.model = AoKarasu(is_4bit=True)
        elif model_name == 'AoKarasuTransformer':
            self.model = AoKarasuTransformer()
        elif model_name == 'Qarasu':
            self.model = Qarasu()
        elif model_name == 'QarasuTransformer':
            self.model = QarasuTransformer()
        elif model_name == 'GPT3':
            self.model = GPT3()
        elif model_name == 'llama':
            self.model = LLaMA(model_version='3', parameter_num='70', instruct=True,tensor_parallel_size=8)
            
    def summarize_review_user(self, max_num=20000, summary_method='aspect',):
        if summary_method=='aspect':
            prompt_review_summary = prompt_review_summary_aspect
        elif summary_method=='general':
            prompt_review_summary = prompt_review_summary_general
        train3 = load_json('./playground/data/v8/train3.json')
        id2ind = {train3[i]['id']:i for i,d in enumerate(train3)}
        users = list(set([d['id'].split('_')[0] for d in train3]))
        prompts = [prompt_review_summary.format('\n'.join(train3[id2ind[f'{user}_review_3_1_False']]['conversations'][0]['value'].split('\n')[:-1])) for user in users]
        print('prompts', prompts[495:500])
        summaries = self.model.generate(prompts[:max_num])
        pd.DataFrame({'user': users[:max_num], 'review_summaries': summaries}).to_csv(f'./preprocess/recommend/user_review_summary_{summary_method}.csv')
        
    def summarize_review_ind(self, max_num=500000):
        df = pd.read_csv('./preprocess/recommend/filtered_review_df.csv')
        prompt_summary_ind = '次のレビューを最大50文字程度で要約してください.ただし元の文体も少し残してください\n\
{}\n'
        prompts = [prompt_summary_ind.format(review) for review in df['review'].values]
        summaries = self.model.generate(prompts[:max_num])
        df = pd.DataFrame({'summary': summaries[:max_num],
                          'review': df['review'][:max_num]})
        df.to_csv('./preprocess/recommend/filtered_review_df_summary.csv')
        
    def rephrase_review(self, max_num=500000):
        df = pd.read_csv('./preprocess/recommend/filtered_review_df.csv')
        prompt_rephrase = '次のレビューを10通りに書き換えてください\n\
{}\n'
        prompts = [prompt_rephrase.format(review) for review in df['review'].values]
        summaries = self.model.generate(prompts[:max_num])
        df = pd.DataFrame({'rephrase': summaries[:max_num],
                          'review': df['review'][:max_num]})
        df.to_csv('./preprocess/recommend/filtered_review_df_rephrase.csv')
        
    def get_1st_step_output_all(self, df_path, target_col):
        df_review = pd.read_csv(df_path)
        llm = self.model
        spots = []
        prompts = []
        for spot in tqdm(df_review[target_col].unique()):
            df_tmp = df_review.query(f'{target_col} == "{spot}"')
            #prompts = []
            chunk_size = 10
            reviews = df_tmp['reviews'].values
            for i in range((len(reviews)-1)//chunk_size+1):
                prompt = copy.copy(prompt_direct_step1[:])
                for j in range(chunk_size*i, min(chunk_size*(i+1), len(reviews))):
                    prompt+=reviews[j]
                prompts.append(prompt)
                spots.append(spot)
            
        #prompts = prompts[:5]
        generated_topics = llm.generate(prompts,)
        #print('generated', generated_topics)
        pd.DataFrame({'spot': spots, 'topics': generated_topics}).to_csv('./preprocess/recommend/output_1st_step.csv')
        return generated_topics
    
    def get_2nd_step_output_all(self, df_path, target_col):
        df_review = pd.read_csv(df_path)
        llm = self.model
        spots, prompts = [], []
        for spot in tqdm(df_review[target_col].unique()):
            df_tmp = df_review.query(f'{target_col} == "{spot}"')
            
            chunk_size = 10
            topics= df_tmp['topics'].values
            for i in range((len(topics)-1)//chunk_size+1):
                prompt = copy.copy(prompt_direct_step1[:])
                for j in range(chunk_size*i, min(chunk_size*(i+1), len(topics))):
                    prompt+=reviews[j]
                prompts.append(prompt)
                spots.append(spot)
            
        prompts = prompts[:5]
        generated_topics = llm.generate(prompts,)
        print('generated', generated_topics)
        pd.DataFrame({'spot': spots, 'topics': generated_topics}).to_csv('./preprocess/recommend/output_1st_step.csv')
        return generated_topics
            
    def get_item_pos_summary_all(self, ):
        llm = self.model
        pos_dict = load_pkl('./preprocess/recommend/pos_dict.pkl')
        spots, prompt_all = [], []
        for spot in tqdm(pos_dict.keys()):
            pos_topic = pos_dict[spot][:100]
            prompt_tmp = prompt_topic_summary[:]
            for topic in pos_topic:
                prompt_tmp+=topic
                prompt_tmp+='\n'
            prompt_all.append(prompt_tmp)
            spots.append(spot)
        #print('prompt_all', prompt_all[:2])
        outputs = llm.generate(prompt_all)
        #spots = spots[10:13]
        #print(outputs)
        with open('./preprocess/recommend/item_pos_summary_karasu.pkl', 'wb') as f:
            pickle.dump(outputs, f)
            
        df = pd.DataFrame({'spot': spots, 'pos_summary': outputs})
        df.to_csv('./preprocess/recommend/pos_summary_karasu.csv')
        
    def get_item_neg_summary_all(self,):
        llm = self.model
        pos_dict = load_pkl('./preprocess/recommend/neg_dict.pkl')
        spots, prompt_all = [], []
        #if os.path.exists(f'./preprocess/recommend/neg_summary{start_ind}_{end_ind}.csv'):
        #   already_spots = pd.read_csv(f'./preprocess/recommend/neg_summary{start_ind}_{end_ind}.csv', names=['spot', 'summary'])['spot'].values
        #else:already_spots=[]
        for spot in tqdm(pos_dict.keys()):
            #if spot in already_spots:continue
            pos_topic = pos_dict[spot][:100]
            prompt_tmp = prompt_topic_summary_neg[:]
            for topic in pos_topic:
                prompt_tmp+=topic
                prompt_tmp+='\n'
            prompt_all.append(prompt_tmp)
            spots.append(spot)
        #print('prompt_all', prompt_all[:2])
        outputs = llm.generate(prompt_all)
        #outputs = llm.generate_and_save(prompt_all[start_ind:end_ind], spots, start_ind, end_ind)
        #spots = spots[10:13]
        #print(outputs)
        #with open('./preprocess/recommend/item_neg_summary3000以降.pkl', 'wb') as f:
        #    pickle.dump(outputs, f)
            
        df = pd.DataFrame({'spot': spots, 'neg_summary': outputs})
        df.to_csv(f'./preprocess/recommend/neg_summary_karasu.csv')
            
    def get_1st_step_output(self, llm, reviews):
        prompt_step1 = prompt_direct_step1

        # 15個づづレビューを処理する
        prompts = []
        chunk_size = 10
        for i in range((len(reviews)-1)//chunk_size+1):
            prompt = copy.copy(prompt_step1[:])
            for j in range(chunk_size*i, min(chunk_size*(i+1), len(reviews))):
                prompt+=reviews[j]
            prompts.append(prompt)
            

        generated_topics = llm.generate(prompts,)
        return generated_topics
    
    
    def get_2nd_step_output(self, llm, summarization_1st):
        prompt_step2 = prompt_direct_step2
        prompts = []
        summary_chunk_size = 10
        for i in range((len(summarization_1st)-1)//summary_chunk_size+1):
            topic_prompt = copy.copy(prompt_step2[:])
            for j in range(summary_chunk_size*i, min(summary_chunk_size*(i+1), len(summarization_1st))):
                topic_prompt+=f'要約{j-summary_chunk_size*i+1}:\n'
                topic_prompt+=summarization_1st[j]
            #print('step1 prompt', topic_prompt)
            prompts.append(topic_prompt)
            
        summarized_topics = llm.generate(prompts,)
        
        # hierachical summarization step2
        if len(summarized_topics)>1:
            prompts = []
            summary_chunk_size = 10
            for i in range((len(summarized_topics)-1)//summary_chunk_size+1):
                topic_final_prompt = prompt_step2[:]
                for j in range(summary_chunk_size*i, min(summary_chunk_size*(i+1), len(summarized_topics))):
                    topic_final_prompt+=f'要約{j-summary_chunk_size*i+1}:\n'
                    topic_final_prompt+=summarized_topics[j]
                prompts.append(topic_final_prompt)
            
            summarized_topics = llm.generate(prompts)
        
        if len(summarized_topics)>1:
            prompts = []
            summary_chunk_size = 10
            for i in range((len(summarized_topics)-1)//summary_chunk_size+1):
                topic_final_prompt = prompt_step2[:]
                for j in range(summary_chunk_size*i, min(summary_chunk_size*(i+1), len(summarized_topics))):
                    topic_final_prompt+=f'要約{j-summary_chunk_size*i+1}:\n'
                    topic_final_prompt+=summarized_topics[j]
                prompts.append(topic_final_prompt)
            
            summarized_topics = llm.generate(prompts)
            
        return summarized_topics
        
    def summarize_review_item(self, llm, reviews) -> List[str]:
        summarization_1st = self.get_1st_step_output(llm, reviews)
        summarization_2nd = self.get_2nd_step_output(llm, summarization_1st)
        return summarization_2nd
    
    def summarize_review_item_all(self, df_review_path, target_col):
        df_review = pd.read_csv(df_review_path)
        llm = self.model
        for spot in tqdm(df_review[target_col].unique()):
            df_tmp = df_review.query(f'{target_col} == "{spot}"')
            summarization = self.summarize_review_item(llm, df_tmp['reviews'].values)
            print('summarization',summarization)
            df = pd.DataFrame({'spot': spot,
                               'summary': summarization})
            df.to_csv(f'./preprocess/recommend/item_summary.csv', mode='a', header=False, index=False)
            
    def inference_review_zs(self, ):
        df = pd.read_csv('./preprocess/recommend/filtered_review_df.csv')
        users = df['name'].unique()[:1000]
        test = load_json('./playground/data/v8/test14.json')
        id2ind = {test[i]['id']: i for i in range(len(test))}
        prompts = [test[id2ind[f'{user}_review_14_1_False']]['conversations'][0]['value'] for user in users[:1000]]
        outputs = self.model.generate_and_save(prompts, users,save_path='./result/rec_review/llava-v1.5-13b-jalan-review-lora-v8.19_0_1000.csv')
        
    def extract_match_aspect(self, mode='val', llm='karasu'):
        test_prompts = []
        user_summary = pd.read_csv('/home/yamanishi/project/airport/src/analysis/LLaVA/preprocess/recommend/user_review_summary.csv')
        user_summaries = user_summary['review_summaries'].values
        users = user_summary['user'].unique()
        if llm == 'karasu':
            df_pos = pd.read_csv('./preprocess/recommend/pos_summary_karasu.csv')
            df_neg = pd.read_csv('./preprocess/recommend/neg_summary_karasu.csv')
            df_pos['summary'] = 'ポジティブな点の要約は以下です\n'+df_pos['pos_summary']+'ネガティブな点の要約は以下です'+df_neg['neg_summary']
        elif llm == 'gpt':
            df_pos = pd.read_csv('./preprocess/recommend/pos_summary.csv')
            df_pos['summary'] = df_pos['pos_summary']
        user2id = {user:i for i,user in enumerate(users)}
        spot2id = {spot:i for i,spot in enumerate(df_pos['spot'])}
        item_summaries = df_pos['summary'].values
        if mode=='val':
            test = load_pkl('./preprocess/recommend/valid.pkl')
            test_interactions = []
        elif mode=='test':
            test = load_pkl('./preprocess/recommend/test.pkl')
            test_interactions = []
        for user in users:
            test_interactions.append((user, test[user]['review'][0][0]))
        for user, spot in test_interactions:
            prompt=''
            prompt+=f'次はユーザー{user}のプロファイルです'
            prompt+=user_summaries[user2id[user]]
            if spot in spot2id:
                prompt+=f'次は観光地{spot}のプロファイルです'
                prompt+=item_summaries[spot2id[spot]]
            prompt+='次のユーザーがアイテムに対してレビューを生成するとき最も重視する観光地の要素を観光地のプロファイルの中から予測してください. ポジティブなこと, ネガティブなことをランキングつけて予測してください.ただし, 観光地のプロファイルは既にランク付けされているわけではありません\n\
出力形式:\n\
ポジティブ:\n\
ランク1:\n\
ランク2: \n\
ランク3:\n\
ネガティブ:\n\
ランク1:\n\
ランク2:\n\
ランク3:\n\
'
            test_prompts.append(prompt)
        match_aspects = self.model.generate(test_prompts)
        save_pkl(match_aspects,
                f'./preprocess/recommend/match_aspects_{mode}.pkl')
        pd.DataFrame({'user': [t[0] for t in test_interactions],
                          'spot': [t[1] for t in test_interactions],
                          'match_aspect': match_aspects}).to_csv(f'./preprocess/recommend/match_aspects_{llm}_{mode}.csv')
    
    def extract_aspect_from_review(self, dataset, prompt_type='aspect', test=False):
        '''
        
        '''
        review_splits = load_pkl(f'./data/p5/data/{dataset}/review_splits.pkl')
        #prompt = prompt_posneg_extract_en
        if prompt_type == 'aspect':
            prompt = prompt_posneg_extract_simple_en[:]
        elif prompt_type == 'fos':
            prompt = prompt_fos_extract[:]
        elif prompt_type == 'cfos':
            prompt = prompt_cfos_extract[:]
        users, items, review_texts, prompts = [], [], [], []
        splits = []
        review_splits['train'] = review_splits['train']
        review_splits['val'] = review_splits['val']
        review_splits['test'] = review_splits['test']
        for split in ['train', 'val', 'test']:
            count = 0
            for reviews in review_splits[split]:
                if dataset == 'tripadvisor':
                    user, item, review_text = reviews['user'], reviews['item'], reviews['reviewText']
                else:
                    user, item, review_text = reviews['reviewerID'], reviews['asin'], reviews['reviewText']
                users.append(user), items.append(item), review_texts.append(review_text), prompts.append(prompt.format(review_text))
                count += 1
                if test:
                    if count==5:break
            splits += [split for _ in range(count)]
        outputs = self.model.generate(prompts)
        df = pd.DataFrame({'user': users, 'item': items, 'review': review_texts, 'extract': outputs, 'split': splits})
        df.to_csv(f'./data/p5/data/{dataset}/extract_{prompt_type}.csv')
        
    def make_unit_reviews(self, dataset):
        user_reviews, item_reviews = defaultdict(list), defaultdict(list)
        review_splits = load_pkl(f'./data/p5/data/{dataset}/review_splits.pkl')
        for datum in review_splits['train']:
            if 'user' in datum:user = datum['user']
            elif 'reviewerID' in datum:user=datum['reviewerID']
            if 'item' in datum:item = datum['item']
            elif 'asin' in datum:item=datum['asin']
            user_reviews[user].append(datum['reviewText'])
            item_reviews[item].append(datum['reviewText'])
            
        save_pkl(item_reviews, f'./data/p5/data/{dataset}/item_reviews.pickle')
        save_pkl(user_reviews, f'./data/p5/data/{dataset}/user_reviews.pickle')
        
    def summarize_1st_step_profile(self, dataset: str, type='user'):
        single_profile_units, single_profiles = [], []
        multi_profile_units, multi_profile_prompts = [], []
        profile_df = pd.read_csv(f'./data/p5/data/{dataset}/{type}_summary.csv')
        profile_df = profile_df.sort_values(type)
        labels = []
        for i,(unit, group) in enumerate(profile_df.groupby(type)):
            if len(group)==1:
                single_profile_units.append(unit)
                single_profiles.append(group['summary'].values[0])
                labels.append('single')
            else:
                current_ind = 0
                if type=='user':
                    current_prompt = prompt_user_merge[:]
                else:
                    current_prompt = item_merge_prompts[dataset][:]
                for profile in group['summary']:
                    # print(profile)
                    if type == 'item':
                        if 'Overall' not in profile:continue
                        profile = profile[profile.find('Overall'):]
                    elif type == 'user':
                        if 'Stylistic features of' not in profile:continue
                        profile = profile[profile.find('Stylistic features of'):]
                    if len(profile)==0:continue
                    current_prompt += f'Profile: {current_ind+1}\n'
                    current_prompt += profile + '\n'
                    current_ind+=1
                    if current_ind == 5:
                        if type=='user':
                            current_prompt = prompt_user_merge[:]
                        else:
                            current_prompt = item_merge_prompts[dataset][:]
                        current_ind = 0
                        multi_profile_prompts.append(current_prompt[:])
                        multi_profile_units.append(unit)
                        labels.append('multi')
                if current_ind!=0:
                    multi_profile_prompts.append(current_prompt[:])
                    multi_profile_units.append(unit)
                    labels.append('multi')
        #print(len(multi_profile_prompts))
        #print('multi profile', multi_profile_prompts)
        summary_of_multi = self.model.generate(multi_profile_prompts)
        units = single_profile_units + multi_profile_units
        summary = single_profiles + summary_of_multi
        pd.DataFrame({type: units, 'summary': summary, 'label': labels}).to_csv(f'../P5/data/{dataset}/{type}_summary_2nd.csv')
        
        
    def summarize_unit_reviews(self, dataset: str, type='user', prompt_type='degree'):
        '''
            input: 
                dataset: [yelp, toys, beauty, sports]
                type: user or item
            output:
                df: [unit, inds, summaries]
        '''
        units, inds, prompts = [], [], []
        if not os.path.exists(f'./data/p5/data/{dataset}/{type}_reviews.pickle'):
            self.make_unit_reviews(dataset)
        unit_reviews = load_pkl(f'./data/p5/data/{dataset}/{type}_reviews.pickle')
        if type=='user':
            # prompt = prompt_review_summary_general_en 
            if prompt_type == 'rank':
                prompt = prompt_user_ranks[dataset]
            elif prompt_type == 'degree':
                prompt = prompt_user_degree[dataset]
        elif type=='item':
            prompt = item_prompts[dataset]
        max_prompt_length = 5000
        for i,(unit, reviews) in enumerate(unit_reviews.items()):
            current_ind = 0
            total_ind = 1
            for review in reviews:
                if len(prompt) + len(review) > max_prompt_length:
                    units.append(unit), inds.append(current_ind), prompts.append(prompt)
                    if type=='user':
                        # prompt = prompt_review_summary_general_en
                        if prompt_type == 'rank':
                            prompt = prompt_user_ranks[dataset]
                        elif prompt_type == 'degree':
                            prompt = prompt_user_degree[dataset]
                    elif type=='item':
                        prompt = item_prompts[dataset]
                    current_ind+=1
                    total_ind = 1
                prompt += f'review {total_ind}:'+ review + "\n"  # Add review to the prompt
                total_ind+=1
            units.append(unit), inds.append(current_ind), prompts.append(prompt)
            if self.args.test:
                if i==5:
                    break
            #if i==10:break
        outputs = self.model.generate(prompts)
        #print('outputs', outputs)
        df = pd.DataFrame({type:units, 'ind': inds, 'summary': outputs})
        # df.to_csv(f'./data/p5/data/{dataset}/{type}_summary.csv')
        print('savepath', f'./data/p5/data/{dataset}/{type}_summary_{prompt_type}.csv')
        df.to_csv(f'./data/p5/data/{dataset}/{type}_summary_{prompt_type}.csv')
        
        
        
    def summarize_short_sentences(self, pos_or_neg_dict, save_path):
        process_size = 100
        user_or_items, inds, prompts = [], [], []
        for user_or_item in list(pos_or_neg_dict.keys()):
            sentences = pos_or_neg_dict[user_or_item]
            for i in range(math.ceil(len(sentences)/process_size)):
                if 'user' in save_path:
                    prompt_tmp = prompt_aspect_summary_user_en[:]
                if 'item' in save_path:
                    prompt_tmp = prompt_aspect_summary_item_en[:]
                sentence_tmp = sentences[process_size*i:process_size*(i+1)]
                for sent in sentence_tmp:
                    prompt_tmp += sent
                    prompt_tmp+='\n'
                user_or_items.append(user_or_item)
                inds.append(i)
                prompts.append(prompt_tmp)
        return user_or_items, inds, prompts
        outputs = self.model.generate(prompts)
        if 'user' in save_path:
            pd.DataFrame({'user': user_or_items, 'ind': inds, 'summary': outputs}).to_csv(save_path)
        else:
            pd.DataFrame({'item': user_or_items, 'ind': inds, 'summary': outputs}).to_csv(save_path)
    
    def summarize_aspect_tripadvisor(self):
        user_positive_dict, user_negative_dict, item_positive_dict, item_negative_dict = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
        extracted_aspect_df = pd.read_csv('./data/p5/data/trip_advisor/extract_aspects.csv')
        count = 0
        for user, item, extract in zip(extracted_aspect_df['user'], extracted_aspect_df['item'],extracted_aspect_df['extract']):
            if pd.isna(extract):
                count+=1
                continue
            positive = extract[extract.find('Positive aspects'):extract.find('Negative aspects')]
            positive = positive.split('・')[1:]
            positive = [p.replace('\n', '') for p in positive]
            negative = extract[extract.find('Negative aspects'):]
            negative = negative.split('・')[1:]
            negative = [n.replace('\n', '') for n in negative]
            # print('positive', positive)
            # print('negative', negative)
            user_positive_dict[user].extend(positive)
            user_negative_dict[user].extend(negative)
            item_positive_dict[item].extend(positive)
            item_negative_dict[item].extend(negative)
            
        user_or_items1, inds1, prompts1 = self.summarize_short_sentences(user_positive_dict, './data/p5/data/trip_advisor/user_positive_summary.csv')
        user_or_items2, inds2, prompts2 = self.summarize_short_sentences(user_negative_dict, './data/p5/data/trip_advisor/user_negative_summary.csv')
        user_or_items3, inds3, prompts3 = self.summarize_short_sentences(item_positive_dict, './data/p5/data/trip_advisor/item_positive_summary.csv')
        user_or_items4, inds4, prompts4 = self.summarize_short_sentences(item_negative_dict, './data/p5/data/trip_advisor/item_negative_summary.csv')
          
        prompts = prompts1 + prompts2 + prompts3 + prompts4
        outputs = self.model.generate(prompts)   
        save_pkl(outputs, './data/p5/data/trip_advisor/aspect_summary.pkl')
        user_or_items = user_or_items1+user_or_items2+user_or_items3+user_or_items4
        inds = inds1 + inds2 + inds3 + inds4
        status = ['user_pos' for _  in range(len(inds1))] + ['user_neg' for _  in range(len(inds2))] + ['item_pos' for _  in range(len(inds3))]+ ['item_neg' for _  in range(len(inds4))]
        pd.DataFrame({'user_or_item': user_or_items, 'ind': inds, 'status': status, 'summary':outputs}).to_csv('./data/p5/data/trip_advisor/aspect_summary.csv')   

        exit()
        for k,v in user_positive_dict.items():
            print('user positive', k, len(v))
        for k,v in user_negative_dict.items():
            print('user negative', k, len(v))
        for k,v in item_positive_dict.items():
            print('item positive', k, len(v))
        for k,v in item_negative_dict.items():
            print('item negative', k, len(v))
        print('count', count)
                
            
        
        
    def generate_pseudo_review(self,):
        pass
        
        

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", type=str)
    parser.add_argument("--dataset", type=str, default='yelp')
    parser.add_argument("--type", type=str, default='user')
    parser.add_argument("--prompt_type", type=str, default='degree')
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--model_name", type=str, default='llama')
    args = parser.parse_args()
    summarizer = Summarizer(args)
    if args.f == 'summarize_unit_reviews':
        summarizer.summarize_unit_reviews(args.dataset, type=args.type, prompt_type=args.prompt_type)
    elif args.f == 'extract_aspect':
        summarizer.extract_aspect_from_review(args.dataset, args.prompt_type, args.test)
        
    #summarizer.summarize_1st_step_profile(args.dataset, type=args.type)
    exit()
    summarizer.summarize_unit_reviews(args.dataset, type=args.type)
    #summarizer.summarize_unit_reviews('yelp', type='item')
    exit()
    for dataset in ['yelp']:
        for data_type in ['user', 'item']:
            summarizer.summarize_unit_reviews('yelp', type='user')
            summarizer.summarize_unit_reviews('yelp', type='item')
    exit()
    summarizer.summarize_item_reviews_tripadvisor()
    exit()
    summarizer.summarize_user_reviews_tripadvisor()
    exit()
    summarizer.summarize_aspect_tripadvisor()
    exit()
    summarizer.extract_aspect_from_review()
    exit()
    summarizer.summarize_user_reviews_tripadvisor()
    exit()
    #summarizer.get_item_pos_summary_all()
    #summarizer.extract_match_aspect(mode='val', llm='gpt')
    summarizer.extract_match_aspect(mode='test', llm='gpt')
    exit()
    summarizer.get_item_neg_summary_all()
    exit()
    summarizer.summarize_review_user(summary_method='general')
    exit()
    summarizer.inference_review_zs()
    exit()
    summarizer.get_item_neg_summary_all(start_ind=0, end_ind=3000)
    exit()
    summarizer.rephrase_review()
    exit()
    summarizer.get_2nd_step_output()
    exit()
    summarizer.get_item_neg_summary_all(start_ind=0, end_ind=3000)
    exit()
    summarizer.rephrase_review()
    exit()
    summarizer.get_item_neg_summary_all()
    exit()
    summarizer.summarize_review_ind(max_num=500000)
    exit()
    summarizer.get_item_pos_summary_all()
    exit()
    summarizer.get_1st_step_output_all('./preprocess/recommend/filtered_review_df_train.csv', 'spot_name')
    exit()
    summarizer.summarize_review_item_all('./preprocess/recommend/filtered_review_df_train.csv', 'spot_name')
    #summarizer.summarize_review_user(max_num=20000)
    summarizer.summarize_review_ind(max_num=200000)
    exit()
        
with open('./playground/data/v8/train2.json') as f:
    train2 = json.load(f)
with open('./playground/data/v8/train3.json') as f:
    train3 = json.load(f)

df = pd.read_csv('./preprocess/recommend/filtered_review_df.csv')
prompt = '次にユーザーの訪問した全ての場所とそこで書いたレビューを渡します。ユーザーについて次の観点からプロファイルしてください。\n\
・文体の特徴\n\
・ユーザの興味、関心があるトピック\n\
・ユーザーがポジティブに感じやすいこと\n\
・ユーザーがネガティブに感じやすいこと\n\
\n\
{}\n\
'
prompt_summary = '次にユーザーの訪問した全ての場所とそこで書いたレビューを渡します。それをれのレビューを1件づず最大50文字程度で要約してください。\n\
{}\n'

prompt1 = 'このレビューを10通りに書き換えて\
若手ホープ、中堅どころからベテランまで、バラエティに富んだアーティストのライブ会場として定番のライブハウス。天井も高く音響的にもグレードが高いと思う。繁華街から近いのも◎'

prompt2 = 'このレビューを10通りに書き換えて\
    4月半ば、桜の通り抜けのシーズンに訪れました。期間が1週間しかない＆比較的天気の良い日曜日だったため、観光客の数が物凄かった！外国の方々も数多く訪れていて、海外のガイド等にも紹介されているのかな、と思います。肝心の桜はちょうど満開～徐々に散りだす、というタイミング。非常にバラエティに富んだ色とりどりの桜を楽しむことができました。'

prompt_summary_ind = '次のレビューを最大50文字程度で要約してください.ただし元の文体も少し残して\n\
{}\n'

sum1 = '\n'.join(train2[0]['conversations'][0]['value'].split('\n')[:-1])
sum2 = '\n'.join(train2[100]['conversations'][0]['value'].split('\n')[:-1])
sums = ['\n'.join(train3[i]['conversations'][0]['value'].split('\n')[:-1]) for i in range(0, 2000, 100)]
#sums = [prompt.format(sm) for sm in sums]
sums = [prompt_summary.format(sm) for sm in sums]

sum_inds = [prompt_summary_ind.format(rv) for rv in df['review'].values[:100000]]
print('sums', sum_inds[:5])
#print('pronpt1', prompt.format(sum1))
#print('pronpt2', prompt.format(sum2))
model = AoKarasu()
#model = AoKarasuTransformer()
#model = QarasuTransformer()
#model = GPT3()
#model = ELYZALLlama()
#result_summary = model.generate(sums)
result_summary = model.generate(sum_inds)
#result_summary = model.generate([prompt.format(sum1),prompt.format(sum2)]) 
                #prompt.format(train2[100]['conversations'][0]['value'])])
print('result_summary', result_summary)
result = model.generate([prompt1, prompt2])
print('result', result)


