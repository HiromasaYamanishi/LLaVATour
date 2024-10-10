import argparse
import copy
import json
import math
import os
from collections import defaultdict
from typing import List

import fire
import pandas as pd
from llm import AoKarasu, AoKarasuTransformer, LLaMA, Qarasu, QarasuTransformer, ElyzaLLama3AWQ, ElyzaLLama3, Calm3, Gemma2, Phi3, SentenceBertJapanese
from prompt import *
from tqdm import tqdm
import random
from utils import *
import argparse
from sklearn.metrics.pairwise import cosine_similarity

class Summarizer:
    def __init__(self, model_name, tensor_parallel_size=4):
        # self.args = args
        # model_name = args.model_name
        self.model_name = model_name
        if model_name == "AoKarasu":
            self.model = AoKarasu()
        elif model_name == "AoKarasu-4B":
            self.model = AoKarasu(is_4bit=True, tensor_parallel_size=tensor_parallel_size)
        elif model_name == "AoKarasuTransformer":
            self.model = AoKarasuTransformer()
        elif model_name == "Qarasu":
            self.model = Qarasu()
        elif model_name == "QarasuTransformer":
            self.model = QarasuTransformer()
        elif model_name == "GPT3":
            self.model = GPT3()
        elif model_name == "llama":
            self.model = LLaMA(
                model_version="3",
                parameter_num="70",
                instruct=True,
                tensor_parallel_size=tensor_parallel_size,
            )
        elif model_name == 'elyzallama3':
            self.model = ElyzaLLama3(tensor_parallel_size=tensor_parallel_size)
        elif model_name == "elyzallama3awq":
            self.model = ElyzaLLama3AWQ(tensor_parallel_size=tensor_parallel_size)
        elif model_name == "calm3":
            self.model = Calm3(tensor_parallel_size=tensor_parallel_size)
        elif model_name == 'gemma2':
            self.model = Gemma2(tensor_parallel_size=tensor_parallel_size)
        elif model_name == 'phi3':
            self.model = Phi3(tensor_parallel_size=tensor_parallel_size)
            
    def parse_rec_result(
            self,
            df_path
    ):
        df = pd.read_csv(df_path)
        prompts = [prompt_rec_parse.format(response=response) for response in df['response']]
        parsed_response = self.model.generate(prompts)
        df['parsed_response'] = parsed_response
        df.to_csv(df_path)

    def summarize_review_user(
        self,
        max_num=20000,
        summary_method="aspect",
    ):
        if summary_method == "aspect":
            prompt_review_summary = prompt_review_summary_aspect[:]
        elif summary_method == "general":
            prompt_review_summary = prompt_review_summary_general
        train3 = load_json("./playground/data/v8/train3.json")
        id2ind = {train3[i]["id"]: i for i, d in enumerate(train3)}
        users = list(set([d["id"].split("_")[0] for d in train3]))
        prompts = [
            prompt_review_summary.format(
                "\n".join(
                    train3[id2ind[f"{user}_review_3_1_False"]]["conversations"][0][
                        "value"
                    ].split("\n")[:-1]
                )
            )
            for user in users
        ]
        print("prompts", prompts[495:500])
        summaries = self.model.generate(prompts[:max_num])
        pd.DataFrame({"user": users[:max_num], "review_summaries": summaries}).to_csv(
            f"./preprocess/recommend/user_review_summary_{summary_method}.csv"
        )

    def summarize_review_ind(self, max_num=500000):
        df = pd.read_csv("./preprocess/recommend/filtered_review_df.csv")
        prompt_summary_ind = (
            "次のレビューを最大50文字程度で要約してください.ただし元の文体も少し残してください\n\
{}\n"
        )
        prompts = [prompt_summary_ind.format(review) for review in df["review"].values]
        summaries = self.model.generate(prompts[:max_num])
        df = pd.DataFrame(
            {"summary": summaries[:max_num], "review": df["review"][:max_num]}
        )
        df.to_csv("./preprocess/recommend/filtered_review_df_summary.csv")

    def rephrase_review(self, max_num=500000):
        df = pd.read_csv("./preprocess/recommend/filtered_review_df.csv")
        prompt_rephrase = "次のレビューを10通りに書き換えてください\n\
{}\n"
        prompts = [prompt_rephrase.format(review) for review in df["review"].values]
        summaries = self.model.generate(prompts[:max_num])
        df = pd.DataFrame(
            {"rephrase": summaries[:max_num], "review": df["review"][:max_num]}
        )
        df.to_csv("./preprocess/recommend/filtered_review_df_rephrase.csv")

    def get_1st_step_output_all(self, df_path, target_col):
        df_review = pd.read_csv(df_path)
        llm = self.model
        spots = []
        prompts = []
        for spot in tqdm(df_review[target_col].unique()):
            df_tmp = df_review.query(f'{target_col} == "{spot}"')
            # prompts = []
            chunk_size = 10
            reviews = df_tmp["reviews"].values
            for i in range((len(reviews) - 1) // chunk_size + 1):
                prompt = copy.copy(prompt_direct_step1[:])
                for j in range(chunk_size * i, min(chunk_size * (i + 1), len(reviews))):
                    prompt += reviews[j]
                prompts.append(prompt)
                spots.append(spot)

        # prompts = prompts[:5]
        generated_topics = llm.generate(
            prompts,
        )
        # print('generated', generated_topics)
        pd.DataFrame({"spot": spots, "topics": generated_topics}).to_csv(
            "./preprocess/recommend/output_1st_step.csv"
        )
        return generated_topics

    def get_2nd_step_output_all(self, df_path, target_col):
        df_review = pd.read_csv(df_path)
        llm = self.model
        spots, prompts = [], []
        for spot in tqdm(df_review[target_col].unique()):
            df_tmp = df_review.query(f'{target_col} == "{spot}"')

            chunk_size = 10
            topics = df_tmp["topics"].values
            for i in range((len(topics) - 1) // chunk_size + 1):
                prompt = copy.copy(prompt_direct_step1[:])
                for j in range(chunk_size * i, min(chunk_size * (i + 1), len(topics))):
                    prompt += reviews[j]
                prompts.append(prompt)
                spots.append(spot)

        prompts = prompts[:5]
        generated_topics = llm.generate(
            prompts,
        )
        print("generated", generated_topics)
        pd.DataFrame({"spot": spots, "topics": generated_topics}).to_csv(
            "./preprocess/recommend/output_1st_step.csv"
        )
        return generated_topics

    def summarize_review_diversity_sample(self, test=False, chunk=None, prompt_type='normal'):
        # sbert = SentenceBertJapanese()
        with open('./playground/data/v4/train.json') as f:
            train = json.load(f)
        spots = set()
        for d in train:
            id = d['id']
            spots.add(id.split('_')[0])
        df_review = pd.read_pickle(
            "/home/yamanishi/project/trip_recommend/data/jalan/review/review_all_period_.pkl"
        )
        model = self.model
        spot2review = defaultdict(list)
        test_reviews = pd.read_csv('./data/df_review_feature_eval.csv')
        for spot, review in zip(df_review['spot'], df_review['review']):
            if spot not in spots:continue
            if review in test_reviews:continue
            spot2review[spot].append(review)

        prompts, spots = [], []
        spots_unique = list(spot2review.keys())
        ls = len(spots_unique)
        chunk_num = ls//3
        spot_pre = spots_unique[:chunk_num]
        spot_middle = spots_unique[chunk_num:chunk_num*2]
        spot_post = spots_unique[chunk_num*2:]
        if chunk is None:
            spots_unique = spots_unique
        elif chunk == 'pre':
            spots_unique = spot_pre
        elif chunk == 'middle':
            spots_unique = spot_middle
        elif chunk == 'post':
            spots_unique = spot_post
        print(spots_unique[:5])
        for spot in spots_unique:
            reviews = spot2review[spot]
            # review_emb = sbert.encode(reviews)

            # kernel_matrix = cosine_similarity(review_emb, review_emb)
            # chosen_inds = dpp_sw(kernel_matrix, window_size=100, max_length=50, epsilon=1E-10)
            # chosen_reviews = [reviews[ind] for ind in chosen_inds]
            if len(reviews)>200:
                reviews = [r for r in reviews if len(r)>40]
            chosen_reviews = random.sample(reviews, min(len(reviews), 50))
            reviews_summary = ''
            for review in chosen_reviews:
                reviews_summary+=review
                reviews_summary+='\n'
                if len(reviews_summary)>4900:
                    break
            #chosen_reviews = '\n'.join(chosen_reviews)
            if prompt_type=='normal':
                prompt_tmp = prompt_summary_review[:]
            elif prompt_type == 'improved':
                prompt_tmp = prompt_summary_improved[:]
            elif prompt_type == 're':
                prompt_tmp = prompt_summary_review_re[:]
            prompts.append(prompt_tmp.format(reviews=reviews_summary))
            spots.append(spot)
        if test:
            prompts = prompts[:5]
            spots = spots[:5]
        outputs = model.generate(prompts)
        if test:
            print(outputs)
        pd.DataFrame({'spot': spots, 'output': outputs}).to_csv(f'./data/spot_review_summary_diverse_re_{chunk}_{prompt_type}.csv')

    def summarize_review_all(
        self, test=False,
    ):
        with open('./playground/data/v4/train.json') as f:
            train = json.load(f)
        spots = set()
        for id in train['id']:
            spots.add(id.split('_')[0])
        df_review = pd.read_pickle(
            "/home/yamanishi/project/trip_recommend/data/jalan/review/review_all_period_.pkl"
        )
        spot2review = defaultdict(list)
        test_reviews = pd.read_csv('../data/df_review_feature_eval.csv')
        for spot, review in zip(df_review['spot'], df_review['review']):
            if spot not in spots:continue
            if review in test_reviews:continue
            spot2review[spot].append(review)

        prompts, spots = [], []
        for spot in spot2review.keys():
            reviews = spot2review[spot]
            prompt_tmp = prompt_summary_all[:]
            for i,review in enumerate(reviews):
                if i==500:break
                prompt_sumary_all+=f'レビュー{i}:' + review
            prompts.append(prompt_summary_all)
            spots.append(spot)

        if test:
            prompts = prompts[:5]
            spots = spots[:5]

        pd.DataFrame({'spot': spots, 'prompt': prompts}).to_csv('./data/spot_summary_all.csv')
        pass
        
    def get_user_profile(
            self, max_review_num=40, test=False, prompt_type='short'
    ):
        train_df = pd.read_csv('./data/dataset_personalize/train.csv')
        test_df = pd.read_csv('./data/dataset_personalize/test.csv')
        test_urls = set(list(test_df['url'].values))
        urls, prompts = [], []
        for i, (url, group) in tqdm(enumerate(train_df.groupby('url'))):
            if url in test_urls:
                review_tmp = group['review'].values[:max_review_num]
                visit_time_tmp = group['visit_time'].values[:max_review_num]
                pref_tmp = group['visit_time'].values[:max_review_num]
                review_user = ''
                for review, visit_time, pref in zip(review_tmp, visit_time_tmp, pref_tmp):
                    review_user+=review
                    review_user+=f'(訪問日: {visit_time}, 都道府県: {pref})'
                    review_user+='\n'
                urls.append(url)
                if prompt_type=='short':
                    prompt_tmp = prompt_user_profile_short[:]
                elif prompt_type == 'detail':
                    prompt_tmp = prompt_user_profile[:]
                prompts.append(prompt_tmp.format(reviews='\n'.join(review_tmp)))
            if test:
                if len(prompts)==5:break
        #print('prompts', prompts)
        outputs = self.model.generate(prompts)
        pd.DataFrame({'url': urls, 'profile': outputs}).to_csv(f'./data/dataset_personalize/user_profile_{prompt_type}.csv')

    def get_user_profile_short_for_review(
            self, max_review_num=40, test=False, prompt_type='short', train_df_path='./data/seq_rec/review.csv'
    ):
        # train_df = pd.read_csv('./data/dataset_personalize/review_train.csv')
        train_df = pd.read_csv(train_df_path)
        with open('./data/dataset_personalize/train_url.pkl', 'rb') as f:
            train_urls = pickle.load(f)
        urls, prompts = [], []
        #df_urls = pd.read_csv('./data/inference_review_attribute_all.csv')
        for i, (url, group) in tqdm(enumerate(train_df.groupby('url'))):
        #for i,url in enumerate(df_urls['url']):
            #if url not in train_urls or len(group)<3:continue
            #group = train_df[train_df['url']==url]
            review_tmp = group['review'].values[:max_review_num]
            visit_time_tmp = group['visit_time'].values[:max_review_num]
            pref_tmp = group['pref'].values[:max_review_num]
            review_user = ''
            for review, visit_time, pref in zip(review_tmp, visit_time_tmp, pref_tmp):
                review_user+=review
                review_user+=f'(訪問日: {visit_time}, 都道府県: {pref})'
                review_user+='\n'
            urls.append(url)
            if prompt_type=='tag':
                prompt_tmp = prompt_user_profile_tag[:]
            elif prompt_type == 'sent':
                prompt_tmp = prompt_user_profile_sent[:]
            prompts.append(prompt_tmp.format(reviews='\n'.join(review_tmp)))
            if test:
                if len(prompts)==5:break
        print('prompts', len(prompts))
        outputs = self.model.generate(prompts)
        if test:
            print('outputs', outputs)
        #pd.DataFrame({'url': urls, 'profile': outputs}).to_csv(f'./data/dataset_personalize/user_profile_{prompt_type}_test.csv')
        pd.DataFrame({'url': urls, 'profile': outputs}).to_csv(f'./data/seq_rec/user_profile_{prompt_type}.csv')

    def get_item_summary_all(
        self, dict_path, save_path, mode='pos', test=False, chunk_num=200
    ):
        llm = self.model
        # pos_dict = load_pkl("./preprocess/recommend/pos_dict.pkl")
        pos_dict = load_pkl(dict_path)
        spots, prompt_all = [], []
        prompt_length = []
        for spot in tqdm(pos_dict.keys()):
            for i in range(min(math.ceil(len(pos_dict[spot])/chunk_num), 5)):
                pos_topic = pos_dict[spot][chunk_num*i:chunk_num*(i+1)]
                if mode=='pos':
                    prompt_tmp = prompt_topic_summary[:]
                else:
                    prompt_tmp = prompt_topic_summary_neg[:]
                for topic in pos_topic:
                    prompt_tmp += topic
                    prompt_tmp += "\n"
                prompt_length.append(len(prompt_tmp))
                prompt_all.append(prompt_tmp)
                spots.append(spot)
        prompt_length = sorted(prompt_length, reverse=True)
        print(prompt_length[:20])
        if test:
            prompt_all = prompt_all[:10]
            spots = spots[:10]
        outputs = llm.generate(prompt_all)
        # spots = spots[10:13]
        if test:
            print(outputs[:5])
        with open(save_path + '.pkl', 'wb') as f:
            pickle.dump(outputs, f)

        #with open("./preprocess/recommend/item_pos_summary_karasu.pkl", "wb") as f:
        #    pickle.dump(outputs, f)

        df = pd.DataFrame({"spot": spots, "summary": outputs})
        df.to_csv(save_path + '.csv')
        #df.to_csv("./preprocess/recommend/pos_summary_karasu.csv")

    def reduce_noise_summary(self, df_path):
        llm = self.model
        # pos_dict = load_pkl("./preprocess/recommend/pos_dict.pkl")
        df = pd.read_csv(df_path)
        prompts = []
        for i,summary in enumerate(df['summary']):
            prompt_tmp = prompt_summary_clean[:]
            prompts.append(prompt_tmp.format(sentence=summary))
        # print('prompt_all', prompt_all[:2])
        print(len(prompts))
        #prompt_all = prompt_all[:5]
        #spots = spots[:5]
        outputs = llm.generate(prompts)

        df['output_clean'] = outputs
        df.to_csv(df_path)


    def get_1st_step_output(self, llm, reviews):
        prompt_step1 = prompt_direct_step1

        # 15個づづレビューを処理する
        prompts = []
        chunk_size = 10
        for i in range((len(reviews) - 1) // chunk_size + 1):
            prompt = copy.copy(prompt_step1[:])
            for j in range(chunk_size * i, min(chunk_size * (i + 1), len(reviews))):
                prompt += reviews[j]
            prompts.append(prompt)

        generated_topics = llm.generate(
            prompts,
        )
        return generated_topics

    def get_2nd_step_output(self, llm, summarization_1st):
        prompt_step2 = prompt_direct_step2
        prompts = []
        summary_chunk_size = 10
        for i in range((len(summarization_1st) - 1) // summary_chunk_size + 1):
            topic_prompt = copy.copy(prompt_step2[:])
            for j in range(
                summary_chunk_size * i,
                min(summary_chunk_size * (i + 1), len(summarization_1st)),
            ):
                topic_prompt += f"要約{j-summary_chunk_size*i+1}:\n"
                topic_prompt += summarization_1st[j]
            # print('step1 prompt', topic_prompt)
            prompts.append(topic_prompt)

        summarized_topics = llm.generate(
            prompts,
        )

        # hierachical summarization step2
        if len(summarized_topics) > 1:
            prompts = []
            summary_chunk_size = 10
            for i in range((len(summarized_topics) - 1) // summary_chunk_size + 1):
                topic_final_prompt = prompt_step2[:]
                for j in range(
                    summary_chunk_size * i,
                    min(summary_chunk_size * (i + 1), len(summarized_topics)),
                ):
                    topic_final_prompt += f"要約{j-summary_chunk_size*i+1}:\n"
                    topic_final_prompt += summarized_topics[j]
                prompts.append(topic_final_prompt)

            summarized_topics = llm.generate(prompts)

        if len(summarized_topics) > 1:
            prompts = []
            summary_chunk_size = 10
            for i in range((len(summarized_topics) - 1) // summary_chunk_size + 1):
                topic_final_prompt = prompt_step2[:]
                for j in range(
                    summary_chunk_size * i,
                    min(summary_chunk_size * (i + 1), len(summarized_topics)),
                ):
                    topic_final_prompt += f"要約{j-summary_chunk_size*i+1}:\n"
                    topic_final_prompt += summarized_topics[j]
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
            summarization = self.summarize_review_item(llm, df_tmp["reviews"].values)
            print("summarization", summarization)
            df = pd.DataFrame({"spot": spot, "summary": summarization})
            df.to_csv(
                "./preprocess/recommend/item_summary.csv",
                mode="a",
                header=False,
                index=False,
            )

    def extract_common_keyword_for_pvqa(self):
        df = pd.read_csv(
            "./data/retrieved_sentence_top5.csv",
            names=["spot", "image_path", "sentence", "ind"],
        )
        count = 0
        prev_sents = ["" for _ in range(5)]
        prompts, image_paths, spots = [], [], []
        sents_count = defaultdict(lambda: defaultdict(int))
        image_path_already = pd.read_csv("./data/pvqa_common_keywords.csv")[
            "image_path"
        ].values
        for i, (image_path, group) in tqdm(
            enumerate(df.groupby("image_path", sort=False))
        ):
            #if image_path in image_path_already:
            #    continue
            sents = group["sentence"].values
            spot = group["spot"].values[0]
            for sent in sents:
                sents_count[spot][sent] += 1
            if len(set(sents) & set(prev_sents)) >= 2:
                continue
            counts = sum(sorted([sents_count[spot][sent] for sent in sents])[-3:])
            if counts >= 50:
                continue
            count += 1
            prev_sents = sents

            prompt = prompt_common_pvqa[:]
            for j, sent in enumerate(sents):
                prompt += f"文{j+1}"
                prompt += sent
                prompt += "\n"
            prompt += "出力:\n"
            prompts.append(prompt), image_paths.append(image_path), spots.append(spot)
            # if i == 100:
            #     break
        print(len(prompts))
        outputs = self.model.generate(prompts)
        pd.DataFrame(
            {"spot": spots, "image_path": image_paths, "keyword": outputs}
        ).to_csv("./data/pvqa_common_keywords.csv", mode='a')
        
    def summarize_for_pvqa2(self, test=False, ind=1):
        print('summarize for pvqa')
        def filter_sentence(spot, keyword):
            return [sent for sent in spot_sentences[spot] if keyword in sent]
        
        df_keyword = pd.read_csv('./data/keyword_image.csv')
        with open('./preprocess/spot_sentences.pkl', 'rb') as f:
            spot_sentences = pickle.load(f)
        prompts, spots, keywords, image_paths, sents_all = [], [], [], [], []
        #df_keyword = df_keyword.drop_duplicates(subset=['spot', 'keyword'])
        print(len(df_keyword))
        if ind==1:
            df_keyword = df_keyword[:200000]
        for i,(spot, keyword, image_path) in enumerate(zip(df_keyword['spot'], df_keyword['keyword'], df_keyword['image_path'])):
            if spot not in spot_sentences:continue
            sents = filter_sentence(spot, keyword)
            if len(sents) < 5:continue
            sents = sorted(sents, key=len)[::-1][:20]
            sents = random.sample(sents, min(len(sents), 5))
            #print('sents', len(sents))
            prompt = prompt_summary_for_pvqa[:].format(keyword)
            for i,sent in enumerate(sents):
                prompt += f'文{i+1}:' + sent
                prompt += '\n'
            prompts.append(prompt), keywords.append(keyword), image_paths.append(image_path), spots.append(spot)
            sents_all.append(' '.join(sents))
        if test:
            test_num = 20
            prompts = random.sample(prompts, test_num)
            keywords = keywords[:test_num]
            sents_all = sents_all[:test_num]
            spots = spots[:test_num]
            image_paths = image_paths[:test_num]
        print(len(prompts))
        outputs = self.model.generate(prompts)
        if ind==1:
            pd.DataFrame({'spot': spots, 
                      'image_path': image_paths, 
                      'keyword': keywords, 
                      'summary': outputs, 
                      'sents': sents_all}).to_csv('./data/pvqa_summary_calm3_0_200000.csv', mode='a')
        elif ind==2:
            pd.DataFrame({'spot': spots, 
                      'image_path': image_paths, 
                      'keyword': keywords, 
                      'summary': outputs, 
                      'sents': sents_all}).to_csv('./data/pvqa_summary_calm3_200000_last.csv', mode='a')
        
    def summarize_for_pvqa(self, test=False):
        print('summarize for pvqa')
        def filter_sentence(spot, keyword):
            return [sent for sent in spot_sentences[spot] if keyword in sent]
        df_keyword = pd.read_csv('./data/pvqa_common_keywords.csv')
        with open('./preprocess/spot_sentences.pkl', 'rb') as f:
            spot_sentences = pickle.load(f)
        #print('spot sentences', spot_sentences)
        prompts, spots, keywords, image_paths, sents_all = [], [], [], [], []
        for i,(spot, keyword, image_path) in enumerate(zip(df_keyword['spot'], df_keyword['keyword'], df_keyword['image_path'])):
            if not spot in spot_sentences:continue
            sents = filter_sentence(spot, keyword)
            if len(sents) < 5:continue
            sents = sorted(sents, key=len)[::-1][:7]
            #print('sents', len(sents))
            prompt = prompt_summary_for_pvqa[:].format(keyword)
            for i,sent in enumerate(sents):
                prompt += f'文{i+1}:' + sent
                prompt += '\n'
            prompts.append(prompt), keywords.append(keyword), image_paths.append(image_path), spots.append(spot)
            sents_all.append(' '.join(sents))
        if test:
            test_num = 20
            prompts = random.sample(prompts, test_num)
            keywords = keywords[:test_num]
            sents_all = sents_all[:test_num]
            spots = spots[:test_num]
            image_paths = image_paths[:test_num]
            
        outputs = self.model.generate(prompts)
        # for prompt, output in zip(prompts, outputs):
        #     print('prompt',prompt)
        #     print('output', output)
        pd.DataFrame({'spot': spots, 'image_path': image_paths, 'keyword': keywords, 'summary': outputs, 'sents': sents_all}).to_csv('./data/pvqa_summary.csv', mode='a')



    def inference_review_zs(
        self,
    ):
        df = pd.read_csv("./preprocess/recommend/filtered_review_df.csv")
        users = df["name"].unique()[:1000]
        test = load_json("./playground/data/v8/test14.json")
        id2ind = {test[i]["id"]: i for i in range(len(test))}
        prompts = [
            test[id2ind[f"{user}_review_14_1_False"]]["conversations"][0]["value"]
            for user in users[:1000]
        ]
        outputs = self.model.generate_and_save(
            prompts,
            users,
            save_path="./result/rec_review/llava-v1.5-13b-jalan-review-lora-v8.19_0_1000.csv",
        )

    def extract_match_aspect(self, mode="val", llm="karasu"):
        test_prompts = []
        user_summary = pd.read_csv(
            "/home/yamanishi/project/airport/src/analysis/LLaVA/preprocess/recommend/user_review_summary.csv"
        )
        user_summaries = user_summary["review_summaries"].values
        users = user_summary["user"].unique()
        if llm == "karasu":
            df_pos = pd.read_csv("./preprocess/recommend/pos_summary_karasu.csv")
            df_neg = pd.read_csv("./preprocess/recommend/neg_summary_karasu.csv")
            df_pos["summary"] = (
                "ポジティブな点の要約は以下です\n"
                + df_pos["pos_summary"]
                + "ネガティブな点の要約は以下です"
                + df_neg["neg_summary"]
            )
        elif llm == "gpt":
            df_pos = pd.read_csv("./preprocess/recommend/pos_summary.csv")
            df_pos["summary"] = df_pos["pos_summary"]
        user2id = {user: i for i, user in enumerate(users)}
        spot2id = {spot: i for i, spot in enumerate(df_pos["spot"])}
        item_summaries = df_pos["summary"].values
        if mode == "val":
            test = load_pkl("./preprocess/recommend/valid.pkl")
            test_interactions = []
        elif mode == "test":
            test = load_pkl("./preprocess/recommend/test.pkl")
            test_interactions = []
        for user in users:
            test_interactions.append((user, test[user]["review"][0][0]))
        for user, spot in test_interactions:
            prompt = ""
            prompt += f"次はユーザー{user}のプロファイルです"
            prompt += user_summaries[user2id[user]]
            if spot in spot2id:
                prompt += f"次は観光地{spot}のプロファイルです"
                prompt += item_summaries[spot2id[spot]]
            prompt += (
                "次のユーザーがアイテムに対してレビューを生成するとき最も重視する観光地の要素を観光地のプロファイルの中から予測してください. ポジティブなこと, ネガティブなことをランキングつけて予測してください.ただし, 観光地のプロファイルは既にランク付けされているわけではありません\n\
出力形式:\n\
ポジティブ:\n\
ランク1:\n\
ランク2: \n\
ランク3:\n\
ネガティブ:\n\
ランク1:\n\
ランク2:\n\
ランク3:\n\
"
            )
            test_prompts.append(prompt)
        match_aspects = self.model.generate(test_prompts)
        save_pkl(match_aspects, f"./preprocess/recommend/match_aspects_{mode}.pkl")
        pd.DataFrame(
            {
                "user": [t[0] for t in test_interactions],
                "spot": [t[1] for t in test_interactions],
                "match_aspect": match_aspects,
            }
        ).to_csv(f"./preprocess/recommend/match_aspects_{llm}_{mode}.csv")

    def extract_aspect_from_review(self, dataset, prompt_type="aspect", test=False):
        """ """
        review_splits = load_pkl(f"./data/p5/data/{dataset}/review_splits.pkl")
        # prompt = prompt_posneg_extract_en
        if prompt_type == "aspect":
            prompt = prompt_posneg_extract_simple_en[:]
        elif prompt_type == "fos":
            prompt = prompt_fos_extract[:]
        elif prompt_type == "acos":
            prompt = prompt_acos_extract[dataset][:]
        users, items, review_texts, prompts = [], [], [], []
        splits = []
        review_splits["train"] = review_splits["train"]
        review_splits["val"] = review_splits["val"]
        review_splits["test"] = review_splits["test"]
        for split in ["train", "val", "test"]:
            count = 0
            for reviews in review_splits[split]:
                if dataset == "tripadvisor":
                    user, item, review_text = (
                        reviews["user"],
                        reviews["item"],
                        reviews["reviewText"],
                    )
                else:
                    user, item, review_text = (
                        reviews["reviewerID"],
                        reviews["asin"],
                        reviews["reviewText"],
                    )
                (
                    users.append(user),
                    items.append(item),
                    review_texts.append(review_text),
                    prompts.append(prompt.format(review_text)),
                )
                count += 1
                if test:
                    if count == 5:
                        break
            splits += [split for _ in range(count)]
        outputs = self.model.generate(prompts)
        df = pd.DataFrame(
            {
                "user": users,
                "item": items,
                "review": review_texts,
                "extract": outputs,
                "split": splits,
            }
        )
        df.to_csv(f"../P5/data/{dataset}/extract_{prompt_type}.csv")

    def make_unit_reviews(self, dataset):
        user_reviews, item_reviews = defaultdict(list), defaultdict(list)
        review_splits = load_pkl(f"./data/p5/data/{dataset}/review_splits.pkl")
        for datum in review_splits["train"]:
            if "user" in datum:
                user = datum["user"]
            elif "reviewerID" in datum:
                user = datum["reviewerID"]
            if "item" in datum:
                item = datum["item"]
            elif "asin" in datum:
                item = datum["asin"]
            user_reviews[user].append(datum["reviewText"])
            item_reviews[item].append(datum["reviewText"])

        save_pkl(item_reviews, f"./data/p5/data/{dataset}/item_reviews.pickle")
        save_pkl(user_reviews, f"./data/p5/data/{dataset}/user_reviews.pickle")

    def summarize_1st_step_profile(self, dataset: str, type="user"):
        single_profile_units, single_profiles = [], []
        multi_profile_units, multi_profile_prompts = [], []
        profile_df = pd.read_csv(f"./data/p5/data/{dataset}/{type}_summary.csv")
        profile_df = profile_df.sort_values(type)
        labels = []
        for i, (unit, group) in enumerate(profile_df.groupby(type)):
            if len(group) == 1:
                single_profile_units.append(unit)
                single_profiles.append(group["summary"].values[0])
                labels.append("single")
            else:
                current_ind = 0
                if type == "user":
                    current_prompt = prompt_user_merge[:]
                else:
                    current_prompt = item_merge_prompts[dataset][:]
                for profile in group["summary"]:
                    # print(profile)
                    if type == "item":
                        if "Overall" not in profile:
                            continue
                        profile = profile[profile.find("Overall") :]
                    elif type == "user":
                        if "Stylistic features of" not in profile:
                            continue
                        profile = profile[profile.find("Stylistic features of") :]
                    if len(profile) == 0:
                        continue
                    current_prompt += f"Profile: {current_ind+1}\n"
                    current_prompt += profile + "\n"
                    current_ind += 1
                    if current_ind == 5:
                        if type == "user":
                            current_prompt = prompt_user_merge[:]
                        else:
                            current_prompt = item_merge_prompts[dataset][:]
                        current_ind = 0
                        multi_profile_prompts.append(current_prompt[:])
                        multi_profile_units.append(unit)
                        labels.append("multi")
                if current_ind != 0:
                    multi_profile_prompts.append(current_prompt[:])
                    multi_profile_units.append(unit)
                    labels.append("multi")
        # print(len(multi_profile_prompts))
        # print('multi profile', multi_profile_prompts)
        summary_of_multi = self.model.generate(multi_profile_prompts)
        units = single_profile_units + multi_profile_units
        summary = single_profiles + summary_of_multi
        pd.DataFrame({type: units, "summary": summary, "label": labels}).to_csv(
            f"../P5/data/{dataset}/{type}_summary_2nd.csv"
        )
        
    def extract_qa_multi(self, review_start=0, review_end=1000000):
        df_review = pd.read_pickle(
            "/home/yamanishi/project/trip_recommend/data/jalan/review/review_all_period_.pkl"
        )
        df_review['length'] = df_review['review'].str.len()
        df_review_sorted = df_review.sort_values('length', ascending=False)[review_start:review_end]
        prompts = []
        for review in df_review_sorted['review']:
            prompt = prompt_qa_multi[:]
            prompts.append(prompt.format(input_review=review))
            
        outputs = self.model.generate(prompts)
        df_review_sorted['qa_pairs'] = outputs
        df_review_sorted.to_csv(f'./data/qa_pairs_jalan_calm_{review_start}_{review_end}.csv')
            
    def convert(self, review_start=0, review_end=1000000, prompt_type='market'):
        df_review = pd.read_pickle(
            "/home/yamanishi/project/trip_recommend/data/jalan/review/review_all_period_.pkl"        
        )
        df_review['length'] = df_review['review'].str.len()
        df_review = df_review[df_review['length']>=10]
        if prompt_type=='market':random_state=42
        elif prompt_type=='search':random_state=44
        else:random_state=45
        df_review = df_review.sample(frac=1, random_state=random_state)[review_start:review_end]
        prompts = []
        for i,row in df_review.iterrows():
            if prompt_type == 'search':
                prompt = make_search_prompt(row)
            elif prompt_type == 'marketing':
                prompt = make_marketing_prompt(row)
            prompts.append(prompt)
        #print(prompts[:5])
        outputs = self.model.generate(prompts)
        # print(outpus[:5])
        df_review['output'] = outputs
        df_review.to_csv(f'./data/{prompt_type}_{self.model_name}.csv')
        # pass

    def summarize_unit_reviews(self, dataset: str, type="user", prompt_type="degree"):
        """
        input:
            dataset: [yelp, toys, beauty, sports]
            type: user or item
        output:
            df: [unit, inds, summaries]
        """
        units, inds, prompts = [], [], []
        if not os.path.exists(f"./data/p5/data/{dataset}/{type}_reviews.pickle"):
            self.make_unit_reviews(dataset)
        unit_reviews = load_pkl(f"./data/p5/data/{dataset}/{type}_reviews.pickle")
        if type == "user":
            # prompt = prompt_review_summary_general_en
            if prompt_type == "rank":
                prompt = prompt_user_ranks[dataset]
            elif prompt_type == "degree":
                prompt = prompt_user_degree[dataset]
        elif type == "item":
            prompt = item_prompts[dataset]
        max_prompt_length = 5000
        for i, (unit, reviews) in enumerate(unit_reviews.items()):
            current_ind = 0
            total_ind = 1
            for review in reviews:
                if len(prompt) + len(review) > max_prompt_length:
                    units.append(unit), inds.append(current_ind), prompts.append(prompt)
                    if type == "user":
                        # prompt = prompt_review_summary_general_en
                        if prompt_type == "rank":
                            prompt = prompt_user_ranks[dataset]
                        elif prompt_type == "degree":
                            prompt = prompt_user_degree[dataset]
                    elif type == "item":
                        prompt = item_prompts[dataset]
                    current_ind += 1
                    total_ind = 1
                prompt += (
                    f"review {total_ind}:" + review + "\n"
                )  # Add review to the prompt
                total_ind += 1
            units.append(unit), inds.append(current_ind), prompts.append(prompt)
            if self.args.test:
                if i == 5:
                    break
            # if i==10:break
        outputs = self.model.generate(prompts)
        # print('outputs', outputs)
        df = pd.DataFrame({type: units, "ind": inds, "summary": outputs})
        # df.to_csv(f'./data/p5/data/{dataset}/{type}_summary.csv')
        print("savepath", f"./data/p5/data/{dataset}/{type}_summary_{prompt_type}.csv")
        df.to_csv(f"./data/p5/data/{dataset}/{type}_summary_{prompt_type}.csv")

    def extract_noun_adj_japanese(self):
        df_review_pred = pd.read_csv('./data/df_review_7review.csv')
        prompts = [prompt_noun_adj_jp.format(review=review) for review in tqdm(df_review_pred['pred'])]
        outputs = self.model.generate(prompts)
        df_review_pred['output'] = outputs
        df_review_pred.to_csv('./data/df_review_7review.csv')

    def extract_fos_japanese(self, ):
        df_review = pd.read_pickle(
            "/home/yamanishi/project/trip_recommend/data/jalan/review/review_all_period_.pkl"
        )
        print('load review')
        prompts = [prompt_fos_japanese.format(review) for review in tqdm(df_review['review'])]
        chunk_size = 500000
        ind =1
        output = self.model.generate(prompts[chunk_size*ind:chunk_size*(ind+1)])
        print(output)
        with open(f'./data/jalan_review_fos_{chunk_size*ind}_{chunk_size*(ind+1)}.pkl', 'wb') as f:
            pickle.dump(output, f)

    def summarize_short_sentences(self, pos_or_neg_dict, save_path):
        process_size = 100
        user_or_items, inds, prompts = [], [], []
        for user_or_item in list(pos_or_neg_dict.keys()):
            sentences = pos_or_neg_dict[user_or_item]
            for i in range(math.ceil(len(sentences) / process_size)):
                if "user" in save_path:
                    prompt_tmp = prompt_aspect_summary_user_en[:]
                if "item" in save_path:
                    prompt_tmp = prompt_aspect_summary_item_en[:]
                sentence_tmp = sentences[process_size * i : process_size * (i + 1)]
                for sent in sentence_tmp:
                    prompt_tmp += sent
                    prompt_tmp += "\n"
                user_or_items.append(user_or_item)
                inds.append(i)
                prompts.append(prompt_tmp)
        return user_or_items, inds, prompts
        outputs = self.model.generate(prompts)
        if "user" in save_path:
            pd.DataFrame(
                {"user": user_or_items, "ind": inds, "summary": outputs}
            ).to_csv(save_path)
        else:
            pd.DataFrame(
                {"item": user_or_items, "ind": inds, "summary": outputs}
            ).to_csv(save_path)

    def extract_aspect_sequence(self):
        df = pd.read_csv('../mrg/data/train.csv')
        #if index==0:
        reviews = df['review'].unique()[280000:]
        prompts = [prompt_aspect_sequence.format(review=review) for review in reviews]
        results = self.model.generate(prompts)
        pd.DataFrame({'review': reviews, 'aspect_sequence': results}).to_csv('./data/aspect_sequece2.csv')

    def extract_sketch(self):
        aspect_sequence1 = pd.read_csv('./data/aspect_sequece1.csv')
        aspect_sequence2 = pd.read_csv('./data/aspect_sequece2.csv')
        aspect_sequence = pd.concat([aspect_sequence1, aspect_sequence2]).reset_index()
        review2seq = {review:seq for review,seq in zip(aspect_sequence['review'], aspect_sequence['aspect_sequence'])}
        df = pd.read_csv('../mrg/data/train.csv')
        #if index==0:
        reviews = df['review'].unique()
        prompts = [prompt_sketch.format(review=review, aspect=review2seq[review]) for review in reviews]
        results = self.model.generate(prompts)
        pd.DataFrame({'review': reviews, 'aspect_sequence': results}).to_csv('./data/sketch.csv')

    def summarize_aspect_tripadvisor(self):
        (
            user_positive_dict,
            user_negative_dict,
            item_positive_dict,
            item_negative_dict,
        ) = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
        extracted_aspect_df = pd.read_csv(
            "./data/p5/data/trip_advisor/extract_aspects.csv"
        )
        count = 0
        for user, item, extract in zip(
            extracted_aspect_df["user"],
            extracted_aspect_df["item"],
            extracted_aspect_df["extract"],
        ):
            if pd.isna(extract):
                count += 1
                continue
            positive = extract[
                extract.find("Positive aspects") : extract.find("Negative aspects")
            ]
            positive = positive.split("・")[1:]
            positive = [p.replace("\n", "") for p in positive]
            negative = extract[extract.find("Negative aspects") :]
            negative = negative.split("・")[1:]
            negative = [n.replace("\n", "") for n in negative]
            # print('positive', positive)
            # print('negative', negative)
            user_positive_dict[user].extend(positive)
            user_negative_dict[user].extend(negative)
            item_positive_dict[item].extend(positive)
            item_negative_dict[item].extend(negative)

        user_or_items1, inds1, prompts1 = self.summarize_short_sentences(
            user_positive_dict, "./data/p5/data/trip_advisor/user_positive_summary.csv"
        )
        user_or_items2, inds2, prompts2 = self.summarize_short_sentences(
            user_negative_dict, "./data/p5/data/trip_advisor/user_negative_summary.csv"
        )
        user_or_items3, inds3, prompts3 = self.summarize_short_sentences(
            item_positive_dict, "./data/p5/data/trip_advisor/item_positive_summary.csv"
        )
        user_or_items4, inds4, prompts4 = self.summarize_short_sentences(
            item_negative_dict, "./data/p5/data/trip_advisor/item_negative_summary.csv"
        )

        prompts = prompts1 + prompts2 + prompts3 + prompts4
        outputs = self.model.generate(prompts)
        save_pkl(outputs, "./data/p5/data/trip_advisor/aspect_summary.pkl")
        user_or_items = (
            user_or_items1 + user_or_items2 + user_or_items3 + user_or_items4
        )
        inds = inds1 + inds2 + inds3 + inds4
        status = (
            ["user_pos" for _ in range(len(inds1))]
            + ["user_neg" for _ in range(len(inds2))]
            + ["item_pos" for _ in range(len(inds3))]
            + ["item_neg" for _ in range(len(inds4))]
        )
        pd.DataFrame(
            {
                "user_or_item": user_or_items,
                "ind": inds,
                "status": status,
                "summary": outputs,
            }
        ).to_csv("./data/p5/data/trip_advisor/aspect_summary.csv")

        exit()
        for k, v in user_positive_dict.items():
            print("user positive", k, len(v))
        for k, v in user_negative_dict.items():
            print("user negative", k, len(v))
        for k, v in item_positive_dict.items():
            print("item positive", k, len(v))
        for k, v in item_negative_dict.items():
            print("item negative", k, len(v))
        print("count", count)


if __name__ == "__main__":
    #summarizer = Summarizer('AoKarasu-4B', tensor_parallel_size=4)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',)
    parser.add_argument('--polarity', type=str, default='pos')
    parser.add_argument('--chunk', type=str, default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--prompt_type', type=str, default='normal')
    parser.add_argument('--tensor_parallel_size', type=int, default=2)
    parser.add_argument('--df_path', type=str)
    args = parser.parse_args()
    
    summarizer = Summarizer(args.model, tensor_parallel_size=args.tensor_parallel_size)
    # summarizer.extract_aspect_sequence()
    summarizer.extract_sketch()
    exit()
    summarizer.summarize_review_diversity_sample(False, chunk=None, prompt_type='re')
    exit()
    summarizer.extract_sketch()
    summarizer.parse_rec_result(args.df_path)
    exit()
    summarizer.parse_rec_result('/home/yamanishi/project/airport/src/analysis/route_recommendation/result/gpt4o/v9.csv')
    #summarizer.get_user_profile_short_for_review(test=args.test, prompt_type=args.prompt_type)
    # summarizer.extract_noun_adj_japanese()
    exit()
    summarizer.get_user_profile_short_for_review(test=args.test, prompt_type=args.prompt_type)
    #summarizer.extract_noun_adj_japanese()
    #summarizer.summarize_review_diversity_sample(test=args.test, chunk=args.chunk, prompt_type=args.prompt_type)
    # summarizer.summarize_review_all()
    # exit()
    exit()
    polarity = args.polarity
    summarizer.get_item_summary_all(dict_path=f'./data/{polarity}_dict_spot.pkl',
                                         save_path=f'./data/{polarity}_spot_summary_{args.model}_again',
                                         test=False,
                                         mode=polarity)
    exit()
    summarizer.get_item_summary_all(dict_path='./data/neg_dict_spot.pkl',
                                         save_path=f'./data/neg_spot_summary_{args.model}_again',
                                         test=False,
                                         mode='neg')
    # summarizer.reduce_noise_summary(df_path='./data/pos_spot_summary_calm.csv')
    #exit()
    # summarizer.get_item_summary_all(dict_path='./data/pos_dict_spot.pkl',
    #                                     save_path='./data/pos_spot_summary_calm_again',
    #                                     mode='pos')
    # exit()
    #summarizer.extract_noun_adj_japanese()
    exit()
    ind=2
    summarizer.convert(review_start=0, review_end=250000, prompt_type='search')
    exit()
    summarizer.summarize_for_pvqa2(test=False, ind=ind)
    exit()
    chunk=250000
    ind = 4
    summarizer.extract_qa_multi(review_start=chunk*ind, review_end=chunk*(ind+1))
    # summarizer.extract_fos_japanese()
    #summarizer.summarize_for_pvqa2(test=False)
    #summarizer.extract_qa_multi(review_num=500000)
    exit()
    summarizer = Summarizer("elyzallama3awq", tensor_parallel_size=8)
    summarizer.extract_fos_japanese()
    exit()
    summarizer.summarize_for_pvqa2(test=False)
    exit()
    summarizer.extract_common_keyword_for_pvqa()
    exit()
    fire.Fire(Summarizer)
    exit()
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", type=str)
    parser.add_argument("--dataset", type=str, default="yelp")
    parser.add_argument("--type", type=str, default="user")
    parser.add_argument("--prompt_type", type=str, default="degree")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--model_name", type=str, default="llama")
    args = parser.parse_args()
    summarizer = Summarizer(args)
    if args.f == "summarize_unit_reviews":
        summarizer.summarize_unit_reviews(
            args.dataset, type=args.type, prompt_type=args.prompt_type
        )
    elif args.f == "extract_aspect":
        summarizer.extract_aspect_from_review(args.dataset, args.prompt_type, args.test)

    # summarizer.summarize_1st_step_profile(args.dataset, type=args.type)
    exit()
    summarizer.summarize_unit_reviews(args.dataset, type=args.type)
    # summarizer.summarize_unit_reviews('yelp', type='item')
    exit()
    for dataset in ["yelp"]:
        for data_type in ["user", "item"]:
            summarizer.summarize_unit_reviews("yelp", type="user")
            summarizer.summarize_unit_reviews("yelp", type="item")
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
    # summarizer.get_item_pos_summary_all()
    # summarizer.extract_match_aspect(mode='val', llm='gpt')
    summarizer.extract_match_aspect(mode="test", llm="gpt")
    exit()
    summarizer.get_item_neg_summary_all()
    exit()
    summarizer.summarize_review_user(summary_method="general")
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
    summarizer.get_1st_step_output_all(
        "./preprocess/recommend/filtered_review_df_train.csv", "spot_name"
    )
    exit()
    summarizer.summarize_review_item_all(
        "./preprocess/recommend/filtered_review_df_train.csv", "spot_name"
    )
    # summarizer.summarize_review_user(max_num=20000)
    summarizer.summarize_review_ind(max_num=200000)
    exit()

with open("./playground/data/v8/train2.json") as f:
    train2 = json.load(f)
with open("./playground/data/v8/train3.json") as f:
    train3 = json.load(f)

df = pd.read_csv("./preprocess/recommend/filtered_review_df.csv")
prompt = (
    "次にユーザーの訪問した全ての場所とそこで書いたレビューを渡します。ユーザーについて次の観点からプロファイルしてください。\n\
・文体の特徴\n\
・ユーザの興味、関心があるトピック\n\
・ユーザーがポジティブに感じやすいこと\n\
・ユーザーがネガティブに感じやすいこと\n\
\n\
{}\n\
"
)
prompt_summary = (
    "次にユーザーの訪問した全ての場所とそこで書いたレビューを渡します。それをれのレビューを1件づず最大50文字程度で要約してください。\n\
{}\n"
)

prompt1 = "このレビューを10通りに書き換えて\
若手ホープ、中堅どころからベテランまで、バラエティに富んだアーティストのライブ会場として定番のライブハウス。天井も高く音響的にもグレードが高いと思う。繁華街から近いのも◎"

prompt2 = "このレビューを10通りに書き換えて\
    4月半ば、桜の通り抜けのシーズンに訪れました。期間が1週間しかない＆比較的天気の良い日曜日だったため、観光客の数が物凄かった！外国の方々も数多く訪れていて、海外のガイド等にも紹介されているのかな、と思います。肝心の桜はちょうど満開～徐々に散りだす、というタイミング。非常にバラエティに富んだ色とりどりの桜を楽しむことができました。"

prompt_summary_ind = (
    "次のレビューを最大50文字程度で要約してください.ただし元の文体も少し残して\n\
{}\n"
)

sum1 = "\n".join(train2[0]["conversations"][0]["value"].split("\n")[:-1])
sum2 = "\n".join(train2[100]["conversations"][0]["value"].split("\n")[:-1])
sums = [
    "\n".join(train3[i]["conversations"][0]["value"].split("\n")[:-1])
    for i in range(0, 2000, 100)
]
# sums = [prompt.format(sm) for sm in sums]
sums = [prompt_summary.format(sm) for sm in sums]

sum_inds = [prompt_summary_ind.format(rv) for rv in df["review"].values[:100000]]
print("sums", sum_inds[:5])
# print('pronpt1', prompt.format(sum1))
# print('pronpt2', prompt.format(sum2))
model = AoKarasu()
# model = AoKarasuTransformer()
# model = QarasuTransformer()
# model = GPT3()
# model = ELYZALLlama()
# result_summary = model.generate(sums)
result_summary = model.generate(sum_inds)
# result_summary = model.generate([prompt.format(sum1),prompt.format(sum2)])
# prompt.format(train2[100]['conversations'][0]['value'])])
print("result_summary", result_summary)
result = model.generate([prompt1, prompt2])
print("result", result)
