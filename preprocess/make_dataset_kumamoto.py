import json
import os
import pickle
import random
import re

import fire
import pandas as pd
import importlib
from importlib import reload
# 必要な変数や関数を再インポート
import prompt
reload(prompt)
from prompt import *
# from prompt import make_training_dict, make_spot_name_prompt,make_tag_prediction_prompt, make_description_prompt
# from prompt import make_context_prompt, make_category_prompt, make_review_context_posneg_prompt, make_review_context_prompt
# from prompt import make_review_prompt, make_caption_prompt, prompt_pvqa
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import torch
from sklearn.model_selection import train_test_split
#import tensorflow as tf

def save_json(obj, json_path):
    with open(json_path, "w") as f:
        json.dump(obj, f)


def load_multiple_dict(dict_names):
    d_all = {}
    for dict_name in dict_names:
        with open(dict_name, "rb") as f:
            d = pickle.load(f)
            d_all.update(d)

    return d_all

def set_seed(seed):
    # Python標準の乱数生成器
    random.seed(seed)
    
    # NumPyの乱数生成器
    np.random.seed(seed)
    
    # PyTorchの乱数生成器
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 全てのGPUに対して設定

    # TensorFlowの乱数生成器
    #tf.random.set_seed(seed)
    
    # 一部のライブラリでは、再現性を完全に保証するための追加の設定が必要です（例：PyTorch）
    # デフォルトでは、PyTorchは非決定的なアルゴリズムを使用することがあります
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DatasetMaker:
    def __init__(self):
        self.text_image_df =  pd.read_csv('/home/yamanishi/project/trip_recommend/data/jalan/spot/kumamoto/text_image_user_pairs.csv',
                names=['image_path', 'caption', 'user', 'spot', 'ind'])
        self.image_save_dir = (
            "/home/yamanishi/project/trip_recommend/data/jalan_image/kumamoto"
        )
        self.graph_dir = "/home/yamanishi/project/trip_recommend/data/jalan/graph/"
        self.experience_df = pd.read_csv(
            "/home/yamanishi/project/airport/src/data/experience_light.csv"
        )
        self.popular_spots = set(
            self.experience_df.sort_values("review_count", ascending=False)[
                "spot_name"
            ].values[3000:10000]
        )
        #self.image_metas = self.text_image_df.set_index("id").T.to_dict()
        self.image_metas = None
        self.review_train = pd.read_csv('../data/dataset_personalize/review_train.csv')
        self.review2url = dict(zip(self.review_train['review'], self.review_train['url']))
        self.user_profile_tag = pd.read_csv('../data/dataset_personalize/user_profile_tag.csv')
        self.user_profile_sent = pd.read_csv('../data/dataset_personalize/user_profile_sent.csv')
        self.url2tag = dict(zip(self.user_profile_tag['url'], self.user_profile_tag['profile']))
        self.url2sent = dict(zip(self.user_profile_sent['url'], self.user_profile_sent['profile']))
        self.url2sent = {k:v[:v.find('旅行者')] for k,v in self.url2sent.items() if not pd.isna(v) and type(v) is str}
        category_all, subcategory_all = set(), set()
        spot_category, spot_subcategory = {}, {}
        for jenre, spot_name in zip(
            self.experience_df["jenre"], self.experience_df["spot_name"]
        ):
            jenre = jenre.split(",")
            category = jenre[0]
            sub_category = jenre[1]
            spot_category[spot_name] = category
            spot_subcategory[spot_name] = sub_category
            category_all.add(category)
            subcategory_all.add(sub_category)
        self.spot_category = spot_category
        self.spot_subcategory = spot_subcategory
        set_seed(42)
        
    @staticmethod
    def shuffle_multiple_lists(*lists):
        if not lists:
            return []
        
        # インデックスを生成してランダムにシャッフル
        indices = list(range(len(lists[0])))
        random.shuffle(indices)
        
        # 全てのリストをシャッフルされたインデックス順に並び替え
        shuffled_lists = []
        for lst in lists:
            shuffled_list = [lst[i] for i in indices]
            shuffled_lists.append(shuffled_list)
        
        return shuffled_lists

    def load_reviews(self, df):
        """レビューデータを読み込み、IDをキーとする辞書でレビューを返す"""
        review_shorts = dict(zip(df["id"], df["review_short"]))
        review_longs = dict(zip(df["id"], df["review_long"]))
        return review_shorts, review_longs

    def make_training_data(self, save_dir, max_train_size=None, shuffle=True, max_size=None, tasks='uqpsrd', review_only=False):
        '''
        tasks:
            u: unused images
            q: question answering
            p: poi visual question answering
            s: sequential recommendation
            r: review generation
            d: desctiption generation
        '''
        training_datas = []
        if 'u' in tasks:
            print('make unused')
            training_datas = self.make_training_data_from_unused(training_datas)
            print('curent train num', len(training_datas))
        if 'q' in tasks:
            print('make qa')
            training_datas = self.make_training_data_from_qa(training_datas)
            print('curent train num', len(training_datas))
        if 'p' in tasks:
            print(len(training_datas))
            print('make pvqa')
            training_datas = self.make_training_data_from_pvqa(training_datas)
            print('curent train num', len(training_datas))
        print(len(training_datas))
        if 's' in tasks:
            print('make sequential')
            training_datas = self.make_training_data_from_sequential_recommendation(training_datas)
            print('curent train num', len(training_datas))

        if 'r' in tasks:
            print('make review')
            training_datas = self.make_training_data_from_retrieved_review(training_datas, review_only=review_only)
            print('curent train num', len(training_datas))

            print('make retrieved image')
            training_datas = self.make_training_data_from_retrieved_images(training_datas, review_only=review_only)
            print('curent train num', len(training_datas))

            print('make posneg')
            training_datas = self.make_training_data_from_posneg(training_datas, review_only=review_only)
            print('curent train num', len(training_datas))

        if 'd' in tasks:
            print('make posneg')
            training_datas = self.make_training_data_from_spot_description(training_datas)

        
        if shuffle:
            random.shuffle(training_datas)
        if max_size is not None:
            random.shuffle(training_datas)
            training_datas = training_datas[:max_size]
        train_data, test_data = self.train_test_split(training_datas)
        random.shuffle(train_data)
        if max_train_size is not None:
            train_data = train_data[:max_train_size]
        print(len(train_data), len(test_data))
        with open(os.path.join(save_dir, 'train.json'), 'w') as f:
            json.dump(train_data, f)
        with open(os.path.join(save_dir, 'test.json'), 'w') as f:
            json.dump(test_data, f)
        # save_json(train_data, os.path.join(save_dir, "train.json"))
        # save_json(test_data, os.path.join(save_dir, "test.json"))
        # training_datas = self.make_training_data_from_context_information(training_datas)

    def train_test_split(self, data):
        def is_test(d, test_images):
            return d.get('image') in test_images

        # Get all unique images in the data
        unique_images = list(set([d.get("image") for d in data if d.get("image") is not None]))

        # Shuffle the images to ensure random distribution
        random.shuffle(unique_images)

        # Calculate the split index for 90% train and 10% test
        split_index = int(len(unique_images) * 0.9)

        # Split the unique images into training and testing sets
        train_images = set(unique_images[:split_index])
        test_images = set(unique_images[split_index:])

        # Split the data based on the train and test images
        train_data = [d for d in data if is_test(d, train_images)]
        test_data = [d for d in data if is_test(d, test_images)]

        print(len(train_data), len(test_data))
        return train_data, test_data

    def make_training_data_from_retrieved_review(
        self,
        training_datas,
        review_only=False
    ):
        '''
        tasks:
            LR: Landmark Recognition
            IPP: Image Popularity Prediction
            CC: Category Classification
            RG: Review Generation
            CRG: Conditional Review Generation
        '''
        print("Making training data from retrieved review")
        retrieved_df = pd.read_csv(
            "../data/kumamoto/retrieved_reviews.csv",
            names=['spot_name', 'url', 'review_short', 'review_long', 'id']
        )
        review_shorts, review_longs = self.load_reviews(
            retrieved_df
        )  # レビューデータを読み込む
        nice_count_less5, nice_count_more5 = 0, 0
        nice_count_thresh = 10
        for i, row in tqdm(retrieved_df.iterrows(), total=retrieved_df.shape[0]):
            id = row["id"]
            spot_name = row['spot_name']
            tasks, instructions, texts, flags = [], [], [], []
            if not os.path.exists(
                f"{self.image_save_dir}/{id}.jpg"
            ):
                continue  # テキストがNAか、画像が存在しない場合はスキップ
            image_path = f"{id}.jpg"
            ### Spot Name Prediction ###
            if not review_only:
                instructions.append(make_spot_name_prompt(row["spot_name"]))
                tasks.append('LR_1_1')
                texts.append(row["spot_name"])
            
            #### Tag Count Prediction ###
            if random.random() < 0.0:
                nice_count = self.image_metas[id]["tag"]
                if (
                    nice_count is not None
                    and nice_count >= nice_count_thresh
                    or (nice_count_less5 < 2 * nice_count_more5 + 10)
                ):
                    caption = self.image_metas[id]["text"]
                    date = self.image_metas[id]["date"]
                    #print(nice_count, caption, date)

                    spot_name = row["spot_name"].replace("/", "")
                    prompt, answer, task_type, data_type = make_tag_prediction_prompt(
                        nice_count, spot_name, date, caption
                    )
                    instructions.extend([prompt])
                    texts.extend([answer])
                    if nice_count < nice_count_thresh:
                        nice_count_less5 += 1
                    else:
                        nice_count_more5 += 1
                    tasks.append(f'IPP_{task_type}_{data_type}')

            ### Category Prediction ###
            if random.random() < 0.0:
                if random.random() < 0.5:
                    if spot_name in self.spot_category:
                        instructions.extend([make_category_prompt(sub=False)])
                        texts.extend([self.spot_category[spot_name]])
                        tasks.append('CC_1_1')
                else:
                    if spot_name in self.spot_subcategory:
                        instructions.extend([make_category_prompt(sub=True)])
                        texts.extend([self.spot_subcategory[spot_name]])
                        tasks.append('CC_2_1')
            # レビューが存在する場合は追加

            ### Review Generation ###
            if id in review_shorts:
                # if random.random()<0.1:
                #     # 画像なし
                #     image_path = None

                
                review = review_longs[id]
                if review in self.review2url and self.review2url[review] in self.url2tag:
                    tag = self.url2tag[self.review2url[review]]
                else:
                    tag = None
                if review in self.review2url and self.review2url[review] in self.url2sent:
                    sent = self.url2sent[self.review2url[review]]
                else:
                    sent = None
                if not pd.isna(review_shorts[id]):
                    review_shorts[id] = review_shorts[id].replace('じゃらんnetで遊び体験済み', '')
                if not pd.isna(review_shorts[id]):
                    review_longs[id] = review_longs[id].replace('じゃらんnetで遊び体験済み', '')
                if type(review_longs[id]) is str:
                    review_length_long = len(review_longs[id])
                else:
                    review_length_long =None
                if type(review_shorts[id]) is str:
                    review_length_short = len(review_shorts[id])
                else:
                    review_length_short =None
                prompt_short, flag_short = make_review_prompt(row["spot_name"], short=True, image_path=image_path, review_length=review_length_short)
                prompt_long, flag_long = make_review_prompt(row["spot_name"], short=False, image_path=image_path, review_length=None
                        )
                flags.extend([flag_short, flag_long])
                #print('prompt long', prompt_long)
                instructions.extend([prompt_short, prompt_long])
                tasks.append(f'RG_2_{flag_short}')
                tasks.append(f'CRG_1_{flag_long}')

                texts.extend([review_shorts[id], review_longs[id]])

            if instructions:
                id = id + "_review"
                
                # instructions, texts, tasks = DatasetMaker.shuffle_multiple_lists(instructions, texts, tasks)
                #print('image_path', image_path)
                datum = make_training_dict(id, image_path, instructions, texts)
                datum['task_ids'] = tasks
                datum['flags'] = flags
                training_datas.append(datum)

        return training_datas
    
    def make_training_data_from_unused(self, training_datas, review_only=False):
        '''
        tasks:
            LR: Landmark Recognition
            IPP: Image Popularity Prediction
            CC: Category Classification
            CRG: Conditional Review Generation
        '''
        print("Making training data from retrieved images")
        retrieved_image_df = pd.read_csv('../data/df_not_used_images.csv',
        names=['spot_name', ])
        nice_count_less5, nice_count_more5 = 0, 0
        nice_count_thresh = 10
        for _, row in tqdm(
            retrieved_image_df.iterrows(), total=retrieved_image_df.shape[0]
        ):
            instructions, tasks, texts = [], [], []
            ind = row["id"]
            spot_name = row["spot_name"]
            if not os.path.exists(os.path.join(self.image_save_dir, f"{ind}.jpg")):
                continue

            ### Spot Name Prediction ###
            image_path = f"{ind}.jpg"
            if not review_only:
                instructions.append(make_spot_name_prompt(row["spot_name"]))
                texts.append(row["spot_name"])
                tasks.append('LR_1_1')
            # テキストがNAまたは画像が存在しない場合はスキップ

            ### Tag Count Prediction ###
            if random.random() < 0.0:
                nice_count = self.image_metas[ind]["tag"]
                if (
                    nice_count is not None
                    and nice_count >= nice_count_thresh
                    or (nice_count_less5 < 2 * nice_count_more5 + 10)
                ):
                    caption = self.image_metas[ind]["text"]
                    date = self.image_metas[ind]["date"]

                    prompt, answer, task_type, data_type = make_tag_prediction_prompt(
                        nice_count, spot_name, date, caption
                    )
                    tasks.append(f'IPP_{task_type}_{data_type}')
                    instructions.extend([prompt])
                    texts.extend([answer])
                    if nice_count < nice_count_thresh:
                        nice_count_less5 += 1
                    else:
                        nice_count_more5 += 1
            id = ind + "_not_used"
            
            # instructions, texts, tasks = DatasetMaker.shuffle_multiple_lists(instructions, texts, tasks)
            datum = make_training_dict(id, image_path, instructions, texts)
            datum['task_ids'] = tasks

            training_datas.append(datum)

        print(f"Current training data num: {len(training_datas)}")
        return training_datas

    def make_training_data_from_retrieved_images(self, training_datas, review_only=False):
        '''
        tasks:
            LR: Landmark Recognition
            IPP: Image Popularity Prediction
            CC: Category Classification
            CRG: Conditional Review Generation
        '''
        print("Making training data from retrieved images")
        retrieved_image_df = pd.read_csv(
            "../data/kumamoto/retrieved_images.csv",
            names=[
                "spot_name",
                "image_path",
                "review",
                "ind",
                "index",
                "title",
                "rating",
                "tag",
                "sex",
                "age",
                "name",
                "url",
                "visit_time",
            ],
        )
        nice_count_less5, nice_count_more5 = 0, 0
        nice_count_thresh = 10
        for _, row in tqdm(
            retrieved_image_df.iterrows(), total=retrieved_image_df.shape[0]
        ):
            instructions, tasks, texts, flags = [], [], [], []
            ind = row["ind"]
            spot_name = row["spot_name"]
            if not os.path.exists(os.path.join(self.image_save_dir, f"{ind}.jpg")):
                continue
            image_path = f"{ind}.jpg"
            ### Spot Name Prediction ###
            if not review_only:
                instructions.append(make_spot_name_prompt(row["spot_name"]))
                texts.append(row["spot_name"])
                tasks.append('LR_1_1')
            # テキストがNAまたは画像が存在しない場合はスキップ

            ### Tag Count Prediction ###
            if random.random() < 0.0:
                nice_count = self.image_metas[ind]["tag"]
                if (
                    nice_count is not None
                    and nice_count >= nice_count_thresh
                    or (nice_count_less5 < 2 * nice_count_more5 + 10)
                ):
                    caption = self.image_metas[ind]["text"]
                    date = self.image_metas[ind]["date"]

                    prompt, answer, task_type, data_type = make_tag_prediction_prompt(
                        nice_count, spot_name, date, caption
                    )
                    tasks.append(f'IPP_{task_type}_{data_type}')
                    instructions.extend([prompt])
                    texts.extend([answer])
                    if nice_count < nice_count_thresh:
                        nice_count_less5 += 1
                    else:
                        nice_count_more5 += 1

            ### Category Prediction ###
            if random.random() < 0.0:
                if random.random() < 0.5:
                    if spot_name in self.spot_category:
                        if spot_name in self.spot_category:
                            instructions.extend([make_category_prompt(sub=False)])
                            texts.extend([self.spot_category[spot_name]])
                            tasks.append('CC_1_1')
                else:
                    if spot_name in self.spot_subcategory:
                        if spot_name in self.spot_subcategory:
                            instructions.extend([make_category_prompt(sub=True)])
                            texts.extend([self.spot_subcategory[spot_name]])
                            tasks.append('CC_2_1')

            # instructions = [make_spot_name_promt(row["spot_name"])]
            # texts = [row["spot_name"]]

            ### Review Prediction
            review = row['review']
            if review in self.review2url and self.review2url[review] in self.url2tag:
                tag = self.url2tag[self.review2url[review]]
            else:
                tag = None
            if review in self.review2url and self.review2url[review] in self.url2sent:
                sent = self.url2sent[self.review2url[review]]
            else:
                sent = None
            # if random.random()<0.1:
            #     # 画像なし
            #     image_path = None

            if not pd.isna(row["review"]):
                row["review"] = row["review"].replace('じゃらんnetで遊び体験済み', '')

            if type(row['review']) is str:
                review_length = len(row['review'])
            prompt, flag = make_review_prompt(row["spot_name"], short=False, image_path=image_path, review_length=None
                    )

            instructions.append(prompt)
            texts.extend([row["review"]])
            tasks.append(f'CRG_1_{flag}')
            flags.append(flag)

            id = ind + "_image"
            
            # instructions, texts, tasks = DatasetMaker.shuffle_multiple_lists(instructions, texts, tasks)
            # print('image_path', image_path)
            datum = make_training_dict(id, image_path, instructions, texts)
            datum['task_ids'] = tasks
            datum['flags'] = flags
            training_datas.append(datum)

        print(f"Current training data num: {len(training_datas)}")
        return training_datas

    def make_training_data_from_posneg(self, training_datas, review_only=False):
        posneg = load_multiple_dict(
            [
                f"/home/yamanishi/project/airport/src/data/review/spot/goodbad_all_period_{i}.pkl"
                for i in range(7)
            ]
        )
        print("Making training data from posneg review")
        retrieved_review_df = pd.read_csv(
            "/home/yamanishi/project/airport/src/analysis/LLaVA/data/kumamoto/retrieved_review_top3.csv",
            names=['spot_name', 'url', 'review', 'id']
        )
        for _, row in tqdm(
            retrieved_review_df.iterrows(), total=retrieved_review_df.shape[0]
        ):
            ind = row["id"]
            if not os.path.exists(os.path.join(self.image_save_dir, f"{ind}.jpg")):
                continue
            # テキストがNAまたは画像が存在しない場合はスキップ
            tasks, instructions, texts, flags = [], [], [], []
            spot_name = row["spot_name"].replace("/", "")
            id = f'{row["id"]}_posneg'
            image_path = f'{row["id"]}.jpg'
            if not review_only:
                instructions.append(make_spot_name_prompt(row["spot_name"]))
                tasks.append('LR_1_1')
                texts.append(row["spot_name"])
            #posneg_tmp = posneg[row["index"]]
            #matches = re.findall(r"「([^」]+)」", posneg_tmp)
            review = row['review']
            
            if review in self.review2url and self.review2url[review] in self.url2tag:
                tag = self.url2tag[self.review2url[review]]
            else:
                tag = None
            if review in self.review2url and self.review2url[review] in self.url2sent:
                sent = self.url2sent[self.review2url[review]]
            else:
                sent = None
            # if random.random()<0.1:
            #     # 画像なし
            #     image_path = None
            if not pd.isna(row['review']):
                row['review'] = row['review'].replace('じゃらんnetで遊び体験済み', '')

            if type(row['review']) is str:
                review_length = len(row['review'])
            prompt, flag= make_review_prompt(row["spot_name"], short=False, image_path=image_path, review_length=None
                    )
            instructions.append(prompt)
            tasks.append(f'CRG_1_{flag}')

            texts.append(row["review"])
            flags.append(flag)
            # instructions, texts, tasks = DatasetMaker.shuffle_multiple_lists(instructions, texts, tasks)
            datum = make_training_dict(id, image_path, instructions, texts)
            datum['task_ids'] = tasks
            datum['flags'] = flags
            training_datas.append(datum)

        print(f"Current training data num: {len(training_datas)}")
        return training_datas
    
    def make_training_data_from_sequential_recommendation(self, training_datas):
        '''
        tasks:
            SR: Sequential Recommendation
        '''
        print('Making training data from sequential recommendation')
        with open('./recommend/sequential_prompts.pkl', 'rb') as f:
            prompts = pickle.load(f)
            
        prompts, answers = prompts['prompt'], prompts['answer']
        ind = np.arange(len(prompts))
        np.random.shuffle(ind)
        prompts, answers = [prompts[i] for i in ind], [answers[i] for i in ind]
        count = 0
        ind = 0
    
        while count < len(prompts):
            seq_count = np.random.choice([1,2,3],size=1,replace=False, p=[0.1, 0.7, 0.2])[0]
            instructions = prompts[count:count+seq_count]
            texts = answers[count:count+seq_count]
            id = f'sequential_{ind}'
            datum = make_training_dict(id, None, instructions, texts)
            datum['task_ids'] = ['SR_1_1' for _ in range(seq_count)]

            training_datas.append(datum)
            count += seq_count
            ind += 1
            
        return training_datas
    
    def make_training_data_from_qa(self, training_datas):
        '''
        tasks:
            QA: question answering
        '''
        # df = pd.read_csv('../data/df_qa_train.csv')
        df = pd.read_csv('../data/df_qa_calm_train.csv')
        df = df.sample(frac=1)
        df_pre, df_post = df[:len(df)//2], df[len(df)//2:]
        spot_qa_pairs = defaultdict(list)
        
        df_post = df[len(df)//2:]
        df_post = df_post.sample(frac=1)
        
        # 初めの前半のdfに関してはshuffleさせない (地名ごとにまとまった)dfを作成する
        spot_qa_pairs = defaultdict(list)
        for spot, prompt, answer in zip(df_pre['spot'], df_pre['prompt'], df_pre['answer']):
            spot_qa_pairs[spot].append((prompt, answer))
            
        pattern = r'次の.+?に関する質問に答えてください。'
        for spot in spot_qa_pairs.keys():
            qa_pairs = spot_qa_pairs[spot]
            current_ind = 0
            index = 0
            
            while True:
                if current_ind>=len(qa_pairs):break
                chunk = random.randint(10, 20)
                qas = qa_pairs[current_ind:current_ind+chunk]
                current_ind+=chunk
                index+=1
                questions = [qa[0] for qa in qas]
                answers = [qa[1] for qa in qas]
                questions = [re.sub(pattern, '', q) for q in questions]
                questions[0] = f'次の{spot}に関する質問に答えてください。' + questions[0]
                id = f'{spot}_qa_{index}'
                d = make_training_dict(id,None, questions, answers)
                training_datas.append(d)
                
        # 後ろの後半のdfに関しては地名の混ざったdfを作成する
        qa_pairs = [(row['prompt'], row['answer']) for i,row in df_post.iterrows()]
        current_ind = 0
        index = 0
        print(qa_pairs[:5])
        while True:
            if current_ind>=len(qa_pairs):break
            chunk = random.randint(10, 20)
            qas = qa_pairs[current_ind:current_ind+chunk]
            current_ind+=chunk
            index+=1
            questions = [qa[0] for qa in qas]
            answers = [qa[1] for qa in qas]
            id = f'shuffle_qa_{index}'
            d = make_training_dict(id,None, questions, answers)
            training_datas.append(d)
            
        return training_datas

    
    def make_training_data_from_pvqa(self, training_datas):
        '''
        tasks:
            LR: Landmark Recognition
            PVQA: Product Visual Question Answering
        '''
        # df = pd.read_csv('../data/pvqa.csv')
        
        df1 = pd.read_csv('../data/pvqa_summary_llama.csv')
        df2 = pd.read_csv('../data/pvqa_summary_calm3_0_200000.csv')
        df = pd.concat([df1, df2])
        keyword_image = pd.read_csv('../data/keyword_image.csv')
        experience_df = pd.read_csv(
            "/home/yamanishi/project/airport/src/data/experience_light.csv"
        )
        top_keyword_df = keyword_image.drop_duplicates(['spot'], keep='first')
        spot_top_keywords = dict(zip(top_keyword_df['spot'], top_keyword_df['keyword']))
        spot_top_keyword_images = defaultdict(list)
        for ind, row in tqdm(keyword_image.iterrows()):
            spot, image_path, keyword = row['spot'], row['image_path'], row['keyword']
            if spot_top_keywords[spot]==keyword:
                spot_top_keyword_images[spot].append(image_path)
        for index,row in tqdm(df.iterrows()):
            # spot, image_path, keyword, sents, summary, keywords = row['spot'], row['image_path'], row['keyword'], row['sents'], row['summary'], row['noun_in_summary']
            spot, image_path, keyword, sents, summary = row['spot'], row['image_path'], row['keyword'], row['sents'], row['summary']
            prompt_type = random.choices([1, 2, 3, 4, 5, 6, 7, 8, 9],k=1, weights=[1/9 for _ in range(9)])[0]
            summary_rand = random.randint(2, 4)
            summary = summary.replace('日本語の自然な形で要約すると、', '')
            summary = "。".join(summary.split('。')[:summary_rand])
            image_path = image_path.split('/')[-1]
            image_paths = [image_path]
            if keyword == spot_top_keywords.get(spot, None):
                image_paths = spot_top_keyword_images[spot][:3]
                
            image_random = random.random()
            if image_random<0.5:
                image_suffix = '画像'
            else:
                image_suffix = '写真'
            exp_random = random.random()
            if exp_random<0.5:
                exp_suffix = '説明'
            else:
                exp_suffix = '解説'
            do_random = random.random()
            if do_random<0.5:
                do_suffix = '生成'
            else:
                do_suffix = '作成'
            keywords = keyword
            for image_path in image_paths:
                image_path = image_path.split('/')[-1]
                image_suffix_ = image_path.split('.')[0]
                id = f'{image_suffix_}_{index}_pvqa'
                sample_num = random.sample([1, 2, 3], k=1)[0]
                sample_sents = ' '.join(random.sample(sents.split(' '), min(sample_num, len(sents.split(' ')))))
                #print(len(sample_sents))
                #print(keywords)
                instructions = [make_spot_name_prompt(image_suffix_.split('_')[0])]
                tasks = ['LR_1_1']
                texts = [image_suffix_.split('_')[0]]
                if pd.isna(keywords):keywords=''
                sample_keywords = ' '.join(random.sample(keywords.split(' '), min(sample_num, len(keywords.split(' ')))))
                
                if prompt_type == 1:
                    prompt = prompt_pvqa.prompt1.format(image_suffix=image_suffix, exp_suffix=exp_suffix)
                elif prompt_type == 2:
                    prompt = prompt_pvqa.prompt2.format(keywords=keywords, image_suffix=image_suffix, exp_suffix=exp_suffix,)
                elif prompt_type == 3:
                    prompt = prompt_pvqa.prompt3.format(reviews=sample_sents, image_suffix=image_suffix, exp_suffix=exp_suffix,)
                elif prompt_type == 4:
                    prompt = prompt_pvqa.prompt4.format(spot=spot, image_suffix=image_suffix, exp_suffix=exp_suffix,)
                elif prompt_type == 5:
                    prompt = prompt_pvqa.prompt5.format(spot=spot, keyword=keyword, image_suffix=image_suffix, exp_suffix=exp_suffix,)
                elif prompt_type == 6:
                    prompt = prompt_pvqa.prompt6.format(spot=spot, keywords=sample_keywords, image_suffix=image_suffix, exp_suffix=exp_suffix, do_suffix=do_suffix)
                elif prompt_type == 7:
                    prompt = prompt_pvqa.prompt7.format(spot=spot, keyword=keyword, keywords=sample_keywords, image_suffix=image_suffix, exp_suffix=exp_suffix, do_suffix=do_suffix)
                elif prompt_type == 8:
                    prompt = prompt_pvqa.prompt8.format(spot=spot, reviews=sample_sents, image_suffix=image_suffix, exp_suffix=exp_suffix, do_suffix=do_suffix)
                elif prompt_type == 9:
                    prompt = prompt_pvqa.prompt9.format(spot=spot, keyword=keyword, reviews=sample_sents, image_suffix=image_suffix, exp_suffix=exp_suffix, do_suffix=do_suffix)
                    
                instructions.extend([prompt])
                texts.extend([summary])
                tasks.extend([f'PVQA_1_{prompt_type}'])
                #print(image_suffix_.split('_')[0])
                # instructions, texts, tasks = DatasetMaker.shuffle_multiple_lists(instructions, texts, tasks)
                # print('image_path', image_path)
                datum = make_training_dict(id, image_path, instructions, texts)
                datum['task_ids'] = tasks
                training_datas.append(datum)
        
        return training_datas
        


    def make_training_data_from_context_information(self, training_datas):
        '''
        その観光地を訪問する観光客の傾向を予測する
        '''
        print("Making training data from spot context information")
        contexts = self.load_context_data()

        for i, spot_name in tqdm(enumerate(self.experiment_df["spot_name"].values)):
            context_data = {
                key: contexts[key][i]
                for _, key in enumerate(["age", "gender", "season", "people", "time"])
            }
            instructions = [make_context_prompt(spot_name)]
            texts = [
                make_context_answer(
                    spot_name,
                    context_data["age"],
                    context_data["gender"],
                    context_data["season"],
                    context_data["people"],
                    context_data["time"],
                )
            ]
            id = f"{spot_name}_context"
            training_datas.append(
                make_training_dict(
                    id, image=None, instructions=instructions, texts=texts
                )
            )

        print(f"Current training data num: {len(training_datas)}")
        return training_datas

    def make_training_data_from_spot_description(self, training_datas):
        '''
        spot descriptionを予測する. 
        '''
        with open("/home/yamanishi/project/airport/src/data/experiment.pkl", "rb") as f:
            experiment_df = pickle.load(f)
        self.experiment_df = experiment_df
        for _, row in tqdm(experiment_df.iterrows(), total=experiment_df.shape[0]):
            spot_name, description = row["spot_name"], row["description"]
            instructions = [make_description_prompt(spot_name)]
            texts = [description]
            id = f"{spot_name}_description"
            training_datas.append(
                make_training_dict(
                    id, image=None, instructions=instructions, texts=texts
                )
            )
        
        print(f"Current training data num: {len(training_datas)}")
        return training_datas


if __name__ == "__main__":
    d = DatasetMaker()
    save_dir = '../playground/data/kumamoto/v1'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    d.make_training_data(save_dir=save_dir, tasks='r', max_train_size=1000000, review_only=True)
    exit()
    fire.Fire(DatasetMaker)
