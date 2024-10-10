import json
from io import BytesIO

import fire
import numpy as np
import glob

# from heron.models.video_blip import VideoBlipForConditionalGeneration, VideoBlipProcessor
import pandas as pd
import requests

# from sumeval.metrics.bleu import BLEUCalculator
from lmm import BLIPInference, LLaVAInference, StableVLMInference, LLaVANext, QwenVLChat, LLaVANextLarge, GPT3, GPT4, LLaVARetrieveInfernce, InternVLInference
from PIL import Image
from tqdm import tqdm
import os
import argparse
import google.generativeai as genai
from preprocess.prompt_inference import make_review_context_prompt, make_review_context_posneg_prompt

# from sumeval.metrics.rouge import RougeCalculator


def load_image(image_file):
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
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

    def get_average(
        self,
    ):
        return np.mean(self.list)


class Inferencer:
    def __init__(
        self,
    ):
        self.experience_df = pd.read_csv(
            "/home/yamanishi/project/airport/src/data/experience_light.csv"
        )
        self.popular_spots = set(
            self.experience_df.sort_values("review_count", ascending=False)[
                "spot_name"
            ].values[:500]
        )
        self.over_100_spots = set(
            self.experience_df.query("review_count >= 100")["spot_name"].values
        )
        self.test_data = load_json(
            "./playground/data/v4/test_conv2.json"
        )
        print(len(self.test_data))
        self.image_dir = "/home/yamanishi/project/trip_recommend/data/jalan_image_with_caption"

    @staticmethod
    def inference(args, model_name, function_name, image_paths, conversations=None, test_df=None, save=False, output_attention=False):
        print('args',args)
        if model_name == "stablevlm":
            model = StableVLMInference()
        elif model_name == "llava":
            args.model_path = "liuhaotian/llava-v1.5-13b"
            args.model_base = None
            args.load_4bit = True
            model = LLaVAInference(args)
        elif model_name == "blip":
            model = BLIPInference()
        elif model_name == "llavatour":
            model = LLaVAInference(args)
            checkpoint_path = args.model_path.split('/')[-1]
        elif model_name == 'llavatour_retrieve':
            model = LLaVARetrieveInfernce(args)
            checkpoint_path = args.model_path.split('/')[-1]
        elif model_name == 'gpt3':
            model = GPT3()
        elif model_name == 'gpt4v':
            model = GPT4(model_name="gpt-4-vision-preview")
        elif model_name == 'gpt4o':
            model = GPT4(model_name='gpt-4o')
        elif model_name == 'internvl':
            model = InternVLInference(tensor_parallel_size=args.tensor_parallel_size)

        if function_name == "inference_spot_name" or function_name == "inference_spot_name_old":
            directory = '_'.join(function_name.split('_')[1:])
            if 'llavatour' in model_name:
                if not os.path.exists(f"./result/{directory}/{model_name}/"): os.makedirs(f"./result/{directory}/{model_name}/")
                save_path = f"./result/{directory}/{model_name}/{checkpoint_path}.csv"
            else:
                save_path = f"./result/{directory}/{model_name}.csv"

            print('save path', save_path)
            print(function_name)
            print(len(image_paths))
            print('output attention', output_attention)
            # result = model.process_images(
            #     image_paths, ["この日本の観光地の名前を教えてください。ただし地名のみ答えて." for _ in range(len(image_paths))], japanese=True, task='spot_name', 
            #     output_attention=output_attention
            # )
            result = model.inference_spot_names(image_paths)
            #result = model.process_images(image_paths, prompts)
            if save:
                pd.DataFrame({"image_path": image_paths, "predicted": result}).to_csv(
                    save_path
                )
        elif function_name == "inference_spot_name_topk":
            result = model.inference_spot_names_topk(image_paths)
            if 'llavatour' in model_name:
                if not os.path.exists(f"./result/spot_name/{model_name}/"):os.makedirs(f"./result/spot_name/{model_name}/")
                save_path = f"./result/spot_name/{model_name}/{checkpoint_path}_topk.csv"
            else:
                save_path = f"./result/spot_name/{model_name}_topk.csv"
            if save:
                pd.DataFrame({"image_path": image_paths, "predicted": result}).to_csv(
                    save_path
                )
                
        elif function_name == 'qa':
            if not os.path.exists('./result/qa'):
                os.makedirs('./result/qa/', exist_ok=True)
            if 'llavatour' in model_name:
                if not os.path.exists(f"./result/qa/{model_name}/"):os.makedirs(f"./result/qa/{model_name}/")
                save_path = f"./result/qa/{model_name}/{checkpoint_path}.csv"
            else:
                save_path = f"./result/qa/{model_name}.csv"
            print('save path', save_path)
            prompts = list(test_df['prompt'].values)
            # GPT3の場合入力160, 回答20トークンx10000で計180万トークン
            # 1000トークンあたり0.002ドルなので180万トークンで3.6ドル=576円
            if 'gpt' in model_name.lower():
                prompts = [prompt + 'ただし, 10-20文字以内で一文で簡潔に回答してください' for prompt in prompts]
                result = model.process_images(image_paths, prompts, test_df=test_df, save_path=save_path, task=function_name)
            else:
                result = model.process_images(image_paths, prompts, task=function_name)
                test_df['predicted'] = result
                if save:
                    test_df.to_csv(save_path)
                
        elif function_name == 'pvqa':
            if 'llavatour' in model_name:
                if not os.path.exists(f"./result/pvqa/{model_name}/"):os.makedirs(f"./result/pvqa/{model_name}/")
                save_path = f"./result/pvqa/{model_name}/{checkpoint_path}.csv"
            else:
                save_path = f"./result/pvqa/{model_name}.csv"
            print('save path', save_path)
            # GPT3の場合入力128, 回答126トークンx1000で計25万トークン
            # 1000トークンあたり0.03ドルなので25万トークンで7.5ドル=1200円
            if 'gpt' in model_name.lower():
                test_df = test_df[:1000]
            prompts = list(test_df['prompt'].values)
            if 'gpt' in model_name.lower():
                prompts = [prompt + 'ただし, 100文字程度で回答してください。 観光地の画像のコンテンツに関する説明をふまえ、2、3文で説明してください。' for prompt in prompts]
            if 'gpt' in model_name.lower():
                result = model.process_images(image_paths, prompts, test_df=test_df, save_path=save_path, task=function_name)
            else:
                result = model.process_images(image_paths, prompts, task=function_name)
                test_df['predicted'] = result
                if not os.path.exists('./result/pvqa'):
                    os.makedirs('./result/pvqa/', exist_ok=True)
                if save:
                    test_df.to_csv(save_path)
                
        elif function_name == 'ipp':
            if 'llavatour' in model_name:
                if not os.path.exists(f"./result/ipp/{model_name}/"):os.makedirs(f"./result/ipp/{model_name}/")
                save_path = f"./result/ipp/llavatour/{checkpoint_path}.csv"
            else:
                save_path = f"./result/ipp/{model_name}.csv"
            print('save path', save_path)
            prompts = list(test_df['prompt'].values)
            result = model.process_images(image_paths, prompts, debug_prompt=False, task=function_name)
            test_df['predicted'] = result
            if not os.path.exists('./result/pvqa'):
                os.makedirs('./result/pvqa/', exist_ok=True)
            if save:
                test_df.to_csv(save_path)
                
        elif 'sequential' in function_name:
            if 'llavatour' in model_name:
                if not os.path.exists(f"./result/sequential/{model_name}/"):os.makedirs(f"./result/sequential/{model_name}/")
                if 'topk' in function_name:
                    save_path = f"./result/sequential/llavatour/{checkpoint_path}_topk.csv"
                else:
                    save_path = f"./result/sequential/llavatour/{checkpoint_path}.csv"
                
            else:
                if 'topk' in function_name:
                    save_path = f"./result/sequential/{model_name}.csv"
                else:
                    save_path = f".result/sequential/{model_name}_topk.csv"
            print('save path', save_path)
            prompts = list(test_df['prompt'].values)
            if 'topk' in function_name:
                result = model.process_images(image_paths, prompts, task='sequential_topk')
            else:
                result = model.process_images(image_paths, prompts, task=function_name)
            test_df['predicted'] = result
            if not os.path.exists('./result/sequential'):
                os.makedirs('./result/sequential/', exist_ok=True)
            if save:
                test_df.to_csv(save_path)
            
        elif function_name == "inference_tag_count":
            prompts = []
            for spot_name, caption, date in zip(test_df['spot'], test_df['text'], test_df['date']):
                if args.prompt_type == 1:
                    prompt = f"この画像の"
                elif args.prompt_type == 2:
                    prompt = f"この画像は{spot_name}で撮られた画像です。この画像の"
                elif args.prompt_type == 3:
                    prompt = f"この画像は{spot_name}で撮られた画像です。 キャプションは{caption}です。この画像の"
                elif args.prompt_type == 4:
                    prompt = f"この画像は{spot_name}で{date}に撮られた画像です。この画像の"
                elif args.prompt_type == 5:
                    prompt = f"この画像は{spot_name}で{date}に撮られた画像です。キャプションは{caption}です。この画像の"
                    
                if args.task_type == 1:
                    prompt += 'いいね数を予測してください'
                elif args.task_type == 2:
                    prompt += 'いいね数が5以上か予測してください'
                elif args.task_type == 3:
                    prompt += 'いいね数が10以上か予測してください'
                prompts.append(prompt)
            result = model.inference_tag_count(image_paths, prompts)
            
        elif function_name == "generate_reviews":
            if 'llavatour' in model_name:
                if not os.path.exists(f"./result/reviews/{model_name}/"):os.makedirs(f"./result/reviews/{model_name}/")
                if 'retrieve' in model_name:
                    save_path = f"./result/reviews/{model_name}/{checkpoint_path}_{args.num_entity}_{args.num_relation}"
                else:
                    save_path = f"./result/reviews/{model_name}/{checkpoint_path}"
            else:
                save_path = f"./result/reviews/{model_name}"
                if args.retrieval:
                    save_path = f"./result/reviews/{model_name}_retrieval_{args.retrieval_num}"
            print('save_path', save_path)
            if args.use_context:
                prompts = []
                for i,row in test_df.iterrows():
                    context = ''
                    age, tag, sex, spot = row['age'], row['tag'], row['sex'], row['spot']
                    if not pd.isna(age):context += f'{age}の'
                    #if not pd.isna(tag):context += f'{tag}の'
                    if not pd.isna(sex):
                        context += f'{sex}の'
                    prompt = f'あなたは{spot}を観光で訪れた{context}観光客です。この画像について観光客のようにレビューを生成してください'
                    prompts.append(prompt)
            elif args.use_feature:
                prompts = []
                for i, row in test_df.iterrows():
                    prompt = f"あなたは{row['spot']}を観光で訪れた観光客です。「{row['feature']}」というキーワードを含めて、この画像についてレビューを生成してください"
                    prompts.append(prompt)
            else:
                prompts = []
                for i,row in test_df.iterrows():
                    #prompt = f"この観光地は{row['spot']}です。観光客になったつもりで画像にあったレビューを生成してください。"
                    prompt = f"あなたは{row['spot']}を訪れた観光客です。与えられた写真についてレビューを書いてください"
                    if args.aspect_cot:
                        prompt+='ただし、段階的に行ってください。まず、レビューに含まれるアスペクトの列を予測してください。アスペクトは次の中のものを用いてください。[トピック,混雑度,景色,歴史,雰囲気,価格,サービス,アクセス,イベント,食事]。その次に、予測したアスペクト列に基づいてレビューを予測してください。'
                    if 'llavatour' not in model_name:
                        prompt+='ただし100文字程度で生成してください'
                    prompts.append(prompt)
            print(' output attention', output_attention)
            if 'gpt4' in model_name:
                result = model.process_images(image_paths, prompts, test_df=test_df, save_path=save_path,  retrieval=args.retrieval, retrieve_num=args.retrieval_num)
            elif 'retrieve' in model_name:
                result = model.process_images(image_paths, prompts, output_attention=output_attention, task=function_name, num_entity=args.num_entity, num_relation=args.num_relation)
            else:
                result = model.process_images(image_paths, prompts, output_attention=output_attention, task=function_name, debug_prompt=True)
            #result = model.generate_reviews(image_paths, prompts)

            if save:

                if args.use_context:
                    pd.DataFrame({"image_path": image_paths, "predicted": result, 'conversations': conversations}).to_csv(
                        save_path + '_context.csv'
                    )
                elif args.use_feature:
                    pd.DataFrame({"image_path": image_paths, "predicted": result, 'conversations': conversations}).to_csv(
                        save_path + '_feature.csv'
                    )
                else:
                    pd.DataFrame({"image_path": image_paths, "predicted": result, 'conversations': conversations}).to_csv(
                        save_path + '.csv'
                    )
            print("result len:", len(result))

        elif function_name == "generate_short_reviews":
            if 'llavatour' in model_name:
                if not os.path.exists(f"./result/short_reviews/{model_name}/"):os.makedirs(f"./result/reviews/{model_name}/")
                if 'retrieve' in model_name:
                    save_path = f"./result/short_reviews/{model_name}/{checkpoint_path}_{args.num_entity}_{args.num_relation}"
                else:
                    save_path = f"./result/short_reviews/{model_name}/{checkpoint_path}"
            else:
                save_path = f"./result/short_reviews/{model_name}"
                if args.retrieval:
                    save_path = f"./result/short_reviews/{model_name}_retrieval_{args.retrieval_num}"
            print('save_path', save_path)

            prompts = []
            for i,row in test_df.iterrows():
                #prompt = f"この観光地は{row['spot']}です。観光客になったつもりで画像にあったレビューを生成してください。"
                prompt = f"あなたは{row['spot']}を訪れた観光客です。与えられた写真について簡潔にレビューを書いてください"
                if 'llavatour' not in model_name:
                    prompt+='ただし30文字程度で生成してください'
                prompts.append(prompt)
            print(' output attention', output_attention)
            if 'gpt4' in model_name:
                result = model.process_images(image_paths, prompts, test_df=test_df, save_path=save_path,  retrieval=args.retrieval, retrieve_num=args.retrieval_num)
            elif 'retrieve' in model_name:
                result = model.process_images(image_paths, prompts, output_attention=output_attention, task=function_name, num_entity=args.num_entity, num_relation=args.num_relation)
            else:
                result = model.process_images(image_paths, prompts, output_attention=output_attention, task=function_name, debug_prompt=True)
            #result = model.generate_reviews(image_paths, prompts)

            if save:
                pd.DataFrame({"image_path": image_paths, "predicted": result, 'conversations': conversations}).to_csv(
                    save_path + '.csv'
                )
            print("result len:", len(result))

        elif function_name == 'generate_review_attribute':
            if 'llavatour' in model_name:
                if not os.path.exists(f"./result/reviews/{model_name}/"):os.makedirs(f"./result/reviews/{model_name}/")
                if 'retrieve' in model_name:
                    save_dir = f"./result/reviews/{model_name}/{checkpoint_path}_{args.num_entity}_{args.num_relation}"
                else:
                    save_dir = f"./result/reviews/{model_name}/{checkpoint_path}"
            else:
                save_dir = f"./result/reviews/{model_name}"
                if args.retrieval:
                    save_dir= f"./result/reviews/{model_name}_retrieval_{args.retrieval_num}"

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            prompts = []
            fixed_value, attribute = args.fixed_value, args.attribute
            for _, row in test_df.iterrows():
                # 使用する属性の値を決定
                gender = row['sex'] if attribute == 'sex' and fixed_value is None else fixed_value if attribute == 'sex' else None
                age = row['age'] if attribute == 'age' and fixed_value is None else fixed_value if attribute == 'age' else None
                tag = row['tag'] if attribute == 'tag' and fixed_value is None else fixed_value if attribute == 'tag' else None
                season = row['season'] if attribute == 'season' and fixed_value is None else fixed_value if attribute == 'season' else None
                month = row['month'] if attribute == 'month' and fixed_value is None else fixed_value if attribute == 'month' else None
                profile_tag = row['profile_tag'] if attribute == 'profile_tag' and fixed_value is None else fixed_value if attribute == 'profile_tag' else None
                profile_long = row['profile_long'] if attribute == 'profile_long' and fixed_value is None else fixed_value if attribute == 'profile_long' else None
                length = row['length'] if attribute == 'length' and fixed_value is None else fixed_value if attribute == 'length' else None
                rating = row['rating'] if attribute == 'rating' and fixed_value is None else fixed_value if attribute == 'rating' else None
                posneg = row['feature'] if attribute == 'posneg' and fixed_value is None else fixed_value if attribute == 'posneg' else None
                print(gender, age, tag, season, month, profile_tag, profile_long, length, rating, posneg)
                # プロンプトを生成
                if attribute=='posneg':
                    prompt, _ = make_review_context_posneg_prompt(
                        spot=row['spot'],
                        tag=tag,
                        gender=gender,
                        age=age,
                        month=month,
                        season=season,
                        rating=rating,
                        prof_tag=profile_tag,
                        prof_sent=profile_long,
                        image_path=row['image_path'],
                        review_length=length,
                        matches=posneg
                    )
                else:
                    prompt, _ = make_review_context_prompt(
                        spot=row['spot'],
                        tag=tag,
                        gender=gender,
                        age=age,
                        month=month,
                        season=season,
                        rating=rating,
                        prof_tag=profile_tag,
                        prof_sent=profile_long,
                        image_path=row['image_path'],
                        review_length=length,
                    )
                if 'llavatour' not in model_name:
                    prompt += 'ただし, 100文字程度で出力してください'
                prompts.append(prompt)
            print(test_df.head())
            print('prompts', prompts[:10])
            #result = model.process_images(image_paths, prompts, debug_prompt=True)
            if 'gpt4' in model_name:
                result = model.process_images(image_paths, prompts, test_df=test_df, save_path=save_path,  retrieval=args.retrieval, retrieve_num=args.retrieval_num)
            elif 'retrieve' in model_name:
                result = model.process_images(image_paths, prompts, output_attention=output_attention, num_entity=args.num_entity, num_relation=args.num_relation)
            else:
                result = model.process_images(image_paths, prompts, output_attention=output_attention, debug_prompt=True)
            test_df['predicted'] = result
            if save:
                test_df.to_csv(os.path.join(save_dir, f'{attribute}_{fixed_value}_re.csv'))
            
        else:
            if not os.path.exists(f'./result/{function_name}'):
                os.makedirs(f'./result/{function_name}/', exist_ok=True)
            if model_name == 'llavatour':
                if not os.path.exists(f"./result/{function_name}/llavatour/"):os.makedirs(f"./result/{function_name}/llavatour/")
                save_path = f"./result/{function_name}/llavatour/{checkpoint_path}.csv"
            else:
                save_path = f"./result/{function_name}/{model_name}.csv"
            print('save path', save_path)
            prompts = list(test_df['prompt'].values)
            # GPT3の場合入力160, 回答20トークンx10000で計180万トークン
        # 1000トークンあたり0.002ドルなので180万トークンで3.6ドル=576円
            if 'gpt4' in model_name.lower():
                result = model.process_images(image_paths, prompts, test_df=test_df, save_path=save_path)
            else:
                result = model.process_images(image_paths, prompts)
            
            test_df['predicted'] = result
            if save:
                test_df.to_csv(save_path)
        del model
        return result
    
    def generate_review_compare_context(self, args, ):
        model = LLaVAInference(args)
        df = pd.read_csv('./data/attribute_compare.csv')
        attributes = ['20代', '40代', '60代']
        for image_path in df['image_path']:
            image_paths = [image_path for _ in range(len(attributes))]
            spot = image_path.split('_')[0]
            prompts = [f'あなたは{spot}を訪れた{attribute}の観光客です。写真からレビューを生成してください'
                       for attribute in attributes]
            
            result = model.generate_reviews(image_paths, prompts)
            pd.DataFrame({'image_path': image_paths, 'predicted': result, 'attribute': attributes}).to_csv('./result/reviews/llavatour_attribute_compare_age.csv', mode='a', header=False)
            
            
    def spot_name_llavanext(self, args):
        model = LLaVANextLarge(args)
        df = pd.read_csv('./data/df_landmark_recog_eval.csv')
        image_paths = df['image_path'].values
        image_paths = [os.path.join(self.image_dir, image_path) for image_path in image_paths]
        result = model.inference(image_paths, ['この画像の観光地名を日本語で答えてください。地名のみ答えてください。' for _ in range(len(image_paths))])
        pd.DataFrame({'image_path': image_paths, 'predicted': result}).to_csv(f'./result/spot_name/{args.model}.csv')
        
    def spot_name_qwen_vl(self,):
        model = QwenVLChat()
        image_paths = glob.glob('./data/spot_name/*.jpg')
        prompts = ['画像の観光地名を日本語で予測してください。地名のみ答えてください。' for _ in range(len(image_paths))]
        results = model.process_images(image_paths, prompts)
        pd.DataFrame({'image_path': image_paths, 'predicted': results}).to_csv(f'./result/spot_name/qwenvl.csv')
        

    def _prepare_image_paths_for_spot_names(self):
        # image_paths = [d.get('image') for d in self.test_data if d['id'].split('_')[0] in self.popular_spots]
        # image_paths = random.sample(image_paths, 1000)
        return pd.read_csv(
            "/home/yamanishi/project/airport/src/analysis/LLaVA/result/spot_name/llava.csv"
        )["image_path"].values
        # return image_paths

    def inference_spot_names(self, args):
        df = pd.read_csv('./data/df_landmark_recog_eval.csv')
        #df = pd.read_csv('./result/spot_name_old/llava.csv')
        image_paths = df['image_path'].values
        result = Inferencer.inference(
            args, args.model_name, "inference_spot_name", image_paths, save=True, output_attention=args.output_attention
        )
        return result
    
    def inference_func(self, args, df_path, function_name):
        df = pd.read_csv(df_path)
        print(df.head())
        image_paths = [None for _ in range(len(df))]
        result = Inferencer.inference(
            args, args.model_name, test_df=df, function_name=function_name, image_paths=image_paths, save=True, 
        )
        return result
    
    def inference_spot_names_old(self, args):
        #df = pd.read_csv('./data/df_landmark_recog_eval.csv')
        df = pd.read_csv('./result/spot_name_old/llava.csv')
        image_paths = df['image_path'].values
        result = Inferencer.inference(
            args, args.model_name, "inference_spot_name_old", image_paths, save=True
        )
        return result
    
    def inference_spot_names_topk(self, args):
        image_paths = self._prepare_image_paths_for_spot_names()
        result = Inferencer.inference(
            args, args.model_name, "inference_spot_name_topk", image_paths, save=True
        )
        return result
        
    def inference_tag_counts(self, args):
        tag_eval_df = pd.read_csv('./data/df_meta_eval.csv')
        image_paths = tag_eval_df['id'].values
        print('eval_df', len(tag_eval_df))
        result = Inferencer.inference(
            args, args.model_name, "inference_tag_count", image_paths, conversations=None,
            test_df=tag_eval_df, save=True
        )
        return result

    def inference_spot_names_all(self, args):
        for method in ["llavatour", "llava", "blip", "stablevlm"]:
            args.model_name = method
            self.inference(args, save=True)

    def _prepare_image_paths_for_review_generation(self):
        # image_paths = [d.get('image') for d in self.test_data if ('retrieved_from_image' in d['id']) and (d['id'].split('_')[0] in self.over_100_spots)]
        # conversations = [d.get('conversations')[1]['value'] for d in self.test_data if ('retrieved_from_image' in d['id']) and (d['id'].split('_')[0] in self.over_100_spots)]
        inds_target = np.load(
            "/home/yamanishi/project/airport/src/analysis/LLaVA/playground/data/v4/review_test_inds_conv2.npy"
        )[:1000]
        image_paths = [d.get("image") for d in self.test_data]
        conversations = [d.get("conversations")[1]["value"] for d in self.test_data]
        # rand_ind = random.sample(inds_target, 2000)
        image_paths = [image_paths[i] for i in inds_target]
        conversations = [conversations[i] for i in inds_target]
        return image_paths, conversations

    def inference_review_generation_attribute(self, args):
        prompts = []
        eval_df = pd.read_csv('./data/inference_review_attribute_all.csv')
        image_paths = eval_df['image_path'].tolist()
        conversations=eval_df['conversations'].values
        result = Inferencer.inference(
            args,
            args.model_name,
            "generate_review_attribute",
            image_paths,
            conversations=conversations,
            test_df = eval_df,
            save=True,
            output_attention=args.output_attention
        )
        return result
    
    def inference_review_generation(self, args):
        '''
        parameters:
            args:
                use_context: 属性情報(年齢・性別など)を使うかどうか
                use_feature: featureを使うか
        '''
        #image_paths, conversations = self._prepare_image_paths_for_review_generation()
        eval_df = pd.read_csv('./data/review_generation_eval.csv')
        image_paths = eval_df['image_path'].values
        conversations=eval_df['conversations'].values
        result = Inferencer.inference(
            args,
            args.model_name,
            "generate_reviews",
            image_paths,
            conversations=conversations,
            test_df = eval_df,
            save=True,
            output_attention=args.output_attention
        )

        return result
    
    def inference_short_review_generation(self, args):
        '''
        parameters:
            args:
                use_context: 属性情報(年齢・性別など)を使うかどうか
                use_feature: featureを使うか
        '''
        #image_paths, conversations = self._prepare_image_paths_for_review_generation()
        eval_df = pd.read_csv('./data/review_generation_eval.csv')
        image_paths = eval_df['image_path'].values
        conversations=eval_df['conversations'].values
        result = Inferencer.inference(
            args,
            args.model_name,
            "generate_short_reviews",
            image_paths,
            conversations=conversations,
            test_df = eval_df,
            save=True,
            output_attention=args.output_attention
        )

        return result

    def inference_review_kumamoto(self, args):
        '''
        parameters:
            args:
                use_context: 属性情報(年齢・性別など)を使うかどうか
                use_feature: featureを使うか
        '''
        #image_paths, conversations = self._prepare_image_paths_for_review_generation()
        eval_df = pd.read_csv('./data/df_review_kumamoto_eval.csv')
        image_paths = eval_df['image_path'].values
        conversations=eval_df['conversations'].values
        result = Inferencer.inference(
            args,
            args.model_name,
            "generate_reviews",
            image_paths,
            conversations=conversations,
            test_df = eval_df,
            save=True,
            output_attention=args.output_attention
        )

        return result

    def inference_sequential(self, args):
        print('inference sequential')
        eval_df = pd.read_csv('./data/df_sequential_eval.csv')
        image_paths = [None for _ in range(len(eval_df))]
        result = Inferencer.inference(args,
                                       args.model_name,
                                       'sequential',
                                       image_paths,
                                       test_df=eval_df,
                                       save=True)
        return result
    
    def inference_sequential_topk(self, args):
        print('inference sequential')
        eval_df = pd.read_csv('./data/df_sequential_eval.csv')
        image_paths = [None for _ in range(len(eval_df))]
        result = Inferencer.inference(args,
                                       args.model_name,
                                       'sequential_topk',
                                       image_paths,
                                       test_df=eval_df,
                                       save=True)
        return result

    
    def inference_ipp(self, args):
        ''''''
        eval_df = pd.read_csv('./data/df_ipp_eval.csv')
        image_paths = eval_df['image_path'].values
        prompts = eval_df['prompt'].values
        result = Inferencer.inference(args, 
                                      args.model_name,
                                      'ipp',
                                      image_paths,
                                      test_df=eval_df,
                                      save=True)
        return result
    
    def inference_qa(self, args):
        ''''''
        eval_df = pd.read_csv('./data/df_qa_eval.csv')
        image_paths = [None for _ in range(len(eval_df))]
        result = Inferencer.inference(args,
                                      args.model_name,
                                      'qa',
                                      image_paths,
                                      test_df=eval_df,
                                      save=True)
        return result

    def inference_pvqa(self, args):
        ''''''
        eval_df = pd.read_csv('./data/df_pvqa_eval.csv')[:3000]
        image_paths = eval_df['image_path'].values
        prompts = eval_df['prompt'].values
        result = Inferencer.inference(args, 
                                      args.model_name,
                                      'pvqa',
                                      image_paths,
                                      test_df=eval_df,
                                      save=True)
        return result

    def inference_review_generation_all(self, args):
        for method in ["llavatour", "llava", "blip", "stablevlm"]:
            args.model_name = method
            self.inference_review_generation(args)

    def llavatour_inference(self, args):
        """
        inference and evaluation with trained model
        """
        llava_inferencer = LLaVAInference(args)
        spot_image_paths = pd.read_csv(
            "/home/yamanishi/project/airport/src/analysis/LLaVA/result/spot_name/llava.csv"
        )["image_path"].values
        review_df_llava = pd.read_csv(
            "/home/yamanishi/project/airport/src/analysis/LLaVA/result/reviews/llava.csv"
        )
        review_image_paths = review_df_llava["image_path"].values
        result_spot_names = llava_inferencer.inference_spot_names(spot_image_paths)
        result_reviews = llava_inferencer.generate_reviews(review_image_paths)
        result_reviews_male = llava_inferencer.generate_review_context(
            review_image_paths, context="男性"
        )
        result_reviews_female = llava_inferencer.generate_review_context(
            review_image_paths, context="女性"
        )
        pd.DataFrame(
            {"image_path": spot_image_paths, "predicted": result_spot_names}
        ).to_csv("./result/spot_name/llavatour.csv")
        pd.DataFrame(
            {
                "image_path": review_image_paths,
                "predicted": result_reviews,
                "conversations": review_df_llava["conversations"],
            }
        ).to_csv("./result/reviews/llavatour.csv")
        pd.DataFrame(
            {
                "image_path": review_image_paths,
                "predicted": result_reviews_male,
                "conversations": review_df_llava["conversations"],
            }
        ).to_csv("./result/reviews/llavatour_male.csv")
        pd.DataFrame(
            {
                "image_path": review_image_paths,
                "predicted": result_reviews_female,
                "conversations": review_df_llava["conversations"],
            }
        ).to_csv("./result/reviews/llavatour_female.csv")

    def llavatour_inference_tag(self, args):
        """
        inference and evaluation with trained model
        """
        llava_inferencer = LLaVAInference(args)
        review_df_llava = pd.read_csv(
            "/home/yamanishi/project/airport/src/analysis/LLaVA/result/reviews/llava.csv"
        )
        review_image_paths = review_df_llava["image_path"].values
        result_reviews_male = llava_inferencer.generate_review_context(
            review_image_paths, context="男性"
        )
        result_reviews_female = llava_inferencer.generate_review_context(
            review_image_paths, context="女性"
        )
        result_reviews_couple = llava_inferencer.generate_review_context(
            review_image_paths, context="カップル・夫婦"
        )
        result_reviews_family = llava_inferencer.generate_review_context(
            review_image_paths, context="家族"
        )
        result_reviews_friend = llava_inferencer.generate_review_context(
            review_image_paths, context="友達同士"
        )
        result_reviews_solo = llava_inferencer.generate_review_context(
            review_image_paths, context="一人"
        )
        pd.DataFrame(
            {
                "image_path": review_image_paths,
                "predicted": result_reviews_male,
                "conversations": review_df_llava["conversations"],
            }
        ).to_csv("./result/reviews/llava_male.csv")
        pd.DataFrame(
            {
                "image_path": review_image_paths,
                "predicted": result_reviews_female,
                "conversations": review_df_llava["conversations"],
            }
        ).to_csv("./result/reviews/llava_female.csv")
        pd.DataFrame(
            {
                "image_path": review_image_paths,
                "predicted": result_reviews_couple,
                "conversations": review_df_llava["conversations"],
            }
        ).to_csv("./result/reviews/llava_couple.csv")
        pd.DataFrame(
            {
                "image_path": review_image_paths,
                "predicted": result_reviews_family,
                "conversations": review_df_llava["conversations"],
            }
        ).to_csv("./result/reviews/llava_family.csv")
        pd.DataFrame(
            {
                "image_path": review_image_paths,
                "predicted": result_reviews_friend,
                "conversations": review_df_llava["conversations"],
            }
        ).to_csv("./result/reviews/llava_friend.csv")
        pd.DataFrame(
            {
                "image_path": review_image_paths,
                "predicted": result_reviews_solo,
                "conversations": review_df_llava["conversations"],
            }
        ).to_csv("./result/reviews/llava_solo.csv")

    def llavatour_inference_one_tag(self, args):
        """
        inference and evaluation with trained model
        """
        llava_inferencer = LLaVAInference(args)
        review_df_llava = pd.read_csv(
            "/home/yamanishi/project/airport/src/analysis/LLaVA/result/reviews/llavatour_epoch2.csv"
        )
        review_image_paths = review_df_llava["image_path"].values
        result = llava_inferencer.generate_review_context(
            review_image_paths, context=args.tag
        )
        pd.DataFrame(
            {
                "image_path": review_image_paths,
                "predicted": result,
                "conversations": review_df_llava["conversations"],
            }
        ).to_csv(f"./result/reviews/llava_{args.tag}.csv")

    def llavatour_inference_tag(self, args):
        """
        inference and evaluation with trained model
        """
        llava_inferencer = LLaVAInference(args)
        review_df_llava = pd.read_csv(
            "/home/yamanishi/project/airport/src/analysis/LLaVA/result/reviews/llava.csv"
        )
        review_image_paths = review_df_llava["image_path"].values
        result_reviews_male = llava_inferencer.generate_review_context(
            review_image_paths, context="男性"
        )
        result_reviews_female = llava_inferencer.generate_review_context(
            review_image_paths, context="女性"
        )
        result_reviews_couple = llava_inferencer.generate_review_context(
            review_image_paths, context="カップル・夫婦"
        )
        result_reviews_family = llava_inferencer.generate_review_context(
            review_image_paths, context="家族"
        )
        result_reviews_friend = llava_inferencer.generate_review_context(
            review_image_paths, context="友達同士"
        )
        result_reviews_solo = llava_inferencer.generate_review_context(
            review_image_paths, context="一人"
        )
        pd.DataFrame(
            {
                "image_path": review_image_paths,
                "predicted": result_reviews_male,
                "conversations": review_df_llava["conversations"],
            }
        ).to_csv("./result/reviews/llava_male.csv")
        pd.DataFrame(
            {
                "image_path": review_image_paths,
                "predicted": result_reviews_female,
                "conversations": review_df_llava["conversations"],
            }
        ).to_csv("./result/reviews/llava_female.csv")
        pd.DataFrame(
            {
                "image_path": review_image_paths,
                "predicted": result_reviews_couple,
                "conversations": review_df_llava["conversations"],
            }
        ).to_csv("./result/reviews/llava_couple.csv")
        pd.DataFrame(
            {
                "image_path": review_image_paths,
                "predicted": result_reviews_family,
                "conversations": review_df_llava["conversations"],
            }
        ).to_csv("./result/reviews/llava_family.csv")
        pd.DataFrame(
            {
                "image_path": review_image_paths,
                "predicted": result_reviews_friend,
                "conversations": review_df_llava["conversations"],
            }
        ).to_csv("./result/reviews/llava_friend.csv")
        pd.DataFrame(
            {
                "image_path": review_image_paths,
                "predicted": result_reviews_solo,
                "conversations": review_df_llava["conversations"],
            }
        ).to_csv("./result/reviews/llava_solo.csv")

    def llavatour_eval(self, args):
        spot_name_metric = Inferencer._calc_spot_name_metric(
            "./result/spot_name/llavatour.csv"
        )
        review_metric = Inferencer._calc_review_generation_metric(
            "./result/reviews/llavatour.csv"
        )
        print("spot_name_metric", spot_name_metric)
        print("review_generation_metric", review_metric)
        review_male_metric = Inferencer._calc_review_generation_metric(
            "./result/reviews/llavatour_male.csv"
        )
        review_female_metric = Inferencer._calc_review_generation_metric(
            "./result/reviews/llavatour_female.csv"
        )
        print("review_male_metric", review_male_metric)
        print("review_female_metric", review_female_metric)

    def captioning(
        self,
    ):
        lmm = BLIPInference()
        df = pd.read_csv("./data/retrieved_direct_reviews_top5_qa.csv")
        image_paths, outputs = [], []
        for image_path in tqdm(df["image_path"].unique()):
            output = lmm.inference(image_path, "")
            image_paths.append(image_path)
            outputs.append(output)
            if len(image_paths) == 100:
                pd.DataFrame({"image_path": image_paths, "output": outputs}).to_csv(
                    "./data/caption.csv", mode="a"
                )
                image_paths, outputs = [], []
        # print(image_path, output)
        
    def gemini_spot_names(self, model_name=''):
        result = []
        df = pd.read_csv('/home/yamanishi/project/airport/src/analysis/LLaVA/result/spot_name/llavatour.csv')
        #limit=1000
        df_gemini = pd.read_csv(f'./result/spot_name/{model_name}.csv', 
                             names=['image_path', 'predicted'])
        already_image_paths = df_gemini['image_path'].values
        key = ''
        genai.configure(api_key=key)
        
        for i in tqdm(range(len(df))):
            try:
                image_path = f"./data/spot_name/{i}.jpg"
                true_image_path = df.loc[i, 'image_path']
                if true_image_path in already_image_paths:
                    continue
                img = Image.open(image_path).convert('RGB')
                print(img.size)
                prompt = 'この観光地の名前を30文字以内で答えて。地名のみ答えて。'
                model = genai.GenerativeModel(model_name)
                response = model.generate_content([prompt, img])
                print(response)
                print(true_image_path, response.text)
                result.append(response.text)
                pd.DataFrame({'image_path': [true_image_path], 'result': [response.text]}).to_csv(f'./result/spot_name/{model_name}.csv', mode='a', index=False, header=False)
            except ValueError:
                continue

if __name__ == "__main__":
    #fire.Fire(Inferencer)
    #exit()
    parser = argparse.ArgumentParser()
    # 実行関数
    parser.add_argument("-f", type=str, required=True, default="spot_name")
    parser.add_argument("--function_name", type=str)
    parser.add_argument("--df_path", type=str)

    # モデル名
    parser.add_argument("--model_name", type=str, required=True, default='llava')
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    #parser.add_argument("--image-file", type=str, required=True)
    # モデルロードパラメータ
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    # モデル生成パラメータ
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    parser.add_argument("--tag", type=str, default='llava')
    parser.add_argument("--task_type", type=int, default=1)
    parser.add_argument("--prompt_type", type=int, default=1)

    parser.add_argument('--output_attention', action='store_true')
    # 属性条件付きレビュ-生成
    parser.add_argument("--use_context", action='store_true')
    parser.add_argument("--use_feature", action='store_true')
    parser.add_argument("--aspect_cot", action='store_true')
    # GPTなどでのレビュー生成におけるretrieval
    parser.add_argument("--retrieval_num", type=int, default=1)
    parser.add_argument('--retrieval', action='store_true')
    # knowledge graphでの推論
    parser.add_argument('--retrieve_method', type=str, default='triplet')
    parser.add_argument('--num_entity', type=int, default=3)
    parser.add_argument('--num_relation', type=int, default=3)
    # 属性固定レビュー生成
    parser.add_argument('--attribute', type=str, default='age')
    parser.add_argument('--fixed_value', type=str, default=None)
    args = parser.parse_args()
    print()
    # inference = LLaVAInference(args)
    # inference.inference_spot_names()
    inferencer = Inferencer()
    #inferencer.evaluate_spot_names(args)
    if args.f == 'spot_name':
        inferencer.inference_spot_names(args)
    elif args.f == 'inference_func':
        inferencer.inference_func(args, args.df_path, args.function_name)
    elif args.f == 'review_generation':
        inferencer.inference_review_generation(args)
    elif args.f == 'short_review_generation':
        inferencer.inference_short_review_generation(args)
    elif args.f == 'review_generation_attribute':
        inferencer.inference_review_generation_attribute(args)
    elif args.f == 'eval_spot_name':
        inferencer.evaluate_spot_names()
    elif args.f == 'eval_review_generation':
        inferencer.evaluate_review_generation()
    elif args.f == 'llavatour_inference':
        inferencer.llavatour_inference(args)
    elif args.f == 'llavatour_inference_one_tag':
        inferencer.llavatour_inference_one_tag(args)
    elif args.f == 'llavatour_inference_tag':
        inferencer.llavatour_inference_tag(args)
    elif args.f == 'inference_tag_counts':
        inferencer.inference_tag_counts(args)
    elif args.f == 'inference_review_kumamoto':
        inferencer.inference_review_kumamoto(args)
    elif args.f == 'llavatour_eval':
        inferencer.llavatour_eval(args)
    elif args.f == 'inference_pvqa':
        inferencer.inference_pvqa(args)
    elif args.f == 'inference_qa':
        inferencer.inference_qa(args)
    elif args.f == 'inference_ipp':
        inferencer.inference_ipp(args)
    elif args.f == 'inference_sequential':
        inferencer.inference_sequential(args)
    elif args.f == 'inference_sequential_topk':
        inferencer.inference_sequential_topk(args)
    elif args.f == 'inference_spot_names':
        inferencer.inference_spot_names(args)
    elif args.f == 'inference_spot_names_topk':
        inferencer.inference_spot_names_topk(args)
    elif args.f == 'inference_spot_names_all':
        inferencer.inference_spot_names_all(args)
    elif args.f == 'inference_name_one':
        inferencer.inference_review_generation_all(args)
    elif args.f == 'llavatour_inference_and_eval':
        inferencer.llavatour_inference_and_eval()
    elif args.f == 'generate_review_compare_context':
        inferencer.generate_review_compare_context(args)
    elif args.f == 'gemini_spot_names':
        inferencer.gemini_spot_names()
    elif args.f == 'spot_name_llavanext':
        inferencer.spot_name_llavanext(args)
    elif args.f == 'spot_name_qwen_vl':
        inferencer.spot_name_qwen_vl()
    else:
      pass
