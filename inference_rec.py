from llava.model.builder import load_pretrained_model
import torch
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.builder import load_pretrained_model, load_pretrained_model_geo
from PIL import Image
import requests
from io import BytesIO
from transformers import TextStreamer
import pandas as pd
from tqdm import tqdm
from tqdm.contrib import tenumerate
import json
from dataclasses import dataclass, field
import os
import numpy as np
import argparse
import pickle
from utils import *

def load_image(image_file):
    image_dir = '/home/yamanishi/project/trip_recommend/data/jalan_image_with_caption'
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        if os.path.exists(os.path.join(image_dir, image_file)):
            image = Image.open(os.path.join(image_dir, image_file)).convert('RGB')
        else:
            image = Image.open(os.path.join(image_dir, '函館山_0.jpg'))
    return image

def load_json(json_file):
    with open(json_file) as f:
        d = json.load(f)
    return d

def decode_lmm_batch(model, tokenizer, image_processor, prompts, image_file, type='review', temperature=0.2, max_new_tokens=512):
    conv_mode = 'llava_v0'
    conv = conv_templates[conv_mode].copy()
    #image_file = '/home/yamanishi/project/trip_recommend/data/jalan_image_with_caption/銀山温泉_4.jpg'
    if image_file is not None:
        if isinstance(image_file, str):
            image = load_image(image_file)
            image_size = image.size
            image_tensor = process_images([image], image_processor, model.config)
        elif isinstance(image_file, list):
            images = [load_image(f) for f in image_file]
            #image_size = images[0].size
            image_tensor = process_images(images, image_processor, model.config)
        if isinstance(image_tensor, list):
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        image = images
    else:
        image = None
        image_tensor=None
        #image_size = (640, 480)
    prompts_new = []
    for prompt in prompts:
        roles = conv.roles
        
        inp = prompt
        if image is not None:
            # first message
            #if model.config.mm_use_im_start_end:
            #    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            #else:
            #    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompts_new.append(conv.get_prompt())
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in prompts_new]
    input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    ).to(model.device)
    
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    #print('model', model, model.__class__.__name__)
    if type!='direct':
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                #image_sizes=[image_size],
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                use_cache=True,)
        outputs = tokenizer.batch_decode(output_ids[:,:], skip_special_tokens=True)#.strip()
    else:
        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    #image_sizes=[image_size],
                    #do_sample=True if temperature > 0 else False,
                    do_sample=False,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    num_beams=6,
                    no_repeat_ngram_size=2,
                    diversity_penalty=0.99, 
                    num_return_sequences= 6,
                    num_beam_groups=3,
                    early_stopping=True,
                    use_cache=True,)
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True,)
            outputs = ','.join(outputs)
        except torch.cuda.OutOfMemoryError:
            outputs=''
    #print('output all', tokenizer.decode(output_ids[0, :]).strip())
    conv.messages[-1][-1] = outputs
    print('outputs', outputs)
    return outputs

def decode_lmm(model, tokenizer, image_processor, prompt, image_file, type='review', temperature=0.2, max_new_tokens=512):
    conv_mode = 'llava_v0'
    conv = conv_templates[conv_mode].copy()
    #image_file = '/home/yamanishi/project/trip_recommend/data/jalan_image_with_caption/銀山温泉_4.jpg'
    if image_file is not None:
        if isinstance(image_file, str):
            image = load_image(image_file)
            image_size = image.size
            image_tensor = process_images([image], image_processor, model.config)
        elif isinstance(image_file, list):
            images = [load_image(f) for f in image_file]
            #image_size = images[0].size
            image_tensor = process_images(images, image_processor, model.config)
        if isinstance(image_tensor, list):
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        image = images
    else:
        image = None
        image_tensor=None
        #image_size = (640, 480)
    roles = conv.roles
    #print('prompt', prompt)
    inp = prompt
    if image is not None:
        # first message
        #if model.config.mm_use_im_start_end:
        #    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        #else:
        #    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        image = None
    else:
        # later messages
        conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    #print('prompt', prompt)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    #print('input_ids', input_ids, input_ids[input_ids==-200], image_file)
    
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    #print('model', model, model.__class__.__name__)
    if type!='direct':
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                #image_sizes=[image_size],
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                streamer=streamer,
                use_cache=True,)
        #print('output_ids', output_ids)
        outputs = tokenizer.decode(output_ids[0, :]).strip()
    else:
        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    #image_sizes=[image_size],
                    #do_sample=True if temperature > 0 else False,
                    do_sample=False,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    num_beams=6,
                    no_repeat_ngram_size=2,
                    diversity_penalty=0.99, 
                    num_return_sequences= 6,
                    num_beam_groups=3,
                    early_stopping=True,
                    use_cache=True,)
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True,)
            outputs = ','.join(outputs)
        except torch.cuda.OutOfMemoryError:
            outputs=''
    #print('output all', tokenizer.decode(output_ids[0, :]).strip())
    conv.messages[-1][-1] = outputs
    print('outputs', outputs)
    return outputs

@dataclass
class ModelArguments:
    geo_tower='graph'

class RecInferencer:
    def __init__(self):
        super().__init__()
        self.df = pd.read_csv('./preprocess/recommend/filtered_review_df.csv')
        self.users = self.df['name'].unique()
        self.geo_ind = [5, 10]
        self.llm_zs_ind = [18]
        self.user_profile_ind = [12, 13, 16, 17, 20]
        self.item_profile_ind = [13, 14, 15, 17, 20, 21]
        self.review_ind = [3, 14, 21,22]
        self.aspect_match_ind = [20, 21, 22]
        self.user_or_item_profile_ind = list(set(self.user_profile_ind).union(set(self.item_profile_ind)))
        self.profile_only_or_image_ind = [4, 12, 13]
        root_dir, split='./data/p5/data', 'trip_advisor'
        datamaps = load_json(os.path.join(root_dir, split, 'datamaps.json'))
        self.user2id = datamaps['user2id']
        self.item2id = datamaps['item2id']
        self.user_list = list(datamaps['user2id'].keys())
        self.item_list = list(datamaps['item2id'].keys())
        self.id2item = datamaps['id2item']
        self.meta_data = load_pkl(os.path.join(root_dir, split, 'meta_data.pkl'))
        self.user_data = load_pkl(os.path.join(root_dir, split, 'user_data.pkl'))
        self.meta_dict = {}
        self.name_dict = {}
        for i, meta_item in enumerate(self.meta_data):
            self.meta_dict[meta_item['hotel_id']] = i
            
        self.item2name = {}
        for i, meta_item in enumerate(self.meta_data):
            self.item2name[meta_item['hotel_id']] = meta_item['name']
            
        self.id2name = {k:self.item2name[v] for k,v in self.id2item.items()}
        self.user_id2name = load_pkl(os.path.join(root_dir, split, 'user_id2name.pkl'))
        self.result_dir = './result'
        
    def _get_model_version(self, model_path):
        return int(model_path.split('.')[-1])
        
    def _inference_ind(self, model_path, save=True, init_user=0, end_user=1000, inference_index=0, data_type='review', temperature=0.1, batch=False, test_ind=None):
        '''
        inference 
        type: direct, review
        '''
        model_base = 'lmsys/vicuna-13b-v1.5'
        index = inference_index
        if test_ind is None:test_ind = index
        if test_ind in self.llm_zs_ind:test_ind = 14
        test = load_json(f'./playground/data/v8/test{test_ind}.json')
        print(test[0])
        id2ind = {test[i]['id']: i for i,d_ in enumerate(test)}
        model_name = get_model_name_from_path(model_path)
        if index in self.llm_zs_ind:
            model_path = 'liuhaotian/llava-v1.5-13b'
            model_base = None
            tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, False, False, 'cuda')
        elif self._get_model_version(model_path) in self.geo_ind:
            tokenizer, model, image_processor, context_len = load_pretrained_model_geo(model_path, model_base, model_name, False, False, 'cuda')
        else:
            tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, False, False, 'cuda')
        outputs, users, gts = [], [], []
        prompts = []
        batch_size = 10
        for i,user in tenumerate(self.users[init_user:]):
            if data_type=='review':target = f'{user}_{data_type}_{test_ind}_1_False'
            else:target = f'{user}_{data_type}_{test_ind}_1'
            prompt_history = test[id2ind[target]]['conversations'][0]['value']
            #print(prompt_history)
            if len(prompt_history)>4000:continue
            #print(test[id2ind[f'{user}_{data_type}_{index}_1']])
            if 'image' in test[id2ind[target]]:
                image_files = test[id2ind[target]]['image']
                if not len(image_files):image_files=None
            else:image_files=None
            #print('image_files', image_files)
            if not batch:
                output = decode_lmm(model, tokenizer, image_processor, prompt_history, image_file=image_files, type=data_type, temperature=temperature, max_new_tokens=512)
                outputs.append(output)
            else:
                prompts.append(prompt_history)
                if len(prompts)==batch_size:
                    output = decode_lmm_batch(model, tokenizer, image_processor, prompt_history, image_file=image_files, type=data_type, temperature=temperature, max_new_tokens=512)
                    outputs.extend(output)
                    prompts = []
                    
            users.append(user)
            gts.append(test[id2ind[target]]['conversations'][1]['value'])
            if i[0]+init_user==end_user:break
        print('prompts', len(prompts))
        if batch:
            output = decode_lmm_batch(model, tokenizer, image_processor, prompts, image_file=image_files, type=data_type, temperature=temperature, max_new_tokens=512)
            outputs.extend(output)
        print(len(outputs), len(users), len(gts))
        return outputs, users, gts
        
    def _inference_all(self, model_path, save=True, init_user=0, end_user=1000, type='review', inference_inds=[0], temperature=0, batch=False, test_ind=None):
        df = {}
        for inference_ind in inference_inds:
            outputs, users, gts = self._inference_ind(model_path, save, init_user, end_user, inference_ind, type, temperature, batch, test_ind)
            df['users'] = users
            if test_ind is None:test_ind = inference_ind
            df[f'output{test_ind}'] = outputs
            df['gt'] = gts
            
        if save:
            if test_ind==inference_ind:
                model_suffix = model_path.split('/')[-1]
                pd.DataFrame(df).to_csv(f'./result/rec_{type}/{model_suffix}_{init_user}_{end_user}.csv')
            else:
                model_suffix = model_path.split('/')[-1]
                pd.DataFrame(df).to_csv(f'./result/rec_{type}/{model_suffix}_{test_ind}_{init_user}_{end_user}.csv')
            
        return outputs
    
    def inference_review(self, model_path, save=True, init_user=0, end_user=1000, inference_inds=[], test_ind=None):
        if isinstance(model_path, str):
            output_reviews = self._inference_all(model_path, save=True, init_user=init_user, end_user=end_user,type='review', inference_inds=inference_inds, batch=False, test_ind=test_ind)
        elif isinstance(model_path, list):
            for pth in model_path:
                outputre_reviews = self._inference_all(pth, save=True, init_user=init_user, end_user=end_user,type='review',inference_inds=inference_inds, batch=False, test_ind=test_ind)
        
    def inference_direct(self, model_path, save=True, init_user=0, end_user=1000,  inference_inds=[]):
        if isinstance(model_path, str):
            output_recs = self._inference_all(model_path, save=True, init_user=init_user, end_user=end_user, type='direct',temperature=0.99, inference_inds=inference_inds,)
        elif isinstance(model_path, list):
            for pth in model_path:
                output_recs = self._inference_all(pth, save=True, init_user=init_user, end_user=end_user, type='direct', temperature=0.99, inference_inds=inference_inds,)

    def make_history(self, d_train, user_history_type,review2summary={}):
        def get_images(in_images, spot):
            out_images = []
            for spot_name, title, url in in_images:
                url = url.split('/')[-1]
                if spot_name==spot:
                    out_images.append(f'{spot}/{url}')
            return out_images
        
        history = ''
        max_num = 10
        review = d_train[-max_num:]
        already_spots = set({})
        images_all = []
        for spot,pref,title,review,star,period in review:
            #review = review[:500]
            if spot in already_spots:continue
            if user_history_type==0:history+=str(self.spot2ind[spot]) + '\n'
            elif user_history_type in [1, 16, 17]:history+=spot + '\n'
            elif user_history_type==2:
                history += spot
                desc = self.spot2desc.get(spot, '')
                history+=f' 説明:{desc}\n'
            elif user_history_type in self.review_ind:
                history+=spot
                history+=f' レビュー: {review}\n'
            elif user_history_type==4:
                history+=spot
                desc = self.spot2desc.get(spot, '')
                #history+=f' 説明:{desc}\n'
                images = get_images(d_train['image'], spot)
                images_all = images_all + images
                if len(images):
                    history+=f' レビュー: {review}'
                    image_text = '<image>'*len(images)
                    history+=f' 写真:{image_text}\n'
                else:history+=f' レビュー: {review}\n'
            elif user_history_type==5:
                history+=spot
                images_all+=get_images(d_train['image'], spot)
                history+=f' レビュー: {review}'
                geo_id = self.spot2ind[spot]
                history+=f' <geo{geo_id}>\n'
            elif user_history_type==6:
                history+=spot
                history+=f' レビュー: {review}'
                geo_id = self.spot2ind[spot]
                if spot in self.spot_coords:
                    coords = str(self.spot_coords[spot])
                    history+=f' 座標:{coords}\n'
            elif user_history_type==7:
                history+=spot
                geo_id = self.spot2ind[spot]
                history+=f' <geo{geo_id}>\n'
            elif user_history_type==8:
                history+=spot
                desc = self.spot2desc.get(spot, '')
                if spot in self.spot_coords:
                    coords = str(self.spot_coords[spot])
                    history+=f' 座標:{coords}\n'
            elif user_history_type ==15:
                history+=spot
                summary = review2summary.get(review, '')[:500]
                #print('summary', summary)
                history+=f' レビュー: {summary}\n'
                
            already_spots.add(spot)
        if user_history_type!=4:
            return history 
        else:
            return history,images_all
        
    def inference_review_tripadvisor(self, model_path, target_num=20000, type='exp'):

        review_splits = load_pkl('./data/p5/data/trip_advisor/review_splits.pkl')
        template_exp = "Generate an explanation for {} about this hotel : {}"
        template_review = "Generate an review for {} about this hotel : {}"
        exp_prompts, review_prompts = [], []
        exp_gt, review_gt = [], []
        users, items = [], []
        for review in review_splits['test'][:target_num]:
            user, item = review['user'], review['item']
            exp_gt.append(review['explanation']), review_gt.append(review['reviewText'])
            user_name, item_name = self.user_id2name[self.user2id[user]], self.item2name[item]
            users.append(user_name), items.append(item_name)
            print(user_name, item_name)
            exp_prompts.append(template_exp.format(user_name, item_name))
            review_prompts.append(template_review.format(user_name, item_name))
            
        model_base = 'lmsys/vicuna-13b-v1.5'
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, False, False, 'cuda')
        if type == 'exp':
            exp_outputs = []
            for prompt in tqdm(exp_prompts[:target_num]):
                output = decode_lmm(model, tokenizer, image_processor, prompt, image_file=None, type='review', temperature=0.1, max_new_tokens=512)
                exp_outputs.append(output)
        
        elif type == 'review':
            review_outputs = []
            for prompt in tqdm(review_prompts[:target_num]):
                output = decode_lmm(model, tokenizer, image_processor, prompt, image_file=None, type='review', temperature=0.1, max_new_tokens=512)
                review_outputs.append(output)
        
        model_suffix = model_path.split('/')[-1]
        if type == 'exp':
            pd.DataFrame({'user': user, 'item': item, 'gt': exp_gt, 'output': exp_outputs}).to_csv(os.path.join(self.result_dir, 'rec_exp', f'{model_suffix}.csv'))
        elif type == 'review':
            pd.DataFrame({'user': user, 'item': item, 'gt': review_gt, 'output': review_outputs}).to_csv(os.path.join(self.result_dir, 'rec_review', f'{model_suffix}.csv'))
        
    def make_profile(self, profile):
        if pd.isna(profile['profile'][0][2]) or profile['profile'][0][2] is None:
            return profile['profile'][0][1]
        elif pd.isna(profile['profile'][0][1]) or profile['profile'][0][1] is None:
            return profile['profile'][0][2]
        else:
            return profile['profile'][0][2]+'の'+profile['profile'][0][1]
        
    def generate_pseudo_review(self, model_path):
        with open('./preprocess/recommend/train.pkl', 'rb') as f:
            train = pickle.load(f)
        model_base = 'lmsys/vicuna-13b-v1.5'
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, False, False, 'cuda')
        item_pos_summary = pd.read_csv('./preprocess/recommend/pos_summary.csv')
        spot2pos = dict(zip(item_pos_summary['spot'], item_pos_summary['pos_summary']))
        prompts, target_reviews = [], []
        for user_name in train.keys():
            reviews = train[user_name]['review'][-10:]
            profile = self.make_profile(train[user_name])
            prompt=f'{user_name}さんは{profile}です\n'
            for i in range(len(reviews)):
                prompt_tmp = prompt[:]
                predict = reviews[i]
                target_spot, target_star, target_review = reviews[i][0], reviews[i][4], reviews[i][3]
                prompt_reviews = [reviews[j] for j in range(len(reviews)) if j!=i]
                history = self.make_history(prompt_reviews, user_history_type=14)
                prompt_tmp+=f'これは{user_name}さんの訪問記録です'
                prompt_tmp+=history
                prompt_tmp+=f'{user_name}さんは観光地{target_spot}を訪問します'
                prompt_tmp+=f'星{target_star}のレビューを生成してください'
                prompt_tmp+=f'ただし、{target_spot}のプロファイルは次です\n'
                item_profile = spot2pos.get(target_spot, '')
                prompt_tmp+=item_profile+'\n'
                prompt_tmp+='それではレビューを生成してください'
                prompts.append(prompt_tmp)
                target_reviews.append(target_review)
        target_num = 500
        outputs = []
        for prompt in tqdm(prompts[:target_num]):
            output = decode_lmm(model, tokenizer, image_processor, prompt, image_file=None, type='review', temperature=0.1, max_new_tokens=512)
            outputs.append(output)
            #print('output', output)
        pd.DataFrame({'target_review': target_reviews[:target_num],
                      'generated_review': outputs[:target_num],
                      'prompt': prompts[:target_num]}).to_csv('./preprocess/recommend/dpo_pairs.csv')
        #print(target_review[:5])
        #print(prompts[:5])
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", type=str, default="spot_name")
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--init_user", type=int, default=0)
    parser.add_argument("--end_user", type=int, default=100000)
    parser.add_argument("--test_ind", type=int, default=None)
    parser.add_argument("--data_type", type=str, default='exp')
    args = parser.parse_args()

    # inference = LLaVAInference(args)
    # inference.inference_spot_names()
    inferencer = RecInferencer()
    #evaluator.evaluate_spot_names(args)
    if args.f == 'inference_review':
        ind = inferencer._get_model_version(args.model_path)
        print('ind', ind)
        inferencer.inference_review(model_path=[args.model_path], init_user=args.init_user, end_user=args.end_user, inference_inds=[ind], test_ind=args.test_ind)
    elif args.f == 'inference_direct':
        ind = inferencer._get_model_version(args.model_path)
        print('ind', ind)
        inferencer.inference_direct(model_path=[args.model_path], init_user=args.init_user, end_user=args.end_user, inference_inds=[ind])
    elif args.f == 'generate_pseudo':
        inferencer.generate_pseudo_review(model_path=args.model_path)
    elif args.f == 'inference_review_ta':
        inferencer.inference_review_tripadvisor(args.model_path, args.end_user, args.data_type)
        
    #inferencer = RecInferencer()
    #ind =  3
    #inferencer.inference_review(model_path=[f'./checkpoints/llava-v1.5-13b-jalan-review-lora-v7.{i}' for i in [ind]], init_user=0, end_user=20000, inference_inds=[ind])
    #ind =  2
    #inferencer.inference_review(model_path=[f'./checkpoints/llava-v1.5-13b-jalan-review-lora-v7.{i}' for i in [ind]], init_user=0, end_user=20000, inference_inds=[ind])
    exit()
    ind = 5
    inferencer.inference_direct(model_path=[f'./checkpoints/llava-v1.5-13b-jalan-review-lora-v7.{i}' for i in [ind]], init_user=0, end_user=20000, inference_inds=[ind])
    #ind = 3
    #inferencer.inference_direct(model_path=[f'./checkpoints/llava-v1.5-13b-jalan-review-lora-v7.{i}' for i in [ind]], init_user=0, end_user=20000, inference_inds=[ind])
    exit()
    inferencer.inference_direct(model_path=[f'./checkpoints/llava-v1.5-13b-jalan-review-lora-v7.{i}' for i in [5]], init_user=0, end_user=1000, type='direct')
    #inferencer.inference_review(model_path=[f'./checkpoints/llava-v1.5-13b-jalan-review-lora-v7.{i}' for i in range(4, 5)], max_user_num=1000)
    #inferencer.inference_review(model_path=[f'./checkpoints/llava-v1.5-13b-jalan-review-lora-v7.{i}' for i in range(3, 4)])
    inferencer.inference_direct(model_path=[f'./checkpoints/llava-v1.5-13b-jalan-review-lora-v7.{i}' for i in [2]], init_user=0, end_user=1000)
    #output_reviews1 = inferencer.inference_review(model_path='./checkpoints/llava-v1.5-13b-jalan-review-lora-v7.1', save=True, max_user_num=1000)
    #output_reviews2 = inferencer.inference_review(model_path='./checkpoints/llava-v1.5-13b-jalan-review-lora-v7.2', save=True, max_user_num=1000) 
    #output_reviews3 = inferencer.inference_review(model_path='./checkpoints/llava-v1.5-13b-jalan-review-lora-v7.3', save=True, max_user_num=1000)          