import pandas as pd
import pickle
from utils import *
from tqdm import tqdm
import re
import json
import random
from collections import defaultdict
import copy
    
class PrepareData:
    def __init__(self):
        '''
        
        '''
        df_exp = pd.read_pickle('/home/yamanishi/project/airport/src/data/experiment.pkl')
        self.df_exp = df_exp
        self.spot2city = dict(zip(df_exp['spot_name'], df_exp['city']))
        self.spot2pref = dict(zip(df_exp['spot_name'], df_exp['prefecture']))
        self.city2spots = df_exp.groupby('city')['spot_name'].apply(list).to_dict()
        self.pref2spots = df_exp.groupby('prefecture')['spot_name'].apply(list).to_dict()
        self.df_review = pd.read_csv('/home/yamanishi/project/airport/src/analysis/LLaVA/preprocess/recommend/filtered_review_df.csv')
        self.spot2desc = dict(zip(self.df_exp['spot_name'], self.df_exp['description']))
        self.spot2desc = {k:v.replace('※掲載されている情報や写真については最新の情報とは限りません。必ずご自身で事前にご確認の上、ご利用ください。', '') for k,v in self.spot2desc.items()}
        self.spot2ind = {spot:i for i,spot in enumerate(self.df_review['spot'].unique())}
        self.spot_coords = {spot:(round(lat, 5), round(lon, 5)) for spot,lat,lon in zip(df_exp['spot_name'], df_exp['latitude'], df_exp['longitude'])}
        self.user_profile_ind = [12, 13, 16, 17, 20]
        self.item_profile_ind = [13, 14, 15, 17, 20, 21]
        self.review_ind = [3, 14, 21,22]
        self.aspect_match_ind = [20, 21, 22]
        self.user_or_item_profile_ind = list(set(self.user_profile_ind).union(set(self.item_profile_ind)))
        self.profile_only_or_image_ind = [4, 12, 13]
        self.profile_and_history_ind = list(set(self.user_profile_ind).difference(self.profile_only_or_image_ind))
        pass
    
    def prepare_pepler_data(self):
        review_data = pd.read_csv('./preprocess/recommend/filtered_review_df.csv')
        d_train = load_pkl('./preprocess/recommend/train.pkl')
        d_val = load_pkl('./preprocess/recommend/valid.pkl')
        d_test = load_pkl('./preprocess/recommend/test.pkl')
        with open('./preprocess/recommend/train.pkl', 'rb') as f:
            d_train = pickle.load(f)
        with open('./preprocess/recommend/valid.pkl', 'rb') as f:
            d_valid = pickle.load(f)
        with open('./preprocess/recommend/test.pkl', 'rb') as f:
            d_test = pickle.load(f)

        user2ind = {user: i for i, user in enumerate(d_train.keys())}
        spot2ind = {spot: i for i, spot in enumerate(review_data['spot'].unique())}

        # Initialize a local index for data entry
        local_index = 0

        # Process each dataset
        data = []
        data, train_indices, local_index = self._process_pepler_data(d_train, local_index, data)
        data, valid_indices, local_index = self._process_pepler_data(d_valid, local_index, data)
        data, test_indices, local_index = self._process_pepler_data(d_test, local_index, data)
        #print('data', len(data))
        #print(train_indices, valid_indices, test_indices)
        with open('../PEPLER/data/jalan_rec/reviews.pickle', 'wb') as f:
            pickle.dump(data, f)
            
        write_numbers(train_indices, '../PEPLER/data/jalan_rec/train.index')
        write_numbers(valid_indices, '../PEPLER/data/jalan_rec/validation.index')
        write_numbers(test_indices, '../PEPLER/data/jalan_rec/test.index')
        
    def _process_pepler_data(self, dataset, index, data):
        indices = []
        for user in dataset.keys():
            for spot_name, pref, title, review, star, time in dataset[user]['review']:
                data.append({
                    'user': user,
                    'item': spot_name,
                    'template': ('', '', review, 1),
                    'predicted': '',
                    'rating': star
                })
                indices.append(index)
                index += 1
                
        return data, indices, index
        
    def prepare_cf_data(self):
        d_trainval = load_pkl('./preprocess/recommend/train.pkl')
        d_test = load_pkl('./preprocess/recommend/test.pkl')
        ind2user = {i:user for i,user in enumerate(d_trainval.keys())}
        spot2ind = {}
        cur_ind = 0
        f_train = open('../LightGCN-PyTorch/data/jalan_rec/train.txt', 'w')
        f_test = open('../LightGCN-PyTorch/data/jalan_rec/test.txt', 'w')
        print('ind2user', ind2user)
        for i in range(len(ind2user)):
            train_tmp, test_tmp = [i], [i]
            train_reviews, test_reviews = d_trainval[ind2user[i]]['review'], d_test[ind2user[i]]['review']
            for spot_name, _, _, _, _, _ in train_reviews: 
                if spot_name not in spot2ind:
                    spot2ind[spot_name] = cur_ind
                    cur_ind += 1
                train_tmp.append(spot2ind[spot_name])
            for spot_name, _, _, _, _, _ in test_reviews: 
                if spot_name not in spot2ind:
                    spot2ind[spot_name] = cur_ind
                    cur_ind += 1
                test_tmp.append(spot2ind[spot_name])
            f_train.write(' '.join([str(i) for i in train_tmp]))
            f_train.write('\n')
            f_test.write(' '.join([str(i) for i in test_tmp]))
            f_test.write('\n')
        f_train.close()
        f_test.close()
        
    def make_history(self, d_train, user_history_type,review2summary):
        def get_images(in_images, spot):
            out_images = []
            for spot_name, title, url in in_images:
                url = url.split('/')[-1]
                if spot_name==spot:
                    out_images.append(f'{spot}/{url}')
            return out_images
        
        history = ''
        max_num = 10
        review = d_train['review'][-max_num:]
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
        
    def make_data(self, d_tr, d_val, history_type, user2match):
        train_data = []
        if history_type in self.user_or_item_profile_ind:
            user_review_summary = pd.read_csv('./preprocess/recommend/user_review_summary.csv')
            user2summary = dict(zip(user_review_summary['user'], user_review_summary['review_summaries']))
            item_pos_summary = pd.read_csv('./preprocess/recommend/pos_summary.csv')
            spot2pos = dict(zip(item_pos_summary['spot'], item_pos_summary['pos_summary']))
        else:spot2pos=None
        if history_type in [15]:
            df = pd.read_csv('./preprocess/recommend/filtered_review_df_summary.csv')
            review2summary = dict(zip(df['review'], df['summary']))
        else:review2summary=None
        for i in tqdm(range(len(list(d_tr.keys())))):
            for u in [history_type]:
                for p in [1,2,3]:
                    user_name = list(d_tr.keys())[i]
                    if history_type in self.user_profile_ind:user_profile = user2summary.get(user_name, '')
                    else:user_profile = None
                    train_data.append(self.make_review_prompt(d_tr, d_val,list(d_tr.keys())[i], user_history_type=u, predict_type=p, predict_star=False, user_profile=user_profile, spot2pos=spot2pos, review2summary=review2summary, user2match=user2match))
                    train_data.append(self.make_review_prompt(d_tr, d_val,list(d_tr.keys())[i], user_history_type=u, predict_type=p, predict_star=True, user_profile=user_profile, spot2pos=spot2pos, review2summary=review2summary, user2match=user2match))
                    if p==3:repeat_num=5
                    else:repeat_num=1
                    for _ in range(repeat_num):
                        train_data.append(self.make_prompt_direct(d_tr, d_val,list(d_tr.keys())[i], user_history_type=u, predict_type=p, user_profile=user_profile,spot2pos=spot2pos, review2summary=review2summary),)
                        train_data.append(self.make_prompt_town(d_tr, d_val,list(d_tr.keys())[i], user_history_type=u, predict_type=p, user_profile=user_profile, spot2pos=spot2pos, review2summary=review2summary),)
        return train_data
    
    def choose_cand_spots(self, test_spot, unit='pref', n=10, m=10):
        if unit=='pref':
            if test_spot in self.spot2pref:cand_pool = self.pref2spots[self.spot2pref[test_spot]]
            else:cand_pool=None
            if cand_pool is None or len(cand_pool)<m-1:
                cand_pool = list(self.df_exp['spot_name'].values)
            if test_spot in cand_pool:cand_pool.remove(test_spot)
            #print(cand_pool)
            cands = random.sample(cand_pool, m-1)
            cands.append(test_spot)
            random.shuffle(cands)
        elif unit=='city':
            if test_spot in self.spot2city:cand_pool =self.city2spots[self.spot2city[test_spot]]
            else:cand_pool=None
            if cand_pool is None or len(cand_pool)<m-1:
                cand_pool = list(self.df_exp['spot_name'].values)
            if test_spot in cand_pool:cand_pool.remove(test_spot)
            #print(cand_pool)
            cands = random.sample(cand_pool, m-1)
            cands.append(test_spot)
            random.shuffle(cands)
        cands = random.sample(cands, n)
        text=''
        for cand in cands:
            text+=cand
            text+='\n'
            #text+=f' 説明文:{spot2desc[test_spot]}
        if n==1:
            text = text.replace('\n', '')
        return text
    
    def make_profile(self, profile):
        if pd.isna(profile['profile'][0][2]) or profile['profile'][0][2] is None:
            return profile['profile'][0][1]
        elif pd.isna(profile['profile'][0][1]) or profile['profile'][0][1] is None:
            return profile['profile'][0][2]
        else:
            return profile['profile'][0][2]+'の'+profile['profile'][0][1]
    
    def make_prompt_direct(self, d_train, d_valid,user_name, user_history_type, predict_type, user_profile=None, spot2pos=None,
                           review2summary=None):
        '''
        user_history_type: 1~4
        predict_type: 1~3
        d[user_name]['review']: (spot_name, pref, title, review, star, time)
        '''
        d = {}
        d['id'] = f'{user_name}_direct_{user_history_type}_{predict_type}'
        d["conversations"]= []
        profile = self.make_profile(d_train[user_name])
        prompt=f'{user_name}さんは{profile}です\n'
        if user_history_type in self.user_profile_ind:
            prompt+=f'これは{user_name}さんのプロファイルです'
            user_profile = user_profile.replace('\n', ',')
            prompt+=user_profile+'\n'
            if user_history_type in self.profile_and_history_ind:
                prompt+=f'これは{user_name}さんの訪問記録です\n'
        else:
            prompt+=f'これは{user_name}さんの訪問記録です\n'
        if user_history_type not in self.profile_only_or_image_ind:
            prompt+=self.make_history(d_train[user_name], user_history_type, review2summary=review2summary)
        elif user_history_type==4:
            history,images = self.make_history(d_train[user_name], user_history_type, review2summary=review2summary)
            prompt+=history
            if len(images):
                #print(user_name)
                d['image'] = images
            else:
                d['image'] = []
        test_spot = d_valid[user_name]['review'][0][0]
        if predict_type==1:
            prompt+=f'{user_name}さんが次に訪問する場所を予測してください'
            response=test_spot
        elif predict_type==2:
            cand_spots = self.choose_cand_spots(test_spot,unit='pref', n=10, m=10)
            prompt+=f'{user_name}さんが訪問する場所を次から選んでください\n{cand_spots}'
            response=test_spot
        elif predict_type==3:
            cand_spot = self.choose_cand_spots(test_spot,unit='pref', n=1, m=5)
            prompt+=f'{user_name}さんが{cand_spot}を訪問するかyesかnoで回答してください'
            # if spot2pos is not None and user_history_type in [13, 14]:
            #     prompt+=f'ただし、{cand_spot}のプロファイルは次です\n'
            #     item_profile = spot2pos.get(cand_spot, '')
            #     item_profile = item_profile.replace('\n', '、')
            #     prompt+=item_profile+'\n'
            #     prompt+='それでは予測してください'
            if test_spot in cand_spot:response='yes'
            else:response='no'
        d['conversations'].append({"from": "human","value": prompt},)
        d['conversations'].append({"from": 'gpt', 'value': response})
        return d
        
    def make_prompt_town(self, d_train, d_valid, user_name, user_history_type,predict_type, user_profile=None, spot2pos=None,
                         review2summary=None):
        '''
        user_history_type: 1~4
        predict_type: 1~3
        d[user_name]['review']: (spot_name, pref, title, review, star, time)
        '''
        d = {}
        d['id'] = f'{user_name}_town_{user_history_type}_{predict_type}'
        d["conversations"]= []
        profile = self.make_profile(d_train[user_name])
        prompt=f'{user_name}さんは{profile}です\n'
        test_spot = d_valid[user_name]['review'][0][0]
        if test_spot in self.spot2pref:pref = self.spot2pref[test_spot]
        else:pref=None
        if user_history_type in self.user_profile_ind:
            prompt+=f'これは{user_name}さんのプロファイルです'
            user_profile = user_profile.replace('\n', '、')
            prompt+=user_profile+'\n'
            if user_history_type in [16, 17]:
                prompt+=f'これは{user_name}さんの訪問記録です\n'
        else:
            prompt+=f'これは{user_name}さんの訪問記録です\n'
        if user_history_type not in self.profile_only_or_image_ind:
            prompt+=self.make_history(d_train[user_name], user_history_type, review2summary=review2summary)
        elif user_history_type==4:
            history,images = self.make_history(d_train[user_name], user_history_type, review2summary=review2summary)
            prompt+=history
            if len(images):
                #print(user_name)
                d['image'] = images
            else:
                d['image'] = []
        if predict_type==1:
            prompt+=f'{user_name}さんが{pref}で次に訪問する場所を予測してください'
            response=test_spot
        elif predict_type==2:
            cand_spots = self.choose_cand_spots(test_spot, unit='pref',n=10, m=10)
            #print(cand_spots)
            prompt+=f'{user_name}さんが{pref}で訪問する場所を次から選んでください\n'
            prompt+=cand_spots
            response=test_spot
        elif predict_type==3:
            cand_spot = self.choose_cand_spots(test_spot, unit='pref',n=1, m=5)
            #print(cand_spots)
            prompt+=f'{user_name}さんが{pref}で{cand_spot}を訪問するかyesかnoで回答してください'
            # if spot2pos is not None and user_history_type in [13, 14]:
            #     prompt+=f'ただし、{cand_spot}のプロファイルは次です\n'
            #     item_profile = spot2pos.get(cand_spot, '')
            #     item_profile = item_profile.replace('\n', '、')
            #     prompt+=item_profile+'\n'
            #     prompt+='それでは予測してください'
            if test_spot in cand_spot:response='yes'
            else:response='no'
        d['conversations'].append({"from": "human","value": prompt},)
        d['conversations'].append({"from": 'gpt', 'value': response})
        return d

    def make_review_prompt(self, d_train, d_valid, user_name, user_history_type, predict_type, predict_star=False, user_profile=None, spot2pos=None,
                           review2summary=None, user2match=None):
        '''
        user_history_type: 1~4
        predict_type: 1~4
        d[user_name]['review']: (spot_name, pref, title, review, star, time)
        '''
        d = {}
        d['id'] = f'{user_name}_review_{user_history_type}_{predict_type}_{predict_star}'
        d['conversations'] = []
        profile = self.make_profile(d_train[user_name])
        prompt=f'{user_name}さんは{profile}です\n'

        if user_history_type in self.user_profile_ind:
            prompt+=f'これは{user_name}さんのプロファイルです'
            user_profile = user_profile.replace('\n', '、')
            prompt+=user_profile+'\n'
            if user_history_type in [16, 17]:
                prompt+=f'これは{user_name}さんの訪問記録です\n'
        else:
            prompt+=f'これは{user_name}さんの訪問記録です\n'
            
        if user_history_type not in self.profile_only_or_image_ind:
            prompt+=self.make_history(d_train[user_name], user_history_type, review2summary=review2summary)
        elif user_history_type==4:
            history,images = self.make_history(d_train[user_name], user_history_type, review2summary=review2summary)
            prompt+=history
            if len(images):
                #print(user_name)
                d['image'] = images
            else:
                d['image'] = []
        test_spot = d_valid[user_name]['review'][0][0]
        if user_history_type==0:
            test_spot = self.spot2ind[test_spot]
        prompt+=f'{user_name}さんは観光地{test_spot}を訪問します' #説明:{spot2desc[test_spot]}を訪問します。'
        if predict_star==True:
            prompt+=f'場所'
        if predict_type==1:
            review_star = d_valid[user_name]['review'][0][4]
            prompt+=f'星{review_star}のレビューを生成してください'
        elif predict_type==2:
            #cand_spots = choose_cand_spots(test_spot, unit='pref')
            title = d_valid[user_name]['review'][0][2]
            prompt+=f'「{title}」のタイトルのレビューを生成してください'
        # elif predict_type==3:
        #     feature_word = get_feature_word(test_review)
        #     prompt+=f'{feature_word}を使ってレビューを生成してください'
        elif predict_type==3:
            prompt+='レビューを生成してください'
        if predict_star:
            prompt+='また、レビューの星の数を予測してください'
        if spot2pos is not None and user_history_type in self.item_profile_ind:
            prompt+=f'ただし、{test_spot}のプロファイルは次です\n'
            item_profile = spot2pos.get(test_spot, '')
            item_profile = item_profile.replace('\n', '、')
            prompt+=item_profile+'\n'
            
        if user2match is not None and user_history_type in self.aspect_match_ind:
            prompt += 'この時、予測されるユーザーとアイテムのマッチするアスペクトは次です.\n'
            prompt += user2match.get(user_name, '')
        if (user2match is not None and user_history_type in self.aspect_match_ind) or (spot2pos is not None and user_history_type in self.item_profile_ind):
            prompt+='それではレビューを生成してください'
            
        d['conversations'].append({"from": "human","value": prompt},)
        review = d_valid[user_name]['review'][0][3]
        star = d_valid[user_name]['review'][0][4]
        response=review
        if predict_star:
            response+=f'\n星{star}'
        d['conversations'].append({"from": 'gpt', 'value': response})
        return d
    
    def make_train_test(self, history_types=[]):
        d_train = load_pkl('./preprocess/recommend/train.pkl')
        d_trainval = load_pkl('./preprocess/recommend/train_val.pkl')
        d_valid = load_pkl('./preprocess/recommend/valid.pkl')
        d_test = load_pkl('./preprocess/recommend/test.pkl')

        for history_type in history_types:
            if history_type in self.aspect_match_ind:
                aspect_match = pd.read_csv('./preprocess/recommend/match_aspects_gpt.csv')
                user2match_val = dict(zip(aspect_match['user'], aspect_match['match_aspect_pos_val']))
                user2match_test = dict(zip(aspect_match['user'], aspect_match['match_aspect_pos_test']))
                train_data = self.make_data(d_train, d_valid, history_type=history_type, user2match=user2match_val)
                test_data = self.make_data(d_trainval, d_test, history_type=history_type, user2match=user2match_test)
                
            else:
                train_data = self.make_data(d_train, d_valid, history_type=history_type, user2match=None)
                test_data = self.make_data(d_trainval, d_test, history_type=history_type, user2match=None)
            print('train data', len(train_data))
            print('test data', len(test_data))
            save_json(train_data, f'./playground/data/v8/train{history_type}.json')
            save_json(test_data, f'./playground/data/v8/test{history_type}.json')
    
    def add_match(self, history_types=[]):
        with open('./preprocess/recommend/poeneg.pkl', 'rb') as f:
            posneg = pickle.load(f)
        v = posneg[0]
        matches = re.findall(r'「([^」]+)」', posneg[0])
        negative_ind = v.find('ネガティブな')
        pn = [(v.find(m)<negative_ind) for m in matches]
        match_dict = {k:re.findall(r'「([^」]+)」', v) for k,v in posneg.items()}
        for history_type in history_types:
            data_train = load_json(f'./playground/data/v8/train{history_type}.json')
            data_test = load_json(f'./playground/data/v8/test{history_type}.json')
            d_train_match, d_test_match = copy.deepcopy(data_train), copy.deepcopy(data_test)
            review2ind = {review:i for i,review in enumerate(self.df_review['review'])}
            pattern = re.compile(r'レビュー: (.*?)\n')
            all_inds = []
            for i in tqdm(range(len(d_train_match))):
                reviews = pattern.findall(d_train_match[i]['conversations'][0]['value'])
                inds = [review2ind.get(review, -1) for review in reviews]
                matched = [match_dict[ind] for ind in inds if ind!=-1]
                d_train_match[i]['match'] = matched
                all_inds.append(inds)
            for i in tqdm(range(len(d_test_match))):
                reviews = pattern.findall(d_test_match[i]['conversations'][0]['value'])
                inds = [review2ind.get(review, -1) for review in reviews]
                matched = [match_dict[ind] for ind in inds if ind!=-1]
                d_test_match[i]['match'] = matched
                all_inds.append(inds)
            save_json(d_train_match, f'./playground/data/v8/train_match{history_type}.json')
            save_json(d_train_match, f'./playground/data/v8/test_match{history_type}.json')
            
    def make_dataset_tripadvisor(self, ):
        tripadvisor_all_texts = load_pkl('./data/p5/data/trip_advisor/all_texts.pkl')
        training_data = []
        counter = defaultdict(int)
        for source_text, target_text, user_name, id in tripadvisor_all_texts:
            d = {}
            count = counter[f'{user_name}_{id}']
            d['id'] = f'{user_name}_{id}_{count}'
            counter[f'{user_name}_{id}'] += 1
            d['conversations'] = []
            d['conversations'].append({"from": "human","value": source_text})
            d['conversations'].append({"from": "gpt","value": target_text})
            training_data.append(d)
        save_json(training_data, './playground/data/trip_advisor/train1.json')
        
            
            
if __name__=='__main__':
    data_prepare = PrepareData()
    data_prepare.make_dataset_tripadvisor()
    exit()
    data_prepare.make_train_test(history_types=[20, 21, 22])
    exit()
    data_prepare.add_match(history_types=[3, 4, 5, 6])
    exit()
    data_prepare.make_train_test(history_types=[0, 6,7,8])
    exit()
    data_prepare.prepare_cf_data()
    exit()
    data_prepare.prepare_pepler_data()
                