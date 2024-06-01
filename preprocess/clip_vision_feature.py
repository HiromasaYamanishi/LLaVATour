from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
import json
from easydict import EasyDict
from PIL import Image
import os
import pickle
from collections import defaultdict
from torch_scatter.scatter import scatter
from collections import Counter
import torch
from tqdm import tqdm

def get_spot_name(image_path):
    if image_path is None:
        return None
    else:
        return image_path.split('_')[0]
    
def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
    
def take_mean(emb1, emb2, count1, count2):
    return (emb1*count1+emb2*count2)/(count1+count2)

class POIVisEMbedding:
    def __init__(self, data_path='../playground/data/jalan_tourism_with_review.json'):
        with open(data_path) as f:
            self.data = json.load(f)
            
        self.image_folder = '/home/yamanishi/project/trip_recommend/data/jalan_image_with_caption'
        self.image_paths = list(set([d['image'] for d in self.data if 'image' in d]))
        self.image_paths = sorted(self.image_paths)
        print(len(set(self.image_paths)))
        self.spot_names = [get_spot_name(p) for p in self.image_paths]
        
        self.args = EasyDict()
        self.args.mm_vision_select_layer = -2
        self.vision_tower = CLIPVisionTower("openai/clip-vit-large-patch14-336", self.args).to('cuda')
        self.image_processor = self.vision_tower.image_processor
        print('initialize done')
        
    def get_poi_vis_emb_batch(self, image_paths):
        images = [Image.open(os.path.join(self.image_folder, image_file)).convert('RGB') 
                  for image_file in image_paths]
        images = [expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean)) for image in images]
        images = self.image_processor(images, return_tensors='pt')['pixel_values'].to('cuda')
        image_features = self.vision_tower(images)
        return image_features
    
    def aggregate_and_update(self, image_features, spot_names_batch, poi_vis_embs, poi_counts):
        spot_counts = Counter(spot_names_batch)
        spot_unique_batch = list(spot_counts.keys())
        spot_unique_counts = list(spot_counts.values())
        spot_name_to_ind = {spot:i for i,spot in enumerate(spot_unique_batch)}
        index = torch.tensor([spot_name_to_ind[spot] for spot in spot_names_batch]).cuda()
        image_feature_aggr_mean = scatter(image_features, index, dim=0, reduce='mean')
        for spot, image_feature, unique_count in zip(spot_unique_batch, image_feature_aggr_mean, spot_unique_counts):
            if spot in poi_vis_embs:
                poi_vis_embs[spot] = take_mean(poi_vis_embs[spot].cuda(), image_feature, poi_counts[spot], unique_count)
            else:
                poi_vis_embs[spot] = image_feature
            poi_vis_embs[spot] = poi_vis_embs[spot].cpu()
            poi_counts[spot]+=unique_count
        return poi_vis_embs, poi_counts
        
    def get_poi_vis_emb_all(self):
        batch_size = 250
        print('image to process', len(self.image_paths))
        poi_vis_embs = {}#defaultdict(list)
        poi_counts = defaultdict(int)
        for i in tqdm(range(0, len(self.image_paths), batch_size)):
            image_paths_batch = self.image_paths[i:i+batch_size]
            spot_names_batch = self.spot_names[i:i+batch_size]
            image_features = self.get_poi_vis_emb_batch(image_paths_batch)
            
            poi_vis_embs, poi_counts = self.aggregate_and_update(image_features, spot_names_batch, poi_vis_embs, poi_counts)
            print(len(poi_vis_embs), len(poi_counts)) 
        infos = {'emb': poi_vis_embs, 'count': poi_counts}   
        #print(poi_vis_embs)   
        with open('../data/poi_embs.pkl', 'wb') as f:
            pickle.dump(infos, f)
            
if __name__=='__main__':
    poivisemb = POIVisEMbedding()
    poivisemb.get_poi_vis_emb_all()