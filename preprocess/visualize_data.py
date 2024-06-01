import json
import matplotlib.pyplot as plt
from collections import defaultdict
import japanize_matplotlib
import pandas as pd
import numpy as np
import os

class DataVisualizer:
    def __init__(self, json_path):
        self.json_path = json_path
        
    def make_pie(self, data):
        count = defaultdict(int)
        for d in data:
            id = d['id']
            if 'retrieved_from_image' in id:
                count['ユーザー条件付きレビュー生成']+=1
                count['地名予測']+=1
            elif 'description' in id:
                count['説明文生成']+=1
            elif 'posneg' in id:
                count['地名予測']+=1
                count['キーワード条件づきレビュー生成']+=1
            elif 'compare' in id:
                count['関係性比較']+=1
            elif 'conversation' in id:
                count['観光VQA']+=1
            else: #no suffix
                count['地名予測']+=1
                count['レビュー生成']+=1
                count['ユーザー条件付きレビュー生成']+=1
               
        tasks = count.keys()
        counts = count.values()
        plt.rcParams['figure.figsize'] = (6, 6)
        plt.rcParams['font.size'] = 18
        plt.pie(counts,labels=None, colors=plt.cm.Pastel2.colors)
        plt.legend(loc='upper left',labels=tasks, bbox_to_anchor=(1, 1))
        plt.savefig('task_pie.png')
        
    def count_static(self, data):
        print('total data num:', len(data))
        image_set = set([d.get('image') for d in data if d.get])
        image_set = {os.path.join('/home/yamanishi/project/trip_recommend/data/jalan_image_with_caption',i) for i in image_set if i is not None}
        print('unique image num:', len(image_set))
        df_retrieved1 = pd.read_csv('../data/retrieved_reviews.csv', names=['spot_name', 'image_path', 'review', 'ind'])
        df_retrieved2 = pd.read_csv('../data/retrieved_images.csv', names=['spot_name', 'image_path', 'review', 'ind', "index", "title","rating", "tag","sex","age", "name","url","visit_time"])
        df_retrieved3 = pd.read_csv('/home/yamanishi/project/airport/src/analysis/LLaVA/data/retrieved_image_per_3text.csv', names=['spot_name', 'image_path', 'review', 'ind', "index", "title","rating", "tag","sex","age", "name","url","visit_time"])
        all_reviews = np.concatenate([df_retrieved1[df_retrieved1['image_path'].isin(image_set)]['review'].values,
                                      df_retrieved2[df_retrieved2['image_path'].isin(image_set)]['review'].values,
                                        df_retrieved3[df_retrieved3['image_path'].isin(image_set)]['review'].values])
        print('unique review num', len(set(all_reviews)))
        
    def category_bar(self, data):
        df = pd.read_csv('/home/yamanishi/project/trip_recommend/data/jalan/spot/experience_light.csv')
        spot_name_set = set([d.get('id').split('_')[0] for d in data])
        df = df[df['spot_name'].isin(spot_name_set)]
        counter = defaultdict(int)
        for jenre in df['jenre']:
            for j in jenre.split(','):
                if 'その他' in j or '名所巡り' in j:continue
                counter[j]+=1
                
        sorted_data = dict(sorted(counter.items(), key=lambda item: item[1], reverse=True))
        plt.rcParams['font.size'] = 30
        # ソートされたデータを棒グラフとしてプロット
        plt.figure(figsize=(8, 6))
        plt.barh(list(sorted_data.keys())[:10][::-1], list(sorted_data.values())[:10][::-1])

        # グラフのタイトルと軸ラベルを設定
        plt.xlabel('Values')
        plt.tight_layout()

        # グラフを表示
        plt.savefig('category_bar.jpg')
        
        
if __name__=='__main__':
    visualizer = DataVisualizer('../playground/data/v4/train.json')
    with open('../playground/data/v4/train_conv2.json') as f:
        train_data = json.load(f)
    with open('../playground/data/v4/test_conv2.json') as f:
        test_data = json.load(f)
    data = train_data + test_data
    #visualizer.category_bar(data)
    visualizer.make_pie(data)
    exit()
    visualizer.make_pie(data)
    visualizer.count_static(data)
         
        
                
                