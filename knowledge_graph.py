import networkx as nx
from collections import defaultdict
import openai
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import fire
from tqdm import tqdm
import spacy
from pyvis.network import Network
from collections import deque
import japanize_matplotlib
from lmm import SentenceBertJapanese
import os
import torch
import torch.distributed as dist
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

class SAKG:
    def __init__(self, load=False):
        
        self.graph_path = './data/kg/sakg.pkl'
        if load:
            self.load_graph()
        else:
            self.df_review = pd.read_pickle(
                "/home/yamanishi/project/trip_recommend/data/jalan/review/review_all_period_.pkl"
            )
            with open('./data/pairs.pkl', 'rb') as f:
                pairs = pickle.load(f)
                
            le_user = LabelEncoder()
            le_spot = LabelEncoder()
            self.df_review['user_id'] = le_user.fit_transform(self.df_review['url'])
            self.df_review['spot_id'] = le_spot.fit_transform(self.df_review['spot'])
            self.df_review['pairs'] = pairs
            self.nlp = spacy.load('ja_ginza')
                
            self.graph = nx.Graph()
            self.build_graph()
            self.clean_up_edges()
            self.visualize_graph(num_nodes=20)
            self.print_graph_statistics()
            self.save_graph()

    def build_graph(self):
        for i, row in tqdm(self.df_review.iterrows()):
            user_id, spot, pairs, review = row['user_id'], row['spot'], row['pairs'], row['review']
            user_node = 'user_' + str(user_id)
            # ノードカテゴリを追加
            self.graph.add_node(user_node, category='user')
            self.graph.add_node(spot, category='spot')
            for adj, noun in pairs:
                self.graph.add_node(noun, category='word')
                self._add_edge_with_count(user_node, noun, adj)
                self._add_edge_with_count(spot, noun, adj)
            #if i==100000:break
            
    def clean_up_edges(self):
        for node in self.graph.nodes:
            if self.graph.nodes[node].get('category') == 'spot':
                edges_to_remove = []
                for neighbor in self.graph.neighbors(node):
                    relations = self.graph[node][neighbor]['relations']
                    relations_to_remove = [rel for rel, count in relations.items() if count < 5]
                    for rel in relations_to_remove:
                        del relations[rel]
                    if not relations:
                        edges_to_remove.append((node, neighbor))
                self.graph.remove_edges_from(edges_to_remove)
    
    def _add_edge_with_count(self, node1, node2, relation):
        if not self.graph.has_edge(node1, node2):
            self.graph.add_edge(node1, node2, relations={})
        # エッジが既に存在する場合、relationごとのカウントを更新
        if relation in self.graph[node1][node2]['relations']:
            self.graph[node1][node2]['relations'][relation] += 1
        else:
            self.graph[node1][node2]['relations'][relation] = 1

    def save_graph(self,):
        with open(self.graph_path, 'wb') as f:
            pickle.dump(self.graph, f)
        # nx.write_gml(self.graph, self.graph_path)
        print(f"Graph saved to {self.graph_path}")
        
    def load_graph(self):
        with open(self.graph_path, 'rb') as f:
            self.graph = pickle.load(f)
        #graph = nx.read_gml(self.graph_path)
        print(f"Graph loaded from {self.graph_path}")

    def visualize_graph(self, num_nodes=20):
        # グラフの一部を抽出するために、ノード数を制限
        subgraph = self.graph.subgraph(list(self.graph.nodes)[:num_nodes])
        
        # ポジションを設定（レイアウトを指定）
        pos = nx.spring_layout(subgraph)
        
        # エッジのラベルを取得
        edge_labels = { (u, v): str(data['relations']) for u, v, data in subgraph.edges(data=True) }

        net = Network(notebook=True, width="100%", height="800px", font_color="black")
    
        net.from_nx(subgraph)
        
        # ノードのラベルを設定
        for node in net.nodes:
            node['label'] = node['id']
            node['title'] = node['id']
            node['font'] = {'size': 10, 'face': 'Hiragino Sans'}
            node['color'] = 'skyblue'
        
        # エッジのラベルを設定
        for edge in net.edges:
            edge['title'] = edge_labels.get((edge['from'], edge['to']), '')
            edge['font'] = {'size': 8, 'face': 'Hiragino Sans'}
            edge['color'] = 'red'
        
        # グラフを HTML ファイルとして保存
        net.show("./data/kg/sakg.html")
        # pyvis_G = Network()
        # pyvis_G.from_nx(self.graph)
        # pyvis_G.show("./data/kg/sakg.html")
        # ノードの描画
        #nx.draw(subgraph, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10,font_family='Hiragino Sans',  font_weight='bold')
        
        # エッジのラベルの描画
        #nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_color='red',font_family='Hiragino Sans', font_size=8)
        
        #plt.savefig('./data/kg/sakg.jpg')

    def get_graph_statistics(self):
        stats = {
            "number_of_nodes": self.graph.number_of_nodes(),
            "number_of_edges": self.graph.number_of_edges(),
            "average_degree": sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes(),
            "graph_density": nx.density(self.graph),
            #"connected_components": nx.number_connected_components(self.graph),
        }
        return stats

    def print_graph_statistics(self):
        stats = self.get_graph_statistics()
        print("Graph Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value}")
    

    def get_n_hop_entities_and_relations(self, start_node, n):
        """
        指定したノードからnホップ離れた全てのエンティティとその関係を取得する

        :param start_node: 開始ノード
        :param n: ホップ数
        :return: nホップ以内のエンティティと関係のリスト
        """
        visited = set()
        queue = deque([(start_node, 0, [])])
        results = []

        while queue:
            node, depth, path = queue.popleft()

            if depth > n:
                continue

            if node not in visited:
                visited.add(node)

                if depth > 0:
                    results.append((node, path))

                for neighbor in self.graph.neighbors(node):
                    if neighbor not in visited:
                        edge_data = self.graph.get_edge_data(node, neighbor)
                        new_path = path + [(node, edge_data['relations'], neighbor)]
                        queue.append((neighbor, depth + 1, new_path))

        return results

    def print_n_hop_results(self, start_node, n):
        """
        結果を見やすく表示するためのヘルパーメソッド
        """
        results = self.get_n_hop_entities_and_relations(start_node, n)
        for entity, path in results:
            print(f"Entity: {entity}")
            print("Path:")
            for source, relation, target in path:
                print(f"  {source} --[{relation}]--> {target}")
            print()
            
    def get_embedding_prompt(self, split=1):
        all_prompt_df = pd.read_csv('./data/all_prompts.csv')
        sbert = SentenceBertJapanese()
        chunk_size = len(all_prompt_df)//3 + 1
        prompt_emb = sbert.encode(all_prompt_df['prompt'][chunk_size*(split-1):chunk_size*(split)], batch_size=100)
        with open(f'./data/all_prompt_emb_{split}.pkl', 'wb') as f:
            pickle.dump(prompt_emb, f)

    def setup(self, rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    def cleanup(self, ):
        dist.destroy_process_group()

    def encode_chunk(self, rank, world_size, all_prompts, chunk_size, model):
        self.setup(rank, world_size)

        # Determine the chunk of data this process will handle
        start_idx = rank * chunk_size
        end_idx = min((rank + 1) * chunk_size, len(all_prompts))
        chunk_prompts = all_prompts[start_idx:end_idx]
        model = SentenceBertJapanese()
        # Encode the prompts
        chunk_embeddings = model.encode(chunk_prompts, show_progress_bar=True, convert_to_tensor=True, device=f'cuda:{rank}')

        # Gather results from all processes
        gathered_embeddings = [torch.zeros_like(chunk_embeddings) for _ in range(world_size)]
        dist.all_gather(gathered_embeddings, chunk_embeddings)

        if rank == 0:
            # Combine gathered embeddings
            embeddings = torch.cat(gathered_embeddings, dim=0).cpu().numpy()
            with open('./data/all_prompt_emb.pkl', 'wb') as f:
                pickle.dump(embeddings, f)
            
    def get_embedding_parallel(self, world_size):    
        all_prompt_df = pd.read_csv('./data/all_prompts.csv')
        all_prompts = all_prompt_df['prompt'].tolist()

        # Initialize SentenceBertJapanese model
        model = SentenceTransformer()

        # world_size = torch.cuda.device_count()
        chunk_size = int(np.ceil(len(all_prompts) / world_size))

        torch.multiprocessing.spawn(self.encode_chunk, args=(world_size, all_prompts, chunk_size, model), nprocs=world_size, join=True)


    def get_embedding_entity(self):
        entity_df = pd.read_csv('./data/sakg_entity.csv')
        sbert = SentenceBertJapanese()
        entity_emb = sbert.encode(entity_df['entity'], batch_size=100)
        with open('./data/all_entity_emb.pkl', 'wb') as f:
            pickle.dump(entity_emb, f)
            
    '''
    def add_user_item_relation(self, user, item, rating):
        if rating > 3:
            relation = 'satisfied'
        elif rating < 3:
            relation = 'dissatisfied'
        else:
            relation = 'purchase'
        self.graph.add_edge(user, item, relation=relation)

    def add_item_word_relation(self, item, word, sentiment):
        if not self.graph.has_edge(item, word):
            self.graph.add_edge(item, word, mentions=0, positive=0, negative=0, neutral=0)
        
        self.graph[item][word]['mentions'] += 1
        self.graph[item][word][sentiment] += 1

    def add_user_word_relation(self, user, word_freq):
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        total_words = len(sorted_words)
        
        for i, (word, freq) in enumerate(sorted_words):
            if i < total_words * 0.1:
                relation = 'mention_a_lot'
            elif freq == 1:
                relation = 'mention_barely'
            else:
                relation = 'mention'
            self.graph.add_edge(user, word, relation=relation)

    def build_graph(self, reviews, users, items, ratings):
        word_freq = defaultdict(lambda: defaultdict(int))
        
        for user, item, review, rating in zip(users, items, reviews, ratings):
            self.add_user_item_relation(user, item, rating)
            
            triplets = self.extract_sentiment(review)
            for f, o, s in triplets:
                self.add_item_word_relation(item, f, s)
                word_freq[user][f] += 1
        
        for user, freq in word_freq.items():
            self.add_user_word_relation(user, freq)
    '''
            


if __name__=='__main__':
    fire.Fire(SAKG)
