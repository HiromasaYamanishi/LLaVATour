import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from collections import defaultdict
from .sentence_transformer import SentenceBertJapanese
import time
from torch_scatter.composite import scatter_softmax
from torch_scatter import scatter_add
from transformers import (
    BertJapaneseTokenizer, 
    BertModel
)
from tqdm import tqdm

class AttentionScorer(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 4, 1)
            #nn.ReLU(),
            #nn.Linear(64, 1)
        )
    
    def forward(self, q_emb, task_emb, relation_emb, neighbor_emb):
        combined = torch.cat([q_emb, task_emb, relation_emb, neighbor_emb], dim=1).to(torch.bfloat16).cuda()
        #print(combined.shape)
        return self.mlp(combined).squeeze(-1)
    
def print_grad(grad):
    print("勾配:", grad)
    return grad

class Retriever(torch.nn.Module):
    def __init__(self, graph_path='./data/kg/sakg.pkl'):
        super().__init__()
        with open(graph_path, 'rb') as f:
            self.graph = pickle.load(f)
            
        # embeddingをロードするのは容量が大きすぎてできないため, retrieve時にencodeする
        # with open('./data/all_entity_emb.pkl') as f:
        #     self.entity_emb_all = pickle.load(f)
            
        # with open('./data/all_prompt_emb.pkl', 'rb') as f:
        #     self.prompt_emb_all = pickle.load(f)
        self.count_embs = torch.nn.Embedding(10000, 768)
        #self.bert = SentenceBertJapanese()
        self.scorer = AttentionScorer(embedding_dim=768)
        
        model_name_or_path = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
        self.model = BertModel.from_pretrained(model_name_or_path)
        self.model.eval()
        
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode(self, sentences, batch_size=20000):
        all_embeddings = []
        # iterator = tqdm(range(0, len(sentences), batch_size))
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest", 
                                           truncation=True, return_tensors="pt") #.to(self.device)
            for k,v in encoded_input.items():
                encoded_input[k] = v.cuda()
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)
        if not len(all_embeddings):
            all_embeddings = [torch.empty(0, 768)]
        # return torch.stack(all_embeddings).numpy()
        return torch.stack(all_embeddings)
        
    def forward(self, batched_data, batched_index, batched_prompt_index, thresh_neighbor=1.2, thresh_relation=1.2):
        '''
        batched_data: [(prompt_1_1, start_entity_1_1, task_1_1)...(prompt_n_m, start_entity_n_m, task_n_m)]
        '''
        print('scorer weight ', self.scorer.mlp[0].weight)
        prompts = [data[0] for data in batched_data]
        start_entities = [data[1] for data in batched_data]
        tasks = [data[2] for data in batched_data]
        
        batched_neighbors = []
        batched_start_entities = []
        batched_start_indices = []
        batched_prompt_indices = []
        batched_thr_scores = []
        
        for i, start_entity in enumerate(start_entities):
            if start_entity not in self.graph:continue
            neighbors = list(self.graph.neighbors(start_entity))
            neighbors = sorted(neighbors, key=lambda x: len(self.graph[start_entity][x]['relations']), reverse=True)[:20]
            
            batched_neighbors.extend(neighbors)
            batched_start_entities.extend([start_entity] * len(neighbors))
            batched_start_indices.extend([batched_index[i]] * len(neighbors))
            batched_prompt_indices.extend([batched_prompt_index[i]] * len(neighbors))
            batched_thr_scores.extend([thresh_neighbor/(len(neighbors)+1e-9)] * len(neighbors))

        assert len(batched_neighbors) == len(batched_start_entities) == len(batched_start_indices)
        batched_relations, batched_counts = [], [] # 全てのrelation, 全てのcount
        batched_start_entities_rel, batched_start_indices_rel, batched_prompt_indices_rel = [], [], [] # 全てのrelationに対応するentity, 全てのrelationに対応するentityのindice
        batched_neighbor_rel, batched_neighbor_indices_rel, batched_neighbor_thr_scores = [], [], []
        
        for i, neighbor in enumerate(batched_neighbors):
            start_entity = batched_start_entities[i]
            if start_entity not in self.graph:continue
            edge_data = self.graph[start_entity][neighbor]['relations']
            sorted_edge_data = sorted(edge_data.items(), key=lambda x: x[1], reverse=True)[:20]
            
            batched_relations.extend([relation for relation, count in sorted_edge_data])
            batched_counts.extend([count for relation, count in sorted_edge_data])
            batched_start_entities_rel.extend([start_entity] * len(sorted_edge_data))
            batched_start_indices_rel.extend([batched_start_indices[i]] * len(sorted_edge_data))
            batched_prompt_indices_rel.extend([batched_prompt_indices[i]] * len(sorted_edge_data))
            batched_neighbor_indices_rel.extend([i] * len(sorted_edge_data))
            batched_neighbor_rel.extend([neighbor] * len(sorted_edge_data))
            batched_neighbor_thr_scores.extend([thresh_relation/(len(sorted_edge_data)+1e-9) for _ in range(len(sorted_edge_data))])
    
        all_text = prompts + start_entities + tasks + batched_relations
        all_emb = self.encode(all_text)
        prompt_emb = all_emb[:len(prompts)]
        start_emb = all_emb[len(prompts): len(prompts)+len(start_entities)]
        task_emb = all_emb[len(prompts)+len(start_entities):len(prompts)+len(start_entities)+len(tasks)]
        relation_emb = all_emb[len(prompts)+len(start_entities)+len(tasks):]
        # print(len(prompts), len(batched_neighbors), len(batched_relations))
        # print(prompt_emb.shape, start_emb.shape, task_emb.shape, relation_emb.shape)
        assert len(batched_relations)==len(batched_counts)==len(batched_start_entities_rel)==len(batched_start_indices_rel)
        #print('relation num', len(batched_relations))
        if len(batched_relations):
            count_emb = self.count_embs(torch.tensor(batched_counts).to(torch.long).cuda()) # [relationの合計 x 768]
            count_emb.register_hook(print_grad)
            start_entity_emb_rel = torch.stack([start_emb[batched_start_indices_rel[i]] for i in range(len(batched_start_entities_rel))])
            relation_emb = relation_emb.cuda() * count_emb.cuda() # [relationの合計 x 768]
            alpha = scatter_softmax(torch.sum(start_entity_emb_rel.cuda()*relation_emb.cuda(), axis=1), torch.tensor(batched_neighbor_indices_rel).cuda()) # [relationの合計]
            chosen_flag = alpha > torch.tensor(batched_neighbor_thr_scores).cuda()
            # print('alpha', alpha, 'neighbor_thr', batched_neighbor_thr_scores, 'chosen_flag', chosen_flag)
            # chosen_relations = [batched_relations[i] for i in range(len(chosen_flag)) if chosen_flag[i]]
            # chosen_start_entity_for_relation = [batched_start_entities_rel[i] for i in range(len(chosen_flag)) if chosen_flag[i]]
            
            # print('chosen')
            # print('alpha', alpha)
            relation_emb = scatter_add(relation_emb.cuda()*alpha.reshape(-1, 1).cuda(), torch.tensor(batched_neighbor_indices_rel).cuda(), dim=0) # [neighborの合計]
        else:
            count_emb = self.count_embs(torch.tensor(batched_counts).to(torch.long).cuda())
            relation_emb = torch.empty(0, 768).cuda()
            chosen_flag = torch.empty(0)
            
        if len(batched_neighbors):
            neighbor_emb = torch.stack([torch.randn(768) for _ in range(len(batched_neighbors))]) # [neighborの合計, 768]
            start_emb = torch.stack([start_emb[i] for i in batched_start_indices]) # [neighborの合計, 768]
            task_emb = torch.stack([task_emb[i] for i in batched_start_indices]) # [neighborの合計, 768]
            prompt_emb = torch.stack([prompt_emb[i] for i in batched_start_indices]) # [neighborの合計, 768]
        else:
            neighbor_emb = torch.empty(0, 768)
            start_emb = torch.empty(0, 768)
            task_emb = torch.empty(0, 768)
            prompt_emb = torch.empty(0, 768)
        
        # print(start_emb.shape, prompt_emb.shape, relation_emb.shape, neighbor_emb.shape)
        score = self.scorer(start_emb.cuda(), prompt_emb.cuda(), relation_emb.cuda(), neighbor_emb.cuda()) # [neighborの合計]
        score.register_hook(print_grad)
        #for k,v in self.scorer.named_parameters():
        #    print(k, v.requires_grad)    
        #print('score', score)
        # print('score', score, 'batched_thr_scores', batched_thr_scores, 'batched_prompt_indices', batched_prompt_indices)
        chosen = score.cuda() > torch.tensor(batched_thr_scores).cuda()
        chosen_entities = [batched_neighbors[i] for i in range(len(batched_neighbors)) if chosen[i]]
        prompt_indices = [batched_prompt_indices[i] for i in range(len(batched_neighbors)) if chosen[i]]

        chosen_relations = [batched_relations[i] for i in range(len(chosen_flag)) if (chosen_flag[i] and batched_neighbor_rel[i] in chosen_entities)]
        chosen_counts = [batched_counts[i] for i in range(len(chosen_flag)) if (chosen_flag[i] and batched_neighbor_rel[i] in chosen_entities)]
        chosen_entities = [batched_neighbor_rel[i] for i in range(len(chosen_flag)) if (chosen_flag[i] and batched_neighbor_rel[i] in chosen_entities)]
        prompt_indices = [batched_prompt_indices_rel[i] for i in range(len(chosen_flag)) if (chosen_flag[i] and batched_neighbor_rel[i] in chosen_entities)]

        assert len(chosen_relations) == len(chosen_counts) == len(chosen_entities)
        # print(len(chosen_relations), len(chosen_counts), len(chosen_relations))
        chosen_triplets = [str(tuple((chosen_entities[i], chosen_relations[i], chosen_counts[i]))) for i in range(len(chosen_entities))]
        
        # print('chosen counts', chosen_counts)
        # print('chosen relations', chosen_relations)
        # print('chosen entities', chosen_entities)
        
        # print('chosen_start_entity_for_relations', chosen_start_entity_for_relation)
        
        # print('chosen prompt indices', prompt_indices)
        #print('score', score)
        #print('chosen', chosen)
        # print('chosen entity', chosen_entities, prompt_indices)
        # print('chosen triplets', chosen_triplets)
        return chosen_triplets, prompt_indices
        
        #batched_relations = [relation for relation,_ in self.graph.neighbors[]]
        
    def forward_(self, prompt, start_entity, task, top_k=3):
        #print('retrieval')
        #print('retrieve', prompt, start_entity, task)
        q_emb = torch.randn(768).cuda() # self.prompt_emb_all[prompt]
        start_emb = torch.randn(768).cuda() # self.entity_emb_all[start_entity]
        task_emb = torch.randn(768).cuda() #self.task_emb[task]
        #print('start_entity_retrieve', start_entity)
        # 1ホップ以内のエンティティとその関係を効率的に取
        neighbors = list(self.graph.neighbors(start_entity))
        if len(neighbors) == 0:
            return []
        #print('neighbors', len(neighbors))
        # relation_data = defaultdict(list)
        relation_emb_batch = []
        neighbor_emb_batch = []
        for i,neighbor in enumerate(neighbors):
            #print(i)
            time1 = time.time()
            relation_data = defaultdict(list)
            edge_data = self.graph[start_entity][neighbor]
            time2 = time.time()
            #print(time2-time1)
            #for relation, count in edge_data['relations'].items():
            #    relation_data[relation].append((neighbor, count))
        
            # バッチ処理のための準備
            #all_relations = list(relation_data.keys())
            #relation_embs = torch.stack([torch.randn(768) for rel in all_relations]).cuda()
            relation_embs = torch.stack([torch.randn(768) for relation, count in edge_data['relations'].items()]).cuda()
            time3 = time.time()
            #print(time3-time2)
            counts = torch.tensor([count for relation, count in edge_data['relations'].items()])
            count_embs = self.count_embs(counts.cuda())
            time4 = time.time()
            #print(time4-time3)
            # α_iの計算（バッチ処理）
            # combined = torch.cat([relation_embs, count_embs], dim=1)
            combined = relation_embs.cuda() * count_embs.cuda()
            time5 = time.time()
            #print(time5-time4)
            #print(relation_embs.shape, count_embs.shape, combined.shape, start_emb.shape)
            alphas = F.softmax(torch.matmul(start_emb, combined.t()), dim=0)
            #print('alphas', alphas)
            time6 = time.time()
            #print(time6-time5)
            # relation_embの計算（バッチ処理）
            relation_emb_weighted = (alphas.unsqueeze(1).cuda() * relation_embs.cuda()).sum(dim=0)
            #print('relation emb', relation_emb_weighted.shape)
            time7 = time.time()
            #print(time7-time6)
            relation_emb_batch.append(relation_emb_weighted)
            neighbor_emb_batch.append(torch.randn(768))
            
        if not len(relation_emb_batch):
            relation_emb_batch = [torch.zeros(768)]
            neighbor_emb_batch = [torch.zeros(768)]
        #print('get emb')
            # Attention weightの計算（バッチ処理）
        q_emb_batch = q_emb.unsqueeze(0).expand(len(neighbor_emb_batch), -1)
        task_emb_batch = task_emb.unsqueeze(0).expand(len(neighbor_emb_batch), -1)
        # relation_emb_batch = relation_emb_weighted.unsqueeze(0).expand(len(neighbors), -1)
        relation_emb_batch = torch.stack(relation_emb_batch)
        neighbor_emb_batch = torch.stack(neighbor_emb_batch)
        
        #print(q_emb_batch.shape, task_emb_batch.shape, relation_emb_batch.shape, neighbor_emb_batch.shape)
        attn_weights = self.scorer(q_emb_batch.cuda(), task_emb_batch.cuda(), relation_emb_batch.cuda(), neighbor_emb_batch.cuda())
        #print('calc attn score')
        # エンティティとスコアのペアを作成
        entity_scores = list(zip(neighbors, attn_weights.tolist()))
        
        # スコアでソートし、上位k個を返す
        top_entities = sorted(entity_scores, key=lambda x: x[1], reverse=True)[:top_k]
        #print('get entity')
        
        return top_entities
    
def build_retriever():
    return Retriever()