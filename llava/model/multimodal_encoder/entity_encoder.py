import torch.nn as nn
import torch
from transformers import (
    BertJapaneseTokenizer, 
    BertModel
)
from torch_scatter.composite import scatter_softmax
from torch_scatter import scatter_add, scatter_mean
import itertools

def convert_to_sequential_with_reset(sequence):
    value_to_seq = {}
    current_seq = -1
    result = []
    last_value = None
    
    for value in sequence:
        if value != last_value:
            # 値が変わったら、連番をリセット
            current_seq += 1
            value_to_seq[value] = current_seq
        
        result.append(value_to_seq[value])
        last_value = value
    
    return result

class EntityEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        model_name_or_path = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
        self.model = BertModel.from_pretrained(model_name_or_path)
        self.model.eval()
        self.scorer = nn.Linear(768*4, 1)
        self.count_emb = torch.nn.Embedding(10000, 768)
        
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
        
        
    def forward(self, triplets, tasks, prompts):
        '''
            triplets: [[(neighbor11, relation11, count11, 1), ....(neighbor1N, relation1N, count1N, 1)..(neighborK1, relationK1, countK1, K), ... (neighborKM, relationKM, countKM. K)]
            ...[(neighbor11, relation11, count11, L), ....(neighbor1Q, relation1Q, count1Q, Q)..(neighborL1, relationL1, countL1, L), ... (neighborLP, relationLP, countLP. L)]]
        '''
        if not len(triplets):return torch.empty(0, 768).cuda(), torch.empty(0, 768).cuda(), []
        len_per_instances = [[len(triplets__) for triplets__ in triplets_] for triplets_ in triplets]
        flattened_triplets = list(itertools.chain.from_iterable(itertools.chain.from_iterable(triplets)))
        tasks = sum(tasks, [])
        prompts = sum(prompts, [])
        for r in flattened_triplets:
            if len(r)!=5:
                print('this is cause', r)
        print(flattened_triplets)
        print(tasks)
        print(prompts)
        if not all(len(triplet) == 5 for triplet in flattened_triplets):
            print('bad triplets', flattened_triplets)
            return torch.empty(0, 768).cuda(), torch.empty(0, 768).cuda(), []
        entities, neighbors, relations, counts, indices = [r[0] for r in flattened_triplets], [r[1] for r in flattened_triplets], [r[2] for r in flattened_triplets], [r[3] for r in flattened_triplets], [r[4] for r in flattened_triplets]
        
        indices = convert_to_sequential_with_reset(indices)
        entity_embs = self.encode(entities)
        neighbor_embs = self.encode(neighbors)
        relation_embs = self.encode(relations)
        task_embs = self.encode(tasks)      
        prompt_embs = self.encode(prompts)
        count_embs = self.count_emb(torch.tensor(counts).cuda())
        score = self.scorer(torch.cat([prompt_embs.cuda(), entity_embs.cuda(), neighbor_embs.cuda(), relation_embs.cuda()*count_embs.cuda()]).to(torch.bfloat16))  
        score = scatter_softmax(score, torch.tensor(indices).cuda())
        relation_embs = scatter_add(relation_embs*count_embs, torch.tensor(indices).cuda())
        entity_embs = scatter_mean(entity_embs, torch.tensor(indices).cuda())
        assert len(relation_embs) == len(entity_embs) == sum([len(i) for i in len_per_instances])
        
        return relation_embs, entity_embs, len_per_instances