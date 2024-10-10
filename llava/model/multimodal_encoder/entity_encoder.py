import torch.nn as nn
import torch
from transformers import (
    BertJapaneseTokenizer, 
    BertModel
)
from torch_scatter.composite import scatter_softmax
from torch_scatter import scatter_add, scatter_mean
import itertools

# def convert_to_sequential_with_reset(sequence):
#     value_to_seq = {}
#     current_seq = -1
#     result = []
#     last_value = None
    
#     for value in sequence:
#         if value != last_value:
#             # 値が変わったら、連番をリセット
#             current_seq += 1
#             value_to_seq[value] = current_seq
        
#         result.append(value_to_seq[value])
#         last_value = value
    
#     return result

def convert_to_sequential_with_reset(sequence, prompt_change_positions):
    value_to_seq = {}
    current_seq = -1
    result = []
    last_value = None
    
    for i, value in enumerate(sequence):
        if value != last_value or i in prompt_change_positions:
            # 値が変わったら、または新しいプロンプトの開始位置の場合は連番をリセット
            current_seq += 1
            value_to_seq[value] = current_seq
        
        result.append(value_to_seq[value])
        last_value = value
    
    return result

def identify_prompt_change_positions(prompts):
    prompt_change_positions = set()
    last_prompt = None
    for i, prompt in enumerate(prompts):
        if prompt != last_prompt:
            prompt_change_positions.add(i)
            last_prompt = prompt
    return prompt_change_positions


class EntityEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        model_name_or_path = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
        self.model = BertModel.from_pretrained(model_name_or_path).cuda()
        self.model.eval()
        self.scorer = nn.Linear(768*5, 1)
        self.count_emb = torch.nn.Embedding(1000, 768)
        self.encode_dim = 768
        self.entity_projector = torch.nn.Linear(768, 5120)
        self.relation_projector = torch.nn.Linear(1536, 5120)
        
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
            3重リスト, batchごとに promptごとにtripletのリスト
        '''
        # print('triplets', triplets)
        flattened_triplets = list(itertools.chain.from_iterable(itertools.chain.from_iterable(triplets)))
        if len(flattened_triplets)==0:
            flattened_triplets = [['なし', 'なし', 'なし', 0, 0, 0]]
            triplets[0][0].append(['なし', 'なし', 'なし', 0, 0, 0])
            tasks[0].append('なし')
            prompts[0].append('なし')
            #triplets = [[[]]]
        len_per_instances = [[len(triplets__) for triplets__ in triplets_] for triplets_ in triplets]

        entity_per_prompts = [[max([e[5] for e in triplets__]) + 1 if triplets__ else 0 for triplets__ in triplets_] for triplets_ in triplets]
        tasks = sum(tasks, [])
        prompts = sum(prompts, [])

        entities, neighbors, relations, counts, prompt_indices, entity_indices = zip(*flattened_triplets)    
        prompt_change_positions = identify_prompt_change_positions(prompt_indices)

        prompt_indices = convert_to_sequential_with_reset(prompt_indices, prompt_change_positions)
        entity_indices = convert_to_sequential_with_reset(entity_indices, prompt_change_positions)

        entity_embs = self.encode(entities).to(torch.bfloat16)
        neighbor_embs = self.encode(neighbors).to(torch.bfloat16)
        relation_embs = self.encode(relations).to(torch.bfloat16)
        task_embs = self.encode(tasks).to(torch.bfloat16) 
        prompt_embs = self.encode(prompts).to(torch.bfloat16)
        prompt_embs = torch.stack([prompt_embs[i] for i in prompt_indices])
        count_embs = self.count_emb(torch.tensor(counts).cuda())
        print('count_embs', count_embs)
        score = self.scorer(torch.cat([prompt_embs.cuda(), entity_embs.cuda(), neighbor_embs.cuda(), relation_embs.cuda(), count_embs.cuda()], dim=1).to(torch.bfloat16))  
        score = scatter_softmax(score.squeeze(1), torch.tensor(entity_indices).cuda())
        relation_embs = scatter_add(torch.cat([relation_embs.cuda(), count_embs.cuda()], dim=1)*score.unsqueeze(1).cuda(), torch.tensor(entity_indices).unsqueeze(1).cuda(), dim=0)
        entity_embs = scatter_mean(entity_embs.cuda(), torch.tensor(entity_indices).unsqueeze(1).cuda(), dim=0)
        #assert len(relation_embs) == len(entity_embs) == sum([len(i) for i in len_per_instances])
        # print('relation_emb', relation_embs.shape, entity_embs.shape,entity_indices,  entity_per_prompts, len_per_instances)
        relation_embs = self.relation_projector(relation_embs)
        entity_embs = self.entity_projector(entity_embs)
        return relation_embs, entity_embs, entity_per_prompts, len_per_instances