import pandas as pd
from collections import defaultdict
import torch.nn as nn
import torch
from transformers import (
    BertJapaneseTokenizer, 
    BertModel
)
import torch.nn.functional as F

class SentimentLearner(nn.Module):
    def __init__(self, input_dim):
        super(SentimentLearner, self).__init__()
        self.Wa = nn.Linear(input_dim, 1, bias=False)
        self.ba = nn.Parameter(torch.zeros(1))
    
    def forward(self, x, attention_mask):
        # x shape: (batch_size, seq_len, input_dim)
        # attention_mask shape: (batch_size, seq_len)
        attention_scores = torch.tanh(self.Wa(x) + self.ba)  # (batch_size, seq_len, 1)
        
        # Apply attention mask
        attention_scores = attention_scores.squeeze(-1)  # (batch_size, seq_len)
        attention_scores = attention_scores.masked_fill(attention_mask == 0, -float('inf'))
        
        attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(-1)  # (batch_size, seq_len, 1)
        
        weighted_sum = torch.sum(x * attention_weights, dim=1)  # (batch_size, input_dim)
        return weighted_sum

class CrossAttention(nn.Module):
    def __init__(self, query_dim, review_dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.query_proj = nn.Linear(query_dim, review_dim)
        self.multihead_attn = nn.MultiheadAttention(review_dim, num_heads, batch_first=True)

    def forward(self, image_embedding, review_embeddings):
        # image_embedding: (1, K)
        # review_embeddings: (N, M)
        
        # Project image embedding to match review dimension
        query = self.query_proj(image_embedding).unsqueeze(0)  # (1, 1, M)
        
        # Prepare key and value from review embeddings
        key = value = review_embeddings.unsqueeze(0)  # (1, N, M)
        
        # Apply multihead attention
        attn_output, _ = self.multihead_attn(query, key, value)
        
        return attn_output.squeeze(0)  # (1,M)

class DifferentiableTopKRetrieval(nn.Module):
    def __init__(self, query_dim, review_dim, temperature=10.0):
        super(DifferentiableTopKRetrieval, self).__init__()
        self.temperature = temperature
        self.query_proj = nn.Linear(query_dim, review_dim)
        self.review_proj = nn.Linear(review_dim, review_dim)

    def forward(self, query_emb, review_emb, topk=5):
        # query_emb: (1, D)
        # review_emb: (N, D)
        
        # 内積の計算
        query_emb = self.query_proj(query_emb)
        review_emb = self.review_proj(review_emb)

        similarity_scores = torch.matmul(query_emb, review_emb.transpose(-2, -1))  # (1, N)
        
        # Top-k スコアの取得

        top_k_scores, _ = torch.topk(similarity_scores, k=min(topk, review_emb.shape[0]), dim=-1)
        kth_score = top_k_scores[:, -1].unsqueeze(-1)
        
        # ソフトマスクの生成
        soft_mask = torch.sigmoid((similarity_scores - kth_score) * self.temperature)
        
        # マスクの適用と正規化
        masked_scores = similarity_scores * soft_mask
        attention_weights = F.softmax(masked_scores, dim=-1)
        
        # 重み付き和の計算
        retrieved_emb = torch.matmul(attention_weights, review_emb)  # (1, D)
        
        return retrieved_emb, attention_weights


class MMRetriever(torch.nn.Module):
    def __init__(self, vision_tower):
        super().__init__()
        df_review = pd.read_pickle('/home/yamanishi/project/trip_recommend/data/jalan/review/review_all_period_.pkl')
        df_test = pd.read_csv('./data/df_review_feature_eval.csv')
        test_reviews = df_test['conversations'].values
        self.spot2reviews = defaultdict(list)
        for spot, review in zip(df_review['spot'], df_review['review']):
            if review in test_reviews:continue
            self.spot2reviews[spot].append(spot)
        self.vision_tower = vision_tower

        model_name_or_path = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
        self.model = BertModel.from_pretrained(model_name_or_path).cuda()
        self.model.eval()
        self.image_dim = 1024
        self.review_dim = 768
        self.sentiment_learner = SentimentLearner(self.review_dim).cuda()
        self.review_projector = torch.nn.Linear(self.review_dim, 4096)
        self.cross_attention = CrossAttention(self.review_dim + self.image_dim, self.review_dim)
        self.topk_retrieval = DifferentiableTopKRetrieval(query_dim=self.review_dim, review_dim=self.review_dim)
        
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
            print('model output', model_output[0].shape)
            sentence_embeddings = self.sentiment_learner(model_output[0], encoded_input["attention_mask"])

            all_embeddings.extend(sentence_embeddings)

        if not len(all_embeddings):
            all_embeddings = [torch.empty(0, 768)]
        # return torch.stack(all_embeddings).numpy()
        return torch.stack(all_embeddings)
        
    def forward(self, images, start_entities, prompts, tasks, topk=5):
        '''
        images: tensor [batch x image]
        start_entities: [['函館山', '函館山', '函館山'], ['東京タワー', '浅草寺']]
        prompts: [['観光地の名前を答えて', 'レビューを書いて', '感想を書いて'], ['レビューを書いて', 'どこにありますか']]
        tasks: [['LR', 'Review', 'Review'], ['Review', 'QA']]
        '''
        outputs = []
        document_num_per_prompts = []
        for i, (start_entity, prompt, task) in enumerate(zip(start_entities, prompts, tasks)):
            tmp = []
            for s, p, t in zip(start_entity, prompt, task):
                image = images[i]
                # print('image', image.shape, sum(image))
                image_feature = self.vision_tower(image.unsqueeze(0))
                image_feature = torch.mean(image_feature, dim=1)

                print('image_feature', image_feature.shape)
                prompt_emb = self.encode([p])
                print('prompt_emb', prompt_emb.shape)

                query_emb = torch.cat([image_feature, prompt_emb], axis=1)
                if s in self.spot2reviews:
                    reviews = self.spot2reviews[s]

                    review_emb = self.encode(reviews)
                else:
                    reviews = [f'これは{start_entity}のレビューです']
                    review_emb = self.encode(reviews)
               
                query_emb = self.cross_attention(query_emb, review_emb)
                print('query_emb', query_emb.shape)
                
                retrieved_emb, attention_weights = self.topk_retrieval(query_emb, review_emb)
                tmp.append(len(retrieved_emb))
                outputs.append(retrieved_emb)
            document_num_per_prompts.append(tmp)

        return torch.cat(outputs, dim=0), document_num_per_prompts