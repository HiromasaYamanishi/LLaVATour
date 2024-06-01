from bleu import compute_bleu
from rouge import rouge
import pandas as pd

class Evaluator:
    def evaluate_review(self, df):
        print(df['gt'][:5])
        print(df['output'][:5])
        bleu_score= compute_bleu(df['gt'].values, df['output'].values)
        rouge_score = rouge(df['gt'].values, df['output'].values)
        print('bleu', bleu_score)
        print('rouge', rouge_score)
        
if __name__ == '__main__':
    evaluator = Evaluator()
    df = pd.read_csv('../result/rec_exp/llava-v1.5-13b-jalan-tripadvisor-lora-v1.csv')
    evaluator.evaluate_review(df)
        