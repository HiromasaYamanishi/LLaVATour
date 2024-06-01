from language_models import LLM
import pandas as pd
from tqdm import tqdm
import pickle
prompt = "また、このレビューが述べるこの観光地のポジティブな部分、ネガティブな部分の対象の名詞を文中からそのまま抜き出してください。\n\
    例1:\n\
    レビュー: 深夜に女性専用ラウンジを利用しました。リクライニングソファがあり寝る事が出来ます。静かで雰囲気は良かったけど暖房が効きすぎているのか暑くて辛かったです。\n\
    出力:\n\
        ポジティブな点: 「リクライニングソファ」「雰囲気」\n\
        ネガティブな点: 「暖房」\n\
    例2:\n\
    レビュー: 函館山からの夜景を楽しみに行ったのに2回とも雲がかかり上からの景色は見ることが出来なかった。ロープウェイで下降する時少し見えただけでした。\n\
    出力:\n\
        ポジティブな点: なし\n\
        ネガティブな点: 「雲」\n\
    例3:\n\
    レビュー: 平日の昼過ぎに訪れました。駅からの道のりは少しわかりにくいですが迷うことなく到着。すぐに鳳凰堂内部見学のため観覧券販売所を見つけ行列の後ろに。１回に５０人程度が入れるのかな。並んだ時に次の時間を買うことができました。鳳凰堂の中に入ってビックリしたのはライトアップされた阿弥陀如来坐像の煌びやかさとたくさんの菩薩像です。２０分程度の説明があっという間でした。仏像を前に身が引き締まる思いです\n\
    出力:\n\
        ポジティブな点: 「ライトアップ」「阿弥陀如来坐像」「たくさんの菩薩像」\n\
        ネガティブな点: 「駅からの道」\n\
それでは次のレビューに対して同様に出力してください。\n\
レビュー: {}。"

df_target = pd.read_csv('./recommend/filtered_review_df.csv')
llm = LLM("lightblue/qarasu-14B-chat-plus-unleashed")
batch_size=100

save_output = {}
start=0
for i in tqdm(range(0, len(df_target), batch_size)):
    prompts = []
    for j in range(batch_size):
        if i+j>=len(df_target):continue
        prompts.append(prompt.format(df_target['review'].values[i+j]))
    
    outputs = llm.generate(prompts)
    for k,output in enumerate(outputs):
        save_output[start+i+k] = output
        
#print(save_output)
with open('./recommend/poeneg.pkl', 'wb') as f:
    pickle.dump(save_output, f)