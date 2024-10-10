import pandas as pd
import numpy as np
from collections import Counter
from janome.tokenizer import Tokenizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import transformers
from transformers import pipeline
import os
import re
from transformers import MarianMTModel, MarianTokenizer
import spacy
trans_dict_1 = {
    '良い': 'Good', ' ': ' ', '場所': 'Place', '思う': 'Think', '綺麗': 'Beautiful', '公園': 'Park', '人': 'People',
    '神社': 'Shrine', '桜': 'Cherry Blossom', '自然': 'Nature', '建物': 'Building', '城': 'Castle', '行ける': 'Accessible',
    '方': 'Way', '歴史': 'History', 'スポット': 'Spot', '車': 'Car', '利用': 'Use', '美術館': 'Museum', '台': 'Platform',
    '乗る': 'Ride', '種類': 'Type', '途中': '途中', '参拝': 'Worship', '季節': 'Season', '素敵': 'Lovely', '上': 'Up',
    '花見': 'Flower Viewing', '大きい': 'Large', '好き': 'Like', '遊具': 'Playground Equipment', '感じる': 'Feel',
    '距離': 'Distance', '橋': 'Bridge', '海岸': 'Coast', '夜': 'Night', '料理': 'Cuisine', 'お参り': 'Visit',
    '階段': 'Stairs', '周り': 'Around', 'ショー': 'Show', '川': 'River', '露天風呂': 'Open-Air Bath', '眺め': 'View',
    '博物館': 'Museum', '市内': 'City', '知る': 'Know', '近い': 'Near', 'スタッフ': 'Staff', '使う': 'Use',
    '作品': 'Work', '芝生': 'Lawn', '物': 'Object', '遊べる': 'Play', '持つ': 'Hold', '街': 'Town', '安心': 'Relief',
    'アクセス': 'Access', '日帰り': 'Day Trip', '湯': 'Hot Water', 'お弁当': 'Lunch Box', '遊歩道': 'Promenade',
    '遊ぶ': 'Play', '満開': 'Full Bloom', '園': 'Garden', '名所': 'Famous Spot', '渡る': 'Cross', '回る': 'Circle',
    '開催': 'Event', '作り': 'Making', '向け': 'For', '長い': 'Long', '風': 'Wind', '頃': 'Time', '夕日': 'Sunset',
    '甘い': 'Sweet', 'キレイ': 'Beautiful', '入浴': 'Bathing', '続く': 'Continue', '遊園地': 'Amusement Park',
    '興味': 'Interest', '走る': 'Run', '旧': 'Old', '街並み': 'Townscape', 'アトラクション': 'Attraction', '満足': 'Satisfaction',
    '本当': 'True', '様々': 'Various', '鹿': 'Deer', '実際': 'Actual', '機会': 'Opportunity', '下': 'Down',
    'リフト': 'Lift', '広場': 'Square', '木々': 'Trees', '勉強': 'Study', '強い': 'Strong', '周': 'Around',
    '得': 'Gain', '程度': 'Degree', '湧水': 'Spring Water', '沿い': 'Along', '停める': 'Park', '渓谷': 'Valley',
    '上がる': 'Rise', 'エリア': 'Area', '説明': 'Explanation', '購入': 'Purchase', '岬': 'Cape', '造り': 'Construction',
    'お守り': 'Amulet', '湖畔': 'Lakeside', '作れる': 'Can Make', '自分': 'Self', '満喫': 'Enjoy', '比較的': 'Relatively',
    '優しい': 'Gentle', 'お部屋': 'Room', '階': 'Floor', '怖い': 'Scary', '大会': 'Tournament', '線': 'Line',
    '競馬場': 'Racecourse', '硫黄': 'Sulfur', '匂い': 'Smell'
}

trans_dict_2 = {
    '子供': 'Child', '夏': 'Summer', '楽しい': 'Fun', '昔': 'Past', '楽しむ': 'Enjoy', '過ごせる': 'Spend',
    'お店': 'Shop', '時間': 'Time', '湖': 'Lake', '買う': 'Buy', '駅': 'Station', '連れ': 'Companion', '敷地': 'Grounds',
    '登る': 'Climb', '大喜び': 'Delighted', '中心': 'Center', '観光': 'Sightseeing', '立派': 'Impressive', '豊か': 'Rich',
    '高い': 'High', '涼しい': 'Cool', '部': 'Part', '小さい': 'Small', '教える': 'Teach', '緑': 'Green', '混む': 'Crowded',
    '動物園': 'Zoo', '美しい': 'Beautiful', '見学': 'Visit', '花': 'Flower', '水族館': 'Aquarium', '迫力': 'Impact',
    '行う': 'Conduct', 'プール': 'Pool', '離れる': 'Leave', 'バス': 'Bus', '残念': 'Regret', '食事処': 'Restaurant',
    '癒す': 'Heal', '子供たち': 'Children', '宿泊': 'Stay', '咲く': 'Bloom', '様': 'Like', '砂浜': 'Sandy Beach',
    '地元': 'Local', '可愛い': 'Cute', '登山': 'Mountain Climbing', '標高': 'Elevation', '大好き': 'Love',
    '夏場': 'Summertime', 'バイキング': 'Buffet', '水遊び': 'Water Play', 'アスレチック': 'Athletic', '見応え': 'Worth Seeing',
    'ロープウェイ': 'Ropeway', '夏休み': 'Summer Vacation', 'パーク': 'Park', '古い': 'Old', '飛行機': 'Airplane',
    '縁結び': 'Matchmaking', '木': 'Tree', '気分': 'Mood', '楽しみ': 'Looking Forward', 'm': 'm', '夜景': 'Night View',
    '毎年': 'Every Year', '波': 'Wave', '足湯': 'Footbath', '有料': 'Charged', 'ソフトクリーム': 'Soft Serve',
    '家': 'House', '売る': 'Sell', '見渡せる': 'Overlook', '残る': 'Remain', '泳ぐ': 'Swim', '広大': 'Vast',
    '疲れる': 'Tired', '町並み': 'Townscape', '暑い': 'Hot', '海鮮丼': 'Seafood Bowl', '方々': 'People', '遠く': 'Far',
    '大': 'Large', '場': 'Place', '忍者': 'Ninja', '違う': 'Different', '周囲': 'Surroundings', '滑り台': 'Slide',
    '宿': 'Inn', '聞く': 'Listen', '寺院': 'Temple', 'テーマ': 'Theme', '用': 'Use', 'バーベキュー': 'Barbecue',
    '電車': 'Train', '石垣': 'Stone Wall', '透明度': 'Transparency', '御朱印': 'Goshuin', 'シャワー': 'Shower',
    '灯台': 'Lighthouse', '高原': 'Plateau', '帰る': 'Return', '派': 'School', '木造': 'Wooden', '本尊': 'Main Image',
    'スタジアム': 'Stadium', '再現': 'Reproduction', 'いちご狩り': 'Strawberry Picking', 'シティ': 'City', '販売': 'Sale',
    'サッカー': 'Soccer', 'コンサート': 'Concert', '迷う': 'Get Lost', '囲む': 'Surround', '食堂': 'Cafeteria',
    '学べる': 'Learn', '入館料': 'Admission Fee', '咲き誇る': 'In Full Bloom', '穏やか': 'Calm', '快適': 'Comfortable',
    '施設内': 'Facility', '茶': 'Tea', '姿': 'Figure', '上る': 'Climb', '変わる': 'Change', '屋根': 'Roof', '足': 'Foot',
    '世界': 'World', '薬師如来': 'Yakushi Nyorai', 'ボール': 'Ball', '数': 'Number', '高層': 'High-Rise',
    '賑やか': 'Lively', '私たち': 'We'
}
trans_dict_3 = {
    '楽しめる': 'Enjoyable', '多い': 'Many', '美味しい': 'Delicious', '紅葉': 'Autumn Leaves', '気持ち': 'Feeling',
    '歩く': 'Walk', '近く': 'Nearby', '最高': 'Best', '土産屋': 'Souvenir Shop', '出来る': 'Can Do', '食事': 'Meal',
    '有名': 'Famous', '雰囲気': 'Atmosphere', '徒歩': 'On Foot', 'オススメ': 'Recommend', '道': 'Road', '日': 'Day',
    '大人': 'Adult', '食べる': 'Eat', '大きな': 'Big', '温泉街': 'Hot Spring Town', '立ち寄る': 'Stop By', 'ホテル': 'Hotel',
    '池': 'Pond', '風呂': 'Bath', '丁寧': 'Polite', '時': 'Time', '感ずる': 'Feel', '休憩': 'Rest', '充実': 'Fulfillment',
    '賑わう': 'Bustling', '滝': 'Waterfall', '庭園': 'Garden', '写真': 'Photo', '天気': 'Weather', '所': 'Place',
    '今回': 'This Time', '魚': 'Fish', '最適': 'Optimal', '訪れる': 'Visit', '整備': 'Maintenance', '秋': 'Autumn',
    '並ぶ': 'Line Up', '位置': 'Position', '空気': 'Air', '際': 'Occasion', '遊覧船': 'Sightseeing Boat', 'トイレ': 'Toilet',
    '展示物': 'Exhibits', '値段': 'Price', '言う': 'Say', '子供達': 'Children', '面白い': 'Interesting', 'ガラス': 'Glass',
    'リフレッシュ': 'Refresh', '感動': 'Impression', '撮る': 'Take (a photo)', '他': 'Others', '旅行': 'Travel',
    '山頂': 'Mountain Top', '肌': 'Skin', '選ぶ': 'Choose', '雨': 'Rain', '通る': 'Pass Through', '注意': 'Attention',
    '学ぶ': 'Learn', '島': 'Island', '味': 'Taste', '文化': 'Culture', '都会': 'City', '試食': 'Sample',
    '以上': 'More Than', 'ドライブ': 'Drive', '眺める': 'Look Out', 'ウォーキング': 'Walking', '跡': 'Trace', 'セット': 'Set',
    '釣り': 'Fishing', '何度': 'Many Times', 'スキー': 'Ski', '建てる': 'Build', '地': 'Ground', 'スペース': 'Space',
    'お子さん': 'Your Child', '平日': 'Weekday', '鯉': 'Carp', '国宝': 'National Treasure', '先端': 'Tip',
    '娘': 'Daughter', '立': 'Stand', '夕食': 'Dinner', '今年': 'This Year', '待つ': 'Wait', '乗れる': 'Can Ride',
    '昼食': 'Lunch', '歴史的': 'Historical', 'パン': 'Bread', 'センター': 'Center'
}

trans_dict_4 = {
    '行く': 'Go', '見る': 'See', '広い': 'Wide', '冬': 'Winter', '中': 'Inside', '時期': 'Season', '散策': 'Walk',
    '海': 'Sea', '静か': 'Quiet', '展示': 'Exhibition', '入る': 'Enter', '見える': 'Visible', '無料': 'Free',
    '寺': 'Temple', '雪': 'Snow', '観光客': 'Tourists', '境内': 'Precincts', '少ない': 'Few', '素晴らしい': 'Wonderful',
    '落ち着く': 'Calm', '便利': 'Convenient', '山': 'Mountain', 'レストラン': 'Restaurant', '寒い': 'Cold',
    '入場料': 'Admission Fee', '見ごたえ': 'Worth Seeing', '周辺': 'Surroundings', '風情': 'Elegance', '動物': 'Animal',
    '大変': 'Difficult', '見れる': 'Can See', '飲食店': 'Restaurant', '暖かい': 'Warm', '安い': 'Cheap', '感じ': 'Feeling',
    '晴れる': 'Sunny', '新鮮': 'Fresh', 'スキー場': 'Ski Resort', 'イベント': 'Event', '昼間': 'Daytime', '後': 'After',
    '来る': 'Come', '野菜': 'Vegetable', '時代': 'Era', '寄る': 'Stop By', 'イルミネーション': 'Illumination',
    '積もる': 'Accumulate', '前': 'Before', '過ごす': 'Spend', '色々': 'Various', '雪景色': 'Snow Scenery',
    '家族': 'Family', '雪質': 'Snow Quality', '出る': 'Exit', '必要': 'Necessary', '一緒': 'Together', '以外': 'Except',
    '鳥居': 'Torii Gate', '堂': 'Hall', '空く': 'Empty', '山々': 'Mountains', 'コーナー': 'Corner', '滑る': 'Slide',
    '岩': 'Rock', '船': 'Boat', '絶景': 'Superb View', '春': 'Spring', '降る': 'Fall', '隣': 'Next', '御': 'Honorable',
    '柱': 'Pillar', '初詣': 'First Shrine Visit', '混雑': 'Congestion', '外': 'Outside', '館': 'Building',
    '売店': 'Shop', 'ランチ': 'Lunch', '気': 'Mood', 'デザート': 'Dessert', '資料館': 'Museum', '冬場': 'Winter Season',
    'イチゴ': 'Strawberry', '上級者': 'Expert', '疲れ': 'Tired', '光景': 'Scenery', 'ジョギング': 'Jogging',
    '果物': 'Fruit', '梨': 'Pear', 'バラ': 'Rose', '吊り橋': 'Suspension Bridge', 'オブジェ': 'Object', '大きさ': 'Size',
    'ライトアップ': 'Illumination', '屋敷': 'Mansion', '直結': 'Direct Connection', 'ショップ': 'Shop', '席': 'Seat',
    '湖面': 'Lake Surface', '映る': 'Reflect', '紫陽花': 'Hydrangea', '閑散': 'Quiet', '狭い': 'Narrow',
    '恐竜': 'Dinosaur', '壁': 'Wall', 'ケーブルカー': 'Cable Car', '形': 'Shape', '宮': 'Shrine', '飛ぶ': 'Fly',
    '普通': 'Normal', '梅': 'Plum', '店内': 'Inside the Store'
}
# fugu_translator = pipeline('translation', model='staka/fugumt-en-ja', device=2)

# encoder_model_name = "cl-tohoku/bert-base-japanese-v2"
# decoder_model_name = "openai-community/gpt2"
# src_tokenizer = transformers.BertJapaneseTokenizer.from_pretrained(encoder_model_name)
# trg_tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(decoder_model_name)
# model = transformers.EncoderDecoderModel.from_pretrained("sappho192/jesc-ja-en-translator")
# def translate_list(text_list):
#     return [translate(text) for text in text_list]

# def translate(text_src):
#     embeddings = src_tokenizer(text_src, return_attention_mask=False, return_token_type_ids=False, return_tensors='pt')
#     embeddings = {k: v for k, v in embeddings.items()}
#     output = model.generate(**embeddings, max_length=512)[0, 1:-1]
#     text_trg = trg_tokenizer.decode(output.cpu())
#     return text_trg

model_name = "Helsinki-NLP/opus-mt-ja-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate(text):
    # 文字数が15文字以上ならNoneを返す
    if len(text) >= 15:
        return None
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs, max_length=512)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def translate_list(text_list):
    # Noneではない結果のみをリストに含める
    return [result for text in text_list if (result := translate(text)) is not None]

# 形態素解析と単語カウントの関数
# GiNZAの日本語モデルをロード
nlp = spacy.load("ja_ginza")

def translate_dict(original_dict, translation_dict):
    translated_dict = {}
    for key, value in original_dict.items():
        translated_key = translation_dict.get(key, key)  # 英訳が存在しない場合、元のキーを使用
        translated_dict[translated_key] = value
    return translated_dict

def count_words(text):
    # テキストをGiNZAで解析
    doc = nlp(text)
    
    words = []
    for token in doc:
        # 名詞、動詞、形容詞を抽出し、基本形に変換
        if token.pos_ in ['NOUN', 'VERB', 'ADJ']:
            words.append(token.lemma_)  # lemma_で基本形に変換

    # 単語のカウント
    return Counter(words)

def split_text(text, max_length=40000):
    """指定されたバイト数以下になるようにテキストを分割"""
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in text.split('。'):  # 文単位で分割
        sentence_length = len(sentence.encode('utf-8'))  # UTF-8でのバイト数を計算
        if current_length + sentence_length > max_length:
            chunks.append('。'.join(current_chunk))
            current_chunk = []
            current_length = 0
        
        current_chunk.append(sentence)
        current_length += sentence_length
    
    if current_chunk:
        chunks.append('。'.join(current_chunk))
    
    return chunks

# 各データフレームの単語カウント
result_dir = '../result/reviews/llavatour/llava-v1.5-13b-jalan-review-lora-v28'
dfs = [pd.read_csv(os.path.join(result_dir, 'season_春.csv')),
        pd.read_csv(os.path.join(result_dir, 'season_夏.csv')),
        pd.read_csv(os.path.join(result_dir, 'season_秋.csv')),
        pd.read_csv(os.path.join(result_dir, 'season_冬.csv')),]
# for df in dfs:
#     print(df.columns)
for df in dfs:
    df['predicted'] = df['predicted'].str.replace('<s>', '').replace('<\s>', '').replace('</', '').replace('じゃらん.netで遊び体験済み', '')
df_names = ['Spring', 'Summer', 'Autumn', 'Winter']
word_counts = []
for df in dfs:
    all_text = ' '.join(df['predicted'])
    chunks = split_text(all_text)
    
    # 各データフレームごとのカウントを保持するCounterを作成
    df_counter = Counter()
    
    for chunk in chunks:
        df_counter.update(count_words(chunk))
    
    # データフレームごとのカウント結果をリストに追加
    word_counts.append(df_counter)

# 他のdfより多く含まれる単語を抽出
unique_words = []
for i, counts in enumerate(word_counts):
    other_counts = [c for j, c in enumerate(word_counts) if j != i]
    unique = {word: count for word, count in counts.items() 
              if all(count > c.get(word, 0) for c in other_counts)}
    unique_words.append(unique)

trans_dicts = [trans_dict_1, trans_dict_2, trans_dict_3, trans_dict_4]
print('prepare done')

fig, axs = plt.subplots(2, 2, figsize=(16, 12))
for i, (unique, ax) in enumerate(zip(unique_words, axs.flatten())):
    unique_new = {}
    
    unique = dict(sorted(unique.items(), key=lambda item: item[1], reverse=True))
    print('unique orig', unique)
    unique = {k: v for k, v in unique.items() if v >= 7}
    unique = {k: v for k, v in unique.items() if not re.fullmatch(r'[!-/:-@[-`{-~]+', k)}
    unique = {k: v for k, v in unique.items() if not re.fullmatch(r'^[\u3040-\u309F]+$', k)}
    unique = dict(sorted(unique.items(), key=lambda item: item[1], reverse=True))
    unique_new = translate_dict(unique, trans_dicts[i])
    #unique = {k: v for k, v in unique.items() if not len(k)==1}
    #print('unique', unique)
    if unique:  # unique_wordsが空でない場合のみWordCloudを生成
        wc = WordCloud(width=800, height=480, background_color='white')
        # unique_en = translate_list(list(unique.keys())[:100])
        # print('unique_en', unique_en)
        # for uni, unie in zip(list(unique.keys())[:100], unique_en):
        #     unique_new[unie] = unique[uni]
        print('unique', unique_new)
        wc.generate_from_frequencies(unique_new)
        ax.imshow(wc, interpolation='bilinear')
    else:
        ax.text(0.5, 0.5, "No unique words", ha='center', va='center')
    ax.axis('off')
    ax.set_title(df_names[i], fontsize=18)

plt.tight_layout()
plt.savefig('./word_cloud_12.jpg')
exit()
fig, axs = plt.subplots(2, 2, figsize=(20, 11))
for i, (unique, ax) in enumerate(zip(unique_words, axs.flatten())):
    unique_new = {}
    
    unique = dict(sorted(unique.items(), key=lambda item: item[1], reverse=True))
    print('unique orig', unique)
    unique = {k: v for k, v in unique.items() if v >= 7}
    unique = {k: v for k, v in unique.items() if not re.fullmatch(r'[!-/:-@[-`{-~]+', k)}
    unique = {k: v for k, v in unique.items() if not re.fullmatch(r'^[\u3040-\u309F]+$', k)}
    unique = dict(sorted(unique.items(), key=lambda item: item[1], reverse=True))
    unique_new = translate_dict(unique, trans_dicts[i])
    #unique = {k: v for k, v in unique.items() if not len(k)==1}
    #print('unique', unique)
    if unique:  # unique_wordsが空でない場合のみWordCloudを生成
        wc = WordCloud(width=1000, height=440, background_color='white')
        # unique_en = translate_list(list(unique.keys())[:100])
        # print('unique_en', unique_en)
        # for uni, unie in zip(list(unique.keys())[:100], unique_en):
        #     unique_new[unie] = unique[uni]
        print('unique', unique_new)
        wc.generate_from_frequencies(unique_new)
        ax.imshow(wc, interpolation='bilinear')
    else:
        ax.text(0.5, 0.5, "No unique words", ha='center', va='center')
    ax.axis('off')
    ax.set_title(df_names[i], fontsize=18)
    
plt.tight_layout()
plt.savefig('./word_cloud_12.jpg')
exit()
fig, axs = plt.subplots(1,4, figsize=(20, 5))
for i, (unique, ax) in enumerate(zip(unique_words, axs.flatten())):
    unique_new = {}
    
    unique = dict(sorted(unique.items(), key=lambda item: item[1], reverse=True))
    print('unique orig', unique)
    unique = {k: v for k, v in unique.items() if v >= 7}
    unique = {k: v for k, v in unique.items() if not re.fullmatch(r'[!-/:-@[-`{-~]+', k)}
    unique = {k: v for k, v in unique.items() if not re.fullmatch(r'^[\u3040-\u309F]+$', k)}
    unique = dict(sorted(unique.items(), key=lambda item: item[1], reverse=True))
    unique_new = translate_dict(unique, trans_dicts[i])
    #unique = {k: v for k, v in unique.items() if not len(k)==1}
    #print('unique', unique)
    if unique:  # unique_wordsが空でない場合のみWordCloudを生成
        wc = WordCloud(width=300, height=270, background_color='white')
        # unique_en = translate_list(list(unique.keys())[:100])
        # print('unique_en', unique_en)
        # for uni, unie in zip(list(unique.keys())[:100], unique_en):
        #     unique_new[unie] = unique[uni]
        print('unique', unique_new)
        wc.generate_from_frequencies(unique_new)
        ax.imshow(wc, interpolation='bilinear')
    else:
        ax.text(0.5, 0.5, "No unique words", ha='center', va='center')
    ax.axis('off')
    ax.set_title(df_names[i], fontsize=20)
    
plt.tight_layout()
plt.savefig('./word_cloud_5.jpg')
exit()

fig, axs = plt.subplots(2, 2, figsize=(20, 7))
for i, (unique, ax) in enumerate(zip(unique_words, axs.flatten())):
    unique_new = {}
    
    unique = dict(sorted(unique.items(), key=lambda item: item[1], reverse=True))
    print('unique orig', unique)
    unique = {k: v for k, v in unique.items() if v >= 7}
    unique = {k: v for k, v in unique.items() if not re.fullmatch(r'[!-/:-@[-`{-~]+', k)}
    unique = {k: v for k, v in unique.items() if not re.fullmatch(r'^[\u3040-\u309F]+$', k)}
    unique = dict(sorted(unique.items(), key=lambda item: item[1], reverse=True))
    unique_new = translate_dict(unique, trans_dicts[i])
    #unique = {k: v for k, v in unique.items() if not len(k)==1}
    #print('unique', unique)
    if unique:  # unique_wordsが空でない場合のみWordCloudを生成
        wc = WordCloud(width=1000, height=360, background_color='white')
        # unique_en = translate_list(list(unique.keys())[:100])
        # print('unique_en', unique_en)
        # for uni, unie in zip(list(unique.keys())[:100], unique_en):
        #     unique_new[unie] = unique[uni]
        print('unique', unique_new)
        wc.generate_from_frequencies(unique_new)
        ax.imshow(wc, interpolation='bilinear')
    else:
        ax.text(0.5, 0.5, "No unique words", ha='center', va='center')
    ax.axis('off')
    ax.set_title(df_names[i], fontsize=18)
    
plt.tight_layout()
plt.savefig('./word_cloud_7.jpg')
exit()


fig, axs = plt.subplots(2, 2, figsize=(20, 8))
for i, (unique, ax) in enumerate(zip(unique_words, axs.flatten())):
    unique_new = {}
    
    unique = dict(sorted(unique.items(), key=lambda item: item[1], reverse=True))
    print('unique orig', unique)
    unique = {k: v for k, v in unique.items() if v >= 5}
    unique = {k: v for k, v in unique.items() if not re.fullmatch(r'[!-/:-@[-`{-~]+', k)}
    unique = {k: v for k, v in unique.items() if not re.fullmatch(r'^[\u3040-\u309F]+$', k)}
    unique = dict(sorted(unique.items(), key=lambda item: item[1], reverse=True))
    unique_new = translate_dict(unique, trans_dicts[i])
    #unique = {k: v for k, v in unique.items() if not len(k)==1}
    #print('unique', unique)
    if unique:  # unique_wordsが空でない場合のみWordCloudを生成
        wc = WordCloud(width=800, height=300, background_color='white')
        # unique_en = translate_list(list(unique.keys())[:100])
        # print('unique_en', unique_en)
        # for uni, unie in zip(list(unique.keys())[:100], unique_en):
        #     unique_new[unie] = unique[uni]
        print('unique', unique_new)
        wc.generate_from_frequencies(unique_new)
        ax.imshow(wc, interpolation='bilinear')
    else:
        ax.text(0.5, 0.5, "No unique words", ha='center', va='center')
    ax.axis('off')
    ax.set_title(df_names[i], fontsize=24)
    
plt.tight_layout()
plt.savefig('./word_cloud_8.jpg')