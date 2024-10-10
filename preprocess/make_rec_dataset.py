import pandas as pd
from collections import defaultdict
import random
from tqdm import tqdm
import pickle

# 都道府県を地方にマッピングする辞書
pref_to_region = {
    '北海道': '北海道', '青森県': '東北', '岩手県': '東北', '宮城県': '東北', '秋田県': '東北', '山形県': '東北', '福島県': '東北',
    '茨城県': '関東', '栃木県': '関東', '群馬県': '関東', '埼玉県': '関東', '千葉県': '関東', '東京都': '関東', '神奈川県': '関東',
    '新潟県': '中部', '富山県': '中部', '石川県': '中部', '福井県': '中部', '山梨県': '中部', '長野県': '中部', '岐阜県': '中部', '静岡県': '中部', '愛知県': '中部',
    '三重県': '近畿', '滋賀県': '近畿', '京都府': '近畿', '大阪府': '近畿', '兵庫県': '近畿', '奈良県': '近畿', '和歌山県': '近畿',
    '鳥取県': '中国', '島根県': '中国', '岡山県': '中国', '広島県': '中国', '山口県': '中国',
    '徳島県': '四国', '香川県': '四国', '愛媛県': '四国', '高知県': '四国',
    '福岡県': '九州', '佐賀県': '九州', '長崎県': '九州', '熊本県': '九州', '大分県': '九州', '宮崎県': '九州', '鹿児島県': '九州', '沖縄県': '沖縄'
}

df = pd.read_pickle('../../../data/review_all_period_.pkl')

def add_region_to_df(df):
    df['region'] = df['pref'].map(pref_to_region)
    return df

def make_profile(user_data):
    sex = user_data[0]['sex']
    age = user_data[0]['age']
    if pd.isna(user_data[0]['tag']):
        tag = ''
    else:
        tag = user_data[0]['tag']
        #tags = set(tag for visit in user_data for tag in visit['tag'].split(','))
    reviews = ' '.join(visit['review'] for visit in user_data)
    
    # ここでは簡単なプロファイル作成を行っていますが、より高度な処理を追加できます
    profile = f"{sex}の{age}。{tag} レビュー特徴: {reviews[:100]}..."
    return profile

def extract_routes(df, time_threshold='1D'):
    print("経路の抽出を開始します...")
    df['visit_time'] = pd.to_datetime(df['visit_time'], format='%Y年%m月%d日', errors='coerce')
    df['visit_time'] = df['visit_time'].fillna(pd.to_datetime(df['visit_time'], format='%Y年%m月', errors='coerce'))
    
    df = df.sort_values(['url', 'visit_time'])
    routes = defaultdict(list)
    current_route = []
    current_user = None
    current_pref = None
    last_visit_time = None

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="経路抽出"):
        if current_user != row['url'] or current_pref != row['pref'] or (last_visit_time and (row['visit_time'] - last_visit_time) > pd.Timedelta(time_threshold)):
            if current_route and len(current_route) >= 3:
                routes[current_user].append(current_route)
            current_route = []
            current_user = row['url']
            current_pref = row['pref']
        
        current_route.append(row)
        last_visit_time = row['visit_time']

    if current_route:
        routes[current_user].append(current_route)

    print(f"経路の抽出が完了しました。{len(routes)}人のユーザーの経路を抽出しました。")
    return routes

def precompute_area_spots(df):
    df = add_region_to_df(df)
    pref_spots = df.groupby(['region', 'pref'])['spot'].apply(set).to_dict()
    region_spots = df.groupby('region')['spot'].apply(set).to_dict()
    all_spots = set(df['spot'])
    return {
        'pref': pref_spots,
        'region': region_spots,
        'all': all_spots
    }
route_prompt_templates = [
    "ユーザープロファイル: {}。{}地方を訪れる予定です。おすすめの観光ルートを提案してください。",
    "{}さんが{}県を旅行します。最適な観光スポットの順序を教えてください。",
    "プロファイル: {}。{}地域での理想的な観光ルートを5つの観光地で作成してください。",
    "{}。{}県内でのおすすめの1日観光プランを立ててください。",
    "ユーザー情報: {}。{}地方での3日間の旅行プランを提案してください。各日の観光スポットを列挙してください。",
    "{}のユーザーが{}を訪れます。以下の観光地から最適なルートを作成してください: {}",
    "ユーザープロファイル: {}。{}県内の観光地リスト: {}。これらの中から最適な周遊ルートを提案してください。",
    "{}さんの{}地方旅行計画。候補地: {}。2泊3日の旅程を組んでください。",
    "プロファイル: {}。{}県の観光スポット候補: {}。日帰り旅行の効率的なルートを提案してください。",
    "{}。{}地域での週末旅行。以下の観光地からベストな順序を選んでください: {}"
]

def determine_restriction(route):
    prefs = set(visit['pref'] for visit in route)
    regions = set(pref_to_region[pref] for pref in prefs)

    if len(prefs) == 1:
        return 'pref', next(iter(prefs))
    elif len(regions) == 1:
        return 'region', next(iter(regions))
    else:
        return 'all', 'Japan'

def generate_route_dataset(routes, precomputed_spots, include_profile=True, include_history=True):
    prompts, answers = [], []

    print("データセットの生成を開始します...")
    for user, user_routes in tqdm(routes.items(), desc="データセット生成"):
        profile = make_profile([visit for route in user_routes for visit in route]) if include_profile else ""
        
        for route in user_routes:
            sequence = [visit['spot'] for visit in route]
            restriction_type, area = determine_restriction(route)
            
            prompt_type = random.randint(0, 9)
            prompt = route_prompt_templates[prompt_type]

            if restriction_type == 'pref':
                region = pref_to_region[area]
                area_spots = precomputed_spots['pref'].get((region, area), set())
            elif restriction_type == 'region':
                area_spots = precomputed_spots['region'].get(area, set())
            else:
                area_spots = precomputed_spots['all']

            if prompt_type <= 4:
                prompt = prompt.format(profile, area)
            else:
                candidate_samples = list(area_spots - set(sequence))
                candidate_num = min(random.randint(19, 99), len(candidate_samples))
                candidate_samples = random.sample(candidate_samples, candidate_num)
                candidate_samples.extend(sequence)
                random.shuffle(candidate_samples)
                candidates_str = ' , '.join(candidate_samples)
                prompt = prompt.format(profile, area, candidates_str)

            if include_history:
                past_visits = [visit['spot'] for user_route in user_routes for visit in user_route if visit['spot'] not in sequence]
                prompt += f" 過去の訪問履歴: {' -> '.join(past_visits)}"

            prompts.append(prompt)
            answers.append(' -> '.join(sequence))

    print(f"データセットの生成が完了しました。{len(prompts)}個のサンプルを生成しました。")
    return prompts, answers

# メイン処理
print("処理を開始します...")

print("データフレームから経路を抽出しています...")
routes = extract_routes(df)

print("エリアごとの観光地を事前計算しています...")
precomputed_spots = precompute_area_spots(df)

print("ユーザーを訓練用とテスト用に分割しています...")
users = list(routes.keys())
random.shuffle(users)
split_point = int(len(users) * 0.7)
train_users = users[:split_point]
test_users = users[split_point:]

print(f"訓練用ユーザー: {len(train_users)}人, テスト用ユーザー: {len(test_users)}人")

train_routes = {user: routes[user] for user in train_users}
test_routes = {user: routes[user] for user in test_users}

print("訓練用データセットを生成しています...")
train_route_prompts, train_route_answers = generate_route_dataset(train_routes, precomputed_spots)

print("テスト用データセットを生成しています...")
test_route_prompts, test_route_answers = generate_route_dataset(test_routes, precomputed_spots)

print(f"処理が完了しました。訓練用サンプル: {len(train_route_prompts)}, テスト用サンプル: {len(test_route_answers)}")


with open('./recommend/rec_route.pkl', 'wb') as f:
    pickle.dump({'train_prompt': train_route_prompts, 'test_prompt': test_route_prompts,
                 'train_answer': train_route_answers, 'test_answer': test_route_answers}, f)