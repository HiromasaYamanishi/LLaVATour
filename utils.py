import pickle
import json
import plotly.graph_objs as go
import umap

color_names = """aliceblue, maroon, aqua, aquamarine, azure,
            ivory, bisque, black, lightyellow, blue,
            blueviolet, brown, burlywood, cadetblue,
            chartreuse, chocolate, coral, cornflowerblue,
            cornsilk, crimson, cyan, darkblue, darkcyan,
            darkgoldenrod, darkgray, darkgrey, darkgreen,
            darkkhaki, darkmagenta, darkolivegreen, darkorange,
            darkorchid, darkred, darksalmon, darkseagreen,
            darkslateblue, darkslategray, darkslategrey,
            darkturquoise, darkviolet, deeppink, deepskyblue,
            dimgray, dimgrey, dodgerblue, firebrick,
            floralwhite, forestgreen, fuchsia, gainsboro,
            ghostwhite, gold, goldenrod, gray, grey, green,
            greenyellow, honeydew, hotpink, indianred, indigo,
            beige, khaki, lavender, lavenderblush, lawngreen,
            lemonchiffon, lightblue, lightcoral, lightcyan,
            lightgoldenrodyellow, lightgray, lightgrey,
            lightgreen, lightpink, lightsalmon, lightseagreen,
            lightskyblue, lightslategray, lightslategrey,
            lightsteelblue, lime, limegreen,
            linen, magenta, antiquewhite, mediumaquamarine,
            mediumblue, mediumorchid, mediumpurple,
            mediumseagreen, mediumslateblue, mediumspringgreen,
            mediumturquoise, mediumvioletred, midnightblue,
            mintcream, mistyrose, moccasin, navajowhite, navy,
            oldlace, olive, olivedrab, orange, orangered,
            orchid, palegoldenrod, palegreen, paleturquoise,
            palevioletred, papayawhip, peachpuff, peru, pink,
            plum, powderblue, purple, red, rosybrown,blanchedalmond,
            royalblue, rebeccapurple, saddlebrown, salmon,
            sandybrown, seagreen, seashell, sienna, silver,
            skyblue, slateblue, slategray, slategrey, snow,
            springgreen, steelblue, tan, teal, thistle, tomato,
            turquoise, violet, wheat, white, whitesmoke,
            yellow, yellowgreen"""

# コンマで分割してリストに変換
color_list = [color.strip() for color in color_names.split(",")]
colors = [color for color in color_list if color]

def write_numbers(array, file):
    with open(file, 'w') as f:
        f.write(' '.join([str(i) for i in array]))
        
def save_pkl(obj, f):
    with open(f, 'wb') as f:
        pickle.dump(obj, f)
        
def load_pkl(f):
    with open(f, 'rb') as f:
        obj = pickle.load(f)
        
    return obj

def save_json(obj, file_name):
    with open(file_name, 'w') as f:
        json.dump(obj, f)
        
def load_json(file_name):
    with open(file_name) as f:
        d = json.load(f)
    return d

def plot_2d(embs, labels, texts, save_path=None):
    pca=umap.UMAP(random_state=0, n_neighbors=10, min_dist=0.05)
    emb_2d = pca.fit_transform(embs)
        
    colors_2d = [colors[l] for l in labels]

    trace = go.Scatter(
        x=emb_2d[:, 0],
        y=emb_2d[:, 1],
        mode='markers',
        text=texts,  # ホバーテキストとして文書の内容を使用
        hoverinfo='text',  # ホバー時にはテキストのみを表示fw∂
        marker=dict(color=colors_2d)
    )
    fig = go.Figure(trace)
    if save_path is not None:
        fig.write_html(save_path)
        
def get_prefecture_to_region():    
    regions = {
        '北海道地方': ['北海道'],
        '東北地方': ['青森県', '岩手県', '宮城県', '秋田県', '山形県', '福島県'],
        '関東地方': ['茨城県', '栃木県', '群馬県', '埼玉県', '千葉県', '東京都', '神奈川県'],
        '中部地方': ['新潟県', '富山県', '石川県', '福井県', '山梨県', '長野県', '岐阜県', '静岡県', '愛知県'],
        '近畿地方': ['三重県', '滋賀県', '京都府', '大阪府', '兵庫県', '奈良県', '和歌山県'],
        '中国地方': ['鳥取県', '島根県', '岡山県', '広島県', '山口県'],
        '四国地方': ['徳島県', '香川県', '愛媛県', '高知県'],
        '九州地方': ['福岡県', '佐賀県', '長崎県', '熊本県', '大分県', '宮崎県', '鹿児島県', '沖縄県']
    }

    # Flipping the dictionary to map prefectures to their region
    prefecture_to_region = {}
    for region, prefectures in regions.items():
        for prefecture in prefectures:
            prefecture_to_region[prefecture] = region

    return prefecture_to_region