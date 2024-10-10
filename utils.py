import pickle
import json
import plotly.graph_objs as go
#import umap
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

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

pref_regions = {
    "北海道": "北海道地方",
    "青森県": "東北地方",
    "岩手県": "東北地方",
    "宮城県": "東北地方",
    "秋田県": "東北地方",
    "山形県": "東北地方",
    "福島県": "東北地方",
    "茨城県": "関東地方",
    "栃木県": "関東地方",
    "群馬県": "関東地方",
    "埼玉県": "関東地方",
    "千葉県": "関東地方",
    "東京都": "関東地方",
    "神奈川県": "関東地方",
    "新潟県": "中部地方",
    "富山県": "中部地方",
    "石川県": "中部地方",
    "福井県": "中部地方",
    "山梨県": "中部地方",
    "長野県": "中部地方",
    "岐阜県": "中部地方",
    "静岡県": "中部地方",
    "愛知県": "中部地方",
    "三重県": "関西地方",
    "滋賀県": "関西地方",
    "京都府": "関西地方",
    "大阪府": "関西地方",
    "兵庫県": "関西地方",
    "奈良県": "関西地方",
    "和歌山県": "関西地方",
    "鳥取県": "中国地方",
    "島根県": "中国地方",
    "岡山県": "中国地方",
    "広島県": "中国地方",
    "山口県": "中国地方",
    "徳島県": "四国地方",
    "香川県": "四国地方",
    "愛媛県": "四国地方",
    "高知県": "四国地方",
    "福岡県": "九州地方",
    "佐賀県": "九州地方",
    "長崎県": "九州地方",
    "熊本県": "九州地方",
    "大分県": "九州地方",
    "宮崎県": "九州地方",
    "鹿児島県": "九州地方",
    "沖縄県": "九州地方"
}

def unique_preserve_order(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

import numpy as np
import math


def dpp(kernel_matrix, max_length, epsilon=1E-10):
    """
    Our proposed fast implementation of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        di2s[selected_item] = -np.inf
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return selected_items


def dpp_sw(kernel_matrix, window_size, max_length, epsilon=1E-10):
    """
    Sliding window version of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param window_size: positive int
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    item_size = kernel_matrix.shape[0]
    v = np.zeros((max_length, max_length))
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    window_left_index = 0
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[window_left_index:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        v[k, window_left_index:k] = ci_optimal
        v[k, k] = di_optimal
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[window_left_index:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        if len(selected_items) >= window_size:
            window_left_index += 1
            for ind in range(window_left_index, k + 1):
                t = math.sqrt(v[ind, ind] ** 2 + v[ind, window_left_index - 1] ** 2)
                c = t / v[ind, ind]
                s = v[ind, window_left_index - 1] / v[ind, ind]
                v[ind, ind] = t
                v[ind + 1:k + 1, ind] += s * v[ind + 1:k + 1, window_left_index - 1]
                v[ind + 1:k + 1, ind] /= c
                v[ind + 1:k + 1, window_left_index - 1] *= c
                v[ind + 1:k + 1, window_left_index - 1] -= s * v[ind + 1:k + 1, ind]
                cis[ind, :] += s * cis[window_left_index - 1, :]
                cis[ind, :] /= c
                cis[window_left_index - 1, :] *= c
                cis[window_left_index - 1, :] -= s * cis[ind, :]
            di2s += np.square(cis[window_left_index - 1, :])
        di2s[selected_item] = -np.inf
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return selected_items

def overlay_attention(image_path, attention_map, output_path):
    # 画像を読み込み
    image = Image.open(image_path).convert("RGB")

    # 画像をテンソルに変換
    transform = transforms.ToTensor()
    image_tensor = transform(image)

    # アテンションマップを正規化
    attention_map = attention_map.detach().cpu().numpy()
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
    #print('attention map', attention_map)
    # アテンションマップをリサイズして、元の画像と同じサイズにする
    #print(attention_map.shape, (image_tensor.shape[2], image_tensor.shape[1]))
    attention_map = attention_map.astype(np.float32) 
    attention_map_resized = cv2.resize(attention_map, (image_tensor.shape[2], image_tensor.shape[1]))
    
    # アテンションマップをガウシアンブラーで滑らかにする
    #attention_map_resized = cv2.GaussianBlur(attention_map_resized, (11, 11), 0)

    # アテンションマップをカラー化
    cmap = plt.get_cmap("jet")
    attention_map_colored = cmap(attention_map_resized)
    attention_map_colored = np.delete(attention_map_colored, 3, 2)  # alphaチャネルを削除

    # オーバーレイのために元の画像とアテンションマップを合成
    overlay = (0.3 * np.array(image) / 255.0) + (0.7 * attention_map_colored)
    overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min())
    overlay = (overlay * 255).astype(np.uint8)

    # オーバーレイ画像を保存
    overlay_image = Image.fromarray(overlay)
    overlay_image.save(output_path)


def overlay_attention_(image_path, attention_map, output_path):
    # 画像を読み込み
    image = Image.open(image_path).convert("RGB")

    # 画像をテンソルに変換
    transform = transforms.ToTensor()
    image_tensor = transform(image)

    # アテンションマップを正規化
    attention_map = attention_map.detach().cpu().numpy()
    # attention_map = np.log10(attention_map)
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
    # print(attention_map.shape)
    # アテンションマップをリサイズして、元の画像と同じサイズにする
    attention_map_resized = Image.fromarray((attention_map * 255).astype(np.uint8))
    attention_map_resized = attention_map_resized.resize((image_tensor.shape[2], image_tensor.shape[1]), resample=Image.BILINEAR)
    attention_map_resized = np.array(attention_map_resized) / 255.0
    #attention_map_resized = cv2.GaussianBlur(attention_map_resized, (11, 11), 0)
    # アテンションマップをカラー化
    cmap = plt.get_cmap("jet")
    attention_map_colored = cmap(attention_map_resized)
    attention_map_colored = np.delete(attention_map_colored, 3, 2)  # alphaチャネルを削除

    # オーバーレイのために元の画像とアテンションマップを合成
    overlay = (0.6 * np.array(image) / 255.0) + (0.4 * attention_map_colored)
    overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min())
    overlay = (overlay * 255).astype(np.uint8)

    overlay_image = Image.fromarray(overlay)
    overlay_image.save(output_path)

