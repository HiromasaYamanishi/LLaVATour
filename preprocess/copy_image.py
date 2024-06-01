import pandas as pd
import os
import shutil
from tqdm import tqdm
import subprocess

# df_orig = pd.read_csv('/home/yamanishi/project/trip_recommend/data/jalan/spot/text_image_pairs.csv',
#                       names=['image_url', 'text', 'spot_name', 'ind'])

# ind2url = {spot_name + '_' + str(ind)+'.jpg':url.split('/')[-1] for spot_name,ind,url in zip(df_orig['spot_name'], df_orig['ind'], df_orig['image_url']) if not pd.isna(spot_name) and not pd.isna(url)}
# print('Start')
# for ind in tqdm(list(ind2url.keys())[689713:]):
#     image_path = os.path.join('/home/yamanishi/project/trip_recommend/data/jalan_image_with_caption', ind)
#     spot_name = ind.split('_')[0]
#     os.makedirs(f'/home/yamanishi/project/airport/src/data/jalan_image/{spot_name}', exist_ok=True)
#     if not os.path.exists(image_path):continue
#     destination_path = os.path.join(f'/home/yamanishi/project/airport/src/data/jalan_image/{spot_name}', ind2url[ind])
#     if os.path.exists(destination_path):continue
#     # if os.path.exists(os.path.join(f'/home/yamanishi/project/airport/src/data/jalan_image/{spot_name}', ind2url[ind])):continue
#     # shutil.copy(image_path, os.path.join(f'/home/yamanishi/project/airport/src/data/jalan_image/{spot_name}', ind2url[ind]))
    
#     subprocess.run(['cp', image_path, destination_path], check=True)
#     #break
    
import subprocess
import os

# ここに`ind2url`の定義や、他の必要なセットアップを追加してください
#@profile
def copy_file():
    df_orig = pd.read_csv('/home/yamanishi/project/trip_recommend/data/jalan/spot/text_image_pairs.csv',
                    names=['image_url', 'text', 'spot_name', 'ind'])

    ind2url = {spot_name + '_' + str(ind)+'.jpg':url.split('/')[-1] for spot_name,ind,url in zip(df_orig['spot_name'], df_orig['ind'], df_orig['image_url']) if not pd.isna(spot_name) and not pd.isna(url)}
    for ind in tqdm(list(ind2url.keys())[689940+20:]):
        image_path = os.path.join('/home/yamanishi/project/trip_recommend/data/jalan_image_with_caption', ind)
        spot_name = ind.split('_')[0]
        os.makedirs(f'/home/yamanishi/project/airport/src/data/jalan_image/{spot_name}', exist_ok=True)
        if not os.path.exists(image_path):
            print(image_path)
            continue
        destination_path = os.path.join(f'/home/yamanishi/project/airport/src/data/jalan_image/{spot_name}', ind2url[ind])
        if os.path.exists(destination_path):continue
        # if os.path.exists(os.path.join(f'/home/yamanishi/project/airport/src/data/jalan_image/{spot_name}', ind2url[ind])):continue
        # shutil.copy(image_path, os.path.join(f'/home/yamanishi/project/airport/src/data/jalan_image/{spot_name}', ind2url[ind]))
        
        subprocess.run(['cp', image_path, destination_path], check=True)

if __name__ == "__main__":
    # ここで`copy_file`を呼び出すためのテストケースを設定
    copy_file()