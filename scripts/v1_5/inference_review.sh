GPU=$1

versions=($2)

# 配列の各要素に対してループ
for version in "${versions[@]}"
do
    # モデルパスのバージョン部分を更新してpythonスクリプトを実行
    CUDA_VISIBLE_DEVICES=${GPU} python inference_rec.py -f inference_review --model_path "./checkpoints/llava-v1.5-13b-jalan-review-lora-v8.$version" --init_user 0 --end_user 1000
done


