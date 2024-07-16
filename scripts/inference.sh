#!/bin/bash

CHECKPOINT_DIR=$1

# 関数: コマンドを実行し、終了を待つ
run_command() {
    echo "Starting: $1"
    eval "$1"
    echo "Finished: $1"
}

# Landmark Recognition
run_command "CUDA_VISIBLE_DEVICES=0 python inference.py --model-path $CHECKPOINT_DIR --model-base lmsys/vicuna-13b-v1.5 --model_name llavatour -f inference_spot_names" &
pid1=$!

# Review Generation
run_command "CUDA_VISIBLE_DEVICES=6 python inference.py --model-path $CHECKPOINT_DIR --model-base lmsys/vicuna-13b-v1.5 -f review_generation --model_name llavatour" &
pid2=$!

# Context Review Generationの終了を待つ
#wait $pid3

# Context Review Generation
run_command "CUDA_VISIBLE_DEVICES=1 python inference.py --model-path $CHECKPOINT_DIR --model-base lmsys/vicuna-13b-v1.5 -f review_generation --model_name llavatour --use_context" &
pid3=$!

run_command "CUDA_VISIBLE_DEVICES=7 python inference.py --model-path $CHECKPOINT_DIR --model-base lmsys/vicuna-13b-v1.5 -f review_generation --model_name llavatour --use_feature" &
pid4=$!

# PVQA
run_command "CUDA_VISIBLE_DEVICES=2 python inference.py --model-path $CHECKPOINT_DIR --model-base lmsys/vicuna-13b-v1.5 -f inference_pvqa --model_name llavatour" &
pid5=$!

# Image Popularity Prediction
run_command "CUDA_VISIBLE_DEVICES=3 python inference.py --model-path $CHECKPOINT_DIR --model-base lmsys/vicuna-13b-v1.5 -f inference_ipp --model_name llavatour" &
pid6=$!

run_command "CUDA_VISIBLE_DEVICES=4 python inference.py --model-path $CHECKPOINT_DIR --model-base lmsys/vicuna-13b-v1.5 -f inference_qa --model_name llavatour" &
pid7=$!

run_command "CUDA_VISIBLE_DEVICES=5 python inference.py --model-path $CHECKPOINT_DIR --model-base lmsys/vicuna-13b-v1.5 -f inference_sequential --model_name llavatour" &
pid8=$!
# Landmark Recognitionの終了を待つ
#wait $pid1

# すべてのバックグラウンドプロセスの終了を待つ
wait