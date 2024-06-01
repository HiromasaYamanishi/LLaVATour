source /home/yamanishi/.pyenv/versions/miniconda3-latest/etc/profile.d/conda.sh
conda activate llava
CUDA_VISIBLE_DEVICES=0 python inference.py -f llavatour_inference --model-path /home/yamanishi/project/airport/src/analysis/LLaVA/checkpoints/llava-v1.5-13b-jalan-review-lora-v4_split --model-base lmsys/vicuna-13b-v1.5
source /home/yamanishi/.pyenv/versions/miniconda3-latest/etc/profile.d/conda.sh
conda activate airp
CUDA_VISIBLE_DEVICES=0 python eval.py -f llavatour_eval --model-path /home/yamanishi/project/airport/src/analysis/LLaVA/checkpoints/llava-v1.5-13b-jalan-review-lora-v4_split --model-base lmsys/vicuna-13b-v1.5
