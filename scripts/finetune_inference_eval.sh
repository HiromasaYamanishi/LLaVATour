OUTPUT_DIR=$1 # e.g ./checkpoints/llava-v1.5-13b-jalan-review-lora-v15_5epoch
RESUME_DIR=$2 # e.g ./checkpoints/llava-v1.5-13b-jalan-review-lora-v14_4epoch
TRAIN_FILE=$3 # e.g ./playground/data/v15/train.json
OUTPUT_SUFFIX=$(basename "$OUTPUT_DIR")
echo $OUTPUT_DIR
echo $RESUME_DIR
echo $OUTPUT_SUFFIX
bash scripts/v1_5/finetune_lora_jalan_resume.sh $OUTPUT_DIR $RESUME_DIR $TRAIN_FILE
bash scripts/inference.sh $OUTPUT_DIR
CONDA_BASE=$(conda info --base)

# Condaを初期化
source "$CONDA_BASE/etc/profile.d/conda.sh"
# 環境をアクティブ化
conda activate airp
bash scripts/evaluate.sh $OUTPUT_SUFFIX