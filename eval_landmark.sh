for epoch in {15..25}
do
    CUDA_VISIBLE_DEVICES=1 python landmark_recognition.py --exp_id exp0 --eval_only --checkpoint ./checkpoint/landmark_recognition/exp0/epoch_${epoch}.pth
done