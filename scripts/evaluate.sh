CHECKPOINT_PATH=$1
python eval.py evaluate_spot_names --checkpoint $CHECKPOINT_PATH >> result/metric/$CHECKPOINT_PATH.txt
python eval.py evaluate_ipp_metric --checkpoint $CHECKPOINT_PATH >> result/metric/$CHECKPOINT_PATH.txt
#python eval.py evaluate_review_metric  --checkpoint ${CHECKPOINT_PATH}
python eval.py evaluate_review_metric  --checkpoint ${CHECKPOINT_PATH}
python eval.py evaluate_review_metric  --checkpoint ${CHECKPOINT_PATH}_feature
python eval.py evaluate_review_metric  --checkpoint ${CHECKPOINT_PATH}_context
python eval.py evaluate_qa_metric  --checkpoint ${CHECKPOINT_PATH} --mode qa
python eval.py evaluate_qa_metric --checkpoint ${CHECKPOINT_PATH} --mode pvqa
python eval.py evaluate_sequential_metric --checkpoint ${CHECKPOINT_PATH} --topk False
