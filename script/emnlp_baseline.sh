
EC=bert-base-uncased # specify encoder
Data=citation_intent # specify dataset
NGPU=0
CHECKPOINT=../ # specify checkpoint


## train at five seed

for S in 1994 1996 2015 1113 777
do
CUDA_VISIBLE_DEVICES=$NGPU python run_classifier.py \
    --dataset $Data \
    --root data \
    --do_train \
    --seed $S \
    --lr 5.0e-5 \
    --gradient_accumulation_step 1 \
    --per_gpu_train_batch_size 16 \
    --n_epoch 10 \
    --mixed_precision \
    --evaluate_during_training \
    --checkpoint_dir $CHECKPOINT \
    --encoder_class $EC ;
done



## test

CUDA_VISIBLE_DEVICES=$NGPU python run_classifier.py \
  --dataset $Data \
  --root data \
  --do_test \
  --lr 5.0e-5 \
  --seed_list 1994 1996 2015 1113 777 \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 256 \
  --mixed_precision \
  --evaluate_during_training \
  --checkpoint_dir $CHECKPOINT \
  --test_log_dir=test_log \
  --encoder_class $EC;
