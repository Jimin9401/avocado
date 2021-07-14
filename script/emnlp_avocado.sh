#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH
EC=allenai/cs_roberta_base # specify encoder
Data=citation_intent # specify dataset
V=10000
NGPU=4
T=3.0
LI=5
AT=simclr
CHECKPOINT=../ # specify checkpoint

for Data in citation_intent
do
  for T in 0.5 1.0 1.5 2.0 2.5 3.0
  do
    for S in 1994 1996 2015 1113 777
      do
      CUDA_VISIBLE_DEVICES=$NGPU python run_classifier.py \
          --dataset $Data \
          --root data \
          --do_train \
          --seed $S \
          --lr 1.0e-5 \
          --gradient_accumulation_step 1 \
          --per_gpu_train_batch_size 16 \
          --n_epoch 50 \
          --vocab_size $V \
          --merge_version \
          --temperature $T \
          --mixed_precision \
          --layer_index $LI \
          --use_fragment \
          --transfer_type average_input \
          --evaluate_during_training \
          --checkpoint_dir $CHECKPOINT \
          --contrastive \
          --align_type $AT \
          --prototype average \
          --encoder_class $EC ;
      done
      CUDA_VISIBLE_DEVICES=$NGPU python run_classifier.py \
      --dataset $Data \
      --root data \
      --do_test \
      --seed 777 \
      --lr 1.0e-5 \
      --seed_list 1994 1996 2015 1113 777 \
      --gradient_accumulation_step 1 \
      --per_gpu_train_batch_size 16 \
      --n_epoch 10 \
      --use_fragment \
      --vocab_size $V \
      --merge_version \
      --contrastive \
      --transfer_type average_input \
      --temperature $T \
      --mixed_precision \
      --align_type $AT \
      --layer_index $LI \
      --prototype average \
      --test_log_dir=test_log \
      --evaluate_during_training \
      --checkpoint_dir $CHECKPOINT \
      --encoder_class $EC ;
  done
done

## test

#for Data in chemprot hyperpartisan_news citation_intent
#do
#  CUDA_VISIBLE_DEVICES=$NGPU python run_classifier.py \
#    --dataset $Data \
#    --root data \
#    --do_test \
#    --seed 777 \
#    --lr 1.0e-5 \
#    --seed_list 1994 1996 2015 1113 777 \
#    --gradient_accumulation_step 1 \
#    --per_gpu_train_batch_size 16 \
#    --n_epoch 10 \
#    --use_fragment \
#    --vocab_size $V \
#    --merge_version \
#    --contrastive \
#    --transfer_type average_input \
#    --temperature $T \
#    --mixed_precision \
#    --align_type $AT \
#    --layer_index $LI \
#    --prototype average \
#    --test_log_dir=test_log \
#    --evaluate_during_training \
#    --checkpoint_dir $CHECKPOINT \
#    --encoder_class $EC ;
#done