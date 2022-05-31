

# AVocaDo : Strategy for Adapting Vocabulary to Downstream Domain

This repository contains the official implementation of the EMNLP 2021 paper "AVocaDo : Strategy for Adapting Vocabulary to Downstream Domain".

An Domain-adaption approach by expanding domain-specified vocabulary using "fragment score". 
AVocaDo can adapt various domain without requiring external resources. Check out our [paper](https://arxiv.org/abs/2110.13434) and more information.




pytorch implementation

| Table of Contents |
|-|
| [Setup](#setup)|
| [Vocabulary](#vocabulary)|
| [Training](#training)|
| [Evaluation](#evaluation)|
| [Result](#result)|


## Setup
### Dependencies

Install other dependecies:
```bash
conda create -n avocado python=3.8
conda activate avocado

pip install -r requirement.txt
mkdir data
mkdir checkpoint
```


We implemented with mixed precision using [apex](https://github.com/NVIDIA/apex) (pytorch-extension library)

```bash
cd ../
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir ./
cd ../avocado
```


* We tested these script using 3090 RTX 24GB gpu in single and training with mixed precision.
Overall descriptions of script:

| implementation                                                 | Script      | Description                                                  |
| ------------------------------------------------------------ | ----------- | ------------------------------------------------------------ |
| finetune vocabulary with AVocaDo | script/emnlp_gen_vocab.sh       | generate downstream-domain-specific vocabulary | 
| baseline | script/emnlp_baseline.sh       |  baseline implementation | 
| * AVocaDo with regularization (Our)| script/emnlp_avocado.sh | require new merge vocab |


## Vocabulary
A script(`script/emnlp_gen_vocab.sh`) performs generate new vocab for each dataset.
```bash
EC=bert-base-uncased
V=10000
Data= citation_intent
python avocado.py --dataset $Data \
  --root data \
  --vocab_size $V \
  --encoder_class $EC;
```




## Training 
A script(`script/emnlp_baseline.sh`) performs baselines.
A script(`script/emnlp_avocado.sh`) performs train with our method
* We tested these script using 3090 RTX 24GB gpu in single and training with mixed precision.
If you get OOM errors, try decreasing ```batch_size```.

```bash
export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH
EC=bert-base-uncased # specify encoder
Data=citation_intent # specify dataset
V=10000
NGPU=0 
T=3.5
LI=5
AT=simclr
CHECKPOINT=../ # specify checkpoint

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
      --n_epoch 10 \
      --vocab_size $V \
      --merge_version \
      --temperature $T \
      --mixed_precision \
      --layer_index $LI \
      --use_fragment \
      --evaluate_during_training \
      --checkpoint_dir $CHECKPOINT \
      --contrastive \
      --align_type $AT \
      --prototype $R \
      --encoder_class $EC ;
```



## Evaluation

* We test at five random seed and calculate average score.

```bash

export PYTHONPATH="${PYTHONPATH}:./"
echo $PYTHONPATH


EC=bert-base-uncased # specify encoder
Data=citation_intent # specify dataset
V=10000
NGPU=0 
T=3.5
LI=5
AT=simclr
CHECKPOINT=../ # specify checkpoint
TESTLOG=test_log

CUDA_VISIBLE_DEVICES=$NGPU python run_classifier.py \
  --dataset $Data \
  --root data \
  --do_test \
  --seed 777 \
  --lr 1.0e-5 \
  --seed_list 1994 1996 2015 1113 777 \ # specify candidate random_seed
  --gradient_accumulation_step 1 \
  --per_gpu_train_batch_size 16 \
  --n_epoch 10 \
  --use_fragment \
  --vocab_size $V \
  --merge_version \
  --contrastive \
  --temperature $T \
  --mixed_precision \
  --align_type $AT \
  --layer_index $LI \
  --prototype average \
  --test_log_dir $TESTLOG \
  --evaluate_during_training \
  --checkpoint_dir $CHECKPOINT \
  --encoder_class $EC ;
```
