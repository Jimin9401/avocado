#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH
# sciie chemprot citation_intent rct-20k

#dmis-lab/biobert-base-cased-v1.1 allenai/scibert_scivocab_uncased nfliu/scibert_basevocab_uncased google/electra-small-discriminator
#bert-base-uncased

EC=roberta-base
V=10000
Data=chemprot

for Data in citation_intent hyperpartisan_news amazon
do
  python avocado.py --dataset $Data \
    --root data \
    --vocab_size $V \
    --use_fragment \
    --encoder_class $EC;
done