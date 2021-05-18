#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH
# sciie chemprot citation_intent rct-20k

#dmis-lab/biobert-base-cased-v1.1 allenai/scibert_scivocab_uncased nfliu/scibert_basevocab_uncased

EC=bert-base-uncased
V=10000
Data=citation_intent

python avocado.py --dataset $Data \
  --root data \
  --vocab_size $V \
  --use_fragment \
  --encoder_class $EC;
