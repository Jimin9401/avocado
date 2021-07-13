from transformers import AutoTokenizer,BertTokenizer,RobertaTokenizer
from corpus_utils.bpe_mapper import Tokenizer
import torch
import pandas as pd
import os
import logging
logger=logging.getLogger(__name__)

def corpuswise_compare(pretrained_tokenizer:BertTokenizer,domain_tokenizer):
    pretrained_vocab=pretrained_tokenizer.get_vocab()
    domain_vocab = domain_tokenizer.get_vocab()

    new_vocab = [k for k, v in domain_vocab.items() if k not in pretrained_vocab]

    logger.info("\n Additional vocab size %d"%(len(new_vocab)))
    return new_vocab
def domain2pretrain(domain_vocab:list,pretrained_tokenizer,vocab_path):

    d2p=dict()
    initial_embedding_id=len(pretrained_tokenizer)

    for embedding_id,key in enumerate(domain_vocab):
        if "##" in key:
            tmp_key = key.replace("##","아")
            values=pretrained_tokenizer.tokenize(tmp_key)
            d2p[key]=(initial_embedding_id+embedding_id,values[2:],pretrained_tokenizer.convert_tokens_to_ids(values[2:]))
        else:
            values=pretrained_tokenizer.tokenize(key)
            d2p[key]=(initial_embedding_id+embedding_id,values,pretrained_tokenizer.convert_tokens_to_ids(values))

    logger.info("\n Save domain vocab to pretrained vocab mapper %s" %(vocab_path))

    pd.to_pickle(d2p,os.path.join(vocab_path,"d2p.pickle"))

    return d2p


def merge_domain_vocab(args,tokenizer:RobertaTokenizer,config,domain_vocab:dict,vocab_path):

    if not os.path.isdir(vocab_path):
        os.makedirs(vocab_path)


    pretrained_vocab=tokenizer.get_vocab()

    new_vocab=[key for key,v in domain_vocab.items() if key not in pretrained_vocab]

    # tokenizer.save_pretrained(vocab_path)
    config.save_pretrained(vocab_path)

    logger.info("\n Merge domain vocab and pretrained vocab at %s" %(vocab_path) )

    f = open(os.path.join(vocab_path, "vocab.txt"), "a")

    for key in new_vocab:
        f.write(key+"\n")
    f.close()

def load_merge_vocab(tokenizer_class,vocab_path):

    return tokenizer_class.from_pretrained(vocab_path)








