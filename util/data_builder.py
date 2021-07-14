import wget
from .preprocessor import *
from .dataset import DATASETS
from transformers import BertTokenizer

import pandas as pd
import logging
from collections import Counter
import numpy as np

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)

jsonl_data = ["train.jsonl", "dev.jsonl", "test.jsonl"]


def parse_data(args):
    url = DATASETS[args.dataset]["data_dir"]

    data_dir = os.path.join(args.root, args.dataset)

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    for jsonl in jsonl_data:
        json_file = os.path.join(data_dir, jsonl)
        wget.download(url + jsonl, json_file)


def get_dataset(args, tokenizer: BertTokenizer):
    res = {}
    data_dir = os.path.join(args.root, args.dataset)
    cache_dir = os.path.join(data_dir, "cache")

    if args.merge_version:
        cache_dir += "_merged"
    elif args.other:
        cache_dir += "_other"

    dataset_dir = os.path.join(cache_dir, args.encoder_class)

    if args.contrastive:
        dataset_dir += "-contrastive-"

    if args.merge_version:
        if args.use_fragment:
            dataset_dir += "{0}_optimized".format(args.vocab_size)
        else:
            dataset_dir += "_{0}".format(args.vocab_size)


    if not os.path.isdir(data_dir):
        parse_data(args)

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)

        # logger.info("tokenize data to %s"%(str(dataset_dir)) )
        for dataset_name in jsonl_data:
            d_name = dataset_name.split(".")[0]
            train = "train" == d_name

            if args.contrastive:
                assert isinstance(tokenizer, tuple)
                pretrained_tokenizer, domain_tokenizer = tokenizer
                builder = ContrastivePreprocessor(data_dir, dataset_name, pretrained_tokenizer, domain_tokenizer,
                                                 train)
            else:
                builder = BasicPreprocesor(data_dir, dataset_name, tokenizer, train)

                if args.exbert:
                    builder._tokenize=builder._tokenize_exbert

            dataset = builder.parse()
            dataset.to_pickle(os.path.join(dataset_dir, dataset_name))
            res[d_name] = dataset

        res["label"] = builder.label_map
    else:
        # logger.info("load data from %s"%(str(dataset_dir)) )

        for f in jsonl_data:
            d_name = f.split(".")[0]
            res[d_name] = pd.read_pickle(os.path.join(dataset_dir, f))
        res["label"] = pd.read_pickle(os.path.join(data_dir, "label.json"))

    return res["train"], res["dev"], res["test"], res["label"]

