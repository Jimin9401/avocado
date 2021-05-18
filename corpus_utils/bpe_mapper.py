import tokenizers
import os
import json
import pandas as pd
from tqdm import tqdm
from util.data_builder import parse_data
import re
from tokenizers import Tokenizer

import logging

logger = logging.getLogger(__name__)



class CustomTokenizer:
    def __init__(self, args, dir_path, dataset_name, vocab_size, encoder_class):
        self.args = args
        self.dir_path = dir_path
        self.prefix = dataset_name
        self.vocab_size = vocab_size
        self.encoder = self.load_encoder(args, encoder_class, dir_path, dataset_name, vocab_size)

    def _jsonl_to_txt(self):

        data_dir = os.path.join(self.dir_path, self.prefix)
        if not os.path.isfile(os.path.join(data_dir, "train.jsonl")):
            logger.info("Download benchmark dataset")
            parse_data(self.args)

        logger.info("Flatten Corpus to text file")
        with open(os.path.join(data_dir, "train.jsonl")) as reader:
            lines = reader.readlines()
            self.data = []
            for line in lines:
                self.data.append(json.loads(line))
        df = pd.DataFrame(self.data)

        txt_file = os.path.join(data_dir, "train.txt")
        f = open(txt_file, "w")
        textlines = []

        for i, row in df.iterrows():
            new_string = re.sub('[^a-zA-Z0-9\n\.]', ' ', row["text"])
            new_string = re.sub(' +', ' ', new_string)
            textlines.append(new_string.replace("\n", " "))

        for textline in tqdm(textlines):
            f.write(textline + "\n")
    # def _ner_to_txt(self):
    #
    #     import glob
    #     data_dir = os.path.join(self.dir_path, self.prefix)
    #
    #     bio_ner_path = os.path.join(data_dir,"raw")+"/*"
    #     all_bio_dataset = glob.iglob(bio_ner_path)
    #     textlines = []
    #
    #     for dataset_path in all_bio_dataset:
    #         print(dataset_path)
    #         name = 'train.tsv'
    #         file_path=os.path.join(dataset_path,name)
    #         with open(file_path,"r") as f:
    #             entities = f.readlines()
    #         holder = []
    #         for entity in entities:
    #             try:
    #                 token, tag= entity.replace("\n","").split("\t")
    #                 holder.append((token,tag))
    #             except:
    #                 textlines.append(holder)
    #                 holder=[]
    #
    #     txt_file = os.path.join(data_dir, "train.txt")
    #     f = open(txt_file, "w")
    #
    #     textlines=[" ".join([token for token, tag in example]) for example in textlines]
    #
    #     for text in textlines:
    #         new_string = re.sub('[^a-zA-Z0-9\n\.]', ' ', text)
    #         new_string = re.sub(' +', ' ', new_string)
    #         f.write(new_string+"\n")

    def train(self):
        # if self.args.dataset=="bio_ner":
        #     self._ner_to_txt()
        # else:
        self._jsonl_to_txt()
        txt_path = os.path.join(self.dir_path, self.prefix, "train.txt")
        self.encoder.train(txt_path, vocab_size=self.vocab_size, min_frequency=1)
        self.encoder.save_model(directory=self.src_dir, prefix="{0}_{1}".format(self.prefix, str(self.vocab_size)))

    def load_encoder(self, args, encoder_class, dir_path, dataset_name, vocab_size):
        self.vocab_path = args.vocab_path
        if "uncased" in args.encoder_class:
            self.encoder = encoder_class()
        else:
            self.encoder = encoder_class(lowercase=False,strip_accents=False)

        self.vocab_size = vocab_size
        # self.vocab_dir=os.path.join(self.dir_path, self.prefix)
        self.prefix = dataset_name
        self.dir_path = dir_path

        # if not os.path.isdir(self.src_dir):
        #     os.makedirs(self.src_dir)
        self.src_dir = self.vocab_path
        base_name = os.path.join(self.src_dir, "{0}_{1}".format(self.prefix, vocab_size))
        vocab_name = base_name + '-vocab.txt'
        if os.path.exists(vocab_name):
            logger.info('\ntrained encoder loaded')
            # self.istrained = True
            return encoder_class.from_file(vocab_name)
        else:
            # self.istrained = False
            logger.info('\nencoder needs to be trained')
            self.train()
            return self.encoder
        # else:
        #     vocab_name = base_name + '-vocab.json'
        #     merge_name = base_name + '-merges.txt'
        #
        #     if os.path.exists(vocab_name) and os.path.exists(merge_name):
        #         logger.info('\ntrained encoder loaded')
        #         self.istrained = True
        #         if encoder_class == tokenizers.SentencePieceBPETokenizer:
        #             return encoder_class(vocab_name, merge_name)
        #         else:
        #             return encoder_class(vocab_name, merge_name)
        #     else:
        #         self.istrained = False
        #         logger.info('\n encoder needs to be trained')
        #         self.train()
        #         if encoder_class == tokenizers.SentencePieceBPETokenizer:
        #             return encoder_class(vocab_name, merge_name)
        #         else:
        #             return self.encoder
