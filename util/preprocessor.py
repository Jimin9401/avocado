import os
from transformers import BertTokenizer
import json
from tqdm import tqdm
import pandas as pd
from os import getpid
from .multi_processor import multi_preprocess
import glob


def load_and_preprocess(dataset_path, tokenizer, featurizer, domain_tokenizer=None):
    key = dataset_path.split("/")[-1]
    dataset = []

    with open(dataset_path, "r") as f:
        entities = f.readlines()
    holder = []
    for entity in entities:
        try:
            token, tag = entity.replace("\n", "").split("\t")
            holder.append((token, tag))
        except:
            dataset.append(holder)
            holder = []
    out = featurizer(dataset, tokenizer, domain_tokenizer)

    return (key, out)

class BasicPreprocesor(object):
    def __init__(self, data_dir, data_name, tokenizer: BertTokenizer, train=False):
        self.data_dir = data_dir
        with open(os.path.join(data_dir, data_name)) as reader:
            lines = reader.readlines()
            self.data = []
            for line in lines:
                self.data.append(json.loads(line))
        self.df = pd.DataFrame(self.data)
        self.tokenizer = tokenizer

        if train:
            label = self.df.label.value_counts().keys()
            self.label_map = {label[i]: i for i in range(len(label))}
            pd.to_pickle(self.label_map, os.path.join(self.data_dir, "label.json"))
        else:
            self.label_map = pd.read_pickle(os.path.join(self.data_dir, "label.json"))

    def parse(self):
        self.df["lens"] = [len(t) for t in self.df["text"].to_list()]
        res = []
        for i, row in tqdm(self.df.iterrows(), total=len(self.df)):

            res.append({"text": self._tokenize(row["text"]), "label": self.label_map[row["label"]]})

        return pd.DataFrame(res)

    def _tokenize(self, text):

        return self.tokenizer.encode(text)

    def _tokenize_exbert(self, text):

        tokens = self.tokenizer.tokenize(text)
        return self.tokenizer.convert_tokens_to_ids(tokens)



class ContrastivePreprocessor(BasicPreprocesor):
    def __init__(self, data_dir, data_name, tokenizer: BertTokenizer, domain_tokenizer, train=False):
        super(ContrastivePreprocessor, self).__init__(data_dir, data_name, tokenizer, train)
        # self.data_dir = data_dir
        self.domain_tokenizer = domain_tokenizer

    def parse(self):
        res = []
        self.df["lens"] = [len(t) for t in self.df["text"].to_list()]

        for i, row in tqdm(self.df.iterrows(), total=len(self.df)):
            res.append(
                {"text": self.tokenizer.encode(row["text"]), "domain_text": self.domain_tokenizer.encode(row["text"]),
                 "label": self.label_map[row["label"]]})

        return pd.DataFrame(res)

    def _tokenize(self, text):
        return self.tokenizer.encode(text)



class ContrastivePreprocessor(BasicPreprocesor):
    def __init__(self, data_dir, data_name, tokenizer: BertTokenizer, domain_tokenizer, train=False):
        super(ContrastivePreprocessor, self).__init__(data_dir, data_name, tokenizer, train)
        # self.data_dir = data_dir
        self.domain_tokenizer = domain_tokenizer

    def parse(self):
        res = []
        self.df["lens"] = [len(t) for t in self.df["text"].to_list()]

        for i, row in tqdm(self.df.iterrows(), total=len(self.df)):
            res.append(
                {"text": self.tokenizer.encode(row["text"]), "domain_text": self.domain_tokenizer.encode(row["text"]),
                 "label": self.label_map[row["label"]]})

        return pd.DataFrame(res)

    def _tokenize(self, text):
        return self.tokenizer.encode(text)
