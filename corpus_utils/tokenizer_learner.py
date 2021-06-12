import tokenizers
from transformers import BertTokenizer
import os
from .merge import load_merge_vocab
import logging
from collections import Counter

logger = logging.getLogger(__name__)
import pandas as pd



class Learner:
    def __init__(self, args, config, pretrain_tokenizer, domain_tokenizer, init_feritility=2):
        self.args = args
        self.config = config

        self.vocab_path = self.args.vocab_path
        self.pretrain_tokenizer = pretrain_tokenizer
        self.domain_tokenizer = domain_tokenizer
        self.pretrain_tokenizer.save_pretrained(self.vocab_path)
        self.unique_corpus = None
        self.init_fertility = init_feritility

        config.save_pretrained(self.vocab_path)

    def init_long_corpus(self, unique_corpus, tokenizer: BertTokenizer):

        out = []
        for w in unique_corpus:
            tokens = tokenizer.tokenize(w)
            if len(tokens) > self.init_fertility:
                out.append(w)
        self.unique_corpus = out

    def compute_fertility(self, unique_corpus: list, tokenizer: BertTokenizer):
        nominator = []
        for w in unique_corpus:
            nominator.extend(tokenizer.tokenize(w))

        return len(nominator) / len(unique_corpus)

    def update_tokenizer(self, unique_words, n_chunk=50):

        pretrain_vocab = self.pretrain_tokenizer.get_vocab()
        domain_vocab = self.domain_tokenizer.get_vocab().items()
        ps = sorted(domain_vocab, key=lambda x: x[-1])

        init = 500
        candidate_vocab = [k for k, _ in ps if k not in pretrain_vocab]
        for_save = []
        init_func = self.init_long_corpus
        update_func = self.compute_fertility

        init_func(unique_words, self.pretrain_tokenizer)
        F = update_func(self.unique_corpus, self.pretrain_tokenizer)
        for_save.append(F)
        logger.info("Initial fertility {0:.6f} ".format(F))
        remains = candidate_vocab
        step = 0

        while F > 3.0 and len(remains)>0:
            step += 1
            domain_one, remains = remains[:init+n_chunk], remains[init+n_chunk:]
            self.pretrain_tokenizer = self.add_domain_vocab(tokenizer=self.pretrain_tokenizer, domain_vocab=domain_one)
            F = update_func(self.unique_corpus, self.pretrain_tokenizer)
            logger.info("Current fertility {0:.10f} ".format(F))
            init += n_chunk
            for_save.append(F)
        print(init)

        pd.to_pickle(F, os.path.join(self.vocab_path, "feritility"))
        return candidate_vocab[:init]

    def add_domain_vocab(self, tokenizer: BertTokenizer, domain_vocab: str):

        if not os.path.isdir(self.vocab_path):
            os.makedirs(self.vocab_path)
        # tokenizer.save_pretrained(self.vocab_path)

        f = open(os.path.join(self.vocab_path, "vocab.txt"), "a")

        for vocab in domain_vocab:
            f.write(vocab + "\n")
        f.close()

        return load_merge_vocab(tokenizer_class=self.pretrain_tokenizer, vocab_path=self.vocab_path)

    def add_vocab_io(self):
        self.pretrain_tokenizer.save_model()

    def return_optimize_tokenizer(self):
        return self.pretrain_tokenizer
