import tokenizers
from transformers import BertTokenizer, RobertaTokenizer, XLNetTokenizer

import json
from tokenizers import BertWordPieceTokenizer, SentencePieceBPETokenizer, ByteLevelBPETokenizer
import os
from .merge import load_merge_vocab
import logging
import pandas as pd

logger = logging.getLogger(__name__)


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

        while F > 3.0 and len(remains) > 0:
            step += 1
            domain_one, remains = remains[:init + n_chunk], remains[init + n_chunk:]
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


class BPELearner:
    def __init__(self, args, config, pretrain_tokenizer: RobertaTokenizer, domain_tokenizer, init_fertility=2):
        self.vocab_path = args.vocab_path
        self.pretrain_tokenizer = pretrain_tokenizer
        self.init_fertility = init_fertility

        if not os.path.isdir(args.vocab_path):
            os.makedirs(args.vocab_path)

        self.pretrain_tokenizer.save_vocabulary(save_directory=self.vocab_path)
        config.save_pretrained(save_directory=self.vocab_path)
        self.vocab_file = os.path.join(self.vocab_path, "vocab.json")
        self.merges_file = os.path.join(self.vocab_path, "merges.txt")
        print(self.vocab_file)
        print(self.merges_file)
        self.init_tokenizer = ByteLevelBPETokenizer(vocab=self.vocab_file, merges=self.merges_file)
        self.domain_tokenizer = domain_tokenizer
        self.init_vocab_merge_pair()

    def init_vocab_merge_pair(self):
        self.domain_tokenizer.save_model(directory=self.vocab_path, prefix="custom")
        # custom_vocab_path = os.path.join(self.vocab_path, "custom-vocab.json")
        custom_merges_path = os.path.join(self.vocab_path, "custom-merges.txt")

        domain_vocab = self.domain_tokenizer.get_vocab()
        sorted_bpe = sorted(domain_vocab.items(), key=lambda x: x[-1])

        with open(custom_merges_path, "r") as reader:
            merges_file = reader.readlines()
        start_index = len(sorted_bpe) - len(merges_file) + 1
        vocab_merge_pair = []

        for bpe, merge in zip(sorted_bpe[start_index:], merges_file[1:]):
            vocab_merge_pair.append({"vocab": bpe, "merge": merge})

        self.vocab_merges = vocab_merge_pair

    def extract_complement(self):
        pretrained_vocab = self.init_tokenizer.get_vocab()
        complement_pairs = [example for example in self.vocab_merges if
                            example["vocab"][0] not in pretrained_vocab]

        return complement_pairs


    def init_long_corpus(self, unique_corpus, tokenizer: ByteLevelBPETokenizer):
        out = []
        for w in unique_corpus:
            tokens = tokenizer.encode(" " + w).tokens
            #             print(tokens)
            if len(tokens) > self.init_fertility:
                out.append(w)
        self.unique_corpus = out

    def update_tokenizer(self, unique_words, n_chunk=50):
        init = 500
        cnt=init

        self.init_long_corpus(unique_words, self.init_tokenizer)
        candidate_pairs = self.extract_complement()
        F = self.compute_fertility(self.unique_corpus, self.init_tokenizer)
        print("Initial fertility {0:.6f} ".format(F))
        remains = candidate_pairs
        step = 0

        domain_one, remains = remains[:init], remains[init:]
        self.init_tokenizer = self.add_domain_vocab(tokenizer=self.init_tokenizer, vocab_merge_pairs=domain_one)

        while F > 3.0 and len(remains) > 0:
            step += 1
            domain_one, remains = remains[:n_chunk], remains[n_chunk:]
            self.init_tokenizer = self.add_domain_vocab(tokenizer=self.init_tokenizer, vocab_merge_pairs=domain_one)
            F = self.compute_fertility(self.unique_corpus, self.init_tokenizer)
            print("Current fertility {0:.10f} ".format(F))
            # init += n_chunk
            cnt+=n_chunk

        return [d["vocab"][0] if "Ġ" not in d["vocab"][0] else d["vocab"][0].replace("Ġ", " ") for d in
                candidate_pairs[:cnt]]

    def compute_fertility(self, unique_corpus: list, tokenizer: ByteLevelBPETokenizer):
        nominator = []
        for w in unique_corpus:
            nominator.extend(tokenizer.encode(w).tokens)

        return len(nominator) / len(unique_corpus)


    def add_domain_vocab(self, tokenizer, vocab_merge_pairs):

        original_vocab = tokenizer.get_vocab()
        print(len(original_vocab))
        writer = open(self.merges_file, "a")

        for vocab_merge_pair in vocab_merge_pairs:

            vocab = vocab_merge_pair["vocab"]
            merge = vocab_merge_pair["merge"]
            original_vocab.update({vocab[0]: len(original_vocab)})
            writer.write(merge)
        writer.close()

        with open(self.vocab_file, "w") as writer:
            json.dump(original_vocab, writer)

        return ByteLevelBPETokenizer(self.vocab_file, self.merges_file)
