import tokenizers
from transformers import BertTokenizer
import os
from .merge import load_merge_vocab

FERTILITY= 1.1

def compute_fertility(unique_corpus,tokenizer:BertTokenizer):

    nominator = []
    for w in unique_corpus:
        nominator.extend(tokenizer.tokenize(w))

    return len(nominator)/len(unique_corpus)


def update_tokenizer(args,pretrain_tokenizer:BertTokenizer,domain_tokenizer,unique_words):

    pretrain_vocab = pretrain_tokenizer.get_vocab()
    domain_vocab = domain_tokenizer.get_vocab().items()
    ps = sorted(domain_vocab, key=lambda x: x[-1])
    candidate_vocab = [k for k, _ in ps if k not in pretrain_vocab]
    F = compute_fertility(unique_words, pretrain_tokenizer)

    while F>FERTILITY:
        domain_one = candidate_vocab.pop()
        F = compute_fertility(unique_words,pretrain_tokenizer)
        add_domain_vocab(args,tokenizer=pretrain_tokenizer,tokenizer_class=pretrain_tokenizer.__class__,\
                         config=pretrain_tokenizer.pretrained_init_configuration,domain_vocab=domain_one)
        print("Feritility {0}".format(F))

    return pretrain_tokenizer




def add_domain_vocab(args,tokenizer:BertTokenizer,tokenizer_class,config,domain_vocab:str,vocab_path):

    if not os.path.isdir(vocab_path):
        os.makedirs(vocab_path)
    tokenizer.save_pretrained(vocab_path)
    config.save_pretrained(vocab_path)



    if args.src_model=="bert":
        f = open(os.path.join(vocab_path, "vocab.txt"), "a")
    elif args.src_model=="roberta":
        import json

        with open(os.path.join(vocab_path, "vocab.json"), "r") as f:
            vocab_file = json.load(f)
            vocab_file.update({domain_vocab:len(vocab_file)})

        with open(os.path.join(vocab_path, "vocab.json"), "w") as f:
            json.dump(vocab_file ,f)
        f = open(os.path.join(vocab_path, "merges.txt"), "a")
    else:
        raise NotImplementedError

    f.write(domain_vocab+"\n")
    f.close()

    return load_merge_vocab(tokenizer_class=tokenizer_class,vocab_path=vocab_path)

def add_vocab_io(pretrain_tokenizer):

    pretrain_tokenizer.save_model("")



def read_corpus(path):
    with open("../data/chemprot/train.txt") as f:
        out = f.read()

    words = out.replace("\n"," ").split(" ")

    return list(set(words))