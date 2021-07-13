from corpus_utils.bpe_mapper import Tokenizer, CustomTokenizer
from tokenizers import SentencePieceBPETokenizer, BertWordPieceTokenizer, ByteLevelBPETokenizer
from util.args import CorpusArgument
import logging
from transformers import AutoTokenizer, AutoConfig, RobertaTokenizer
from corpus_utils.merge import domain2pretrain, merge_domain_vocab, corpuswise_compare
from corpus_utils.tokenizer_learner import Learner,BPELearner
import re
import os

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":

    args = CorpusArgument()

    if args.encoder_class == "roberta-base":
        encoder_class = ByteLevelBPETokenizer
    else:
        encoder_class = BertWordPieceTokenizer
    domain_tokenizer = CustomTokenizer(args=args, dir_path=args.root, encoder_class=encoder_class,
                                       dataset_name=args.dataset, vocab_size=args.vocab_size)

    pretrained_tokenizer = AutoTokenizer.from_pretrained(args.encoder_class)
    pretrained_tokenizer.save_pretrained(args.vocab_path)

    if "biobert" in args.encoder_class:
        f = open(os.path.join(args.vocab_path, "vocab.txt"), "a")
        f.write("아" + "\n")
        f.close()
        pretrained_tokenizer = AutoTokenizer.from_pretrained(args.vocab_path)
        print(pretrained_tokenizer.get_vocab()["아"])

    pretrained_config = AutoConfig.from_pretrained(args.encoder_class)

    if args.use_fragment:

        txt_file = os.path.join(args.root, args.dataset, "train.txt")
        with open(txt_file, "r") as f:
            out = f.read()
        new_string = re.sub('[^a-zA-Z0-9\n\.]', ' ', out)
        new_string = re.sub(' +', ' ', new_string)

        unique_words = list(new_string.strip().replace("\n", " ").split(" "))

        if args.encoder_class=="roberta-base":
            learner = BPELearner(args, pretrained_tokenizer, domain_tokenizer.encoder)
        else:
            learner = Learner(args, pretrained_config, pretrained_tokenizer, domain_tokenizer.encoder, )
        added_vocab = learner.update_tokenizer(unique_words, 50)

        d2p = domain2pretrain(added_vocab, pretrained_tokenizer, vocab_path=args.vocab_path)
        # merge_domain_vocab(args, pretrained_tokenizer, pretrained_config, d2p, args.vocab_path)

    else:
        new_vocab = corpuswise_compare(pretrained_tokenizer, domain_tokenizer.encoder)
        d2p = domain2pretrain(new_vocab, pretrained_tokenizer, vocab_path=args.vocab_path)
        # merge_domain_vocab(args, pretrained_tokenizer, pretrained_config, d2p, args.vocab_path)
