from util.trainer import CFTrainer, ContrastiveTrainer
from util.data_builder import get_dataset
from util.args import ExperimentArgument
from tqdm import tqdm
import pandas as pd
from embedding_utils.embedding_initializer import transfer_embedding
from transformers import BertTokenizer, AdamW, BertConfig
# from pytorch_transformers import WarmupLinearSchedule
import apex
from util.batch_generator import CFBatchFier, ContrastiveBatchFier
from model.classification_model import PretrainedTransformer

from transformers import get_scheduler

import torch.nn as nn
import torch
import random
from util.logger import *
import logging

logger = logging.getLogger(__name__)


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def get_trainer(args, model, train_batchfier, test_batchfier):
    # optimizer = torch.optim.AdamW(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)

    # optimizer=RAdam(model.parameters(),args.learning_rate,weight_decay=args.weight_decay)
    optimizer = AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)

    if args.mixed_precision:
        print('mixed_precision')
        opt_level = 'O2'
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level=opt_level)
        # from apex.parallel import DistributedDataParallel as DDP
        # model=DDP(model,delay_allreduce=True)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        # model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.gpu])

    # decay_step = args.decay_step
    # decay_step=0
    # scheduler = WarmupLinearSchedule(optimizer, args.warmup_step, args.decay_step)

    # lr_scheduler = get_scheduler(
    #     name=args.lr_scheduler_type,
    #     optimizer=optimizer,
    #     num_warmup_steps=args.num_warmup_steps,
    #     num_training_steps=args.max_train_steps,
    # )

    criteria = nn.CrossEntropyLoss(ignore_index=-100)

    if args.contrastive:
        trainer = ContrastiveTrainer(args, model, train_batchfier, test_batchfier, optimizer,
                                     args.gradient_accumulation_step, criteria, args.clip_norm, args.mixed_precision,
                                     args.n_label)


    else:
        trainer = CFTrainer(args, model, train_batchfier, test_batchfier, optimizer,
                            args.gradient_accumulation_step, criteria, args.clip_norm, args.mixed_precision,
                            args.n_label)

    return trainer


def get_batchfier(args, tokenizer):
    n_gpu = torch.cuda.device_count()
    train, dev, test, label = get_dataset(args, tokenizer)

    if isinstance(tokenizer, tuple):
        _, domain_tokenizer = tokenizer
        padding_idx = domain_tokenizer.pad_token_id
        mask_idx = domain_tokenizer.pad_token_id
    else:
        padding_idx = tokenizer.pad_token_id
        mask_idx = tokenizer.pad_token_id
    if args.contrastive:
        train_batch = ContrastiveBatchFier(args, train, batch_size=args.per_gpu_train_batch_size * n_gpu,
                                           maxlen=args.seq_len,
                                           padding_index=padding_idx, mask_idx=mask_idx)
        dev_batch = ContrastiveBatchFier(args, dev, batch_size=args.per_gpu_eval_batch_size * n_gpu,
                                         maxlen=args.seq_len,
                                         padding_index=padding_idx, mask_idx=mask_idx)
        test_batch = ContrastiveBatchFier(args, test, batch_size=args.per_gpu_eval_batch_size * n_gpu,
                                          maxlen=args.seq_len,
                                          padding_index=padding_idx, mask_idx=mask_idx)

    else:
        train_batch = CFBatchFier(args, train, batch_size=args.per_gpu_train_batch_size * n_gpu, maxlen=args.seq_len,
                                  padding_index=padding_idx)
        dev_batch = CFBatchFier(args, dev, batch_size=args.per_gpu_eval_batch_size * n_gpu, maxlen=args.seq_len,
                                padding_index=padding_idx)
        test_batch = CFBatchFier(args, test, batch_size=args.per_gpu_eval_batch_size * n_gpu, maxlen=args.seq_len,
                                 padding_index=padding_idx)

    # from torch.utils.data import DataLoader
    # train_batchfier=DataLoader(train_batch,batch_size=train_batch.size,collate_fn=train_batch.collate)
    # dev_batchfier=DataLoader(dev_batch, batch_size=dev_batch.size*4, collate_fn=dev_batch.collate,)

    return train_batch, dev_batch, test_batch, label


def expand_token_embeddings(model, tokenizer):
    new_vocab_size = len(tokenizer)
    model.resize_token_embeddings(new_num_tokens=new_vocab_size)


def embedding(args, model: PretrainedTransformer, d2p):
    transfer_embedding(model, d2p, args.transfer_type)


def main():
    args = ExperimentArgument()
    args.aug_ratio = 0.0
    set_seed(args.seed)
    gpu = 0
    args.gpu = gpu
    print(args.__dict__)
    from transformers import AutoConfig,AutoTokenizer
    pretrained_config = AutoConfig.from_pretrained(args.encoder_class)

    if args.merge_version:
        tokenizer = AutoTokenizer.from_pretrained(args.vocab_path)

        if args.contrastive:
            pretrained_tokenizer = AutoTokenizer.from_pretrained(args.encoder_class)
        if "uncased" in args.encoder_class:
            args.original_vocab_size = 30522
            args.extended_vocab_size = len(tokenizer) - args.original_vocab_size

        elif "roberta" in args.encoder_class:
            args.original_vocab_size = 50265
            args.extended_vocab_size = len(tokenizer) - args.original_vocab_size

        else:
            args.original_vocab_size = 28996
            args.extended_vocab_size = len(tokenizer) - args.original_vocab_size
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.encoder_class)
        args.extended_vocab_size = 0

    logger.info("\nNew merged Vocabulary size is %s" % (args.extended_vocab_size))

    if args.contrastive:
        train_gen, dev_gen, test_gen, label = get_batchfier(args, (pretrained_tokenizer, tokenizer))
    else:
        train_gen, dev_gen, test_gen, label = get_batchfier(args, tokenizer)

    print(label)
    args.n_label = len(label)

    inverse_label_map = {v: k for k, v in label.items()}
    args.label_list = inverse_label_map

    model = PretrainedTransformer(args, args.encoder_class, n_class=args.n_label)

    if args.merge_version:
        d2p = pd.read_pickle(os.path.join(args.vocab_path, "d2p.pickle"))
        expand_token_embeddings(model, tokenizer)
        embedding(args, model, d2p)

    model.cuda(args.gpu)
    optimal_score = -1.0

    trainer = get_trainer(args, model, train_gen, dev_gen)
    best_dir = os.path.join(args.savename, "best_model")

    if not os.path.isdir(best_dir):
        os.makedirs(best_dir)

    results = []
    not_improved = 0

    if args.do_train:
        for e in tqdm(range(0, args.n_epoch)):
            print("Epoch : {0}".format(e))
            trainer.train_epoch()
            save_path = os.path.join(args.savename, "epoch_{0}".format(e))
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            if args.evaluate_during_training:
                accuracy, macro_f1 = trainer.test_epoch()
                results.append({"eval_acc": accuracy, "eval_f1": macro_f1})

                if optimal_score < macro_f1:
                    optimal_score = macro_f1
                    torch.save(model.state_dict(), os.path.join(best_dir, "best_model.bin"))
                    print("Update Model checkpoints at {0}!! ".format(best_dir))
                    not_improved = 0
                else:
                    not_improved += 1

            if not_improved >= 5:
                break


        log_full_eval_test_results_to_file(args, config=pretrained_config, results=results)


    if args.do_eval:
        accuracy, macro_f1 = trainer.test_epoch()
        descriptions = os.path.join(args.savename, "eval_results.txt")
        writer = open(descriptions, "w")
        writer.write("accuracy: {0:.4f}, macro f1 : {1:.4f}".format(accuracy, macro_f1) + "\n")
        writer.close()

    if args.do_test:
        original_tokenizer = AutoTokenizer.from_pretrained(args.encoder_class)
        args.aug_word_length = len(tokenizer) - len(original_tokenizer)
        trainer.test_batchfier = test_gen
        results = []


        if args.model_path_list == "":
            raise EnvironmentError("require to clarify the argment of model_path")
        for model_path in args.model_path_list:
            print(model_path, "best_model", "best_model.bin")
            state_dict = torch.load(os.path.join(model_path, "best_model", "best_model.bin"))
            model.load_state_dict(state_dict)
            model.eval()

            accuracy, macro_f1 = trainer.test_epoch()
            results.append(macro_f1)

        log_full_test_results_to_file(args, config=pretrained_config, results=results)


if __name__ == "__main__":
    main()
