import argparse, random, sys
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch
from transformers import BertTokenizer, AdamW, BertModel
from model.classification_model import PretrainedTransformer
import pandas as pd


def overlap_with_bert(w2v_vocab, tokenizer: BertTokenizer):
    bert_vocab = tokenizer.get_vocab()

    overlapped_vocab = [{"w2v": (word, idx), "bert": (word, bert_vocab[word])} for word, idx in w2v_vocab.items() if
                        word in bert_vocab]

    return overlapped_vocab


def generate_batchfier(overlapped_pairs, batch_size=32):
    w2v_examples = torch.FloatTensor()
    bert_examples = torch.FloatTensor()
    for pair in overlapped_pairs:
        w2v_embedding, bert_embedding = pair["w2v"], pair["bert"]

        w2v_examples = torch.cat([w2v_examples, w2v_embedding], dim=0)
        bert_examples = torch.cat([bert_examples, bert_embedding], dim=0)
    total_size = w2v_examples.size(0)

    for idx in range(0, total_size, batch_size):
        yield w2v_examples[idx:idx + batch_size], bert_examples[idx:idx + batch_size]


def align_w2v_with_bert(bert_embedding_layer, w2v_embedding_matrix, overlapped_pairs, batch_size=32):
    d_w2v = w2v_embedding_matrix.shape[-1]
    d_bert = 768

    w_matrix = nn.Parameter(torch.randn(d_w2v, d_bert))
    # bert_embedding_layer = model.main_net.embeddings.word_embeddings
    optimizer = AdamW(w_matrix.parameters())
    l2_loss = nn.MSELoss(size_average=True)
    loader = generate_batchfier(overlapped_pairs, batch_size=batch_size)
    pbar = tqdm(loader, total=len(loader))

    for pair in pbar:
        w2v_matrix, bert_matrix = pair
        transformed = torch.matmul(w2v_matrix, w_matrix)
        loss = l2_loss(transformed, bert_matrix)

        loss.backward()
        optimizer.step()

        pbar.set_description(
            "training loss : %f  " % (loss), )
        pbar.update()

    return w_matrix


def inference_unoverlapped_word(w2v_vocab, bert_vocab, w_matrix, w2v_embedding_matrix):
    out = {}
    unoverlapped_vocab = [{"w2v": (word, idx)} for word, idx in w2v_vocab.items() if
                          word not in bert_vocab]

    for pairs in unoverlapped_vocab:
        w2v_idx = pairs["w2v"][-1]
        word = pairs["w2v"][0]
        pairs["w2v"] = w2v_vocab[w2v_idx]
        transformed = torch.matmul(w2v_embedding_matrix[w2v_idx], w_matrix)
        out[word] = (w2v_idx, transformed)

    pd.to_pickle("inexpensive_vocab_embedding.pkl")



def load_w2v(file_path) -> dict:
    raise NotImplementedError


if __name__ == "__main__":
    args = "asd"
    w2v_pair = load_w2v(args)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    model = BertModel.from_pretrained("bert-base-uncased")

    bert_layer = model.embeddings.word_embeddings
    overlapped = overlap_with_bert(w2v_vocab=w2v_pair, tokenizer=tokenizer)

    w_matrix = align_w2v_with_bert(overlapped,w2v_embedding_matrix=w2v_pair,overlapped_pairs=overlapped)

    inference_unoverlapped_word(w2v_vocab=w2v_pair,bert_vocab=tokenizer.get_vocab(),w_matrix=w_matrix,w2v_embedding_matrix=w2v_pair)

