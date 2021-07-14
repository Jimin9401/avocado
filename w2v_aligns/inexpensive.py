import argparse, random, sys
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch
from transformers import BertTokenizer, AdamW, BertModel
import pandas as pd
import os
from gensim.models import Word2Vec  # for pair in pbar:



def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)



class LinearTransform(nn.Module):
    def __init__(self):
        super(LinearTransform, self).__init__()
        self.weight_matrix = nn.Linear(768, 768, bias=False)

    def forward(self, x):
        return self.weight_matrix(x)


def overlap_with_bert(w2v_vocab, tokenizer: BertTokenizer):
    bert_vocab = tokenizer.get_vocab()

    overlapped_vocab = [{"w2v": (word, idx), "bert": (word, bert_vocab[word])} for word, idx in w2v_vocab.items() if
                        word in bert_vocab]

    unoverlapped_vocab = [(word, idx) for word, idx in w2v_vocab.items() if
                          word not in bert_vocab]

    return overlapped_vocab, unoverlapped_vocab


def construct_embedding_matrix(overlapped_pairs, bert_matrix, w2v_matrix):
    # w2v_examples = torch.FloatTensor()
    # bert_examples = torch.FloatTensor()

    w2v_examples = None
    bert_examples = None

    for pair in overlapped_pairs:
        w2v, bert = pair["w2v"], pair["bert"]
        # print(w2v)
        # print(bert)
        w2v_embedding = w2v_matrix[w2v[1]].view(-1, 768)
        bert_embedding = bert_matrix.weight.data[bert[1]].view(-1, 768)

        if w2v_examples is None:
            w2v_examples = w2v_embedding
            bert_examples = bert_embedding

        else:
            w2v_examples = torch.cat([w2v_examples, w2v_embedding], dim=0)
            bert_examples = torch.cat([bert_examples, bert_embedding], dim=0)

    # print(w2v_examples.shape)
    # print(bert_examples.shape)

    return w2v_examples, bert_examples


def align_w2v_with_bert(bert_matrix, w2v_embedding_matrix, overlapped_pairs, batch_size=512):
    w_matrix = LinearTransform().cuda()
    optimizer = AdamW(w_matrix.parameters(),lr=1e-5)
    l2_loss = nn.MSELoss(size_average=True)

    w2v_examples, bert_examples = construct_embedding_matrix(overlapped_pairs, bert_matrix, w2v_embedding_matrix)
    # print("construct")
    # loader = generate_batchfier(w2v_examples, bert_examples, batch_size=batch_size)
    # pbar = tqdm(loader)

    # len(w2v_examples)
    for e in range(50) :
        epoch_loss = []
        for idx in range(0, len(w2v_examples), batch_size):
            w2v_matrix, bert_matrix = w2v_examples[idx:idx + batch_size].to("cuda"), bert_examples[
                                                                                     idx:idx + batch_size].cuda()

            transformed = w_matrix(w2v_matrix)
            loss = l2_loss(transformed, bert_matrix)

            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        print(f"Epoch : {e} loss: {np.mean(epoch_loss)}")

    return w_matrix


def inference_unoverlapped_word(w2v_vocab, bert_vocab, w_matrix, w2v_embedding_matrix, dataset_name):
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



#     w2v_matrix, bert_matrix = pair


def load_w2v_vocab(args):
    file_path = os.path.join("../data", "word2vec_" + args.dataset + ".vec")
    w2v = Word2Vec.load(file_path)

    return w2v.wv.key_to_index, w2v.wv.vectors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["chemprot", "citation_intent", "hyperpartisan_news",
                                              "amazon"], required=True, type=str)

    args = parser.parse_args()
    set_seed(777)
    w2v_pair, embedding_matrix = load_w2v_vocab(args)

    embedding_matrix = torch.FloatTensor(embedding_matrix)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # tokenizer.add_tokens(list(w2v_pair.keys()))

    model = BertModel.from_pretrained("bert-base-uncased")
    model.resize_token_embeddings(len(tokenizer))
    # model.to("cuda")

    bert_layer = model.embeddings.word_embeddings
    overlapped, unoverlapped = overlap_with_bert(w2v_vocab=w2v_pair, tokenizer=tokenizer)
    print("find overlapped")

    w_matrix = align_w2v_with_bert(bert_matrix=bert_layer, w2v_embedding_matrix=embedding_matrix,
                                   overlapped_pairs=overlapped)
    out_path = os.path.join("../data/",args.dataset,"newly_added.pkl")

    out= {}

    for word,idx in unoverlapped:
        with torch.no_grad():
            transformed = w_matrix(embedding_matrix[idx].cuda())
        out[word]=transformed.cpu().numpy()

    pd.to_pickle(out,out_path)



