import argparse, random, sys
import numpy as np

import cupy
from cupy import cuda

def parse_embeddings(embed_path):
    embed_dict = {}
    with open(embed_path, errors='ignore') as f_embed:
        next(f_embed)  # skip header
        for line in f_embed:
            pieces = line.rstrip().split(" ")
            embed_dict[pieces[0]] = np.array([float(weight) for weight in pieces[1:]])
    return embed_dict


def search_neighbors(src_batch_emb, tgt_all_emb,
                     word2id, batch_words,
                     include_self=False, num_neighbors=10):

    # Normalize batch embeddings
    src_batch_emb_norm = src_batch_emb / cupy.linalg.norm(src_batch_emb, axis=1)[:, None]

    # Compute cosine similarity
    cos_score = src_batch_emb_norm.dot(tgt_all_emb.T) # [batch_size, num_words]

    # Ignore exact matching words
    if not include_self:
        # indexはbatchの各単語のword indexをもっている
        word_index = cupy.array([word2id[word] for word in batch_words if word in word2id], dtype=cupy.int32)
        batch_index = cupy.array([i for i, word in enumerate(batch_words) if word in word2id], dtype=cupy.int32)
        # Set the score of matching words to very small
        cos_score[batch_index, word_index] = -100
    sim_indices = cupy.argsort(-cos_score, axis=1)[:, :num_neighbors] # [batch_size, num_neighbors]
    # conc = cupy.concatenate([cupy.expand_dims(cos_score[i][sim_indices[i]], axis=0) for i in range(len(sim_indices))], axis=0)
    sim_cos_scores = cupy.concatenate([cupy.expand_dims(cos_score[i][sim_indices[i]], axis=0) for i in range(len(sim_indices))], axis=0)
    sim_cos_scores = cupy.asnumpy(sim_cos_scores)
    return sim_indices, sim_cos_scores

def compute_similarity(src_batch_emb, tgt_all_emb,
                       word2id, batch_words, indices,
                       include_self=False, num_neighbors=10):

    # Normalize batch embeddings
    src_batch_emb_norm = src_batch_emb / cupy.linalg.norm(src_batch_emb, axis=1)[:, None]

    # Compute cosine similarity
    cos_score = src_batch_emb_norm.dot(tgt_all_emb.T) # [batch_size, num_words]

    sim_indices = indices
    # conc = cupy.concatenate([cupy.expand_dims(cos_score[i][sim_indices[i]], axis=0) for i in range(len(sim_indices))], axis=0)
    sim_cos_scores = cupy.concatenate([cupy.expand_dims(cos_score[i][sim_indices[i]], axis=0) for i in range(len(sim_indices))], axis=0)
    sim_cos_scores = cupy.asnumpy(sim_cos_scores)
    return sim_cos_scores



