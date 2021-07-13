import pandas as pd
import random
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader, Dataset
import math
from tqdm import tqdm
import random

from torch.nn.utils.rnn import pad_sequence


class Dataset(Dataset):
    def __init__(self, X, y):
        """Reads source and target sequences from txt files."""
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        data = {}
        data["X"] = self.X[index]
        data["y"] = self.y[index]
        return data


class Base_Batchfier(IterableDataset):
    def __init__(self, args, batch_size: int = 32, seq_len=512, minlen=50, maxlen: int = 512,
                 criteria: str = 'lens', padding_index=70000, epoch_shuffle=True, device='cuda'):
        super(Base_Batchfier).__init__()
        self.args = args
        self.maxlen = maxlen
        self.minlen = minlen
        self.size = batch_size
        self.criteria = criteria
        self.seq_len = seq_len
        self.padding_index = padding_index
        self.epoch_shuffle = epoch_shuffle
        self.device = device
        # self.size = len(self.df) / num_buckets

    def truncate_small(self, df, criteria='lens'):
        lens = np.array(df[criteria])
        indices = np.nonzero((lens < self.minlen).astype(np.int64))[0]
        return df.drop(indices)

    def truncate_large(self, texts, lens):
        new_texts = []
        new_lens = []
        for i in range(len(texts)):
            text = texts[i]
            if len(text) > self.maxlen:
                new_texts.append(text[:self.maxlen])
                new_lens.append(self.maxlen)
            else:
                remainder = len(text) % self.seq_len
                l = lens[i]
                if remainder and remainder < 10:
                    text = text[:-remainder]
                    l = l - remainder
                new_texts.append(text)
                new_lens.append(l)
        return new_texts, new_lens

    def shuffle(self, df, num_buckets):
        dfs = []
        for bucket in range(num_buckets - 1):
            new_df = df.iloc[bucket * self.size: (bucket + 1) * self.size]
            dfs.append(new_df)
        random.shuffle(dfs)
        dfs.append(df.iloc[num_buckets - 1 * self.size: num_buckets * self.size])
        df = pd.concat(dfs)
        return df


class CFBatchFier(Base_Batchfier):
    def __init__(self, args, df: pd.DataFrame, batch_size: int = 32, seq_len=512, minlen=50, maxlen: int = 512,
                 criteria: str = 'lens', padding_index=0, epoch_shuffle=True, device='cuda'):
        super(CFBatchFier, self).__init__(args, batch_size, seq_len, minlen, maxlen, criteria, padding_index,
                                          epoch_shuffle,
                                          device)

        self.size = batch_size
        self.df = df
        self.df["lens"] = [len(text) for text in self.df.text]
        self.num_buckets = len(self.df) // self.size + (len(self.df) % self.size != 0)
        self.df = self.sort(self.df, criteria="lens")

        if epoch_shuffle:
            self.df = self.shuffle(self.df, self.num_buckets)

    def _maxlens_in_first_batch(self, df):
        first_batch = df.iloc[0:self.size]

        return first_batch

    def shuffle(self, df, num_buckets):
        dfs = []
        for bucket in range(1, num_buckets):
            new_df = df.iloc[bucket * self.size: (bucket + 1) * self.size]
            dfs.append(new_df)

        random.shuffle(dfs, )
        # dfs.append(df.iloc[(num_buckets-1) * self.size: num_buckets * self.size])
        first_batch = self._maxlens_in_first_batch(df)
        dfs.insert(0, first_batch)
        df = pd.concat(dfs)

        return df.reset_index(drop=True)

    def sort(self, df, criteria="lens"):
        return df.sort_values(criteria, ascending=False).reset_index(drop=True)

    def truncate_large(self):
        lens = np.array(self.df["lens"])
        indices = np.nonzero((lens > self.maxlen).astype(np.int64))[0]

        self.df = self.df.drop(indices).reset_index(drop=True)

    def __iter__(self):
        # num_buckets = len(self.df) // self.size if len(self.df) % self.size == 0 else len(self.df) // self.size + 1
        # self.df = self.sort(self.df)
        # self.df = self.shuffle(self.df, self.size)

        for _, row in self.df.iterrows():
            if isinstance(row["label"],list):
                yield row["text"][:self.maxlen], row["label"][:self.maxlen]
            else:
                yield row["text"][:self.maxlen], row["label"]

    def __len__(self):
        return self.num_buckets



    def collate(self, batch):

        text_ids = [torch.LongTensor(item[0]) for item in batch]
        labels = [torch.LongTensor([item[1]]) for item in batch]
        text_ids = pad_sequence(text_ids, batch_first=True, padding_value=self.padding_index)
        labels = pad_sequence(labels, batch_first=True, padding_value=self.padding_index)

        attention_mask = text_ids == self.padding_index

        return text_ids, attention_mask, labels


    def collate_ner(self, batch):

        text_ids = [torch.LongTensor(item[0]) for item in batch]
        labels = [torch.LongTensor(item[1]) for item in batch]
        text_ids = pad_sequence(text_ids, batch_first=True, padding_value=self.padding_index)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        attention_mask = text_ids == self.padding_index

        return text_ids, attention_mask, labels


class ContrastiveBatchFier(CFBatchFier):
    def __init__(self, args, df: pd.DataFrame, batch_size: int = 32, seq_len=1024, minlen=50, maxlen: int = 512,
                 criteria: str = 'lens', padding_index=0, epoch_shuffle=False, device='cuda', mask_idx=-1,masked_prob=0.1):
        super(ContrastiveBatchFier, self).__init__(args, df, batch_size, seq_len, minlen, maxlen, criteria,
                                                   padding_index,
                                                   epoch_shuffle, device)
        self.mask_prob = masked_prob
        self.mask_idx = mask_idx

    def __iter__(self):
        for _, row in self.df.iterrows():
            origin_text = row["text"][:self.maxlen]
            domain_text = row["domain_text"][:self.maxlen]

            if isinstance(row["label"],list):
                yield origin_text[:self.maxlen], domain_text[:self.maxlen], row["label"][:self.maxlen]
            else:
                yield origin_text[:self.maxlen], domain_text[:self.maxlen], row["label"]

    def _create_selective_mask(self, original_text, domain_text):

        original_text = np.array(original_text)
        domain_text = np.array(domain_text)
        id_set = []

        for idx, word in enumerate(original_text):
            if word in domain_text:
                id_set.append(idx)

        n_mask = int(len(id_set) * self.mask_prob)
        if n_mask==0:
            n_mask+=1

        id_set = random.sample(id_set, k=n_mask)
        original_text[np.array(id_set)] = self.mask_idx

        return original_text, domain_text

    def __len__(self):
        return self.num_buckets

    def collate(self, batch):
        text_ids = [torch.LongTensor(item[0]) for item in batch]
        domain_text_ids = [torch.LongTensor(item[1]) for item in batch]

        labels = [torch.LongTensor([item[-1]]) for item in batch]
        text_ids = pad_sequence(text_ids, batch_first=True, padding_value=self.padding_index)
        domain_text_ids = pad_sequence(domain_text_ids, batch_first=True, padding_value=self.padding_index)

        # attention_mask=text_ids==self.padding_index

        return text_ids, domain_text_ids, pad_sequence(labels)


    def collate_ner(self, batch):
        text_ids = [torch.LongTensor(item[0]) for item in batch]
        domain_text_ids = [torch.LongTensor(item[1]) for item in batch]

        labels = [torch.LongTensor(item[-1]) for item in batch]
        text_ids = pad_sequence(text_ids, batch_first=True, padding_value=self.padding_index)
        domain_text_ids = pad_sequence(domain_text_ids, batch_first=True, padding_value=self.padding_index)
        labels = pad_sequence(labels,batch_first=True,padding_value=-100)
        # attention_mask=text_ids==self.padding_index

        return text_ids, domain_text_ids,labels