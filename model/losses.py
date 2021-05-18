import torch
from torch.nn.modules.loss import _Loss
import torch.nn as nn
import numpy as np


class NTXentLoss(_Loss):
    def __init__(self, args, batch_size, temperature=1.0, use_cosine_similarity=True, hidden_size=768, device="cuda"):
        super(NTXentLoss, self).__init__(args)
        self.device = device
        self.temperature = temperature
        self.batch_size = batch_size
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion_simclr = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):

        self.batch_size = zis.shape[0]
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[
            self.mask_samples_from_same_repr[:(self.batch_size * 2), :(self.batch_size * 2)]].view(2 * self.batch_size,
                                                                                                   -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion_simclr(logits, labels)

        return loss / (2 * self.batch_size)


class AlignLoss(_Loss):
    def __init__(self, args, batch_size, temperature=1.0, device="cuda"):
        super(AlignLoss, self).__init__(args)
        self.device = device
        self.temperature = temperature
        self.batch_size = batch_size
        self._cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, zis, zjs):
        self.batch_size = zis.shape[0]
        a = self._cosine_similarity(zis, zjs)

        return torch.mean(1 - a)