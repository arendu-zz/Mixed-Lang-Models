#!/usr/bin/env python
__author__ = 'arenduchintala'
import linecache
import numpy as np
import pickle
import torch

import pdb

from torch.utils.data import Dataset


def my_collate(batch):
    # batch is a list of tuples (l, data)
    batch = sorted(batch, reverse=True)
    l, s = zip(*batch)
    l = list(l)
    max_len = l[0]
    t = torch.zeros(len(batch), max_len).long()
    for _idx in range(len(l)):
        t[_idx, :l[_idx]] = torch.LongTensor(s[_idx])
    return l, t


class LazyTextDataset(Dataset):
    def __init__(self, corpus_file, v2idx):
        self.corpus_file = corpus_file
        self.UNK = '<UNK>'
        self.BOS = '<BOS>'
        self.EOS = '<EOS>'
        self.PAD = '<PAD>'
        self.v2idx = v2idx
        with open(self.corpus_file, "r", encoding="utf-8") as f:
            self._total_data = len(f.readlines()) - 1

    def __getitem__(self, idx):
        line = linecache.getline(self.corpus_file, idx + 1)
        data = [self.v2idx[self.BOS]] + \
               [self.v2idx.get(w, self.v2idx[self.UNK]) for w in line.strip().split()] +\
               [self.v2idx[self.EOS]]
        return (len(data), data)

    def __len__(self):
        return self._total_data
