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
    l, d, i = zip(*batch)
    l = list(l)
    max_len = l[0]
    torch_d = torch.zeros(len(batch), max_len).long()
    torch_i = torch.zeros(len(batch), max_len).long()
    for _idx in range(len(l)):
        torch_d[_idx, :l[_idx]] = torch.LongTensor(d[_idx])
        torch_i[_idx, :l[_idx]] = torch.LongTensor(i[_idx])
    return l, torch_d, torch_i


class LazyTextDataset(Dataset):
    def __init__(self, corpus_file, v2idx, gv2idx=None):
        self.corpus_file = corpus_file
        self.UNK = '<UNK>'
        self.BOS = '<BOS>'
        self.EOS = '<EOS>'
        self.PAD = '<PAD>'
        self.v2idx = v2idx
        self.gv2idx = gv2idx
        with open(self.corpus_file, "r", encoding="utf-8") as f:
            self._total_data = len(f.readlines()) - 1

    def __getitem__(self, idx):
        line = linecache.getline(self.corpus_file, idx + 1)
        line_items = line.split('|||')
        if len(line_items) == 2 and self.gv2idx is not None:
            l1_line = line_items[0].strip().split()
            l2_line = line_items[1].strip().split()
            ind = np.random.randint(1, 3, (len(l1_line),)).tolist()
        else:
            l1_line = line_items[0].strip().split()
            l2_line = [None] * len(l1_line)
            ind = [1] * len(l1_line)
        data = [self.v2idx[self.BOS]] + \
               [self.v2idx.get(w, self.v2idx[self.UNK]) if i == 1 else self.gv2idx.get(g, self.gv2idx[self.UNK]) for w, g, i in zip(l1_line, l2_line, ind)] +\
               [self.v2idx[self.EOS]]
        ind = [1] + ind + [1]
        return (len(data), data, ind)

    def __len__(self):
        return self._total_data
