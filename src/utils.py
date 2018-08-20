#!/usr/bin/env python
__author__ = 'arenduchintala'
import linecache
import numpy as np
import torch
import pdb

from torch.utils.data import Dataset

from model import CBiLSTM



class text_effect:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


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


def parallel_collate(batch):
    # batch is a list of tuples (l, data)
    batch = sorted(batch, reverse=True)
    l, l1_d, l2_d = zip(*batch)
    l = list(l)
    max_len = l[0]
    torch_l1_d = torch.zeros(len(batch), max_len).long()
    torch_l2_d = torch.zeros(len(batch), max_len).long()
    for _idx in range(len(l)):
        torch_l1_d[_idx, :l[_idx]] = torch.LongTensor(l1_d[_idx])
        torch_l2_d[_idx, :l[_idx]] = torch.LongTensor(l2_d[_idx])
    return l, torch_l1_d, torch_l2_d


class ParallelTextDataset(Dataset):
    def __init__(self, corpus_file, s2i, t2i, data_mode='train'):
        self.corpus_file = corpus_file
        self.BOS = '<BOS>'
        self.EOS = '<EOS>'
        self.PAD = '<PAD>'
        self.UNK = '<UNK>'
        self.s2i = s2i
        self.t2i = t2i
        self.data_mode = data_mode
        assert self.data_mode in ['train', 'test']
        with open(self.corpus_file, "r", encoding="utf-8") as f:
            self._total_data = len(f.readlines())

    def __getitem__(self, idx):
        line = linecache.getline(self.corpus_file, idx + 1)
        if self.data_mode == 'train':
            l1_line, l2_line = line.strip().split('|||')
            l1_line = l1_line.strip().split()
            l2_line = l2_line.strip().split()
        elif self.data_mode == 'test':
            l1_line = line.strip().split()
            l2_line = [self.UNK for i in l1_line]
        else:
            pass

        l1_data = [self.s2i[self.BOS]] + \
                  [self.s2i.get(w, self.s2i[self.UNK]) for w in l1_line] + \
                  [self.s2i[self.EOS]]
        l2_data = [self.t2i[self.BOS]] + \
                  [self.t2i.get(w, self.t2i[self.PAD]) for w in l2_line] + \
                  [self.t2i[self.EOS]]
        assert len(l1_data) == len(l2_data)
        return (len(l1_data), l1_data, l2_data)

    def __len__(self):
        return self._total_data


class LazyTextDataset(Dataset):
    def __init__(self, corpus_file, v2idx, gv2idx, mode, swap_prob=0.25):
        self.corpus_file = corpus_file
        self.UNK = '<UNK>'
        self.BOS = '<BOS>'
        self.EOS = '<EOS>'
        self.PAD = '<PAD>'
        self.v2idx = v2idx
        self.gv2idx = gv2idx
        self.mode = mode
        self.threshold = 1.0 - swap_prob
        with open(self.corpus_file, "r", encoding="utf-8") as f:
            self._total_data = len(f.readlines()) - 1

    def __getitem__(self, idx):
        line = linecache.getline(self.corpus_file, idx + 1)
        line_items = line.split('|||')
        if self.mode == CBiLSTM.L12_LEARNING:
            assert len(line_items) == 2
            l1_line = line_items[0].strip().split()
            l2_line = line_items[1].strip().split()
            ind = np.random.rand(len(l1_line),)
            ind[ind >= self.threshold] = 2
            ind[ind < self.threshold] = 1
            ind = ind.astype(int).tolist()
        else:
            l1_line = line_items[0].strip().split()
            l2_line = [None] * len(l1_line)
            ind = [1] * len(l1_line)
        data = [self.v2idx[self.BOS]] + \
               [self.v2idx.get(w, self.v2idx[self.UNK]) if i == 1 else self.gv2idx.get(g, self.gv2idx[self.UNK])
                for w, g, i in zip(l1_line, l2_line, ind)] + \
               [self.v2idx[self.EOS]]
        ind = [1] + ind + [1]
        return (len(data), data, ind)

    def __len__(self):
        return self._total_data
