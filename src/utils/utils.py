#!/usr/bin/env python
__author__ = 'arenduchintala'
import random
import torch


class SPECIAL_TOKENS:
    UNK = '<unk>'
    BOS = '<bos>'
    EOS = '<eos>'
    PAD = '<pad>'
    # spl sym for chars
    UNK_C = '<unkc>'
    BOW = '<bow>'
    EOW = '<eow>'
    NULL = '<null>'


class TEXT_EFFECT:
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


class LazyTextBatcher(object):
    def __init__(self,
                 file_name,
                 shuffle,
                 sort_by_len,
                 min_batch_size,
                 min_batch_by_num_tokens=True,
                 max_batch_size=100000):
        assert max_batch_size >= min_batch_size
        self.file_name = file_name
        self.shuffle = shuffle
        self.sort_by_len = sort_by_len
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.min_batch_by_num_token = min_batch_by_num_tokens

    def max_batch_iter(self,):
        with open(self.file_name, 'r', encoding='utf-8') as f:
            max_batch = []
            for line in f:
                lines = line.strip().split('|||')
                l1_line = lines[0]
                if len(lines) == 2:
                    l2_line = lines[1]
                else:
                    l2_line = None
                len_line = len(l1_line.split())
                max_batch.append((len_line, l1_line, l2_line))
                if len(max_batch) >= self.max_batch_size:
                    copy_max_batch = max_batch
                    max_batch = []
                    if self.sort_by_len:
                        yield sorted(copy_max_batch, reverse=True)
                    else:
                        yield copy_max_batch
            if len(max_batch) > 0:
                if self.sort_by_len:
                    yield sorted(max_batch, reverse=True)
                else:
                    yield max_batch

    def __iter__(self,):
        for max_batch in self.max_batch_iter():
            if self.min_batch_by_num_token:
                lengths, _, _ = zip(*max_batch)
                lengths_span = [(self.min_batch_size // (_i + 2)) for _i in lengths]
                min_batches = []
                i = 0
                while i < len(lengths):
                    mb = max_batch[i: lengths_span[i] + i]
                    i = lengths_span[i] + i
                    min_batches.append(mb)
            else:
                num_min_batches = len(max_batch) // self.min_batch_size
                min_batches = [max_batch[i * self.min_batch_size: (i+1) * self.min_batch_size]
                               for i in range(num_min_batches)]
            if self.shuffle:
                random.shuffle(min_batches)
            for min_batch in min_batches:
                yield min_batch


class TextDataset(object):
    def __init__(self,
                 corpus_file,
                 v2idx,
                 shuffle,
                 sort_by_len,
                 min_batch_size,
                 lower_text=True):
        if min_batch_size > 1:
            assert sort_by_len
        self.v2idx = v2idx
        assert self.v2idx[SPECIAL_TOKENS.PAD] == 0
        self.lower_text = lower_text
        self.lazy_batcher = LazyTextBatcher(corpus_file, shuffle, sort_by_len, min_batch_size)
        self.num_batches = 0
        self.p = 0.0
        # for _ in self.lazy_batcher:
        #    self.num_batches += 1

    def set_mask_rate(self, p):
        self.p = p

    def __iter__(self,):
        for min_batch in self.lazy_batcher:
            lengths, l1_text_data, l2_text_data = zip(*min_batch)
            l1_text_data = list(l1_text_data)
            lengths = [i + 2 for i in lengths]
            torch_data = torch.zeros(len(l1_text_data), lengths[0]).long()
            for _idx, l in enumerate(lengths): # range(len(lengths)):
                if self.lower_text:
                    _tmp = [self.v2idx.get(i.lower(), self.v2idx[SPECIAL_TOKENS.UNK]) for i in
                            [SPECIAL_TOKENS.BOS] + l1_text_data[_idx].split() + [SPECIAL_TOKENS.EOS]]
                else:
                    _tmp = [self.v2idx.get(i, self.v2idx[SPECIAL_TOKENS.UNK]) for i in
                            [SPECIAL_TOKENS.BOS] + l1_text_data[_idx].split() + [SPECIAL_TOKENS.EOS]]
                torch_data[_idx, :lengths[_idx]] = torch.LongTensor(_tmp)
            yield lengths, torch_data, l1_text_data


class ParallelTextDataset(object):
    def __init__(self,
                 corpus_file,
                 l1_v2idx,
                 l2_v2idx,
                 lower_text=True):
        self.l1_v2idx = l1_v2idx
        self.l2_v2idx = l2_v2idx
        assert self.l1_v2idx[SPECIAL_TOKENS.PAD] == 0
        assert self.l2_v2idx[SPECIAL_TOKENS.PAD] == 0
        self.lower_text = lower_text
        self.lazy_batcher = LazyTextBatcher(corpus_file, False, False, 1, min_batch_by_num_tokens=False)

    def __iter__(self,):
        for min_batch in self.lazy_batcher:
            lengths, l1_text_data, l2_text_data = zip(*min_batch)
            lengths = [i + 2 for i in lengths]
            l1_text_data = list(l1_text_data)
            l2_text_data = list(l2_text_data)
            l1_torch_data = torch.zeros(len(l1_text_data), lengths[0]).long()
            l2_torch_data = torch.zeros(len(l2_text_data), lengths[0]).long()
            for _idx, l in enumerate(lengths):  # range(len(lengths)):
                if self.lower_text:
                    _tmp = [self.l1_v2idx.get(i.lower(), self.l1_v2idx[SPECIAL_TOKENS.UNK]) for i in
                            [SPECIAL_TOKENS.BOS] + l1_text_data[_idx].split() + [SPECIAL_TOKENS.EOS]]
                else:
                    _tmp = [self.l1_v2idx.get(i, self.l1_v2idx[SPECIAL_TOKENS.UNK]) for i in
                            [SPECIAL_TOKENS.BOS] + l1_text_data[_idx].split() + [SPECIAL_TOKENS.EOS]]
                l1_torch_data[_idx, :lengths[_idx]] = torch.LongTensor(_tmp)
                if self.lower_text:
                    _tmp = [self.l2_v2idx.get(i.lower(), self.l2_v2idx[SPECIAL_TOKENS.UNK]) for i in
                            [SPECIAL_TOKENS.BOS] + l2_text_data[_idx].split() + [SPECIAL_TOKENS.EOS]]
                else:
                    _tmp = [self.l2_v2idx.get(i, self.l2_v2idx[SPECIAL_TOKENS.UNK]) for i in
                            [SPECIAL_TOKENS.BOS] + l2_text_data[_idx].split() + [SPECIAL_TOKENS.EOS]]
                l2_torch_data[_idx, :lengths[_idx]] = torch.LongTensor(_tmp)
            yield lengths, l1_torch_data, l2_torch_data, l1_text_data, l2_text_data
