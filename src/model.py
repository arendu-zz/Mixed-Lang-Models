#!/usr/bin/env python
__author__ = 'arenduchintala'
import math
import torch
import torch.nn as nn

from torch.autograd import Variable

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import pdb


def get_unsort_idx(sort_idx):
    unsort_idx = torch.zeros_like(sort_idx).long().scatter_(0, sort_idx, torch.arange(sort_idx.size(0)).long())
    return unsort_idx


class WordRepresenter(nn.Module):
    def __init__(self, word2spelling, char2idx, cv_size, ce_size, cp_idx, cr_size, we_size,
                 bidirectional=False, dropout=0.3):
        super(WordRepresenter, self).__init__()
        self.word2spelling = word2spelling
        self.sorted_spellings, self.sorted_lengths, self.unsort_idx = self.init_word2spelling()
        self.char2idx = char2idx
        self.ce_size = ce_size
        self.cv_size = cv_size
        self.cr_size = cr_size
        self.ce_layer = torch.nn.Embedding(self.cv_size, self.ce_size, padding_idx=cp_idx)
        self.ce_layer.weight = nn.Parameter(
            torch.FloatTensor(self.cv_size, self.ce_size).uniform_(-0.5 / self.ce_size, 0.5 / self.ce_size))
        self.c_rnn = torch.nn.LSTM(self.ce_size, self.cr_size,
                                   bidirectional=bidirectional, batch_first=True,
                                   dropout=dropout)
        if self.cr_size * (2 if bidirectional else 1) != we_size:
            self.c_proj = torch.nn.Linear(self.cr_size * (2 if bidirectional else 1), we_size)
        else:
            self.c_proj = None
        print('WordRepresenter init complete.')

    def init_word2spelling(self,):
        spellings = None
        for v, s in self.word2spelling.items():
            if spellings is not None:
                spellings = torch.cat((spellings, torch.LongTensor(s).unsqueeze(0)), dim=0)
            else:
                spellings = torch.LongTensor(s).unsqueeze(0)
        lengths = spellings[:, -1]
        spellings = spellings[:, :-1]
        sorted_lengths, sort_idx = torch.sort(lengths, 0, True)
        unsort_idx = get_unsort_idx(sort_idx)
        sorted_lengths = sorted_lengths.numpy().tolist()
        sorted_spellings = spellings[sort_idx, :]
        sorted_spellings = Variable(sorted_spellings, requires_grad=False)
        return sorted_spellings, sorted_lengths, unsort_idx

    def init_cuda(self,):
        self.sorted_spellings = self.sorted_spellings.cuda()
        self.unsort_idx = self.unsort_idx.cuda()

    def forward(self,):
        emb = self.ce_layer(self.sorted_spellings)
        packed_emb = pack(emb, self.sorted_lengths, batch_first=True)
        output, (ht, ct) = self.c_rnn(packed_emb, None)
        # output, l = unpack(output)
        del output, ct
        if ht.size(0) == 2:
            # concat the last ht from fwd RNN and first ht from bwd RNN
            ht = torch.cat([ht[0, :, :], ht[1, :, :]], dim=1)
        else:
            ht = ht.squeeze()
        if self.c_proj is not None:
            word_embeddings = self.c_proj(ht)
        else:
            word_embeddings = ht
        unsorted_word_embeddings = word_embeddings[self.unsort_idx, :]
        return unsorted_word_embeddings


class VarLinear(nn.Module):
    def __init__(self, word_representer):
        super(VarLinear, self).__init__()
        self.word_representer = word_representer

    def matmul(self, data):
        var = self.word_representer()
        if data.dim() > 1:
            assert data.size(-1) == var.size(-1)
            return torch.matmul(data, var.transpose(0, 1))
        else:
            raise BaseException("data should be at least 2 dimensional")

    def forward(self, data):
        return self.matmul(data)


class VarEmbedding(nn.Module):
    def __init__(self, word_representer):
        super(VarEmbedding, self).__init__()
        self.word_representer = word_representer

    def forward(self, data):
        return self.lookup(data)

    def lookup(self, data):
        var = self.word_representer()
        # vocab_size = var.size(0)
        embedding_size = var.size(1)
        if data.dim() == 2:
            batch_size = data.size(0)
            seq_len = data.size(1)
            data = data.contiguous()
            data = data.view(-1)  # , data.size(0), data.size(1))
            var_data = var[data]
            var_data = var_data.view(batch_size, seq_len, embedding_size)
        else:
            var_data = var[data]
        return var_data


class CBiLSTM(nn.Module):
    def __init__(self,  rnn_size, input_size, vocab_size, encoder, decoder,
                 dropout=0.3, max_grad_norm=5.0):
        super(CBiLSTM, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.dropout = nn.Dropout(dropout)
        self.max_grad_norm = max_grad_norm
        self.input_size = input_size
        self.rnn_size = rnn_size  # self.encoder.weight.size(1) // 2
        self.vocab_size = vocab_size  # self.encoder.weight.size(0)
        self.rnn = nn.LSTM(self.input_size, self.rnn_size, dropout=dropout,
                           batch_first=True,
                           bidirectional=True)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = torch.optim.Adam(self.parameters())
        self.eos_sym = None
        self.bos_sym = None
        self.z = Variable(torch.zeros(1, 1, self.rnn_size), requires_grad=False)
        # .expand(batch_size, 1, self.rnn_size), requires_grad=False)

    def init_cuda(self,):
        self.z = self.z.cuda()

    def is_cuda(self,):
        return self.rnn.weight_hh_l0.is_cuda

    def forward(self, batch):
        lengths, data = batch
        # data = Variable(data, requires_grad=False)
        # if self.is_cuda():
        #     data = data.cuda()
        batch_size = data.size(0)
        # max_seq_len = data.size(1)
        assert data.dim() == 2

        # data = (batch_size x seq_len)
        encoded = self.dropout(self.encoder(data))
        packed_encoded = pack(encoded, lengths, batch_first=True)
        # encoded = (batch_size x seq_len x embedding_size)
        packed_hidden, (h_t, c_t) = self.rnn(packed_encoded)
        hidden, lengths = unpack(packed_hidden, batch_first=True)

        z = self.z.expand(batch_size, 1, self.rnn_size)
        fwd_hidden = torch.cat((z, hidden[:, :-1, :self.rnn_size]), dim=1)
        bwd_hidden = torch.cat((hidden[:, 1:, self.rnn_size:], z), dim=1)
        # bwd_hidden = (batch_size x seq_len x rnn_size)
        # fwd_hidden = (batch_size x seq_len x rnn_size)
        final_hidden = torch.cat((fwd_hidden, bwd_hidden), dim=2)
        out = self.decoder(self.dropout(final_hidden))
        loss = self.loss(out.view(-1, self.vocab_size), data.view(-1))
        return loss

    def do_backprop(self, batch, freeze=None):
        self.optimizer.zero_grad()
        l = self(batch)
        l.backward()
        grad_norm = torch.nn.utils.clip_grad_norm(self.parameters(),
                                                  self.max_grad_norm)
        if math.isnan(grad_norm):
            print('skipping update grad_norm is nan')
        else:
            if freeze is None:
                self.optimizer.step()
            else:
                for p in self.rnn.parameters():
                    p.grad *= 0
                self.encoder.weight.grad[freeze] *= 0
        if self.is_cuda():
            np_loss = l.data.clone().cpu().numpy()[0]
        else:
            np_loss = l.data.clone().numpy()[0]
        del l, batch
        return np_loss, grad_norm
