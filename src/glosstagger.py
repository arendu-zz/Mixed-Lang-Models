#!/usr/bin/env python
__author__ = 'arenduchintala'
import math
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from highway import ResLSTM


class GlossTagger(nn.Module):
    def __init__(self, inp_vocab,
                 out_vocab,
                 rnn_size,
                 num_layers,
                 embedding_size,
                 dropout_prob=0.2,
                 pad_idx=0,
                 encoder=None,
                 decoder=None,
                 rnn_type='lstm'):
        super(GlossTagger, self).__init__()
        self.inp_vocab = inp_vocab
        self.out_vocab = out_vocab
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.dropout_prob = dropout_prob
        self.bidirectional = True
        self.encoder = encoder
        self.rnn_type = rnn_type
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.embedding_size, self.rnn_size, self.num_layers,
                               dropout=self.dropout_prob, batch_first=True, bidirectional=self.bidirectional)
        elif self.rnn_type == 'reslstm':
            self.rnn = ResLSTM(self.embedding_size, self.rnn_size, self.num_layers,
                               bidirectional=self.bidirectional)
        elif self.rnn_type == 'conv':
            self.rnn = nn.Conv1d(self.embedding_size, 2 * self.rnn_size, kernel_size=7, stride=1, padding=(7//2))
        else:
            raise NotImplementedError("unknown rnn_type")
        self.decoder = decoder
        self.dropout = torch.nn.Dropout(self.dropout_prob)
        print(pad_idx, 'ignore_index')
        self.loss = torch.nn.CrossEntropyLoss(size_average=True, ignore_index=pad_idx)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()))
        self.max_grad_norm = 5.

    def init_cuda(self,):
        self = self.cuda()
        self.encoder = self.encoder.cuda()
        self.decoder = self.decoder.cuda()

    def predict(self, data, lengths):
        #data = (batch_size, seq_size)
        emb = self.encoder(data)
        if self.rnn_type == 'lstm' or self.rnn_type == 'reslstm':
            packed_emb = pack(emb, lengths, batch_first=True)
            packed_hidden, (h_t, c_t) = self.rnn(packed_emb)
            hidden, lengths = unpack(packed_hidden, batch_first=True)
        elif self.rnn_type == 'conv':
            emb_ = emb.transpose(1, 2)
            hidden_ = self.rnn(emb_)
            hidden = hidden_.transpose(1, 2)
        else:
            raise NotImplementedError("unknown rnn type")
        predictions = self.decoder(hidden)
        log_preds = torch.nn.functional.log_softmax(predictions, dim=2)
        max_vals, max_idxs = torch.max(log_preds, dim=2)
        max_vals = max_vals.detach()
        return max_idxs, max_vals

    def forward(self, batch):
        lengths, data, labels = batch
        emb = self.encoder(data)
        emb = self.dropout(emb)
        if self.rnn_type == 'lstm' or self.rnn_type == 'reslstm':
            packed_emb = pack(emb, lengths, batch_first=True)
            packed_hidden, (h_t, c_t) = self.rnn(packed_emb)
            hidden, lengths = unpack(packed_hidden, batch_first=True)
        elif self.rnn_type == 'conv':
            emb_ = emb.transpose(1, 2)
            hidden_ = self.rnn(emb_)
            hidden = hidden_.transpose(1, 2)
        else:
            raise NotImplementedError("unknown rnn type")
        hidden = self.dropout(hidden)
        p = self.decoder(hidden)
        p = p.view(-1, p.size(2))  # (batch_size * seq_len, vocab_size)
        labels = labels.view(-1)
        p = p[labels != 0, :]
        labels = labels[labels != 0]
        ls = self.loss(p, labels)
        return ls

    def do_backprop(self, batch):
        self.optimizer.zero_grad()
        ls = self(batch)
        ls.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.parameters()),
                                                   self.max_grad_norm)
        if math.isnan(grad_norm):
            print('skipping update grad_norm is nan')
        else:
            self.optimizer.step()
        loss = ls.item()
        del ls, batch, grad_norm
        return loss

    def is_cuda(self,):
        return next(self.rnn.parameters()).is_cuda

    def save_model(self, path):
        torch.save(self, path)
