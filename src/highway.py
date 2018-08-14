#!/usr/bin/env python
__author__ = 'arenduchintala'
import numpy as np
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class Highway(nn.Module):
    def __init__(self, input_size):
        super(Highway, self).__init__()
        self.highway = nn.Sequential(nn.Linear(input_size, input_size),
                                     nn.Sigmoid())

    def forward(self, x):
        t = self.highway(x)
        return t


class ResLSTM(nn.Module):
    def __init__(self, input_size, rnn_size, num_layers, bidirectional):
        super(ResLSTM, self).__init__()
        lstms = []
        for i in range(num_layers):
            if i == 0:
                lstm = nn.LSTM(input_size, rnn_size, num_layers=1,
                               batch_first=True, bidirectional=bidirectional)
            else:
                m = 2 if bidirectional else 1
                lstm = nn.LSTM(input_size + (m * rnn_size), rnn_size, num_layers=1,
                               batch_first=True, bidirectional=bidirectional)
            lstms.append(lstm)
        self.lstms = nn.ModuleList(lstms)

    def forward(self, packed_input):
        input, lengths = unpack(packed_input, batch_first=True)
        hidden = None
        packed_hidden = None
        for lstm_idx, lstm in enumerate(self.lstms):
            if lstm_idx == 0:
                packed_hidden, (h_t, c_t) = lstm(packed_input)
                hidden, lengths = unpack(packed_hidden, batch_first=True)
            else:
                hidden = torch.cat([hidden, input], dim=2)
                packed_hidden = pack(hidden, lengths, batch_first=True)
                packed_hidden, (h_t, c_t) = lstm(packed_hidden)
                hidden, lengths = unpack(packed_hidden, batch_first=True)  # for next lstm
        return packed_hidden, (h_t, c_t)


if __name__ == '__main__':
    bs = 3
    rnn_size = 10
    inp_size = 5
    seq_len = 22
    reslstm = ResLSTM(inp_size, rnn_size, 2, True)
    rand_inp = torch.Tensor(np.random.rand(bs, seq_len, inp_size))
    lengths = [22, 22, 22]
    packed_input = pack(rand_inp, lengths, batch_first=True)
    packed_out, (h_t, c_t) = reslstm(packed_input)
    out, lengths = unpack(packed_out, batch_first=True)  # for next lstm
    print(out.shape)
    if next(reslstm.parameters()).is_cuda:
        print('reslstm on cuda')
    else:
        print('reslstm not on cuda')
