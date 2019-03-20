#!/usr/bin/env python
__author__ = 'arenduchintala'

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from src.utils.utils import SPECIAL_TOKENS


class RGL(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

    def backward(self, grad):
        return -grad


class Discriminator(nn.Module):
    def __init__(self, decoder_size):
        super().__init__()
        self.decoder_size = decoder_size
        self.rgl = RGL()
        self.ff = torch.nn.Linear(self.decoder_size, 2)

    def forward(self, x):
        x = self.rgl(x)
        o = self.ff(x)
        return o


class LM(nn.Module):
    def __init__(self,
                 emb_size,
                 vocab_size,
                 num_layers,
                 dropout,
                 dictionary,
                 adv_lambda=0.0,
                 adv_labels=('<en>', '<de>'),
                 max_grad_norm=5.0,
                 tie_weights=True):
        super(LM, self).__init__()
        self.emb_size = emb_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.dictionary = dictionary
        self.idx_dictionary = {v: k for k, v in self.dictionary.items()}
        self.encoder = torch.nn.Embedding(vocab_size, emb_size)
        self.decoder = torch.nn.Linear(emb_size, vocab_size, bias=False)
        self.tie_weights = tie_weights
        if tie_weights:
            self.decoder.weight = self.encoder.weight
        self.max_grad_norm = max_grad_norm
        self.lstm = torch.nn.LSTM(
            emb_size, emb_size, self.num_layers, dropout=dropout, bidirectional=False)
        self.discriminator = Discriminator(emb_size)
        self.adv_labels = adv_labels
        self.adv_classes = [self.dictionary[self.adv_labels[0]], self.dictionary[self.adv_labels[1]]]
        self.adv_lambda = adv_lambda
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()))

    def is_cuda(self,):
        return self.lstm.weight_hh_l0.is_cuda

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if not self.tie_weights:
            self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, inp, lengths, hidden=None):
        encoded = self.encoder(inp)
        packed = pack(encoded, lengths, batch_first=True)
        hiddens, (h_t, c_t) = self.lstm(packed)
        hiddens, _ = unpack(hiddens, batch_first=True)
        preds = self.decoder(hiddens)
        adv_preds = self.discriminator(hiddens)
        return preds, hiddens, adv_preds

    def combine_loss(self, loss1, loss2, epoch):
        #e_scale = min((float(epoch) - 1) / 10.0, 1.0)
        #e_scale = 0.0 if e_scale < 0.0 else e_scale
        e_scale = 1.0
        loss = loss1 + (self.adv_lambda * e_scale * loss2)
        return loss

    def main_loss(self, preds, targets):
        preds = preds[targets != self.dictionary[SPECIAL_TOKENS.PAD], :]
        targets = targets[targets != self.dictionary[SPECIAL_TOKENS.PAD]]
        lprobs = F.log_softmax(preds, dim=1)
        _l = F.nll_loss(lprobs, targets, reduction='sum') 
        _a = preds.argmax(1).eq(targets).sum().item()
        _a = float(_a) / float(targets.size(0))
        return _l, _a

    def adv_loss(self, adv_preds, adv_targets):
        adv_mask = (adv_targets == self.adv_classes[0]) + (adv_targets == self.adv_classes[1])
        adv_masked_target = adv_targets[adv_mask]
        adv_masked_target[adv_masked_target == self.adv_classes[0]] = 0
        adv_masked_target[adv_masked_target == self.adv_classes[1]] = 1
        adv_lprobs = F.log_softmax(adv_preds, dim=1)
        adv_masked_lprobs = adv_lprobs[adv_mask, :]
        _l = F.nll_loss(adv_masked_lprobs, adv_masked_target, reduction='sum')
        return _l

    def train_step(self, inps, targets, lengths, adv_targets, epoch, hidden=None):
        self.optimizer.zero_grad()

        preds, hiddens, adv_preds = self(inps, lengths, hidden)

        main_loss, acc = self.main_loss(preds, targets)
        if self.adv_lambda > 0.0:
            adv_loss = self.adv_loss(adv_preds, adv_targets)
            loss = self.combine_loss(main_loss, adv_loss, epoch)
        else:
            adv_loss = torch.tensor(0.0)
            loss = main_loss

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.parameters()),
                                                   self.max_grad_norm)
        if math.isnan(grad_norm):
            print('skipping update grad_norm is nan!')
        else:
            self.optimizer.step()
        return loss.item(), main_loss.item(), adv_loss.item(), grad_norm, acc

    def generate(self, num_sentences=4, temp=1.0, max_len=80):
        inp_sym = torch.Tensor(num_sentences * [[self.dictionary[SPECIAL_TOKENS.BOS]]]).long()
        if self.is_cuda():
            inp_sym = inp_sym.cuda()
        generated = inp_sym.clone()
        with torch.no_grad():
            recurrent = None
            for _ in range(80):
                inp = self.encoder(inp_sym)
                out, recurrent = self.lstm(inp, recurrent)
                pred = self.decoder(out)
                pred = pred.squeeze(1).div(temp).exp()
                inp_sym = torch.multinomial(pred, 1)
                generated = torch.cat([generated, inp_sym], dim=1)
        generated = generated.cpu().numpy()
        txt = []
        for row in generated:
            sent = ['SAMPLE:'] + [self.idx_dictionary[i] for i in row if i not in
                                  [self.dictionary[SPECIAL_TOKENS.PAD],
                                      self.dictionary[SPECIAL_TOKENS.BOS],
                                      self.dictionary[SPECIAL_TOKENS.EOS]]]
            txt.append(' '.join(sent))
        return '\n'.join(txt)

    def save_model(self, path):
        torch.save(self, path)
