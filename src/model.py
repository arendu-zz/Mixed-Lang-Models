#!/usr/bin/env python
__author__ = 'arenduchintala'
import math
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from rewards import batch_cosine_sim
from rewards import score_embeddings
from rewards import prob_score_embeddings

import pdb

def get_unsort_idx(sort_idx):
    unsort_idx = torch.zeros_like(sort_idx).long().scatter_(0, sort_idx, torch.arange(sort_idx.size(0)).long())
    return unsort_idx


class WordRepresenter(nn.Module):
    def __init__(self, word2spelling, char2idx, cv_size, ce_size, cp_idx, cr_size, we_size,
                 bidirectional=False, dropout=0.3,
                 is_extra_feat_learnable=False, num_required_vocab=None, char_composition='RNN', pool='Ave'):
        super(WordRepresenter, self).__init__()
        self.word2spelling = word2spelling
        self.sorted_spellings, self.sorted_lengths, self.unsort_idx = self.init_word2spelling()
        self.v_size = len(self.sorted_lengths)
        self.char2idx = char2idx
        self.ce_size = ce_size
        self.we_size = we_size
        self.cv_size = cv_size
        self.cr_size = cr_size
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.ce_layer = torch.nn.Embedding(self.cv_size, self.ce_size, padding_idx=cp_idx)
        self.vocab_idx = torch.arange(self.v_size, requires_grad=False).long()
        self.ce_layer.weight = nn.Parameter(
            torch.FloatTensor(self.cv_size, self.ce_size).uniform_(-0.5 / self.ce_size, 0.5 / self.ce_size))
        self.char_composition = char_composition
        self.pool = pool
        if self.char_composition == 'RNN':
            self.c_rnn = torch.nn.LSTM(self.ce_size + 1, self.cr_size,
                                       bidirectional=bidirectional, batch_first=True,
                                       dropout=self.dropout)
            if self.cr_size * (2 if bidirectional else 1) != self.we_size:
                self.c_proj = torch.nn.Linear(self.cr_size * (2 if bidirectional else 1), self.we_size)
                print('using Linear c_proj layer')
            else:
                print('no Linear c_proj layer')
                self.c_proj = None
        elif self.char_composition == 'CNN':
            assert self.we_size % 4 == 0
            self.c1d_3g = torch.nn.Conv1d(self.ce_size + 1, self.we_size // 4, 3)
            self.c1d_4g = torch.nn.Conv1d(self.ce_size + 1, self.we_size // 4, 4)
            self.c1d_5g = torch.nn.Conv1d(self.ce_size + 1, self.we_size // 4, 5)
            self.c1d_6g = torch.nn.Conv1d(self.ce_size + 1, self.we_size // 4, 6)
            if self.pool == 'Ave':
                self.max_3g = torch.nn.AvePool1d(self.sorted_spellings.size(1) - 3 + 1)
                self.max_4g = torch.nn.AvePool1d(self.sorted_spellings.size(1) - 4 + 1)
                self.max_5g = torch.nn.AvePool1d(self.sorted_spellings.size(1) - 5 + 1)
                self.max_6g = torch.nn.AvePool1d(self.sorted_spellings.size(1) - 6 + 1)
            elif self.pool == 'Max':
                self.max_3g = torch.nn.MaxPool1d(self.sorted_spellings.size(1) - 3 + 1)
                self.max_4g = torch.nn.MaxPool1d(self.sorted_spellings.size(1) - 4 + 1)
                self.max_5g = torch.nn.MaxPool1d(self.sorted_spellings.size(1) - 5 + 1)
                self.max_6g = torch.nn.MaxPool1d(self.sorted_spellings.size(1) - 6 + 1)
            else:
                raise BaseException("uknown pool")
        else:
            raise BaseException("Unknown seq model")

        self.num_required_vocab = num_required_vocab if num_required_vocab is not None else self.v_size
        self.extra_ce_layer = torch.nn.Embedding(self.v_size, 1)
        self.extra_ce_layer.weight = nn.Parameter(torch.ones(self.v_size, 1))
        print('WordRepresenter init complete.')

    def set_extra_feat_learnable(self, is_extra_feat_learnable):
        self.is_extra_feat_learnable = is_extra_feat_learnable
        self.extra_ce_layer.weight.requires_grad = is_extra_feat_learnable

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
        #sorted_spellings = Variable(sorted_spellings, requires_grad=False)
        return sorted_spellings, sorted_lengths, unsort_idx

    def init_cuda(self,):
        self = self.cuda()
        self.sorted_spellings = self.sorted_spellings.cuda()
        self.unsort_idx = self.unsort_idx.cuda()
        self.vocab_idx = self.vocab_idx.cuda()

    def cnn_representer(self, emb):
        # (batch, seq_len, char_emb_size)
        emb = emb.transpose(1, 2)
        m_3g = self.max_3g(self.c1d_3g(emb)).squeeze()
        m_4g = self.max_4g(self.c1d_4g(emb)).squeeze()
        m_5g = self.max_5g(self.c1d_5g(emb)).squeeze()
        m_6g = self.max_6g(self.c1d_6g(emb)).squeeze()
        word_embeddings = torch.cat([m_3g, m_4g, m_5g, m_6g], dim=1)
        del emb, m_3g, m_4g, m_5g, m_6g
        return word_embeddings

    def rnn_representer(self, emb):
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
        return word_embeddings

    def forward(self,):
        emb = self.ce_layer(self.sorted_spellings)
        extra_emb = self.extra_ce_layer(self.vocab_idx).unsqueeze(1)
        extra_emb = extra_emb.expand(extra_emb.size(0), emb.size(1), extra_emb.size(2))
        emb = torch.cat((emb, extra_emb), dim=2)
        if not hasattr(self, 'char_composition'):  # for back compatability
            word_embeddings = self.rnn_representer(emb)
        elif self.char_composition == 'RNN':
            word_embeddings = self.rnn_representer(emb)
        elif self.char_composition == 'CNN':
            word_embeddings = self.cnn_representer(emb)
        else:
            raise BaseException("unknown char_composition")

        unsorted_word_embeddings = word_embeddings[self.unsort_idx, :]
        if self.num_required_vocab > unsorted_word_embeddings.size(0):
            e = unsorted_word_embeddings[0].unsqueeze(0)
            e = e.expand(self.num_required_vocab - unsorted_word_embeddings.size(0), e.size(1))
            unsorted_word_embeddings = torch.cat([unsorted_word_embeddings, e], dim=0)
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
    L1_LEARNING = 'L1_LEARNING'
    L2_LEARNING = 'L2_LEARNING'
    L3_LEARNING = 'L3_LEARNING'

    def __init__(self, input_size,
                 encoder, decoder,
                 g_encoder, g_decoder,
                 mode,
                 dropout=0.3, max_grad_norm=5.0):
        super(CBiLSTM, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.g_encoder = g_encoder
        self.g_decoder = g_decoder
        self.dropout = nn.Dropout(dropout)
        self.max_grad_norm = max_grad_norm
        self.input_size = input_size
        assert self.input_size % 2 == 0
        self.rnn_size = self.input_size // 2
        self.rnn = nn.LSTM(self.input_size, self.rnn_size, dropout=dropout,
                           batch_first=True,
                           bidirectional=True)
        self.init_param_freeze(mode)
        self.loss = torch.nn.CrossEntropyLoss(size_average=True, ignore_index=0)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()))
        self.eos_sym = None
        self.bos_sym = None
        #self.z = Variable(torch.zeros(1, 1, self.rnn_size), requires_grad=False)
        self.z = torch.zeros(1, 1, self.rnn_size, requires_grad=False)
        # .expand(batch_size, 1, self.rnn_size), requires_grad=False)

    def init_optimizer(self, type='Adam'):
        if type == 'Adam':
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()))
        elif type == 'SGD':
            self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=1.0)
        elif type == 'LBFGS':
            self.optimizer = torch.optim.LBFGS(filter(lambda p: p.requires_grad, self.parameters()),
                                               max_iter=3, history_size=2)
        else:
            raise NotImplementedError("unknonw optimizer option")

    def init_cuda(self,):
        self = self.cuda()
        self.z = self.z.cuda()

    def init_param_freeze(self, mode):
        self.mode = mode
        if self.mode == CBiLSTM.L2_LEARNING or self.mode == CBiLSTM.L3_LEARNING:
            assert self.g_encoder is not None
            assert self.g_decoder is not None
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.decoder.parameters():
                p.requires_grad = False
            for p in self.rnn.parameters():
                p.requires_grad = False
            print('L2_LEARNING, L1 Parameters frozen')
            for p in self.g_encoder.parameters():
                p.requires_grad = True
            for p in self.g_decoder.parameters():
                p.requires_grad = True
            if isinstance(self.g_encoder, VarEmbedding):
                self.g_encoder.word_representer.set_extra_feat_learnable(True)
                assert isinstance(self.g_decoder, VarLinear)
                assert self.g_decoder.word_representer.is_extra_feat_learnable
        elif self.mode == CBiLSTM.L1_LEARNING:
            if self.g_encoder is not None:
                for p in self.g_encoder.parameters():
                    p.requires_grad = False
            if self.g_decoder is not None:
                for p in self.g_decoder.parameters():
                    p.requires_grad = False
            print('L1_LEARNING, L2 Parameters frozen')
            for p in self.encoder.parameters():
                p.requires_grad = True
            for p in self.decoder.parameters():
                p.requires_grad = True
            for p in self.rnn.parameters():
                p.requires_grad = True
            if isinstance(self.encoder, VarEmbedding):
                self.encoder.word_representer.set_extra_feat_learnable(False)
                assert isinstance(self.decoder, VarLinear)
                assert not self.decoder.word_representer.is_extra_feat_learnable

    def is_cuda(self,):
        return self.rnn.weight_hh_l0.is_cuda

    def score_embeddings(self, l2_key, l1_key):
        if isinstance(self.encoder, VarEmbedding):
            raise NotImplementedError("only word level scores")
        else:
            l1_embedding = self.encoder.weight.data
            l2_embedding = self.g_encoder.weight.data
            l2_key = l2_key.cuda() if self.is_cuda() else l2_key
            l1_key = l1_key.cuda() if self.is_cuda() else l1_key
            s = score_embeddings(l2_embedding, l1_embedding, l2_key, l1_key)
            # ps = prob_score_embeddings(l2_embedding, l1_embedding, l2_key, l1_key)
            return s.item()

    def forward(self, batch, seen=None):
        lengths, data, ind = batch
        v_ind = ind % 2
        g_ind = ind - 1
        g_ind[g_ind < 0] = 0
        if seen is not None:
            seen, seen_offset, seen_set = seen
        # if self.is_cuda():
        #     data = data.cuda()
        batch_size = data.size(0)
        # max_seq_len = data.size(1)
        assert data.dim() == 2

        # data = (batch_size x seq_len)
        v_encoded = self.encoder(data)
        v_inp_ind = v_ind.unsqueeze(2).expand(v_ind.size(0), v_ind.size(1), v_encoded.size(2)).float()
        if self.mode == CBiLSTM.L2_LEARNING:
            g_encoded = self.g_encoder(data)
            g_inp_ind = g_ind.unsqueeze(2).expand(g_ind.size(0), g_ind.size(1), g_encoded.size(2)).float()
            encoded = v_inp_ind * v_encoded + g_inp_ind * g_encoded
            encoded = self.dropout(encoded)
        else:
            encoded = self.dropout(v_encoded)

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
        final_hidden = self.dropout(final_hidden)

        v_out = self.decoder(final_hidden)
        # v_out = (batch_size, seq_len, v_vocab_size)
        if self.mode == CBiLSTM.L2_LEARNING or self.mode == CBiLSTM.L3_LEARNING:
            g_out = self.g_decoder(final_hidden)
        # g_out = (batch_size, seq_len, g_vocab_size)

        # flatten tensors for loss computation
        if self.mode == CBiLSTM.L1_LEARNING:
            data = data.view(-1)
            v_ind = v_ind.view(-1)
            d_idx = torch.arange(data.size(0)).type_as(data.data).long()
            v_idx = d_idx[v_ind.data == 1]
            v_data = data[v_idx]
            v_out = v_out.view(-1, v_out.size(2))[v_idx, :]
            loss = self.loss(v_out, v_data)

        if self.mode == CBiLSTM.L2_LEARNING:
            data = data.view(-1)
            v_ind = v_ind.view(-1)
            d_idx = torch.arange(data.size(0)).type_as(data.data).long()
            v_idx = d_idx[v_ind.data == 1]
            v_data = data[v_idx]
            v_out = v_out.view(-1, v_out.size(2))[v_idx, :]
            loss = self.loss(v_out, v_data)

            g_ind = g_ind.view(-1)
            g_idx = d_idx[g_ind.data == 1]
            if g_idx.dim() > 0:
                g_data = data[g_idx]
                g_out = g_out.view(-1, g_out.size(2))[g_idx, :]
                loss += self.loss(g_out, g_data)

        if self.mode == CBiLSTM.L3_LEARNING:
            all_out = torch.cat([v_out, g_out[:, :, seen_set]], dim=2)
            all_data = data.clone()
            all_data[g_ind == 1] = seen_offset + v_out.size(2)
            all_data = all_data.view(-1)
            all_out = all_out.view(-1, all_out.size(2))
            loss = self.loss(all_out, all_data)

        return loss

    def do_backprop(self, batch, seen=None):
        self.optimizer.zero_grad()
        l = self(batch, seen)
        l.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.parameters()),
                                                   self.max_grad_norm)
        if math.isnan(grad_norm):
            print('skipping update grad_norm is nan')
        else:
            self.optimizer.step()
        loss = l.item()
        del l, batch
        return loss, grad_norm

    def save_model(self, path):
        torch.save(self, path)
