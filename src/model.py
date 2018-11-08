#!/usr/bin/env python
__author__ = 'arenduchintala'
import math
import pdb
import torch
import torch.nn as nn
import numpy as np

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from rewards import batch_cosine_sim
from rewards import score_embeddings
from rewards import prob_score_embeddings


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


def log_gaussian(x, mu, sigma, log_sigma):
    s = -0.5 * float(np.log(2 * np.pi)) - log_sigma - (((x - mu) ** 2) / (2 * sigma ** 2))
    #return -0.5 * np.log(2 * np.pi) - torch.log(sigma) - (x - mu) ** 2 / (2 * sigma ** 2)
    return s

#def log_gaussian(x, mu, sigma):
#    return float(-0.5 * np.log(2 * np.pi) - np.log(np.abs(sigma))) - (x - mu)**2 / (2 * sigma**2)


#def log_gaussian_logsigma(x, mu, logsigma):
#    return float(-0.5 * np.log(2 * np.pi)) - logsigma - (x - mu)**2 / (2 * torch.exp(logsigma)**2)


class VariationalEmbeddings(nn.Module):
    def __init__(self, mean, rho, sigma_prior=1.):
        super(VariationalEmbeddings, self).__init__()
        self.mean = mean
        self.rho = rho
        self.log_p_w = 0.
        self.log_q_w = 0.
        self.sigma_prior = sigma_prior

    def forward(self, data):
        return self.lookup(data)

    def reparameterize(self,):
        if self.training:
            #std = torch.exp(0.5 * self.rho)
            std = torch.log(1. + torch.exp(self.rho))  # softplus instead of exp
            eps = torch.randn_like(std)
            eps.requires_grad = False
            embeddings = self.mean + std * eps
            self.log_p_w = log_gaussian(embeddings, 0., self.sigma_prior, float(np.log(self.sigma_prior))).sum()
            self.log_q_w = log_gaussian(embeddings, self.mean, std, torch.log(std)).sum()
            return embeddings
        else:
            return self.mean

    def sample_lookup(self, data):
        batch_size, seq_len = data.shape

    def lookup(self, data):
        embeddings = self.reparameterize()
        embedding_size = embeddings.size(1)
        if data.dim() == 2:
            batch_size = data.size(0)
            seq_len = data.size(1)
            data = data.contiguous()
            data = data.view(-1)  # , data.size(0), data.size(1))
            var_data = embeddings[data]
            var_data = var_data.view(batch_size, seq_len, embedding_size)
        else:
            var_data = embeddings[data]
        return var_data


class VariationalLinear(nn.Module):
    def __init__(self, mean, rho):
        super(VariationalLinear, self).__init__()
        self.mean = mean
        self.rho = rho

    def forward(self, data):
        return self.matmul(data)

    def reparameterize(self,):
        if self.training:
            std = torch.log(1. + torch.exp(self.rho))
            eps = torch.randn_like(std)
            eps.requires_grad = False
            embeddings = self.mean + std * eps
            return embeddings
        else:
            return self.mean

    def matmul(self, data):
        embeddings = self.reparameterize()
        if data.dim() > 1:
            assert data.size(-1) == embeddings.size(-1)
            return torch.matmul(data, embeddings.transpose(0, 1))
        else:
            raise BaseException("data should be at least 2 dimensional")


class CBiLSTM(nn.Module):
    L1_LEARNING = 'L1_LEARNING'  # updates only l1 params i.e. base language model
    L12_LEARNING = 'L12_LEARNING'  # updates both l1 params and l2 params (novel vocab embeddings)
    L2_LEARNING = 'L2_LEARNING'  # update only l2 params

    def __init__(self, input_size, rnn_size, layers,
                 encoder, decoder,
                 g_encoder, g_decoder,
                 mode,
                 dropout=0.3,
                 max_grad_norm=10.0,
                 size_average=False):
        super(CBiLSTM, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.g_encoder = g_encoder
        self.g_decoder = g_decoder
        self.dropout_val = dropout
        self.dropout = nn.Dropout(self.dropout_val)
        self.max_grad_norm = max_grad_norm
        self.input_size = input_size
        self.rnn_size = rnn_size
        # self.rnn = nn.LSTM(self.input_size, self.rnn_size, dropout=dropout,
        #                   num_layers=layers,
        #                   batch_first=True,
        #                   bidirectional=True)
        self.fwd_rnn = nn.LSTM(self.input_size, self.rnn_size, dropout=dropout,
                               num_layers=layers,
                               batch_first=True,
                               bidirectional=False)
        self.bwd_rnn = nn.LSTM(self.input_size, self.rnn_size, dropout=dropout,
                               num_layers=layers,
                               batch_first=True,
                               bidirectional=False)

        self.linear = nn.Linear(2 * self.rnn_size, self.input_size)
        self.init_param_freeze(mode)
        self.loss = torch.nn.CrossEntropyLoss(size_average=size_average, reduce=True, ignore_index=0)
        self.eos_sym = None
        self.bos_sym = None
        #self.z = Variable(torch.zeros(1, 1, self.rnn_size), requires_grad=False)
        self.z = torch.zeros(1, 1, self.rnn_size, requires_grad=False)
        # .expand(batch_size, 1, self.rnn_size), requires_grad=False)
        self.l1_key = None
        self.l2_key = None
        self.init_optimizer(type='SGD')

    def init_optimizer(self, type='Adam'):
        #grad_params = [p for p in self.parameters() if p.requires_grad]
        #total_params = [p for p in self.parameters()]
        #print(str(len(grad_params)) + '/' + str(len(total_params)) + ' params requires_grad')
        if type == 'Adam':
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()))
        elif type == 'SGD':
            self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=1.0)
        else:
            raise NotImplementedError("unknown optimizer option")

    def set_key(self, l1_key, l2_key):
        self.l1_key = l1_key
        self.l2_key = l2_key

    def init_key(self,):
        if self.l1_key is not None:
            if self.is_cuda():
                self.l1_key = self.l1_key.cuda()
            else:
                pass
        if self.l2_key is not None:
            if self.is_cuda():
                self.l2_key = self.l2_key.cuda()
            else:
                pass

    def init_cuda(self,):
        self = self.cuda()
        self.z = self.z.cuda()

    def init_param_freeze(self, mode):
        self.mode = mode
        if self.mode == CBiLSTM.L12_LEARNING or self.mode == CBiLSTM.L2_LEARNING:
            self.dropout = nn.Dropout(0.0)
            assert self.g_encoder is not None
            assert self.g_decoder is not None
            for p in self.parameters():
                p.requires_grad = False
            for p in self.g_encoder.parameters():
                p.requires_grad = True
            for p in self.g_decoder.parameters():
                p.requires_grad = True
            if isinstance(self.g_encoder, VarEmbedding):
                self.g_encoder.word_representer.set_extra_feat_learnable(True)
                assert isinstance(self.g_decoder, VarLinear)
                assert self.g_decoder.word_representer.is_extra_feat_learnable
            # print('L2_LEARNING, L1 Parameters frozen')
        elif self.mode == CBiLSTM.L1_LEARNING:
            self.dropout = nn.Dropout(0.0)
            for p in self.parameters():
                p.requires_grad = True
            if self.g_encoder is not None:
                for p in self.g_encoder.parameters():
                    p.requires_grad = False
            if self.g_decoder is not None:
                for p in self.g_decoder.parameters():
                    p.requires_grad = False
            # print('L1_LEARNING, L2 Parameters frozen')
            if isinstance(self.encoder, VarEmbedding):
                self.encoder.word_representer.set_extra_feat_learnable(False)
                assert isinstance(self.decoder, VarLinear)
                assert not self.decoder.word_representer.is_extra_feat_learnable

    def is_cuda(self,):
        return self.fwd_rnn.weight_hh_l0.is_cuda

    def set_reset_weight(self,):
        self.reset_weight = self.g_encoder.weight.detach().clone()

    def update_g_weights(self, weights):
        if self.is_cuda():
            weights = weights.clone().cuda()
        else:
            weights = weights.clone()
        self.g_encoder.weight.data = weights
        self.g_decoder.weight.data = weights

    def score_embeddings(self,):
        if isinstance(self.encoder, VarEmbedding):
            raise NotImplementedError("only word level scores")
        else:
            l1_embedding = self.encoder.weight.data
            l2_embedding = self.g_encoder.weight.data
            # l2_key = l2_key.cuda() if self.is_cuda() else l2_key
            # l1_key = l1_key.cuda() if self.is_cuda() else l1_key
            s = score_embeddings(l2_embedding, l1_embedding, self.l2_key, self.l1_key)

            # ps = prob_score_embeddings(l2_embedding, l1_embedding, l2_key, l1_key)
            return s.item()

    def forward(self, batch, l2_seen=None):
        lengths, l1_data, l2_data, ind = batch
        rev_idx_col = torch.zeros(l1_data.size(0), l1_data.size(1)).long()
        for _idx, l in enumerate(lengths):
            rev_idx_col[_idx, :] = torch.LongTensor(list(range(l - 1, -1, -1)) + list(range(l, lengths[0])))

        rev_idx_row = torch.arange(len(lengths)).long()
        rev_idx_row = rev_idx_row.unsqueeze(1).expand(l1_data.shape[0], l1_data.shape[1])
        #if data.is_cuda:
        #    rev_idx_row = rev_idx_row.cuda()
        v_ind = ind % 2
        g_ind = ind - 1
        g_ind[g_ind < 0] = 0
        #if seen is not None:
        #    seen, seen_offset, seen_set = seen
        # if self.is_cuda():
        #     data = data.cuda()
        batch_size = l1_data.size(0)
        # max_seq_len = data.size(1)

        # l1_data = (batch_size x seq_len)
        # l2_data = (batch_size x seq_len)
        if self.mode == CBiLSTM.L2_LEARNING:
            l1_encoded = self.encoder(l1_data)
            l2_encoded = self.g_encoder(l2_data)
            g_inp_ind = g_ind.unsqueeze(2).expand(g_ind.size(0), g_ind.size(1), l2_encoded.size(2)).float()
            v_inp_ind = v_ind.unsqueeze(2).expand(v_ind.size(0), v_ind.size(1), l1_encoded.size(2)).float()
            tmp_encoded = v_inp_ind * l1_encoded + g_inp_ind * l2_encoded
            encoded = l1_encoded * v_ind.unsqueeze(2).expand_as(l1_encoded).float() + \
                l2_encoded * g_ind.unsqueeze(2).expand_as(l2_encoded).float()
            assert (encoded - tmp_encoded).sum().item() == 0
            encoded = self.dropout(encoded)
        elif self.mode == CBiLSTM.L12_LEARNING:
            raise NotImplementedError("no longer supported")
            l1_encoded = self.encoder(l1_data)
            l2_encoded = self.g_encoder(l2_data)
            g_inp_ind = g_ind.unsqueeze(2).expand(g_ind.size(0), g_ind.size(1), l2_encoded.size(2)).float()
            v_inp_ind = v_ind.unsqueeze(2).expand(v_ind.size(0), v_ind.size(1), l1_encoded.size(2)).float()
            tmp_encoded = v_inp_ind * l1_encoded + g_inp_ind * l2_encoded
            encoded = l1_encoded * v_ind.unsqueeze(2).expand_as(l1_encoded).float() + \
                l2_encoded * g_ind.unsqueeze(2).expand_as(l2_encoded).float()
            assert (encoded - tmp_encoded).sum().item() == 0
            encoded = self.dropout(encoded)
        else:
            l1_encoded = self.encoder(l1_data)
            encoded = self.dropout(l1_encoded)

        # bwd_encoded = torch.zeros_like(fwd_encoded)
        # for _idx, l in enumerate(lengths):
        #    bwd_encoded[_idx, :, :] = fwd_encoded[_idx, rev_idx_col[_idx, :], :]

        rev_encoded = encoded[rev_idx_row, rev_idx_col, :]
        # assert (tmp - bwd_encoded).sum().item() == 0

        fwd_packed_encoded = pack(encoded, lengths, batch_first=True)
        bwd_packed_encoded = pack(rev_encoded, lengths, batch_first=True)
        # encoded = (batch_size x seq_len x embedding_size)
        #packed_hidden, (h_t, c_t) = self.rnn(packed_encoded)
        #hidden, lengths = unpack(packed_hidden, batch_first=True)
        fwd_packed_hidden, (fwd_h_t, fwd_c_t) = self.fwd_rnn(fwd_packed_encoded)
        fwd_hidden, fwd_lengths = unpack(fwd_packed_hidden, batch_first=True)

        bwd_packed_hidden, (bwd_h_t, bwd_c_t) = self.bwd_rnn(bwd_packed_encoded)
        bwd_hidden, bwd_lengths = unpack(bwd_packed_hidden, batch_first=True)
        # rev_bwd_hidden = torch.zeros_like(bwd_hidden)
        # for _idx, l in enumerate(lengths):
        #    rev_bwd_hidden[_idx, :, :] = bwd_hidden[_idx, rev_idx_col[_idx, :], :]

        rev_bwd_hidden = bwd_hidden[rev_idx_row, rev_idx_col, :]
        #assert(tmp - rev_bwd_hidden).sum() == 0

        # hidden = (batch_size x seq_len x rnn_size)
        z = self.z.expand(batch_size, 1, self.rnn_size)
        fwd_hidden = torch.cat((z, fwd_hidden[:, :-1, :]), dim=1)
        rev_bwd_hidden = torch.cat((rev_bwd_hidden[:, 1:, :], z), dim=1)
        # bwd_hidden = (batch_size x seq_len x rnn_size)
        # fwd_hidden = (batch_size x seq_len x rnn_size)
        final_hidden = torch.cat((fwd_hidden, rev_bwd_hidden), dim=2)
        final_hidden = self.dropout(final_hidden)
        final_hidden = self.linear(final_hidden)

        if self.mode == CBiLSTM.L1_LEARNING:
            l1_final_hidden = final_hidden[v_ind == 1, :]
            l1_out = self.decoder(l1_final_hidden)
            loss = self.loss(l1_out, l1_data[v_ind == 1])
        elif self.mode == CBiLSTM.L2_LEARNING or self.mode == CBiLSTM.L12_LEARNING:
            l1_final_hidden = final_hidden[v_ind == 1, :]
            l1_out = self.decoder(l1_final_hidden)
            l1_loss = self.loss(l1_out, l1_data[v_ind == 1])
            l2_final_hidden = final_hidden[v_ind == 0, :]
            if l2_final_hidden.shape[0] > 0:
                l2_out = self.g_decoder(l2_final_hidden)
                l2_loss = self.loss(l2_out, l2_data[v_ind == 0])
                loss = l1_loss + l2_loss
            else:
                loss = l1_loss
        else:
            raise BaseException("unknown learning type")
        return loss

    def do_backprop(self, batch, l2_seen=None, total_batches=None):
        self.optimizer.zero_grad()
        _l = self(batch, l2_seen=l2_seen)
        if isinstance(self.encoder, VariationalEmbeddings):
            # print('encoder mu', self.encoder.mean[10].sum())
            # print('decoder mu', self.decoder.mean[10].sum())
            # print('encoder rho', self.encoder.rho[10].sum())
            # print('decoder rho', self.decoder.rho[10].sum())
            # print('\nlpw', self.encoder.log_p_w.item())
            # print('lqw', self.encoder.log_q_w.item())
            kl_loss = (1. / total_batches) * (self.encoder.log_q_w - self.encoder.log_p_w)
            # print('kl', kl_loss.item())
            # print('l', l.item())
            _l += kl_loss
            # print('full', l.item())

        _l.backward()

        if self.mode == CBiLSTM.L2_LEARNING:
                keep_grad = torch.zeros_like(self.g_encoder.weight.grad)
                keep_grad[l2_seen, :] = 1.0
                self.g_encoder.weight.grad *= keep_grad
        # print(self.encoder.weight[10], self.encoder.weight[10].sum())
        #grad_sum = sum([p.grad.data.sum().item() for p in self.parameters() if p.requires_grad])
        grad_norm = torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.parameters()),
                                                   self.max_grad_norm)
        if math.isnan(grad_norm):
            print('skipping update grad_norm is nan!')
        else:
            self.optimizer.step()
        loss = _l.item()
        #del _l
        #del batch
        return loss, grad_norm

    def save_model(self, path):
        torch.save(self, path)
