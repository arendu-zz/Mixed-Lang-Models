#!/usr/bin/env python
__author__ = 'arenduchintala'
import math
import torch
import torch.nn as nn
import numpy as np
import pdb

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from src.utils.utils import SPECIAL_TOKENS
from src.opt.noam import NoamOpt

from src.rewards import batch_cosine_sim
from src.rewards import score_embeddings
from src.rewards import prob_score_embeddings
from src.rewards import rank_score_embeddings


from .transformer_encoder_layer import TransformerEncoderLayer

def get_unsort_idx(sort_idx):
    unsort_idx = torch.zeros_like(sort_idx).long().scatter_(0, sort_idx, torch.arange(sort_idx.size(0)).long())
    return unsort_idx


def make_vl_encoder(mean, rho, sigma_prior):
    print('making VariationalEmbeddings with', sigma_prior)
    variational_embedding = VariationalEmbeddings(mean, rho, sigma_prior)
    return variational_embedding


def make_vl_decoder(mean, rho):
    variational_linear = VariationalLinear(mean, rho)
    return variational_linear


def make_cl_encoder(word_representer):
    e = VarEmbedding(word_representer)
    return e


def make_cl_decoder(word_representer):
    d = VarLinear(word_representer)
    return d


def make_wl_encoder(vocab_size=None, embedding_size=None, wt=None):
    if wt is None:
        assert vocab_size is not None
        assert embedding_size is not None
        e = torch.nn.Embedding(vocab_size, embedding_size)
        torch.nn.init.xavier_uniform_(e.weight)
        #e.weight = torch.nn.Parameter(torch.FloatTensor(vocab_size, embedding_size).uniform_(-0.01 / embedding_size,
        #                                                                                     0.01 / embedding_size))
    else:
        e = torch.nn.Embedding(wt.size(0), wt.size(1))
        e.weight = torch.nn.Parameter(wt)
    return e


def make_wl_decoder(encoder):
    decoder = torch.nn.Linear(encoder.weight.size(1), encoder.weight.size(0), bias=False)
    decoder.weight = encoder.weight
    #torch.nn.init.xavier_uniform_(decoder.weight)
    return decoder


class SinusoidalPositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, max_len, embed_size):
        super(SinusoidalPositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() *
                             -(math.log(10000.0) / embed_size)).unsqueeze(0).float()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = nn.Embedding(max_len, embed_size)
        self.pe.weight.data = pe
        self.pe.weight.requires_grad = False

    def forward(self, x):
        print(x.shape)
        pos = torch.arange(x.shape[1]).expand_as(x).type_as(x)
        return self.pe(pos)


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


class MapLinear(nn.Module):
    def __init__(self, l1_weights, map_weights):
        super(MapLinear, self).__init__()
        self.l1_weights = l1_weights
        self.l1_weights.requires_grad = False
        #shape = l1_vocab x emb_size
        #l1_voc, l2_voc = self.map.shape
        self.map_weights = map_weights
        self.map_weights.requires_grad = True
        self.register_buffer('l1_weight', self.l1_weights)
        self.register_buffer('map_weight', self.map_weights)

    def get_l2_weights(self,):
        l2_weights = torch.nn.functional.softmax(self.map_weights, dim=1)
        l2_weights = l2_weights.matmul(self.l1_weights)
        return l2_weights

    def forward(self, x):
        l2_weights = self.get_l2_weights()
        if x.dim() > 1:
            assert x.size(-1) == l2_weights.size(-1)
            return torch.matmul(x, l2_weights.transpose(0, 1))
        else:
            raise BaseException("x should be at least 2 dimensional")
        return None


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


class CEncoderModel(nn.Module):
    L1_LEARNING = 'L1_LEARNING'  # updates only l1 params i.e. base language model
    L12_LEARNING = 'L12_LEARNING'  # updates both l1 params and l2 params (novel vocab embeddings)
    L2_LEARNING = 'L2_LEARNING'  # update only l2 params

    def __init__(self,
                 encoder,
                 decoder,
                 l2_encoder,
                 l2_decoder,
                 mode,
                 l1_dict,
                 l2_dict,
                 dropout=0.3,
                 max_grad_norm=5.,
                 size_average=False,
                 use_positional_embeddings=False):
        super().__init__()
        self.mode = mode
        self.l1_dict = l1_dict
        self.l2_dict = l2_dict
        self.encoder = encoder
        self.decoder = decoder
        self.l2_encoder = l2_encoder
        self.l2_decoder = l2_decoder
        self.dropout_val = dropout
        self.dropout = nn.Dropout(self.dropout_val)
        self.max_grad_norm = max_grad_norm
        self.emb_size = self.encoder.weight.shape[1]
        self.emb_max = self.encoder.weight.max().item()
        self.emb_min = self.encoder.weight.min().item()
        self.use_positional_embeddings = use_positional_embeddings
        self.mask_unseen_l2 = 0

    def join_l2_weights(self,):
        #print(id(self.encoder.weight))
        #print(id(self.decoder.weight))
        #print(id(self.l2_encoder.weight))
        #print(id(self.l2_decoder.weight))
        l1_wt = self.encoder.weight.data.clone()
        l2_wt = self.l2_encoder.weight.data.clone()
        l2_l1_wt = torch.cat([l2_wt, l1_wt], dim=0)
        self.l2_encoder = make_wl_encoder(None, None, l2_l1_wt)
        self.l2_decoder = make_wl_decoder(self.l2_encoder)
        #print(id(self.l2_encoder.weight))
        #print(id(self.l2_decoder.weight))
        return True

    def forward(self, batch, l2_seen, l2_unseen):
        raise NotImplementedError
    
    def do_backprop(self, batch, l2_seen, total_batches=None):
        if l2_seen is not None and self.mask_unseen_l2 == 1:
            l2_unseen = set(range(len(self.l2_dict))) - set(l2_seen.cpu().tolist())
            l2_unseen = torch.Tensor(list(l2_unseen)).type_as(l2_seen)
        else:
            l2_unseen = None
        self.zero_grad()
        _l, _a = self(batch, l2_seen, l2_unseen)
        if isinstance(self.encoder, VariationalEmbeddings):
            kl_loss = (1. / total_batches) * (self.encoder.log_q_w - self.encoder.log_p_w)
            _l += kl_loss

        _l.backward()
        if self.mode == CEncoderModel.L2_LEARNING:
            keep_grad = torch.zeros_like(self.l2_encoder.weight.grad)
            keep_grad[l2_seen, :] = 1.0
            self.l2_encoder.weight.grad *= keep_grad
        grad_norm = torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.parameters()),
                                                   self.max_grad_norm)
        if math.isnan(grad_norm):
            print('skipping update grad_norm is nan!')
        else:
            self.optimizer.step()
        loss = _l.item()
        #del _l
        #del batch
        return loss, grad_norm, _a

    def init_param_freeze(self, mode):
        self.mode = mode
        if self.mode == CEncoderModel.L12_LEARNING or self.mode == CEncoderModel.L2_LEARNING:
            self.dropout = nn.Dropout(0.0) ## we do this because cudnn RNN backward does not work in eval model...
            assert self.l2_encoder is not None
            assert self.l2_decoder is not None
            for p in self.parameters():
                p.requires_grad = False
            for p in self.l2_encoder.parameters():
                p.requires_grad = True
            for p in self.l2_decoder.parameters():
                p.requires_grad = True
            if isinstance(self.l2_encoder, VarEmbedding):
                self.l2_encoder.word_representer.set_extra_feat_learnable(True)
                assert isinstance(self.l2_decoder, VarLinear)
                assert self.l2_decoder.word_representer.is_extra_feat_learnable
            #print('L2_LEARNING, L1 Parameters frozen')
        elif self.mode == CEncoderModel.L1_LEARNING:
            self.dropout = nn.Dropout(self.dropout_val)
            for p in self.parameters():
                p.requires_grad = True
            if self.l2_encoder is not None:
                for p in self.l2_encoder.parameters():
                    p.requires_grad = False
            if self.l2_decoder is not None:
                for p in self.l2_decoder.parameters():
                    p.requires_grad = False
            #print('L1_LEARNING, L2 Parameters frozen')
            if isinstance(self.encoder, VarEmbedding):
                self.encoder.word_representer.set_extra_feat_learnable(False)
                assert isinstance(self.decoder, VarLinear)
                assert not self.decoder.word_representer.is_extra_feat_learnable

    def init_optimizer(self, type='Adam', lr=1.0):
        if type == 'Adam':
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()))
        elif type == 'SGD':
            self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        elif type == 'noam':
            _optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()))
            self.optimizer = NoamOpt(self.emb_size, 100, _optimizer)
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
        raise NotImplementedError

    def is_cuda(self,):
        raise NotImplementedError

    def set_reset_weight(self,):
        self.reset_weight = self.l2_encoder.weight.detach().clone()

    def get_weight(self,):
        if self.is_cuda:
            weights = self.l2_encoder.weight.clone().detach().cpu()
        else:
            weights = self.l2_encoder.weight.clone().detach()
        return weights

    def update_g_weights(self, weights):
        if self.is_cuda():
            weights = weights.clone().cuda()
        else:
            weights = weights.clone()
        self.l2_encoder.weight.data = weights
        self.l2_decoder.weight.data = weights

    def score_embeddings(self, l2_embedding):
        if isinstance(self.encoder, VarEmbedding):
            raise NotImplementedError("only word level scores")
        else:
            l1_embedding = self.encoder.weight.data
            #l2_embedding = self.l2_encoder.weight.data
            #rank_score_embeddings(l2_embedding, l1_embedding, self.l2_key, self.l1_key)
            s = score_embeddings(l2_embedding, l1_embedding, self.l2_key, self.l1_key)
            return s

    def save_model(self, path):
        torch.save(self, path)


class CTransformerEncoder(CEncoderModel):
    def __init__(self, input_size, model_size, layers,
                 encoder, decoder,
                 l2_encoder, l2_decoder,
                 mode,
                 l1_dict,
                 l2_dict,
                 size_average=False,
                 max_positional_embeddings=500,
                 positional_embeddings_type='none'):
        super().__init__(encoder, decoder, l2_encoder, l2_decoder, mode, l1_dict, l2_dict)
        self.model_size = model_size
        self.l1_embed_scale = math.sqrt(self.encoder.weight.shape[1])
        if self.l2_encoder is not None:
            self.l2_embed_scale = math.sqrt(self.l2_encoder.weight.shape[1])
        else:
            self.l2_embed_scale = None
        self.input_size = input_size
        self.max_positional_embeddings = max_positional_embeddings
        if positional_embeddings_type == 'sinusoidal':
            self.positional_embeddings = SinusoidalPositionalEncoding(max_positional_embeddings,
                                                                      self.encoder.weight.shape[1])
        elif positional_embeddings_type == 'learned':
            self.positional_embeddings = nn.Embedding(max_positional_embeddings,
                                                      self.encoder.weight.shape[1])
        else:
            self.positional_embeddings = None

        self.layers = nn.ModuleList(
                [TransformerEncoderLayer(input_size, model_size, 1, self.dropout_val) for _ in range(layers)]
                )
        self.init_param_freeze(mode)
        self.loss = torch.nn.CrossEntropyLoss(size_average=size_average, reduce=True, ignore_index=0)
        self.l1_key = None
        self.l2_key = None
        #self.init_optimizer(type='SGD', lr=1.0)  # , lr=0.01)
        self.init_optimizer(type='Adam')

    def init_cuda(self,):
        self = self.cuda()

    def is_cuda(self,):
        return self.layers[0].self_attn.k_linear[0].weight.is_cuda
    
    def forward(self, batch, l2_seen, l2_unseen):
        lengths, l1_data, l2_data, ind, word_mask = batch
        batch_size, seq_len = l1_data.shape
        l1_idxs = ind.eq(1).long()
        l2_idxs = ind.eq(2).long()  # make all the -1s into 0, keep old 0s as 0
        for st in [SPECIAL_TOKENS.PAD, SPECIAL_TOKENS.UNK, SPECIAL_TOKENS.EOS, SPECIAL_TOKENS.BOS]:
            if st in self.l1_dict:
                l1_idxs[l1_data.eq(self.l1_dict[st])] = 0
                l2_idxs[l1_data.eq(self.l1_dict[st])] = 0

        if self.positional_embeddings is not None:
            pos_encoded = self.positional_embeddings(l1_data)
        else:
            pos_encoded = None
        if self.mode == CTransformerEncoder.L1_LEARNING:
            l1_encoded = self.encoder(l1_data)
            rand = torch.zeros_like(l1_encoded[word_mask.eq(1), :]).uniform_(self.emb_min, self.emb_max)
            rand.requires_grad = False
            l1_encoded[word_mask.eq(1), :] = rand
            if self.positional_embeddings is not None:
                encoded = l1_encoded + pos_encoded
            else:
                encoded = l1_encoded

            diag_mask = l1_data.eq(self.l1_dict[SPECIAL_TOKENS.PAD])
            d = torch.arange(1, seq_len)
            diag_mask = diag_mask.unsqueeze(1).repeat(1, seq_len, 1)
            diag_mask[:, d, d] = 1

        elif self.mode == CTransformerEncoder.L2_LEARNING:
            l1_encoded = self.encoder(l1_data)
            l2_encoded = self.l2_encoder(l2_data)
            g_inp_ind = l2_idxs.unsqueeze(2).expand(l2_idxs.size(0), l2_idxs.size(1), l2_encoded.size(2)).float()
            v_inp_ind = l1_idxs.unsqueeze(2).expand(l1_idxs.size(0), l1_idxs.size(1), l1_encoded.size(2)).float()
            encoded = v_inp_ind * l1_encoded + g_inp_ind * l2_encoded
            diag_mask = l1_data.eq(self.l1_dict[SPECIAL_TOKENS.PAD])
            #encoded = l1_encoded * l1_idxs.unsqueeze(2).expand_as(l1_encoded).float() + \
            #    l2_encoded * l2_idxs.unsqueeze(2).expand_as(l2_encoded).float()
            #assert (encoded - tmp_encoded).sum().item() == 0
            if self.positional_embeddings is not None:
                encoded = encoded + pos_encoded
            else:
                pass
        else:
            raise NotImplementedError("L12_LEARNING not supported")

        x = self.dropout(encoded)  #.transpose(0, 1)  # BS, SL, EMB -- > SL, BS, EMB
        for layer in self.layers:
            x, attn_probs = layer(x, diag_mask)  # all the transformer magic
        final_hidden = self.dropout(x)  #x.transpose(1, 0)  # SL, BS, EMB ---> BS, SL, EMB

        if self.mode == CTransformerEncoder.L1_LEARNING:
            l1_final_hidden = final_hidden[l1_idxs == 1, :]
            l1_out = self.decoder(l1_final_hidden)
            l1_pred = l1_out.argmax(1)
            acc = l1_pred.eq(l1_data[l1_idxs == 1]).sum().item()
            acc = float(acc) / float(l1_pred.size(0))
            loss = self.loss(l1_out, l1_data[l1_idxs == 1])
        else:
            l1_final_hidden = final_hidden[l1_idxs == 1, :]
            if l1_final_hidden.shape[0] > 0:
                l1_out = self.decoder(l1_final_hidden)
                l1_loss = self.loss(l1_out, l1_data[l1_idxs == 1])
            else:
                l1_loss = 0.
            l2_final_hidden = final_hidden[l2_idxs == 1, :]
            if l2_final_hidden.shape[0] > 0:
                l2_out = self.l2_decoder(l2_final_hidden)
                if l2_unseen is not None:
                    l2_out[:, l2_unseen] = float('-inf')
                l2_loss = self.loss(l2_out, l2_data[l2_idxs == 1])
            else:
                l2_loss = 0.
            loss = l1_loss + l2_loss
            acc = None  # TODO
        return loss, acc


class CBiLSTMFast(CEncoderModel):

    def __init__(self, input_size, rnn_size, layers,
                 encoder, decoder,
                 l2_encoder, l2_decoder,
                 mode,
                 l1_dict,
                 l2_dict,
                 size_average=False):
        super().__init__(encoder, decoder, l2_encoder, l2_decoder, mode, l1_dict, l2_dict)
        self.input_size = input_size
        self.rnn_size = rnn_size
        self.rnn = nn.LSTM(self.input_size, self.rnn_size,
                           num_layers=1,
                           batch_first=True,
                           bidirectional=True)
        #self.max_positional_embeddings = 100 #max_positional_embeddings
        #self.positional_embeddings = nn.Embedding(self.max_positional_embeddings, self.encoder.weight.shape[1])
        #self.layer_norm = nn.LayerNorm(input_size)
        self.linear = nn.Linear(2 * self.rnn_size, self.input_size)
        self.init_param_freeze(mode)
        self.loss = torch.nn.CrossEntropyLoss(size_average=size_average, reduce=True, ignore_index=0)
        #self.z = Variable(torch.zeros(1, 1, self.rnn_size), requires_grad=False)
        self.z = torch.zeros(1, 1, self.rnn_size, requires_grad=False)
        # .expand(batch_size, 1, self.rnn_size), requires_grad=False)
        self.l1_key = None
        self.l2_key = None
        self.init_optimizer(type='Adam')

    def init_cuda(self,):
        self = self.cuda()
        self.z = self.z.cuda()

    def is_cuda(self,):
        return self.rnn.weight_hh_l0.is_cuda

    def get_hiddens(self, encoded, lengths, batch_size):
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
        final_hidden = self.linear(final_hidden)
        return final_hidden

    def get_mixed_input_encoding(self, l1_data, l2_data, l1_idxs, l2_idxs, l2_encoder):
        l1_encoded = self.encoder(l1_data)
        l2_encoded = l2_encoder(l2_data)
        g_inp_ind = l2_idxs.unsqueeze(2).expand(l2_idxs.size(0), l2_idxs.size(1), l2_encoded.size(2)).float()
        v_inp_ind = l1_idxs.unsqueeze(2).expand(l1_idxs.size(0), l1_idxs.size(1), l1_encoded.size(2)).float()
        tmp_encoded = v_inp_ind * l1_encoded + g_inp_ind * l2_encoded
        encoded = l1_encoded * l1_idxs.unsqueeze(2).expand_as(l1_encoded).float() + \
            l2_encoded * l2_idxs.unsqueeze(2).expand_as(l2_encoded).float()
        assert (encoded - tmp_encoded).sum().item() == 0
        encoded = self.dropout(encoded)
        return encoded

    def get_l1_output_predictions(self, final_hidden, l1_data, l1_idxs, l2_idxs):
        final_hidden = final_hidden.squeeze(0) # we assume batch size == 1
        l1_out = self.decoder(final_hidden)
        l1_pred = l1_out.argmax(1)
        l1_probs = torch.nn.functional.softmax(l1_out, dim=1)
        return l1_probs, l1_pred

    def bp_iter(self, lengths, batch_size, l1_data, l2_data, l1_idxs, l2_idxs, l2_weights):
        self.update_g_weights(l2_weights)
        encoded = self.get_mixed_input_encoding(l1_data, l2_data, l1_idxs, l2_idxs, self.l2_encoder)
        final_hiddens = self.get_hiddens(encoded, lengths, batch_size)
        l1_probs, l1_pred = self.get_l1_output_predictions(final_hiddens, l1_data, l1_idxs, l2_idxs)
        pdb.set_trace()
        print('l1_preds at l2 words', l1_pred[l2_idxs[0] == 1])
        print('l2_words', l2_data[l2_idxs == 1])
        pdb.set_trace()
        l1_pred_embeddings = self.encoder.weights.data[l1_pred[l2_idxs == 1]]
        l2_weights[l2_data[l2_idxs == 1]] = l1_pred_embeddings
        pdb.set_trace()
        #TODO: do something with l1_probs and l1_pred and get new l2_weights

    def do_bp_forward(self, batch, l2_weights):
        lengths, l1_data, l2_data, ind, word_mask = batch
        l1_idxs = ind.eq(1).long()
        l2_idxs = ind.eq(2).long()
        for st in [SPECIAL_TOKENS.PAD, SPECIAL_TOKENS.UNK, SPECIAL_TOKENS.EOS, SPECIAL_TOKENS.BOS]:
            if st in self.l1_dict:
                l1_idxs[l1_data.eq(self.l1_dict[st])] = 0
                l2_idxs[l1_data.eq(self.l1_dict[st])] = 0
        batch_size = l1_data.size(0)
        for _ in range(10):
            print('bp iter', _)
            l2_weights = self.bp_iter(lengths, batch_size, l1_data, l2_data, l1_idxs, l2_idxs, l2_weights)
        return l2_weights

    def forward(self, batch, l2_seen, l2_unseen):
        lengths, l1_data, l2_data, ind, word_mask = batch
        l1_idxs = ind.eq(1).long()
        l2_idxs = ind.eq(2).long()
        for st in [SPECIAL_TOKENS.PAD, SPECIAL_TOKENS.UNK, SPECIAL_TOKENS.EOS, SPECIAL_TOKENS.BOS]:
            if st in self.l1_dict:
                l1_idxs[l1_data.eq(self.l1_dict[st])] = 0
                l2_idxs[l1_data.eq(self.l1_dict[st])] = 0
        batch_size = l1_data.size(0)
        # l1_data = (batch_size x seq_len)
        # l2_data = (batch_size x seq_len)
        if self.mode == CEncoderModel.L2_LEARNING:
            encoded = self.get_mixed_input_encoding(l1_data, l2_data, l1_idxs, l2_idxs, self.l2_encoder)
        elif self.mode == CEncoderModel.L12_LEARNING:
            raise NotImplementedError("no longer supported")
            #l1_encoded = self.encoder(l1_data)
            #l2_encoded = self.l2_encoder(l2_data)
            #g_inp_ind = l2_idxs.unsqueeze(2).expand(l2_idxs.size(0), l2_idxs.size(1), l2_encoded.size(2)).float()
            #v_inp_ind = l1_idxs.unsqueeze(2).expand(l1_idxs.size(0), l1_idxs.size(1), l1_encoded.size(2)).float()
            #encoded = v_inp_ind * l1_encoded + g_inp_ind * l2_encoded
            #encoded = l1_encoded * l1_idxs.unsqueeze(2).expand_as(l1_encoded).float() + \
            #    l2_encoded * l2_idxs.unsqueeze(2).expand_as(l2_encoded).float()
            #assert (encoded - tmp_encoded).sum().item() == 0
            #encoded = self.dropout(encoded)
        elif self.mode == CEncoderModel.L1_LEARNING:
            #pos_data = torch.arange(lengths[0]).expand_as(l1_data).type_as(l1_data)
            #pos_data[pos_data > self.max_positional_embeddings - 1] = self.max_positional_embeddings - 1
            l1_encoded = self.encoder(l1_data)
            rand = torch.zeros_like(l1_encoded[word_mask == 1, :]).uniform_(self.emb_min, self.emb_max)
            rand.requires_grad = False
            l1_encoded[word_mask == 1, :] = rand
            #l1_pos_encoded = self.positional_embeddings(pos_data)
            #l1_encoded += l1_pos_encoded
            encoded = self.dropout(l1_encoded)
        else:
            raise NotImplementedError("unknown mode")

        final_hidden = self.get_hiddens(encoded, lengths, batch_size)

        if self.mode == CEncoderModel.L1_LEARNING:
            l1_final_hidden = final_hidden[l1_idxs == 1, :]
            l1_out = self.decoder(l1_final_hidden)
            l1_pred = l1_out.argmax(1)
            acc = l1_pred.eq(l1_data[l1_idxs == 1]).sum().item()
            acc = float(acc) / float(l1_pred.size(0))
            loss = self.loss(l1_out, l1_data[l1_idxs == 1])
        elif self.mode == CEncoderModel.L2_LEARNING or self.mode == CEncoderModel.L12_LEARNING:
            l1_final_hidden = final_hidden[l1_idxs == 1, :]
            if l1_final_hidden.shape[0] > 0:
                l1_out = self.decoder(l1_final_hidden)
                #pdb.set_trace()
                #l1_probs = torch.nn.functional.softmax(l1_out)
                l1_loss = self.loss(l1_out, l1_data[l1_idxs == 1])
            else:
                l1_loss = 0.
            l2_final_hidden = final_hidden[l2_idxs == 1, :]
            if l2_final_hidden.shape[0] > 0:
                l2_out = self.l2_decoder(l2_final_hidden)
                if l2_unseen is not None:
                    l2_out[:, l2_unseen] = float('-inf')
                l2_loss = self.loss(l2_out, l2_data[l2_idxs == 1])
            else:
                l2_loss = 0.
            loss = l1_loss + l2_loss
            acc = None  # TODO
        else:
            raise BaseException("unknown learning type")
        return loss, acc


class CBiLSTM(CEncoderModel):

    def __init__(self, input_size, rnn_size, layers,
                 encoder, decoder,
                 l2_encoder, l2_decoder,
                 mode,
                 l1_dict,
                 l2_dict,
                 size_average=False):
        super().__init__(encoder, decoder, l2_encoder, l2_decoder, mode, l1_dict, l2_dict)
        self.input_size = input_size
        self.rnn_size = rnn_size
        #self.rnn = nn.LSTM(self.input_size, self.rnn_size,
        #                   num_layers=1,
        #                   batch_first=True,
        #                   bidirectional=True)
        self.fwd_rnn = nn.LSTM(self.input_size, self.rnn_size, dropout=self.dropout_val,
                               num_layers=layers,
                               batch_first=True,
                               bidirectional=False)
        self.bwd_rnn = nn.LSTM(self.input_size, self.rnn_size, dropout=self.dropout_val,
                               num_layers=layers,
                               batch_first=True,
                               bidirectional=False)
        #self.max_positional_embeddings = 100 #max_positional_embeddings
        #self.positional_embeddings = nn.Embedding(self.max_positional_embeddings, self.encoder.weight.shape[1])
        #self.layer_norm = nn.LayerNorm(input_size)
        self.linear = nn.Linear(2 * self.rnn_size, self.input_size)
        self.init_param_freeze(mode)
        self.loss = torch.nn.CrossEntropyLoss(size_average=size_average, reduce=True, ignore_index=0)
        #self.z = Variable(torch.zeros(1, 1, self.rnn_size), requires_grad=False)
        self.z = torch.zeros(1, 1, self.rnn_size, requires_grad=False)
        # .expand(batch_size, 1, self.rnn_size), requires_grad=False)
        self.l1_key = None
        self.l2_key = None
        self.init_optimizer(type='Adam')

    def init_cuda(self,):
        self = self.cuda()
        self.z = self.z.cuda()

    def is_cuda(self,):
        return self.fwd_rnn.weight_hh_l0.is_cuda

    def forward(self, batch, l2_seen, l2_unseen):
        lengths, l1_data, l2_data, ind, word_mask = batch
        rev_idx_col = torch.zeros(l1_data.size(0), l1_data.size(1)).long()
        for _idx, l in enumerate(lengths):
            rev_idx_col[_idx, :] = torch.LongTensor(list(range(l - 1, -1, -1)) + list(range(l, lengths[0])))

        rev_idx_row = torch.arange(len(lengths)).long()
        rev_idx_row = rev_idx_row.unsqueeze(1).expand(l1_data.shape[0], l1_data.shape[1])
        #if data.is_cuda:
        #    rev_idx_row = rev_idx_row.cuda()
        l1_idxs = ind.eq(1).long()
        l2_idxs = ind.eq(2).long()
        for st in [SPECIAL_TOKENS.PAD, SPECIAL_TOKENS.UNK, SPECIAL_TOKENS.EOS, SPECIAL_TOKENS.BOS]:
            if st in self.l1_dict:
                l1_idxs[l1_data.eq(self.l1_dict[st])] = 0
                l2_idxs[l1_data.eq(self.l1_dict[st])] = 0
        #if seen is not None:
        #    seen, seen_offset, seen_set = seen
        # if self.is_cuda():
        #     data = data.cuda()
        batch_size = l1_data.size(0)
        # max_seq_len = data.size(1)

        # l1_data = (batch_size x seq_len)
        # l2_data = (batch_size x seq_len)
        if self.mode == CEncoderModel.L2_LEARNING:
            l1_encoded = self.encoder(l1_data)
            l2_encoded = self.l2_encoder(l2_data)
            g_inp_ind = l2_idxs.unsqueeze(2).expand(l2_idxs.size(0), l2_idxs.size(1), l2_encoded.size(2)).float()
            v_inp_ind = l1_idxs.unsqueeze(2).expand(l1_idxs.size(0), l1_idxs.size(1), l1_encoded.size(2)).float()
            tmp_encoded = v_inp_ind * l1_encoded + g_inp_ind * l2_encoded
            encoded = l1_encoded * l1_idxs.unsqueeze(2).expand_as(l1_encoded).float() + \
                l2_encoded * l2_idxs.unsqueeze(2).expand_as(l2_encoded).float()
            assert (encoded - tmp_encoded).sum().item() == 0
            encoded = self.dropout(encoded)
        elif self.mode == CEncoderModel.L12_LEARNING:
            raise NotImplementedError("no longer supported")
            l1_encoded = self.encoder(l1_data)
            l2_encoded = self.l2_encoder(l2_data)
            g_inp_ind = l2_idxs.unsqueeze(2).expand(l2_idxs.size(0), l2_idxs.size(1), l2_encoded.size(2)).float()
            v_inp_ind = l1_idxs.unsqueeze(2).expand(l1_idxs.size(0), l1_idxs.size(1), l1_encoded.size(2)).float()
            encoded = v_inp_ind * l1_encoded + g_inp_ind * l2_encoded
            #encoded = l1_encoded * l1_idxs.unsqueeze(2).expand_as(l1_encoded).float() + \
            #    l2_encoded * l2_idxs.unsqueeze(2).expand_as(l2_encoded).float()
            #assert (encoded - tmp_encoded).sum().item() == 0
            encoded = self.dropout(encoded)
        else:
            #pos_data = torch.arange(lengths[0]).expand_as(l1_data).type_as(l1_data)
            #pos_data[pos_data > self.max_positional_embeddings - 1] = self.max_positional_embeddings - 1
            l1_encoded = self.encoder(l1_data)
            rand = torch.zeros_like(l1_encoded[word_mask == 1, :]).uniform_(self.emb_min, self.emb_max)
            rand.requires_grad = False
            l1_encoded[word_mask == 1, :] = rand
            #l1_pos_encoded = self.positional_embeddings(pos_data)
            #l1_encoded += l1_pos_encoded
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
        #final_hidden = self.layer_norm(final_hidden)

        if self.mode == CEncoderModel.L1_LEARNING:
            l1_final_hidden = final_hidden[l1_idxs == 1, :]
            l1_out = self.decoder(l1_final_hidden)
            l1_pred = l1_out.argmax(1)
            acc = l1_pred.eq(l1_data[l1_idxs == 1]).sum().item()
            acc = float(acc) / float(l1_pred.size(0))
            loss = self.loss(l1_out, l1_data[l1_idxs == 1])
        elif self.mode == CEncoderModel.L2_LEARNING or self.mode == CEncoderModel.L12_LEARNING:
            l1_final_hidden = final_hidden[l1_idxs == 1, :]
            if l1_final_hidden.shape[0] > 0:
                l1_out = self.decoder(l1_final_hidden)
                l1_loss = self.loss(l1_out, l1_data[l1_idxs == 1])
            else:
                l1_loss = 0.
            l2_final_hidden = final_hidden[l2_idxs == 1, :]
            if l2_final_hidden.shape[0] > 0:
                l2_out = self.l2_decoder(l2_final_hidden)
                if l2_unseen is not None:
                    l2_out[:, l2_unseen] = float('-inf')
                l2_loss = self.loss(l2_out, l2_data[l2_idxs == 1])
            else:
                l2_loss = 0.
            loss = l1_loss + l2_loss
            acc = None  # TODO
        else:
            raise BaseException("unknown learning type")
        return loss, acc
