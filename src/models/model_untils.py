#!/usr/bin/env python
__author__ = 'arenduchintala'
import math
import torch
import torch.nn as nn
import numpy as np

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from src.utils.utils import SPECIAL_TOKENS
from src.opt.noam import NoamOpt

from src.rewards import score_embeddings
from src.rewards import rank_score_embeddings


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


class MapEmbedding(nn.Module):
    def __init__(self, l1_weights, map_weights):
        super(MapEmbedding, self).__init__()
        self.l1_weights = l1_weights
        self.l1_weights.requires_grad = False
        self.map_weights = map_weights
        self.map_weights.requires_grad = True
        self.register_buffer('l1_weight', self.l1_weights)
        self.register_buffer('map_weight', self.map_weights)

    def get_l2_weights(self,):
        l2_weights = torch.nn.functional.softmax(self.map_weights, dim=1)
        l2_weights = l2_weights.matmul(self.l1_weights)
        return l2_weights

    def get_map_param(self,):
        return self.map_weights

    def forward(self, x):
        l2_weights = self.get_l2_weights()
        embedding_dim = l2_weights.size(1)
        if x.dim() == 2:
            batch_size, seq_len = x.shape
            l2_emb = l2_weights[x.view(-1)]
            l2_emb = l2_emb.view(batch_size, seq_len, embedding_dim)
            return l2_emb
        elif x.dim() == 1:
            return l2_weights[x]

