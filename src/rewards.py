#!/usr/bin/env python
__author__ = 'arenduchintala'
import torch
import torch.nn.functional as F
import pdb

from torch.autograd import Variable

import numpy as np
np.set_printoptions(precision=4, linewidth=float('inf'))



def sigmoid(v, s):
    return 1.0 / (1.0 + torch.exp(-s * v))

def batch_cosine_sim(a, b):
    eps = 1e-8
    dots = torch.matmul(a, b.transpose(0, 1))
    a_norm = torch.norm(a, 2, 1, keepdim=True)
    b_norm = torch.norm(b, 2, 1, keepdim=True)
    ab_norm = torch.matmul(a_norm, b_norm.transpose(0, 1))
    ab_norm[ab_norm <= 0.0] = eps
    return torch.div(dots, ab_norm)

def score_embeddings(l2_embedding, l1_embedding, l2_key, l1_key, l2_swap_types=None):
    assert l2_embedding.size(1) == l1_embedding.size(1)
    l2_sub = l2_embedding[l2_key]
    l1_sub = l1_embedding[l1_key]
    cs = batch_cosine_sim(l2_sub, l1_sub)  # _embedding, l1_embedding)
    #cs_tmp = batch_cosine_sim(l2_embedding, l1_embedding)
    #assert cs_tmp.min().item() > -1
    #assert cs_tmp.max().item() < 1
    #cs_tmp = cs_tmp[l2_key, l1_key]
    cs = cs.diag()
    cs = torch.nn.functional.relu(cs - 0.0) ** 2
    #cs_tmp = torch.nn.functional.relu(cs_tmp - 0.0) ** 2
    #scores = sigmoid(cs, S)
    #print(cs.diag().view(-1).cpu().numpy(), score)
    #cs_list = ','.join(['%.3f'%i for i in cs.tolist()])
    #print(cs_list)
    #print(cs_tmp.sum().item() - cs.sum().item())
    #assert cs_tmp.sum().item() == cs.sum().item()
    score = cs.sum().item()
    del cs
    return score


def get_nearest_neighbors(l2_embedding, l1_embedding, l2_swap_types, rank_threshold):
    l2_types = torch.Tensor(list(l2_swap_types)).type_as(l1_embedding).long()
    cs = batch_cosine_sim(l2_embedding[l2_types], l1_embedding)
    cs_top, arg_top = torch.topk(cs, rank_threshold, 1, sorted=True)
    del cs
    return arg_top, cs_top


def get_nearest_neighbors_simple(a_emb, b_emb, r):
    cs = batch_cosine_sim(a_emb, b_emb)
    _, arg_top = torch.topk(cs, r, 1, sorted=True)
    del cs, _
    return arg_top


def mrr_score_embeddings(l2_embedding, l1_embedding, l2_key, l1_key, rank_threshold, l2_key_wt):
    l2_key = l2_key.type_as(l1_embedding).long()
    l1_key = l1_key.type_as(l1_embedding).long()
    assert l2_key.shape == l2_key_wt.shape
    cs = batch_cosine_sim(l2_embedding[l2_key], l1_embedding)
    _, arg_top_S = torch.topk(cs, rank_threshold, 1, sorted=True)
    reciprocal_ranks = []
    for idx, (l2_kw, l2_k, l1_k) in enumerate(zip(l2_key_wt, l2_key, l1_key)):
        m = (arg_top_S[idx] == l1_k.item()).nonzero()
        if m.numel() > 0:
            r = float(m.min().item()) + 1.0
        else:
            r = np.inf
        reciprocal_ranks.append(l2_kw.item() * (1.0 / r))
    score = sum(reciprocal_ranks) / (float(len(reciprocal_ranks)) + 1e-4)
    return score


def mmr_score_embedding_with_assist_check(l2_embedding, l1_embedding, l2_key, l1_key,
                                          l2_seq, l1_seq, l2_seq_idx, rank_threshold, l2_key_wt):
    if l2_seq_idx.nonzero().numel() > 0:
        l2_seq = l2_seq.type_as(l1_embedding).long()
        l2_seq_idx = l2_seq_idx.type_as(l1_embedding).long()
        l1_seq = l1_seq.type_as(l1_embedding).long()
        l2_seq_key_used = l2_seq[l2_seq_idx == 1]
        l1_seq_key_used = l1_seq[l2_seq_idx == 1]
        if (l1_seq_key_used == 3).nonzero().numel() > 0:
            token_score = -1.0 #-np.inf  # if the configuration wants to swap a rare English word, we prevent it here...
        else:
            cs_seq = batch_cosine_sim(l2_embedding[l2_seq_key_used], l1_embedding)
            _, arg_top_S = torch.topk(cs_seq, rank_threshold, 1, sorted=True)
            l1_exp_S = l1_seq_key_used.unsqueeze(1).expand_as(arg_top_S)
            nz = (arg_top_S == l1_exp_S).nonzero()
            if nz.shape[0] == arg_top_S.shape[0]:
                token_score = 1.0
            elif nz.shape[0] < arg_top_S.shape[0]:
                token_score = -1.0 #-np.inf  # there has been a assist
            elif nz.shape[0] > arg_top_S.shape[0]:
                raise BaseException("Can this even happen??")
            else:
                raise BaseException("all bases covered, why am i here??")
    else:
        token_score = 1.0

    type_score = mrr_score_embeddings(l2_embedding, l1_embedding, l2_key, l1_key, rank_threshold, l2_key_wt)
    final_score = token_score * type_score
    return max(0, final_score)


def token_mrr_score_embeddings(l2_embedding, l1_embedding, l2_key, l1_key, l2_idx, rank_threshold):
    if l2_idx.nonzero().numel() > 0:
        l2_key = l2_key.type_as(l1_embedding).long()
        l2_idx = l2_idx.type_as(l1_embedding).long()
        l1_key = l1_key.type_as(l1_embedding).long()
        l2_key_used = l2_key[l2_idx == 1]
        l1_key_used = l1_key[l2_idx == 1]
        if (l1_key_used == 3).nonzero().numel() > 0:
            score = -10000
        else:
            cs = batch_cosine_sim(l2_embedding[l2_key_used], l1_embedding)
            _, arg_top_S = torch.topk(cs, rank_threshold, 1, sorted=True)
            l1_exp_S = l1_key_used.unsqueeze(1).expand_as(arg_top_S)
            nz = (arg_top_S == l1_exp_S).nonzero()
            _bad = arg_top_S.shape[0]
            if nz.numel() > 0:
                assert nz.shape[1] == 2
                rr = {}
                for n in nz:
                    n0, n1 = n[0].item(), n[1].item() + 1.0
                    rr[n0] = max(rr.get(n[0].item(), 0.0), 1.0 / n1)
                _bad -= len(rr)
                #reciprocal_ranks = []
                #for idx, (l2_k, l1_k) in enumerate(zip(l2_key_used, l1_key_used)):
                #    m = (arg_top_S[idx] == l1_k.item()).nonzero()
                #    if m.numel() > 0:
                #        r = float(m.min().item()) + 1.0
                #    else:
                #        r = -0.1 #np.inf
                #    reciprocal_ranks.append(1.0 / r)
                #score = sum(reciprocal_ranks) # + 0.1 * top_M_l2.shape[0] + 0.01 * top_L_l2.shape[0]
                #print(reciprocal_ranks, 'ranks')
                score = sum(rr.values()) - 10000 * float(_bad)  # scoring function
            else:
                #print('herererer', _bad)
                score = 0.0 - (10000 * _bad)  # made swaps and non of them got mrr result
    else:
        score = 0.0  # did not make any swaps
    return score

def token_rank_score_embeddings(l2_embedding, l1_embedding, l2_key, l1_key, l2_idx, rank_threshold):
    if l2_idx.nonzero().numel() > 0:
        l2_key = l2_key.type_as(l1_embedding).long()
        l2_idx = l2_idx.type_as(l1_embedding).long()
        l1_key = l1_key.type_as(l1_embedding).long()
        l2_key_used = l2_key[l2_idx == 1]
        l1_key_used = l1_key[l2_idx == 1]
        cs = batch_cosine_sim(l2_embedding[l2_key_used], l1_embedding)
        _, arg_top_S = torch.topk(cs, rank_threshold, 1, sorted=True)
        arg_top_S_l2 = arg_top_S
        l1_exp_S = l1_key_used.unsqueeze(1).expand_as(arg_top_S_l2)
        top_S_l2 = (arg_top_S_l2 == l1_exp_S).sum(1).nonzero().view(-1)
        bad = l2_key_used.numel() - top_S_l2.numel()
        score = float(top_S_l2.numel() - bad)  # + 0.1 * top_M_l2.shape[0] + 0.01 * top_L_l2.shape[0]
    else:
        score = 0.0
    return score

def rank_score_embeddings(l2_embedding, l1_embedding, l2_key, l1_key, l2_swap_types=None):
    l2_key = l2_key.type_as(l1_embedding).long()
    l1_key = l1_key.type_as(l1_embedding).long()
    cs = batch_cosine_sim(l2_embedding[l2_key], l1_embedding)
    _, arg_top_S = torch.topk(cs, 7, 1, sorted=True)
    #_, arg_top_M = torch.topk(cs, 3, 1)
    #_, arg_top_L = torch.topk(cs, 10, 1)
    arg_top_S_l2 = arg_top_S
    #arg_top_M_l2 = arg_top_M[l2_key]
    #arg_top_L_l2 = arg_top_L[l2_key]
    l1_exp_S = l1_key.unsqueeze(1).expand_as(arg_top_S_l2)
    #l1_exp_M = l1_key.unsqueeze(1).expand_as(arg_top_M_l2)
    #l1_exp_L = l1_key.unsqueeze(1).expand_as(arg_top_L_l2)
    top_S_l2 = (arg_top_S_l2 == l1_exp_S).sum(1).nonzero().view(-1)
    #top_M_l2 = (arg_top_M_l2 == l1_exp_M).sum(1).nonzero().view(-1)
    #top_L_l2 = (arg_top_L_l2 == l1_exp_L).sum(1).nonzero().view(-1)
    score = top_S_l2.shape[0] #+ 0.1 * top_M_l2.shape[0] + 0.01 * top_L_l2.shape[0]
    return score
