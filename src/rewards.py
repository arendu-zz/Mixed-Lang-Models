#!/usr/bin/env python
__author__ = 'arenduchintala'
import pdb
import torch
import torch.nn.functional as F

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

def prob_score_embeddings(l2_embedding, l1_embedding, l2_key, l1_key):
    assert l2_embedding.size(1) == l1_embedding.size(1)
    cs = batch_cosine_sim(l2_embedding, l1_embedding)  # _embedding, l1_embedding)
    cs = F.softmax(Variable(cs), dim=1).data
    score = cs[l2_key, l1_key]
    score = score.mean()
    return score

def score_embeddings(l2_embedding, l1_embedding, l2_key, l1_key):
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
    return cs.sum().item()


def get_nearest_neighbors(l2_embedding, l1_embedding):
    cs = batch_cosine_sim(l2_embedding, l1_embedding)
    _, arg_top = torch.topk(cs, 3, 1)
    return arg_top

def rank_score_embeddings(l2_embedding, l1_embedding, l2_key, l1_key):
    cs = batch_cosine_sim(l2_embedding, l1_embedding)
    _, arg_top_S = torch.topk(cs, 3, 1)
    #_, arg_top_M = torch.topk(cs, 3, 1)
    #_, arg_top_L = torch.topk(cs, 10, 1)
    arg_top_S_l2 = arg_top_S[l2_key]
    #arg_top_M_l2 = arg_top_M[l2_key]
    #arg_top_L_l2 = arg_top_L[l2_key]
    l1_exp_S = l1_key.unsqueeze(1).expand_as(arg_top_S_l2)
    #l1_exp_M = l1_key.unsqueeze(1).expand_as(arg_top_M_l2)
    #l1_exp_L = l1_key.unsqueeze(1).expand_as(arg_top_L_l2)
    top_S_l2 = (arg_top_S_l2 == l1_exp_S).sum(1).nonzero().view(-1)
    #top_M_l2 = (arg_top_M_l2 == l1_exp_M).sum(1).nonzero().view(-1)
    #top_L_l2 = (arg_top_L_l2 == l1_exp_L).sum(1).nonzero().view(-1)
    score = top_S_l2.shape[0] #+ 0.1 * top_M_l2.shape[0] + 0.01 * top_L_l2.shape[0]
    #print('top_S_l2', top_S_l2)
    #print(l1_key[top_S_l2], 'l1_key correct')
    #print(l2_key[top_S_l2], 'l2_key correct')
    #if score == 0.:
    #    score += 0.75 * top_M_l2.shape[0]
    #if score == 0.:
    #    score += 0.25 * top_L_l2.shape[0]
    return score
