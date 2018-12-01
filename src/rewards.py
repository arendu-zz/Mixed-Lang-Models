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
    cs = cs.diag()
    cs = torch.nn.functional.relu(cs - 0.0) ** 2
    #scores = sigmoid(cs, 10)
    #print(cs.diag().view(-1).cpu().numpy(), score)
    #cs_list = ','.join(['%.3f'%i for i in cs.tolist()])
    #print(cs_list)
    return cs.sum().item()

def rank_score_embeddings(l2_embedding, l1_embedding, l2_key, l1_key):
    cs = batch_cosine_sim(l2_embedding, l1_embedding)
    pdb.set_trace()
    pass
