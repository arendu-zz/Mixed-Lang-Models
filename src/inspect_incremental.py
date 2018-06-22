#!/usr/bin/env python
__author__ = 'arenduchintala'
import argparse
import torch

import pdb
import pickle

from model import CBiLSTM
from model import VarEmbedding
from utils import batch_cosine_sim
from utils import LazyTextDataset
from utils import my_collate


global PAD, EOS, BOS, UNK
PAD = '<PAD>'
UNK = '<UNK>'
BOS = '<BOS>'
EOS = '<EOS>'

if __name__ == '__main__':
    opt = argparse.ArgumentParser(description="write program description here")
    # insert options here
    opt.add_argument('--gpuid', action='store', type=int, dest='gpuid', default=-1)
    opt.add_argument('--v2i', action='store', dest='v2i', required=True,
                     help='vocab to index pickle obj')
    opt.add_argument('--v2spell', action='store', dest='v2spell', required=True,
                     help='vocab to spelling pickle obj')
    opt.add_argument('--c2i', action='store', dest='c2i', required=True,
                     help='character (corpus and gloss)  to index pickle obj')
    opt.add_argument('--gv2i', action='store', dest='gv2i', required=False, default=None,
                     help='gloss vocab to index pickle obj')
    opt.add_argument('--gv2spell', action='store', dest='gv2spell', required=False, default=None,
                     help='gloss vocab to index pickle obj')
    opt.add_argument('--trained_model', action='store', dest='trained_model', required=True)
    options = opt.parse_args()
    print(options)
    if options.gpuid > -1:
        torch.cuda.set_device(options.gpuid)
        tmp = torch.ByteTensor([0])
        tmp.cuda()
        print("using GPU", options.gpuid)
    else:
        print("using CPU")

    v2i = pickle.load(open(options.v2i, 'rb'))
    i2v = dict((v, k) for k, v in v2i.items())
    v2c = pickle.load(open(options.v2spell, 'rb'))
    gv2i = pickle.load(open(options.gv2i, 'rb')) if options.gv2i is not None else None
    i2gv = dict((v, k) for k, v in gv2i.items())
    gv2c = pickle.load(open(options.gv2spell, 'rb')) if options.gv2spell is not None else None
    c2i = pickle.load(open(options.c2i, 'rb'))
    v_max_vocab = len(v2i)
    g_max_vocab = len(gv2i) if gv2i is not None else 0
    max_vocab = max(v_max_vocab, g_max_vocab)
    cbilstm = torch.load(options.trained_model, map_location=lambda storage, loc: storage)
    pdb.set_trace()
    if isinstance(cbilstm.encoder, VarEmbedding):
        fff = cbilstm.g_encoder.word_representer.extra_ce_layer.weight.data[cbilstm.g_encoder.word_representer.unsort_idx]
        for i in i2gv.keys():
            print(fff[i].numpy()[0], i2gv[i], i)
    else:
        pdb.set_trace()
        v_data = cbilstm.encoder.weight.data
        v_data = v_data[torch.arange(v_max_vocab).long(), :]
        g_data = cbilstm.g_encoder.weight.data
        g_data = g_data[torch.arange(g_max_vocab).long(), :]
        cs_sim = batch_cosine_sim(g_data, v_data)
        top_cs, top_args = torch.topk(cs_sim, 10, dim=0)
        for vi in range(cs_sim.size(0)):
            top_args_vi = ','.join([i2v.get(i, '<UNK>') for i in top_args[:, vi]])
            print(i2gv[vi], top_args_vi)
            pdb.set_trace()
