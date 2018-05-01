#!/usr/bin/env python
__author__ = 'arenduchintala'
import argparse
import numpy as np
import os
import pickle
import time
import torch

from model import CBiLSTM
from model import VarEmbedding
from model import WordRepresenter
from torch.utils.data import DataLoader
from train import make_cl_decoder
from train import make_cl_encoder
from train import make_wl_decoder
from train import make_wl_encoder
from utils import LazyTextDataset
from utils import my_collate

from torch.autograd import Variable


global PAD, EOS, BOS, UNK
PAD = '<PAD>'
UNK = '<UNK>'
BOS = '<BOS>'
EOS = '<EOS>'

if __name__ == '__main__':
    opt = argparse.ArgumentParser(description="write program description here")
    # insert options here
    opt.add_argument('--data_dir', action='store', dest='data_folder', required=True)
    opt.add_argument('--save_dir', action='store', dest='save_folder', required=True,
                     help='folder to save the model after every epoch')
    opt.add_argument('--train_corpus', action='store', dest='train_corpus', required=True)
    opt.add_argument('--train_mode', action='store', dest='train_mode', required=True, type=int, choices=set([1]))
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
    opt.add_argument('--batch_size', action='store', type=int, dest='batch_size', default=20)
    opt.add_argument('--gpuid', action='store', type=int, dest='gpuid', default=-1)
    opt.add_argument('--char_based', action='store', type=int, dest='char_based', default=0, choices=set([0, 1]))
    opt.add_argument('--trained_model', action='store', dest='trained_model', required=True)
    opt.add_argument('--epochs', action='store', type=int, dest='epochs', default=100)
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
    v2c = pickle.load(open(options.v2spell, 'rb'))
    gv2i = pickle.load(open(options.gv2i, 'rb')) if options.gv2i is not None else None
    gv2c = pickle.load(open(options.gv2spell, 'rb')) if options.gv2spell is not None else None
    c2i = pickle.load(open(options.c2i, 'rb'))
    train_mode = CBiLSTM.L2_LEARNING
    dataset = LazyTextDataset(options.train_corpus, v2i, gv2i, train_mode)
    dataloader = DataLoader(dataset, batch_size=options.batch_size,  shuffle=True, collate_fn=my_collate)
    total_batches = int(np.ceil(len(dataset) / options.batch_size))
    v_max_vocab = len(v2i)
    g_max_vocab = len(gv2i) if gv2i is not None else 0
    max_vocab = max(v_max_vocab, g_max_vocab)
    cbilstm = torch.load(options.trained_model, map_location=lambda storage, loc: storage)
    if isinstance(cbilstm.encoder, VarEmbedding):
        wr = cbilstm.encoder.word_representer
        we_size = 200  # wr.we_size
        learned_weights = cbilstm.encoder.word_representer()
        g_wr = WordRepresenter(gv2c, c2i, len(c2i), wr.ce_size,
                               c2i[PAD], wr.cr_size, we_size,
                               bidirectional=wr.bidirectional, dropout=wr.dropout,
                               num_required_vocab=max_vocab)
        for (name_op, op), (name_p, p) in zip(wr.named_parameters(), g_wr.named_parameters()):
            assert name_p == name_op
            if name_op == 'extra_ce_layer.weight':
                pass
            else:
                p.data.copy_(op.data)
        g_wr.set_extra_feat_learnable(True)
        if options.gpuid > -1:
            g_wr.init_cuda()
        g_cl_encoder = make_cl_encoder(g_wr)
        g_cl_decoder = make_cl_decoder(g_wr)
        encoder = make_wl_encoder(max_vocab, we_size, learned_weights.data.clone())
        decoder = make_wl_decoder(max_vocab, we_size, encoder)
        cbilstm.encoder = encoder
        cbilstm.decoder = decoder
        cbilstm.g_encoder = g_cl_encoder
        cbilstm.g_decoder = g_cl_decoder
        cbilstm.init_param_freeze(CBiLSTM.L2_LEARNING)
    else:
        learned_weights = cbilstm.encoder.weight
        g_wl_encoder = make_wl_encoder(max_vocab, options.w_embedding_size)
        g_wl_decoder = make_wl_decoder(max_vocab, options.w_embedding_size, g_wl_encoder)
        cbilstm.g_encoder = g_wl_encoder
        cbilstm.g_decoder = g_wl_decoder
        cbilstm.init_param_freeze(CBiLSTM.L2_LEARNING)
    cbilstm.init_optimizer()
    if options.gpuid > -1:
        cbilstm.init_cuda()
    print(cbilstm)
    ave_time = 0.
    s = time.time()
    for epoch in range(options.epochs):
        cbilstm.train()
        train_losses = []
        for batch_idx, batch in enumerate(dataloader):
            l, data, ind = batch
            data = Variable(data, requires_grad=False)
            ind = Variable(ind, requires_grad=False)
            if cbilstm.is_cuda():
                data = data.cuda()
                ind = ind.cuda()
            batch = l, data, ind
            loss, grad_norm = cbilstm.do_backprop(batch)
            if batch_idx % 10 == 0 and batch_idx > 0:
                e = time.time()
                ave_time = (e - s) / 10.
                s = time.time()
                print("e{:d} b{:5d}/{:5d} loss:{:7.4f} ave_time:{:7.4f}\r".format(epoch, batch_idx + 1,
                                                                                  total_batches, loss, ave_time))
            else:
                print("e{:d} b{:d}/{:d} loss:{:7.4f}\r".format(epoch, batch_idx + 1, total_batches, loss))
            train_losses.append(loss)
        save_name = "e_{:d}_train_loss_{:.4f}".format(epoch, np.mean(train_losses))
        print("Ending e{:d} AveTrainLoss:{:7.4f}\r".format(epoch, np.mean(train_losses)))
        if options.save_folder is not None:
            cbilstm.save_model(os.path.join(options.save_folder, save_name + '.model'))
