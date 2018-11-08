#!/usr/bin/env python
__author__ = 'arenduchintala'
import argparse
import numpy as np
import os
import pickle
import random
import time
import torch
import pdb

from model import CBiLSTM
from model import VarEmbedding
from model import VariationalEmbeddings
from model import VarLinear
from model import VariationalLinear
from model import WordRepresenter

from utils.utils import TextDataset
from utils.utils import SPECIAL_TOKENS


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
        e.weight = torch.nn.Parameter(torch.FloatTensor(vocab_size, embedding_size).uniform_(-0.01 / embedding_size,
                                                                                             0.01 / embedding_size))
    else:
        e = torch.nn.Embedding(wt.size(0), wt.size(1))
        e.weight = torch.nn.Parameter(wt)
    return e


def make_wl_decoder(encoder):
    decoder = torch.nn.Linear(encoder.weight.size(0), encoder.weight.size(1), bias=False)
    decoder.weight = encoder.weight
    return decoder


if __name__ == '__main__':
    opt = argparse.ArgumentParser(description="write program description here")
    # insert options here
    opt.add_argument('--save_dir', action='store', dest='save_folder', required=True,
                     help='folder to save the model after every epoch')
    opt.add_argument('--train_corpus', action='store', dest='train_corpus', required=True)
    opt.add_argument('--dev_corpus', action='store', dest='dev_corpus', required=False, default=None)
    opt.add_argument('--v2i', action='store', dest='v2i', required=True,
                     help='vocab to index pickle obj')
    opt.add_argument('--v2spell', action='store', dest='v2spell', required=True,
                     help='vocab to spelling pickle obj')
    opt.add_argument('--c2i', action='store', dest='c2i', required=True,
                     help='character (corpus and gloss)  to index pickle obj')
    opt.add_argument('--w_embedding_size', action='store', type=int, dest='w_embedding_size', default=500)
    opt.add_argument('--layers', action='store', type=int, dest='layers', default=1)
    opt.add_argument('--rnn_size', action='store', type=int, dest='rnn_size', default=250)
    opt.add_argument('--c_embedding_size', action='store', type=int, dest='c_embedding_size', default=20)
    opt.add_argument('--batch_size', action='store', type=int, dest='batch_size', default=20)
    opt.add_argument('--gpuid', action='store', type=int, dest='gpuid', default=-1)
    opt.add_argument('--epochs', action='store', type=int, dest='epochs', default=50)
    opt.add_argument('--char_composition', action='store', type=str,
                     dest='char_composition', default='None',
                     choices=set(['RNN', 'CNN', 'None', 'Variational']))
    opt.add_argument('--char_bidirectional', action='store', type=int, dest='char_bidirectional', default=1,
                     required=False, choices=set([0, 1]))
    opt.add_argument('--lsp', action='store', type=float, dest='lsp', default=0.)
    opt.add_argument('--seed', action='store', dest='seed', default=1234, type=int)
    options = opt.parse_args()
    print(options)
    torch.manual_seed(options.seed)
    random.seed(options.seed)
    np.random.seed(options.seed)
    if options.gpuid > -1:
        torch.cuda.set_device(options.gpuid)
        tmp = torch.ByteTensor([0])
        tmp.cuda()
        print("using GPU", options.gpuid)
    else:
        print("using CPU")

    v2i = pickle.load(open(options.v2i, 'rb'))
    v2c = pickle.load(open(options.v2spell, 'rb'))
    c2i = pickle.load(open(options.c2i, 'rb'))

    train_mode = CBiLSTM.L1_LEARNING
    train_dataset = TextDataset(options.train_corpus, v2i, shuffle=True, sort_by_len=True,
                                min_batch_size=options.batch_size, max_batch_size=10000)
    if options.dev_corpus is not None:
        dev_dataset = TextDataset(options.dev_corpus, v2i, shuffle=False, sort_by_len=True,
                                  min_batch_size=options.batch_size, max_batch_size=10000)
    vocab_size = len(v2i)
    if options.char_composition == 'None':
        encoder = make_wl_encoder(vocab_size, options.w_embedding_size, None)
        decoder = make_wl_decoder(encoder)
        cbilstm = CBiLSTM(options.w_embedding_size, options.rnn_size, options.layers,
                          encoder, decoder, None, None, mode=CBiLSTM.L1_LEARNING)
    elif options.char_composition == 'Variational':
        mean = torch.FloatTensor(vocab_size, options.w_embedding_size).uniform_(-0.01 / options.w_embedding_size,
                                                                                0.01 / options.w_embedding_size)
        mean = torch.nn.Parameter(mean)
        rho = 0.01 * torch.randn(vocab_size, options.w_embedding_size)
        rho = torch.nn.Parameter(rho)
        mean.requires_grad = True
        rho.requires_grad = True
        encoder = make_vl_encoder(mean, rho, float(np.exp(options.lsp)))  # log sigma prior
        decoder = make_vl_decoder(mean, rho)
        cbilstm = CBiLSTM(options.w_embedding_size, options.rnn_size, options.layers,
                          encoder, decoder, None, None, mode=CBiLSTM.L1_LEARNING)
    else:
        wr = WordRepresenter(v2c, c2i, len(c2i), options.c_embedding_size, c2i[SPECIAL_TOKENS.PAD],
                             options.w_embedding_size // (2 if options.char_bidirectional else 1),
                             options.w_embedding_size,
                             bidirectional=options.char_bidirectional == 1,
                             is_extra_feat_learnable=False, num_required_vocab=vocab_size,
                             char_composition=options.char_composition)
        if options.gpuid > -1:
            wr.init_cuda()
        cl_encoder = make_cl_encoder(wr)
        cl_decoder = make_cl_decoder(wr)
        cbilstm = CBiLSTM(options.w_embedding_size, options.rnn_size, options.layers,
                          cl_encoder, cl_decoder, None, None, mode=CBiLSTM.L1_LEARNING)
    if options.gpuid > -1:
        cbilstm.init_cuda()

    print(cbilstm)
    ave_time = 0.
    s = time.time()
    total_batches = 0 #train_dataset.num_batches
    for epoch in range(options.epochs):
        cbilstm.train()
        train_losses = []
        for batch_idx, batch in enumerate(train_dataset):
            l, data, text_data = batch
            ind = torch.ones_like(data).long()
            if cbilstm.is_cuda():
                data = data.cuda()
                ind = ind.cuda()
            cuda_batch = l, data, data, ind
            loss, grad_norm = cbilstm.do_backprop(cuda_batch, total_batches=total_batches)
            if batch_idx % 10 == 0 and batch_idx > 0:
                e = time.time()
                ave_time = (e - s) / 10.
                s = time.time()
                print("e{:d} b{:5d}/{:5d} loss:{:7.6f} ave_time:{:7.6f}\r".format(epoch, batch_idx + 1,
                                                                                  total_batches, loss, ave_time))
            else:
                pass
                #print("e{:d} b{:d}/{:d} loss:{:7.6f}\r".format(epoch, batch_idx + 1, total_batches, loss))
            train_losses.append(loss)
        total_batches = batch_idx
        dev_losses = []
        assert options.dev_corpus is not None
        cbilstm.eval()
        for batch_idx, batch in enumerate(dev_dataset):
            l, data, text_data = batch
            ind = torch.ones_like(data).long()
            if cbilstm.is_cuda():
                data = data.cuda()
                ind = ind.cuda()
            cuda_batch = l, data, data, ind
            loss = cbilstm(cuda_batch)
            if cbilstm.is_cuda():
                loss = loss.item()  # .data.cpu().numpy()[0]
            else:
                loss = loss.item()  # .data.numpy()[0]
            dev_losses.append(loss)
        print("Ending e{:d} AveTrainLoss:{:7.6f} AveDevLoss:{:7.6f}\r".format(epoch, np.mean(train_losses),
                                                                              np.mean(dev_losses)))
        save_name = "e_{:d}_train_loss_{:.6f}_dev_loss_{:.6f}".format(epoch, np.mean(train_losses),
                                                                      np.mean(dev_losses))
        if options.save_folder is not None:
            cbilstm.save_model(os.path.join(options.save_folder, save_name + '.model'))
