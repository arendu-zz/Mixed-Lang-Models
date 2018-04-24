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
from model import VarLinear
from model import WordRepresenter
from torch.utils.data import DataLoader
from utils import LazyTextDataset
from utils import my_collate

from torch.autograd import Variable

global PAD, EOS, BOS, UNK
PAD = '<PAD>'
UNK = '<UNK>'
BOS = '<BOS>'
EOS = '<EOS>'


def make_cl_encoder(word_representer):
    e = VarEmbedding(word_representer)
    return e


def make_cl_decoder(word_representer):
    d = VarLinear(word_representer)
    return d


def make_wl_encoder(vocab_size, embedding_size):
    e = torch.nn.Embedding(vocab_size, embedding_size)
    e.weight = torch.nn.Parameter(torch.FloatTensor(vocab_size, embedding_size).uniform_(-0.5 / embedding_size,
                                                                                         0.5 / embedding_size))
    return e


def make_wl_decoder(vocab_size, embedding_size, encoder=None):
    decoder = torch.nn.Linear(embedding_size, vocab_size, bias=False)
    if encoder is not None:
        decoder.weight = encoder.weight
    return decoder

if __name__ == '__main__':
    opt = argparse.ArgumentParser(description="write program description here")
    # insert options here
    opt.add_argument('--data_dir', action='store', dest='data_folder', required=True)
    opt.add_argument('--train_corpus', action='store', dest='train_corpus', required=True)
    opt.add_argument('--train_mode', action='store', dest='train_mode', required=True, type=int, choices=set([0, 1]))
    opt.add_argument('--dev_corpus', action='store', dest='dev_corpus', required=False, default=None)
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
    opt.add_argument('--w_embedding_size', action='store', type=int, dest='w_embedding_size', default=200)
    opt.add_argument('--c_embedding_size', action='store', type=int, dest='c_embedding_size', default=20)
    opt.add_argument('--batch_size', action='store', type=int, dest='batch_size', default=20)
    opt.add_argument('--gpuid', action='store', type=int, dest='gpuid', default=-1)
    opt.add_argument('--epochs', action='store', type=int, dest='epochs', default=50)
    opt.add_argument('--c_rnn_size', action='store', type=int, dest='c_rnn_size', default=100)
    opt.add_argument('--char_based', action='store', type=int, dest='char_based', default=0, choices=set([0, 1]))
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
    train_mode = CBiLSTM.L1_LEARNING if options.train_mode == 0 else CBiLSTM.L2_LEARNING
    dataset = LazyTextDataset(options.train_corpus, v2i, gv2i)
    dataloader = DataLoader(dataset, batch_size=options.batch_size, shuffle=True, collate_fn=my_collate)
    if options.dev_corpus is not None:
        dataset_dev = LazyTextDataset(options.dev_corpus, v2i)
        dataloader_dev = DataLoader(dataset_dev, batch_size=options.batch_size, shuffle=False, collate_fn=my_collate)
    total_batches = int(np.ceil(len(dataset) / options.batch_size))
    v_max_vocab = len(v2i)
    g_max_vocab = len(gv2i) if gv2i is not None else 0
    max_vocab = max(v_max_vocab, g_max_vocab)
    if options.char_based == 0:
        encoder = make_wl_encoder(max_vocab, options.w_embedding_size)
        decoder = make_wl_decoder(max_vocab, options.w_embedding_size, encoder)
        if g_max_vocab > 0:
            g_encoder = make_wl_encoder(max_vocab, options.w_embedding_size)
            g_decoder = make_wl_decoder(max_vocab, options.w_embedding_size, g_encoder)
        else:
            g_encoder = None
            g_decoder = None
        cbilstm = CBiLSTM(options.w_embedding_size,
                          encoder, decoder, g_encoder, g_decoder, mode=train_mode)
    else:
        wr = WordRepresenter(v2c, c2i, len(c2i), options.c_embedding_size,
                             c2i[PAD], options.c_rnn_size, options.w_embedding_size)
        cl_encoder = make_cl_encoder(wr)
        cl_decoder = make_cl_decoder(wr)
        if g_max_vocab > 0:
            g_wr = WordRepresenter(gv2c, c2i, len(c2i), options.c_embedding_size,
                                   c2i[PAD], options.c_rnn_size, options.w_embedding_size)
            g_cl_encoder = make_cl_encoder(g_wr)
            g_cl_decoder = make_cl_decoder(g_wr)
        else:
            g_cl_encoder = None
            g_cl_decoder = None
        cbilstm = CBiLSTM(options.w_embedding_size,
                          cl_encoder, cl_decoder, g_cl_encoder, g_cl_decoder, mode=train_mode)
    if options.gpuid > -1:
        cbilstm = cbilstm.cuda()
        cbilstm.init_cuda()
        if options.char_based == 1:
            wr.init_cuda()

    print(cbilstm)
    ave_time = 0.
    s = time.time()
    for epoch in range(options.epochs):
        cbilstm.train()
        ave_train_loss = []
        ave_dev_loss = []
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
                print("e{:d} b{:5d}/{:5d} loss:{:7.4f}\r".format(epoch, batch_idx + 1, total_batches, loss))
            ave_train_loss.append(loss)
        if options.dev_corpus is not None:
            cbilstm.eval()
            for batch_idx, batch in enumerate(dataloader_dev):
                l, data, ind = batch
                data = Variable(data, requires_grad=False, volatile=True)
                ind = Variable(ind, requires_grad=False, volatile=True)
                if cbilstm.is_cuda():
                    data = data.cuda()
                    ind = ind.cuda()
                batch = l, data, ind
                loss = cbilstm(batch)
                if cbilstm.is_cuda():
                    loss = loss.data.cpu().numpy()[0]
                else:
                    loss = loss.data.numpy()[0]
                ave_dev_loss.append(loss)
            print("Ending e{:d} AveTrainLoss:{:7.4f} AveDevLoss:{:7.4f}\r".format(epoch, np.mean(ave_train_loss),
                                                                                  np.mean(ave_dev_loss)))
        else:
            print("Ending e{:d} AveTrainLoss:{:7.4f}\r".format(epoch, np.mean(ave_train_loss)))
