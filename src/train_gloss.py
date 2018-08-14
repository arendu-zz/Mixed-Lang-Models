#!/usr/bin/env python
__author__ = 'arenduchintala'
import argparse
import os
import pickle
import pdb
import time
import torch
from glosstagger import GlossTagger

from utils import ParallelTextDataset
from utils import my_collate
from torch.utils.data import DataLoader

import numpy as np


if __name__ == '__main__':
    opt = argparse.ArgumentParser(description="write program description here")

    # insert options here
    opt.add_argument('-t', action='store', dest='gloss_train', required=True)
    opt.add_argument('-d', action='store', dest='gloss_dev', required=True)
    opt.add_argument('--batch_size', action='store', type=int, dest='batch_size', default=1)
    opt.add_argument('--s2i', action='store', dest='s2i', required=True)
    opt.add_argument('--s2spell', action='store', dest='s2spell', required=True)
    opt.add_argument('--sc2i', action='store', dest='sc2i', required=True)
    opt.add_argument('--t2i', action='store', dest='t2i', required=True)
    opt.add_argument('--t2spell', action='store', dest='t2spell', required=True)
    opt.add_argument('--tc2i', action='store', dest='tc2i', required=True)
    opt.add_argument('--rnn_type', action='store', type=str, dest='rnn_type', required=True)
    opt.add_argument('--gpuid', action='store', type=int, dest='gpuid', default=-1)
    opt.add_argument('--save_model', action='store', dest='save_model', required=True)
    opt.add_argument('--rnn_size', action='store', type=int, dest='rnn_size', default=100)
    opt.add_argument('--rnn_layers', action='store', type=int, dest='rnn_layers', default=1)
    opt.add_argument('--embedding_size', action='store', type=int, dest='embedding_size', default=500)
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

    s2i = pickle.load(open(options.s2i, 'rb'))
    t2i = pickle.load(open(options.t2i, 'rb'))
    assert s2i['<PAD>'] == t2i['<PAD>']
    sc2i = pickle.load(open(options.s2i, 'rb'))
    tc2i = pickle.load(open(options.t2i, 'rb'))
    s2spell = pickle.load(open(options.s2spell, 'rb'))
    t2spell = pickle.load(open(options.t2spell, 'rb'))
    dataset = ParallelTextDataset(options.gloss_train, s2i, t2i)
    dev_dataset = ParallelTextDataset(options.gloss_dev, s2i, t2i)
    dataloader = DataLoader(dataset, batch_size=options.batch_size, shuffle=True, collate_fn=my_collate)
    dataloader_dev = DataLoader(dataset, batch_size=options.batch_size, shuffle=False, collate_fn=my_collate)
    gloss_encoder = torch.nn.Embedding(len(s2i), options.embedding_size, padding_idx=s2i['<PAD>'])
    gloss_decoder = torch.nn.Linear(options.rnn_size * 2, len(t2i))
    glosstagger = GlossTagger(inp_vocab=len(s2i),
                              out_vocab=len(t2i),
                              rnn_size=options.rnn_size,
                              num_layers=options.rnn_layers,
                              embedding_size=options.embedding_size,
                              dropout_prob=0.3,
                              pad_idx=s2i['<PAD>'],
                              encoder=gloss_encoder,
                              decoder=gloss_decoder,
                              rnn_type=options.rnn_type)
    if options.gpuid > -1:
        glosstagger.init_cuda()

    print(glosstagger)
    total_batches = int(np.ceil(len(dataset) / options.batch_size))
    ave_time = 0.
    s = time.time()
    for epoch in range(options.epochs):
        glosstagger.train()
        train_losses = []
        dev_losses = []
        for batch_idx, batch in enumerate(dataloader):
            lengths, data, labels = batch
            if glosstagger.is_cuda():
                data = data.cuda()
                labels = labels.cuda()
            batch = lengths, data, labels
            loss = glosstagger.do_backprop(batch)
            if batch_idx % 10 == 0 and batch_idx > 0:
                e = time.time()
                ave_time = (e - s) / 10.
                s = time.time()
                print("e{:d} b{:5d}/{:5d} loss:{:7.4f} ave_time:{:7.4f}\r".format(epoch, batch_idx + 1,
                                                                                  total_batches, loss, ave_time))
            else:
                print("e{:d} b{:d}/{:d} loss:{:7.4f}\r".format(epoch, batch_idx + 1, total_batches, loss))
            train_losses.append(loss)
        if dataloader_dev is not None:
            glosstagger.eval()
            for batch_idx, batch in enumerate(dataloader_dev):
                l, data, labels = batch
                if glosstagger.is_cuda():
                    data = data.cuda()
                    labels = labels.cuda()
                batch = l, data, labels
                loss = glosstagger(batch)
                loss = loss.item()  # .data.cpu().numpy()[0]
                dev_losses.append(loss)
            print("Ending e{:d} AveTrainLoss:{:7.4f} AveDevLoss:{:7.4f}\r".format(epoch, np.mean(train_losses),
                                                                                  np.mean(dev_losses)))
            save_name = "e_{:d}_train_loss_{:.4f}_dev_loss_{:.4f}".format(epoch, np.mean(train_losses),
                                                                          np.mean(dev_losses))
        else:
            save_name = "e_{:d}_train_loss_{:.4f}".format(epoch, np.mean(train_losses))
            print("Ending e{:d} AveTrainLoss:{:7.4f}\r".format(epoch, np.mean(train_losses)))
        if options.save_model is not None:
            glosstagger.save_model(os.path.join(options.save_model, save_name + '.model'))
