#!/usr/bin/env python
__author__ = 'arenduchintala'
import argparse
import numpy as np
import pickle
import random
import time
import torch
from src.utils.utils import TextDataset

from src.models.adv_lm_model import LM

if __name__ == '__main__':
    opt = argparse.ArgumentParser(description="write program description here")
    # insert options here
    opt.add_argument('--save_dir', action='store', dest='save_folder', required=True,
                     help='folder to save the model after every epoch')
    opt.add_argument('--train_corpus', action='store', dest='train_corpus', required=True)
    opt.add_argument('--dev_corpus', action='store', dest='dev_corpus', required=False, default=None)
    opt.add_argument('--adv_labels', action='store', dest='adv_labels', required=False, default=None)
    opt.add_argument('--v2i', action='store', dest='v2i', required=True,
                     help='vocab to index pickle obj')
    opt.add_argument('--w_embedding_size', action='store', type=int, dest='w_embedding_size', default=128)
    opt.add_argument('--num_layers', action='store', type=int, dest='num_layers', default=2)
    opt.add_argument('--model_size', action='store', type=int, dest='model_size', default=128)
    opt.add_argument('--batch_size', action='store', type=int, dest='batch_size', default=200)
    opt.add_argument('--gpuid', action='store', type=int, dest='gpuid', default=-1)
    opt.add_argument('--adv_lambda', action='store', type=float, dest='adv_lambda', default=0.0)
    opt.add_argument('--epochs', action='store', type=int, dest='epochs', default=100)
    opt.add_argument('--seed', action='store', dest='seed', default=1234, type=int)
    opt.add_argument('--use_early_stop', action='store', dest='use_early_stop', default=1, type=int, choices=[0, 1])
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
    i2v = {v: k for k, v in v2i.items()}
    assert len(v2i) == len(i2v)

    train_dataset = TextDataset(options.train_corpus, v2i, shuffle=True, sort_by_len=True,
                                min_batch_size=options.batch_size)
    if options.dev_corpus is not None:
        dev_dataset = TextDataset(options.dev_corpus, v2i, shuffle=False, sort_by_len=True,
                                  min_batch_size=options.batch_size)
    vocab_size = len(v2i)
    lm_model = LM(emb_size=options.w_embedding_size,
                  vocab_size=vocab_size,
                  num_layers=options.num_layers,
                  dropout=0.3,
                  dictionary=v2i,
                  adv_lambda=options.adv_lambda)

    if options.gpuid > -1:
        lm_model = lm_model.cuda()

    print(lm_model)
    ave_time = 0.
    s = time.time()
    total_batches = 0
    early_stops = []
    for epoch in range(options.epochs):
        lm_model.train()
        for batch_idx, batch in enumerate(train_dataset):
            lengths, inps, targets, text_data, adv_targets = batch
            if lm_model.is_cuda():
                inps = inps.cuda()
                targets = targets.cuda()
                adv_targets = adv_targets.cuda()
            loss, mloss, aloss, grad_norm, acc = lm_model.train_step(inps, targets, lengths, adv_targets, epoch)
            if batch_idx % 100 == 0 and batch_idx > 0:
                e = time.time()
                ave_time = (e - s) / 100.
                s = time.time()
                print("e{:d} b{:d}/{:d} loss:{:7.3f} mloss:{:7.3f} aloss:{:7.3f} acc:{:.2f} ave_time:{:7.6f}\r".format(epoch,
                                                                                                                   batch_idx + 1,
                                                                                                                   total_batches,
                                                                                                                   loss,
                                                                                                                   mloss,
                                                                                                                   aloss,
                                                                                                                   acc,
                                                                                                                   ave_time))
            else:
                #print("e{:d} b{:d}/{:d} loss:{:7.6f} acc:{:.3f}\r".format(epoch,
                #                                                          batch_idx + 1,
                #                                                          total_batches,
                #                                                          loss,
                #                                                          acc))
                pass
        total_batches = batch_idx
        if epoch % 5 == 0:
            print('-------------EPOCH ' + str(epoch) + '------------------------')
            print(lm_model.generate())
            print('--------------------------------------------------------------')

