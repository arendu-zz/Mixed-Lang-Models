#!/usr/bin/env python
__author__ = 'arenduchintala'
import argparse
import numpy as np
import os
import pickle
import random
import time
import torch

from model import CBiLSTM
from model import CTransformerEncoder
from model import VarEmbedding
from model import VariationalEmbeddings
from model import VarLinear
from model import VariationalLinear
from model import WordRepresenter

from utils.utils import TextDataset
from utils.utils import SPECIAL_TOKENS

import pdb


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

def make_random_mask(data, lengths, mask_val, pad_idx):
    drop_num = int(lengths[-1] * mask_val)
    mask = data.eq(pad_idx)
    if drop_num > 0:
        drop_samples_col = torch.multinomial(torch.ones(data.shape[0], lengths[-1]), drop_num)
        drop_samples_row = torch.arange(data.shape[0]).unsqueeze(1).expand_as(drop_samples_col)
        mask[drop_samples_row, drop_samples_col] = 1
    return mask


def word_emb_quality(encoder, idx2voc):
    print('--------------------------word-emb---------------------')
    l1_weights = encoder.weight.data
    l1_l1 = l1_weights.matmul(l1_weights.transpose(0, 1))
    _, l1_l1_topk_idx = torch.topk(l1_l1, 10)
    for i in list(range(50)) + list(range(1000, 1050)) + list(range(10000, 10050)):
        if i in idx2voc:
            v = idx2voc[i]
            k = ' '.join([idx2voc[j] for j in l1_l1_topk_idx[i].tolist()])
            print(v + ':' + k)
    print('------------------------------------------------------')


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
    opt.add_argument('--model_type', action='store', dest='model_type', default='lstm',
                     choices=set(['lstm', 'transformer']), help='type of contextual model to use')
    opt.add_argument('--w_embedding_size', action='store', type=int, dest='w_embedding_size', default=500)
    opt.add_argument('--layers', action='store', type=int, dest='layers', default=1)
    opt.add_argument('--model_size', action='store', type=int, dest='model_size', default=250)
    opt.add_argument('--c_embedding_size', action='store', type=int, dest='c_embedding_size', default=20)
    opt.add_argument('--batch_size', action='store', type=int, dest='batch_size', default=20)
    opt.add_argument('--gpuid', action='store', type=int, dest='gpuid', default=-1)
    opt.add_argument('--epochs', action='store', type=int, dest='epochs', default=50)
    opt.add_argument('--mask_val', action='store', type=float, dest='mask_val', default=0.2)
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
    i2v = {v: k for k, v in v2i.items()}
    assert len(v2i) == len(i2v)
    v2c = pickle.load(open(options.v2spell, 'rb'))
    c2i = pickle.load(open(options.c2i, 'rb'))

    train_mode = CBiLSTM.L1_LEARNING
    train_dataset = TextDataset(options.train_corpus, v2i, shuffle=True, sort_by_len=True,
                                min_batch_size=options.batch_size)
    if options.dev_corpus is not None:
        dev_dataset = TextDataset(options.dev_corpus, v2i, shuffle=False, sort_by_len=True,
                                  min_batch_size=options.batch_size)
    vocab_size = len(v2i)
    if options.char_composition == 'None':
        encoder = make_wl_encoder(vocab_size, options.w_embedding_size, None)
        decoder = make_wl_decoder(encoder)
        encoder.padding_idx = v2i[SPECIAL_TOKENS.PAD]
        if options.model_type == 'lstm':
            cloze_model = CBiLSTM(options.w_embedding_size, options.model_size, options.layers,
                                  encoder, decoder, None, None, mode=CBiLSTM.L1_LEARNING, l1_dict=v2i, l2_dict=None)
        elif options.model_type == 'transformer':
            cloze_model = CTransformerEncoder(options.w_embedding_size, options.model_size, options.layers,
                                              encoder, decoder, None, None, mode=CTransformerEncoder.L1_LEARNING, l1_dict=v2i, l2_dict=None)
        else:
            raise NotImplementedError("unknown model type")
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
        if options.model_type == 'lstm':
            cloze_model = CBiLSTM(options.w_embedding_size, options.model_size, options.layers,
                                  encoder, decoder, None, None, mode=CBiLSTM.L1_LEARNING, l1_dict=v2i, l2_dict=None)
        elif options.model_type == 'transformer':
            cloze_model = CTransformerEncoder(options.w_embedding_size, options.model_size, options.layers,
                                              encoder, decoder, None, None, mode=CTransformerEncoder.L1_LEARNING,
                                              l1_dict=v2i, l2_dict=None)
        else:
            raise NotImplementedError("unknown model type")
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
        cloze_model = CBiLSTM(options.w_embedding_size, options.model_size, options.layers,
                              cl_encoder, cl_decoder, None, None, mode=CBiLSTM.L1_LEARNING,
                              l1_dict=v2i,l2_dict=None)
    if options.gpuid > -1:
        cloze_model.init_cuda()

    print(cloze_model)
    ave_time = 0.
    s = time.time()
    total_batches = 0 #train_dataset.num_batches
    mask_val = options.mask_val
    early_stops = []
    for epoch in range(options.epochs):
        cloze_model.train()
        train_losses = []
        train_accs = []
        for batch_idx, batch in enumerate(train_dataset):
            l, data, text_data = batch
            ind = data.ne(v2i[SPECIAL_TOKENS.PAD]).long()
            mask = make_random_mask(data, l, mask_val, v2i[SPECIAL_TOKENS.PAD])
            if cloze_model.is_cuda():
                data = data.cuda()
                ind = ind.cuda()
                mask = mask.cuda()
            cuda_batch = l, data, data, ind, mask
            loss, grad_norm, acc = cloze_model.do_backprop(cuda_batch, total_batches=total_batches)
            if batch_idx % 100 == 0 and batch_idx > 0:
                e = time.time()
                ave_time = (e - s) / 100.
                s = time.time()
                print("e{:d} b{:d}/{:d} loss:{:7.6f} acc:{:.3f} ave_time:{:7.6f}\r".format(epoch,
                                                                                           batch_idx + 1,
                                                                                           total_batches,
                                                                                           loss,
                                                                                           acc,
                                                                                           ave_time))
            else:
                #print("e{:d} b{:d}/{:d} loss:{:7.6f} acc:{:.3f}\r".format(epoch, batch_idx + 1, total_batches, loss, acc))
                pass
            train_losses.append(loss)
            train_accs.append(acc)
        total_batches = batch_idx
        dev_losses = []
        dev_accs = []
        word_emb_quality(cloze_model.encoder, i2v)
        assert options.dev_corpus is not None
        cloze_model.eval()
        for batch_idx, batch in enumerate(dev_dataset):
            l, data, text_data = batch
            mask = make_random_mask(data, l, mask_val, v2i[SPECIAL_TOKENS.PAD])
            ind = torch.ones_like(data).long()
            if cloze_model.is_cuda():
                data = data.cuda()
                ind = ind.cuda()
                mask = mask.cuda()
            cuda_batch = l, data, data, ind, mask
            with torch.no_grad():
                loss, acc = cloze_model(cuda_batch)
                loss = loss.item()  # .data.cpu().numpy()[0]
            dev_losses.append(loss)
            dev_accs.append(acc)

        dev_acc_mu = np.mean(dev_accs)
        dev_losses_mu = np.mean(dev_losses)
        train_acc_mu = np.mean(train_accs)
        train_losses_mu = np.mean(train_losses)
        print("Ending e{:d} AveTrainLoss:{:7.6f} AveTrainAcc{:.3f} AveDevLoss:{:7.6f} AveDecAcc:{:.3f}\r".format(epoch,
                                                                                                       train_losses_mu,
                                                                                                       train_acc_mu,
                                                                                                       dev_losses_mu,
                                                                                                       dev_acc_mu))
        save_name = "e_{:d}_train_loss_{:.6f}_dev_loss_{:.6f}_dev_acc_{:.3f}".format(epoch,
                                                                                     train_losses_mu,
                                                                                     dev_losses_mu,
                                                                                     dev_acc_mu)
        if options.save_folder is not None:
            cloze_model.save_model(os.path.join(options.save_folder, save_name + '.model'))
        if epoch > 5 and dev_losses_mu > max(early_stops[-3:]):
            print('early stopping...')
            exit()
        else:
            early_stops.append(dev_losses_mu)
