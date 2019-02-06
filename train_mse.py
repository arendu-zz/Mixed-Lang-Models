#!/usr/bin/env python
__author__ = 'arenduchintala'
import argparse
import numpy as np
import os
import pickle
import random
import time
import torch
from src.utils.utils import SPECIAL_TOKENS
from src.utils.utils import TextDataset

from src.models.mse_model import MSE_CLOZE
from src.models.model_untils import make_context_encoder

def make_random_mask(data, lengths, mask_val, pad_idx):
    drop_num = int(lengths[-1] * mask_val)
    mask = data.eq(pad_idx)
    if drop_num > 0:
        drop_samples_col = torch.multinomial(torch.ones(data.shape[0], lengths[-1]), drop_num)
        drop_samples_row = torch.arange(data.shape[0]).unsqueeze(1).expand_as(drop_samples_col)
        mask[drop_samples_row, drop_samples_col] = 1
    return mask


if __name__ == '__main__':
    opt = argparse.ArgumentParser(description="write program description here")
    # insert options here
    opt.add_argument('--save_dir', action='store', dest='save_folder', required=True,
                     help='folder to save the model after every epoch')
    opt.add_argument('--train_corpus', action='store', dest='train_corpus', required=True)
    opt.add_argument('--dev_corpus', action='store', dest='dev_corpus', required=False, default=None)
    opt.add_argument('--v2i', action='store', dest='v2i', required=True,
                     help='vocab to index pickle obj')
    opt.add_argument('--vmat', action='store', dest='vmat', required=True,
                     help='l1 word embeddings')
    opt.add_argument('--v2spell', action='store', dest='v2spell', required=True,
                     help='vocab to spelling pickle obj')
    opt.add_argument('--c2i', action='store', dest='c2i', required=True,
                     help='character (corpus and gloss)  to index pickle obj')
    opt.add_argument('--model_size', action='store', type=int, dest='model_size', default=250)
    opt.add_argument('--context_encoder_type', action='store', type=str, dest='context_encoder_type',
                     required=True, choices=['RNN', 'Attention'])
    opt.add_argument('--batch_size', action='store', type=int, dest='batch_size', default=20)
    opt.add_argument('--loss_type', action='store', type=str,
                     choices=['cs', 'cs_margin', 'mse', 'huber'], required=True)
    opt.add_argument('--gpuid', action='store', type=int, dest='gpuid', default=-1)
    opt.add_argument('--epochs', action='store', type=int, dest='epochs', default=50)
    opt.add_argument('--mask_val', action='store', type=float, dest='mask_val', default=0.2)
    opt.add_argument('--use_rand_hiddens', action='store', type=int, dest='use_rand_hiddens',
                     choices=[0, 1], default=0)
    opt.add_argument('--use_orthographic_model', action='store', type=int, dest='use_orthographic_model',
                     choices=[0, 1, 2, 3, 4], default=0)
    opt.add_argument('--num_highways', action='store', type=int, dest='num_highways', default=1)
    opt.add_argument('--nn_mat', action='store', type=str, dest='nn_mat')
    opt.add_argument('--nn_mat_size', action='store', type=int, dest='nn_mat_size', default=20)
    opt.add_argument('--seed', action='store', dest='seed', default=1234, type=int)
    opt.add_argument('--use_early_stop', action='store',
                     dest='use_early_stop', default=1, type=int, choices=set([0, 1]))
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
    vocab_size = len(v2i)
    v2i = pickle.load(open(options.v2i, 'rb'))
    i2v = {v: k for k, v in v2i.items()}
    assert len(v2i) == len(i2v)

    vmat = torch.load(options.vmat)
    assert vmat.shape[0] == vocab_size
    emb_dim = vmat.shape[1]
    l1_encoder = torch.nn.Embedding(vocab_size, emb_dim)
    l1_encoder.weight.data = vmat

    train_dataset = TextDataset(options.train_corpus, v2i, shuffle=True, sort_by_len=True,
                                min_batch_size=options.batch_size)
    if options.dev_corpus is not None:
        dev_dataset = TextDataset(options.dev_corpus, v2i, shuffle=False, sort_by_len=True,
                                  min_batch_size=2000)
    #if options.use_orthographic_model == 1:
    #    print('MSE_ORTHOGRAPHIC_CLOZE mode 1')
    #    cloze_model = MSE_ORTHOGRAPHIC_CLOZE(input_size=emb_dim,
    #                                         rnn_size=options.model_size,
    #                                         encoder=l1_encoder,
    #                                         nn_mapper=None,
    #                                         l1_dict=v2i,
    #                                         loss_type=options.loss_type)
    #elif options.use_orthographic_model == 2:
    #    print('MSE_ORTHOGRAPHIC_CLOZE mode 2')
    #    nn_mat = torch.load(options.nn_mat)
    #    nn_mat = nn_mat[:, :options.nn_mat_size]
    #    nn_embedding = torch.nn.Embedding(nn_mat.shape[0], nn_mat.shape[1])
    #    nn_embedding.weight.data = nn_mat
    #    cloze_model = MSE_ORTHOGRAPHIC_CLOZE(input_size=emb_dim,
    #                                         rnn_size=options.model_size,
    #                                         encoder=l1_encoder,
    #                                         nn_mapper=nn_embedding,
    #                                         l1_dict=v2i,
    #                                         loss_type=options.loss_type)
    #else:
    if options.use_orthographic_model == 3 or options.use_orthographic_model == 4:
        nn_mat = torch.load(options.nn_mat)
        nn_mat = nn_mat[:, :options.nn_mat_size]
        nn_mapper = torch.nn.Embedding(nn_mat.shape[0], nn_mat.shape[1])
        nn_mapper.weight.data = nn_mat
        nn_mapper.requires_grad = False
    else:
        nn_mapper = None
    context_encoder = make_context_encoder(options.context_encoder_type, emb_dim, options.model_size, v2i[SPECIAL_TOKENS.PAD])
    cloze_model = MSE_CLOZE(emb_dim,
                            l1_encoder,
                            context_encoder,
                            v2i,
                            options.loss_type,
                            options.use_orthographic_model,
                            options.use_rand_hiddens,
                            nn_mapper=nn_mapper,
                            num_highways=options.num_highways)

    if options.gpuid > -1:
        cloze_model.init_cuda()

    print(cloze_model)
    ave_time = 0.
    s = time.time()
    total_batches = 0  # train_dataset.num_batches
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
            loss, grad_norm, acc = cloze_model.do_backprop(cuda_batch)
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
                print("e{:d} b{:d}/{:d} loss:{:7.6f} acc:{:.3f}\r".format(epoch, batch_idx + 1, total_batches, loss, acc))
                pass
            train_losses.append(loss)
            train_accs.append(acc)
        total_batches = batch_idx
        dev_losses = []
        dev_accs = []
        assert options.dev_corpus is not None
        cloze_model.eval()
        print('completed epoch', epoch)
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
                _loss, acc = cloze_model(cuda_batch)
                loss = _loss.item()
                del _loss
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
        if options.use_early_stop == 1:
            if epoch > 5 and dev_losses_mu > max(early_stops[-3:]):
                print('early stopping...')
                exit()
            else:
                early_stops.append(dev_losses_mu)
