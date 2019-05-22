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

from src.models.ce_model import CE_CLOZE
from src.models.ce_model import spell2mat
from src.models.ce_model import TiedEncoderDecoder, CharTiedEncoderDecoder, CGramTiedEncoderDecoder
from src.models.ce_model import ClozeContextEncoder, ClozeMaskContextEncoder, LMContextEncoder
from src.models.model_untils import make_context_encoder

import pdb


def do_dev_batch(model, dev_data, nsc, prev_dev_acc_mu):
    assert options.dev_corpus is not None
    dev_losses = []
    dev_accs = []
    model.eval()
    print('num_updates', num_updates)
    for batch_idx, batch in enumerate(dev_data):
        l, data, text_data = batch
        ind = torch.ones_like(data).long()
        if model.is_cuda():
            data = data.cuda()
            ind = ind.cuda()
        cuda_batch = l, data, data, ind
        with torch.no_grad():
            _loss, acc = model(cuda_batch)
            loss = _loss.item()
            del _loss
        dev_losses.append(loss)
        dev_accs.append(acc)
    dev_acc_mu = np.mean(dev_accs)
    dev_losses_mu = np.mean(dev_losses)
    print("AveDevLoss:{:7.6f} AveDecAcc:{:.3f}\r".format(dev_losses_mu, dev_acc_mu))
    save_name = 'best'
    if options.save_folder is not None:
        if dev_acc_mu > prev_dev_acc_mu:
            print('saving model...', dev_acc_mu, 'greater than', prev_dev_acc_mu)
            model.save_model(os.path.join(options.save_folder, save_name + '.model'))
            prev_dev_acc_mu = dev_acc_mu
            nsc = 0
        else:
            nsc += 1
            print('not saving model...', dev_acc_mu, 'less than', prev_dev_acc_mu)
    if options.use_early_stop == 1 and nsc > 3:
        return False, prev_dev_acc_mu, nsc
    else:
        return True, prev_dev_acc_mu, nsc


if __name__ == '__main__':
    opt = argparse.ArgumentParser(description="write program description here")
    # insert options here
    opt.add_argument('--save_dir', action='store', dest='save_folder', required=True,
                     help='folder to save the model after every epoch')
    opt.add_argument('--train_corpus', action='store', dest='train_corpus', required=True)
    opt.add_argument('--dev_corpus', action='store', dest='dev_corpus', required=False, default=None)
    opt.add_argument('--v2i', action='store', dest='v2i', required=True,
                     help='vocab to index pickle obj')
    opt.add_argument('--v2cgramspell', action='store', dest='v2cgramspell', required=True,
                     help='vocab to spelling pickle obj')
    opt.add_argument('--cgram2i', action='store', dest='cgram2i', required=True,
                     help='character to index pickle obj comma sepeated list of files')
    opt.add_argument('--embedding_size', action='store', type=int,
                     dest='embedding_size', default=100)
    opt.add_argument('--embedding_pretrain', action='store', type=int,
                     dest='embedding_pretrain', choices=[1, 0], default=0, required=True)
    opt.add_argument('--model_size', action='store',
                     type=int, dest='model_size', default=100)
    opt.add_argument('--batch_size', action='store', type=int, dest='batch_size', default=20)
    opt.add_argument('--loss_at', action='store', type=str,
                     choices=['all', 'noise'], required=True)
    opt.add_argument('--gpuid', action='store', type=int, dest='gpuid', default=-1)
    opt.add_argument('--epochs', action='store', type=int, dest='epochs', default=50)
    opt.add_argument('--noise_profile', action='store', type=int, dest='noise_profile',
                     choices=[1, 2, 3], default=1)
    opt.add_argument('--num_layers', action='store', type=int, dest='num_layers', default=1)
    opt.add_argument('--seed', action='store', dest='seed', default=1234, type=int)
    opt.add_argument('--use_early_stop', action='store', dest='use_early_stop',
                     default=1, type=int, choices=set([0, 1]))
    opt.add_argument('--context_encoder', action='store', dest='context_encoder',
                     choices=['cloze', 'cloze_mask', 'lm'], required=True)
    opt.add_argument('--checkpoint_freq', action='store', required=True,
                     dest='checkpoint_freq', type=int, default=10)
    opt.add_argument('--char_aware', action='store', required=False, choices=[0, 1, 2],
                     dest='char_aware', type=int, default=0)
    opt.add_argument('--pool_type', action='store',
                     choices=['None', 'RNN', 'CNNAvg', 'CNNMax', 'CNNLP', '0-4', '1-4', '2-4', '3-4', '4-4'],
                     dest='pool_type', type=str)
    opt.add_argument('--lang_bit_ratio', action='store', dest='lang_bit_ratio', type=float)
    opt.add_argument('--learn_main_embs', action='store', dest='learn_main_embs',
                     type=int, choices=[0, 1], required=True)
    opt.add_argument('--vmat', action='store', dest='vmat', required=True,
                     help='l1 word embeddings')
    options = opt.parse_args()
    print(options)
    assert 0.0 <= options.lang_bit_ratio <= 1.0, " lang_bit_ratio should be [0,1.0]"
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

    #load cgrams
    cgram_files = options.cgram2i.split(',')
    cgram2i_list = []
    i2cgram_list = []
    for cgram_file in cgram_files:
        cngram2i = pickle.load(open(cgram_file, 'rb'))
        i2cngram = {v: k for k, v in cngram2i.items()}
        cgram2i_list.append(cngram2i)
        i2cgram_list.append(i2cngram)
    v2cgramspell_files = options.v2cgramspell.split(',')
    cgram_spelling_mat_list = []
    for v2cgramspell_file in v2cgramspell_files:
        f = pickle.load(open(v2cgramspell_file, 'rb'))
        m = spell2mat(f)
        assert vocab_size == m.shape[0]
        cgram_spelling_mat_list.append(m)

    if options.embedding_pretrain == 1:
        vmat = torch.load(options.vmat)
    else:
        vmat = None

    if options.char_aware == 1:
        tied_encoder_decoder = CharTiedEncoderDecoder(char_vocab_size=len(cgram2i_list[0]),
                                                      char_embedding_size=options.embedding_size,
                                                      word_vocab_size=vocab_size,
                                                      word_embedding_size=options.embedding_size,
                                                      spelling_mat=cgram_spelling_mat_list[0],
                                                      mode='l1',
                                                      pool=options.pool_type,
                                                      num_lang_bits=int(options.embedding_size * options.lang_bit_ratio))
        tied_encoder_decoder.param_type = 'l1'
    elif options.char_aware == 2:
        l1_cgram_vocab_sizes = [len(i) for i in cgram2i_list]
        l1_cgram_spelling_mats = cgram_spelling_mat_list
        min_cgram = int(options.pool_type.split('-')[0])
        max_cgram = int(options.pool_type.split('-')[1])
        l1_cgram_vocab_sizes = l1_cgram_vocab_sizes[min_cgram:max_cgram]
        l1_cgram_spelling_mats = l1_cgram_spelling_mats[min_cgram:max_cgram]
        tied_encoder_decoder = CGramTiedEncoderDecoder(l1_cgram_vocab_sizes=l1_cgram_vocab_sizes,
                                                       l1_cgram_spelling_mats=l1_cgram_spelling_mats,
                                                       l1_word_vocab_size=vocab_size,
                                                       l2_cgram_vocab_sizes=[],
                                                       l2_cgram_spelling_mats=[],
                                                       l2_cgram_by_l1_spelling_mats=[],
                                                       l2_word_vocab_size=None,
                                                       word_embedding_size=options.embedding_size,
                                                       mode='l1',
                                                       min_cgram=min_cgram,
                                                       max_cgram=max_cgram,
                                                       param_type='l1',
                                                       learn_main_l1_embs=options.learn_main_embs == 1)
    else:
        tied_encoder_decoder = TiedEncoderDecoder(vocab_size=vocab_size,
                                                  embedding_size=options.embedding_size,
                                                  mode='l1', vmat=vmat)
        tied_encoder_decoder.param_type = 'l1'

    train_dataset = TextDataset(options.train_corpus, v2i, shuffle=True, sort_by_len=True,
                                min_batch_size=options.batch_size)
    if options.dev_corpus is not None:
        dev_dataset = TextDataset(options.dev_corpus, v2i, shuffle=False, sort_by_len=True,
                                  min_batch_size=2000)
    if options.context_encoder == 'cloze_mask':
        context_encoder = ClozeMaskContextEncoder(input_size=options.embedding_size,
                                                  rnn_size=options.model_size,
                                                  num_layers=options.num_layers)
    elif options.context_encoder == 'cloze':
        assert options.num_layers == 1
        context_encoder = ClozeContextEncoder(input_size=options.embedding_size,
                                              rnn_size=options.model_size,
                                              num_layers=1)
    elif options.context_encoder == 'lm':
        context_encoder = LMContextEncoder(input_size=options.embedding_size,
                                           rnn_size=options.model_size,
                                           num_layers=options.num_layers)
    else:
        raise NotImplementedError("unknown context_encoder")

    simulation_model = CE_CLOZE(tied_encoder_decoder=tied_encoder_decoder,
                                context_encoder=context_encoder,
                                l1_dict=v2i,
                                loss_at=options.loss_at)

    if options.gpuid > -1:
        simulation_model.init_cuda()

    print(simulation_model)
    for n, p in simulation_model.named_parameters():
        print(n, p.requires_grad)
    print(sum([p.numel() for p in simulation_model.parameters() if p.requires_grad]), ' learnable parameters')
    print(sum([p.numel() for p in simulation_model.parameters()]), ' parameters ( if char_aware spelling_mat is not learnable)')
    ave_time = 0.
    s = time.time()
    total_batches = 0  # train_dataset.num_batches
    early_stops = []
    num_updates = 0
    prev_dev_acc_mu = 0
    nsc = 0
    for epoch in range(options.epochs):
        for batch_idx, batch in enumerate(train_dataset):
            simulation_model.train()
            l, data, text_data = batch
            ind = data.ne(v2i[SPECIAL_TOKENS.PAD]).long()
            if simulation_model.is_cuda():
                data = data.cuda()
                ind = ind.cuda()
            cuda_batch = l, data, data, ind
            loss, grad_norm, acc = simulation_model.train_step(cuda_batch)
            num_updates += 1
            if batch_idx % 100 == 0 and batch_idx > 0:
                e = time.time()
                ave_time = (e - s) / 100.
                s = time.time()
                print("e{:d} b{:d}/{:d} loss:{:7.6f} acc:{:.3f} ave_time:{:7.6f} updates:{:d} nsc:{:d}\r".format(epoch,
                                                                                           batch_idx + 1,
                                                                                           total_batches,
                                                                                           loss,
                                                                                           acc,
                                                                                           ave_time,
                                                                                           num_updates,
                                                                                           nsc))
            else:
                print("e{:d} b{:d}/{:d} loss:{:7.6f} acc:{:.3f} updates:{:d} nsc:{:d}\r".format(epoch,
                                                                                                batch_idx + 1,
                                                                                                total_batches,
                                                                                                loss,
                                                                                                acc,
                                                                                                num_updates,
                                                                                                nsc))
            if num_updates % options.checkpoint_freq == 0 and epoch > 1:
                continue_train, prev_dev_acc_mu, nsc = do_dev_batch(simulation_model, dev_dataset, nsc, prev_dev_acc_mu)
                if not continue_train:
                    exit()
        total_batches = batch_idx
        _, prev_dev_acc_mu, nsc = do_dev_batch(simulation_model, dev_dataset, nsc, prev_dev_acc_mu)
        #if not continue_train:
        #    exit()
