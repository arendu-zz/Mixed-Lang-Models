#!/usr/bin/env python
import argparse
import pickle
import sys
import torch

from model import CBiLSTM
from model import VarEmbedding
from model import WordRepresenter
from search import apply_swap
from search import MacaronicSentence
from torch.utils.data import DataLoader
from train import make_cl_decoder
from train import make_cl_encoder
from train import make_wl_decoder
from train import make_wl_encoder
from utils import ParallelTextDataset
from utils import parallel_collate
from utils import TEXT_EFFECT
import pdb


global PAD, EOS, BOS, UNK
PAD = '<PAD>'
UNK = '<UNK>'
BOS = '<BOS>'
EOS = '<EOS>'

if __name__ == '__main__':
    print(sys.stdout.encoding)
    torch.manual_seed(1234)
    opt = argparse.ArgumentParser(description="write program description here")
    # insert options here
    opt.add_argument('--data_dir', action='store', dest='data_folder', required=True)
    opt.add_argument('--train_corpus', action='store', dest='train_corpus', required=True)
    opt.add_argument('--train_mode', action='store', dest='train_mode', default=1, type=int, choices=set([1]))
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
    opt.add_argument('--batch_size', action='store', type=int, dest='batch_size', default=1)
    opt.add_argument('--gpuid', action='store', type=int, dest='gpuid', default=-1)
    opt.add_argument('--trained_model', action='store', dest='trained_model', required=True)
    opt.add_argument('--key', action='store', dest='key', required=True)
    opt.add_argument('--max_steps', action='store', dest='max_steps', default=100, type=int)
    opt.add_argument('--penalty', action='store', dest='penalty', default=0.0, type=float)
    opt.add_argument('--improvement', action='store', dest='improvement_threshold', default=0.01, type=float)
    opt.add_argument('--verbose', action='store_true', dest='verbose', default=False)
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
    l2_key, l1_key = zip(*pickle.load(open(options.key, 'rb')))
    l2_key = torch.LongTensor(list(l2_key))
    l1_key = torch.LongTensor(list(l1_key))
    train_mode = CBiLSTM.L2_LEARNING
    dataset = ParallelTextDataset(options.train_corpus, v2i, gv2i)
    dataloader = DataLoader(dataset, batch_size=options.batch_size,  shuffle=False, collate_fn=parallel_collate)
    total_batches = len(dataset)
    v_max_vocab = len(v2i)
    g_max_vocab = len(gv2i) if gv2i is not None else 0
    max_vocab = max(v_max_vocab, g_max_vocab)
    cbilstm = torch.load(options.trained_model, map_location=lambda storage, loc: storage)
    cbilstm.set_key(l1_key, l2_key)
    cbilstm.init_key()
    if isinstance(cbilstm.encoder, VarEmbedding):
        wr = cbilstm.encoder.word_representer
        we_size = wr.we_size
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
        encoder = make_wl_encoder(None, None, learned_weights.data.clone())
        decoder = make_wl_decoder(encoder)
        cbilstm.encoder = encoder
        cbilstm.decoder = decoder
        cbilstm.g_encoder = g_cl_encoder
        cbilstm.g_decoder = g_cl_decoder
        cbilstm.init_param_freeze(CBiLSTM.L2_LEARNING)
    else:
        learned_weights = cbilstm.encoder.weight.data.clone()
        we_size = cbilstm.encoder.weight.size(1)
        max_vocab = cbilstm.encoder.weight.size(0)
        encoder = make_wl_encoder(None, None, learned_weights)
        decoder = make_wl_decoder(encoder)
        g_wl_encoder = make_wl_encoder(max_vocab, we_size, None)
        g_wl_decoder = make_wl_decoder(g_wl_encoder)
        cbilstm.encoder = encoder
        cbilstm.decoder = decoder
        cbilstm.g_encoder = g_wl_encoder
        cbilstm.g_decoder = g_wl_decoder
        cbilstm.init_param_freeze(CBiLSTM.L2_LEARNING)
    if options.gpuid > -1:
        cbilstm.init_cuda()
    print(cbilstm)
    hist_flip_l2 = {}
    hist_limit = 1
    penalty = options.penalty #* ( 1.0 / 8849.0)
    if cbilstm.is_cuda:
        sent_init_weights = cbilstm.g_encoder.weight.clone().detach().cpu()
    else:
        sent_init_weights = cbilstm.g_encoder.weight.clone().detach()

    for batch_idx, batch in enumerate(dataloader):
        old_g = cbilstm.g_encoder.weight.clone()
        lens, l1_data, l2_data = batch
        l1_tokens = [i2v[i.item()] for i in l1_data[0, :]]
        l2_tokens = [i2gv[i.item()] for i in l2_data[0, :]]
        swapped = set([])
        swappable = set(range(1, l1_data[0, :].size(0) - 1))
        macaronic_0 = MacaronicSentence(l1_tokens, l2_tokens, l1_data.clone(), l2_data.clone(), swapped, swappable)
        go_next = False
        while not go_next:
            l1_str = ' '.join([str(_idx) + ':' + i for _idx, i in enumerate(macaronic_0.tokens_l1)][1:-1])
            print('\n' + l1_str)
            swaps_selected = input('swqp (1,' + str(lens[0]-2) + '): ').split(',')
            swaps_selected = set([int(i) for i in swaps_selected])
            swap_str = ' '.join([(i2v[l1_data[0, i].item()]
                                 if i not in swaps_selected else (TEXT_EFFECT.UNDERLINE + i2gv[l2_data[0, i].item()] + TEXT_EFFECT.END))
                                 for i in range(1, l1_data.size(1) - 1)])
            new_macaronic = macaronic_0.copy() #.deepcopy(curr_hyp.macaronic_sentence)
            for a in swaps_selected:
                new_macaronic.swapped.add(a)
                new_macaronic.swappable.remove(a)
            if options.verbose:
                print(new_macaronic)
            swap_score, new_weights = apply_swap(new_macaronic,
                                                 cbilstm,
                                                 sent_init_weights,
                                                 penalty)
            print('swap score', swap_score)
            go_next = input('next line or retry (n/r):')
            go_next = go_next == 'n'
            if go_next:
                print('going to next...')
                if cbilstm.is_cuda:
                    sent_init_weights = cbilstm.g_encoder.weight.clone().detach().cpu()
                else:
                    sent_init_weights = cbilstm.g_encoder.weight.clone().detach()
            else:
                #g_wl_encoder = make_wl_encoder(max_vocab, we_size, old_g.data.clone())
                #g_wl_decoder = make_wl_decoder(max_vocab, we_size, g_wl_encoder)
                #cbilstm.g_encoder = g_wl_encoder
                #cbilstm.g_decoder = g_wl_decoder
                #cbilstm.init_param_freeze(CBiLSTM.L2_LEARNING)
                pass
