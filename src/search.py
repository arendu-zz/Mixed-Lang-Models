#!/usr/bin/env python
import argparse
import collections
import copy
from operator import attrgetter
import pickle
import sys
import torch


from model import CBiLSTM
from model import VarEmbedding
from model import WordRepresenter
from torch.utils.data import DataLoader
from train import make_cl_decoder
from train import make_cl_encoder
from train import make_wl_decoder
from train import make_wl_encoder
from utils import ParallelTextDataset
from utils import parallel_collate
from utils import text_effect

import numpy as np

global PAD, EOS, BOS, UNK
PAD = '<PAD>'
UNK = '<UNK>'
BOS = '<BOS>'
EOS = '<EOS>'

Hyp = collections.namedtuple('Hyp', 'score swaps remaining_swaps weights sent_str')


def get_stochastic_neighbors(hyp, num_neighbors=5):
    all_idx = hyp.swaps.union(hyp.remaining_swaps)
    neighbors = []
    for _ in range(num_neighbors):
        swap = hyp.swaps.copy()
        remaining_swap = hyp.remaining_swaps.copy()
        num_swap_additions = np.random.randint(1, 1 + (len(remaining_swap) // 2))
        num_swap_removals = np.random.randint(1, 1 + (len(swap) // 2))
        # take from remaining swaps randomly and add to this list
        swap_additions = set(np.random.choice(list(remaining_swap),
                                              num_swap_additions,
                                              replace=False).tolist())
        # take from existing swaps randomly and add this this list
        swap_removals = set(np.random.choice(list(swap),
                                             num_swap_removals,
                                             replace=False).tolist())
        neighbor_swap = swap.difference(swap_removals)
        neighbor_swap = neighbor_swap.union(swap_additions)

        neighbor_remaining_swap = all_idx.difference(neighbor_swap)
        print('neighbor_swap', neighbor_swap)
        print('neighbor_remaining_swap', neighbor_remaining_swap)
        neighbors.append((neighbor_swap, neighbor_remaining_swap))
        all_neighbor_idx = neighbor_swap.union(neighbor_remaining_swap)
        assert len(all_neighbor_idx.difference(all_idx)) == 0
    return neighbors


def prep_swap(swap_ind, l1_d, l2_d):
    indicator = torch.LongTensor([1] * l1_d.size(1)).unsqueeze(0)
    indicator[:, swap_ind] = 2
    flip_l1 = l1_d[indicator == 2]  # .numpy().tolist()
    flip_l2 = l2_d[indicator == 2]  # .numpy().tolist()
    flip_set = set([(i, j) for i, j in zip(flip_l1.numpy().tolist(), flip_l2.numpy().tolist())])
    flip_l2_set, flip_l2_offset = torch.unique(flip_l2, sorted=True, return_inverse=True)
    flip_l2_set = flip_l2_set.long()
    flip_l2_offset = flip_l2_offset.long()

    # for reward_computation
    flip_l1_key, flip_l2_key = zip(*flip_set)
    flip_l1_key = torch.Tensor(list(flip_l1_key)).long()
    flip_l2_key = torch.Tensor(list(flip_l2_key)).long()

    l1_d[indicator == 2] = l2_d[indicator == 2]
    # swap_str = ' '.join([(i2v[i.item()]
    #                     if indicator[0, _idx] == 1 else (text_effect.UNDERLINE + i2gv[i.item()] + text_effect.END))
    #                     for _idx, i in enumerate(l1_d[0, :])][1:-1])
    return l1_d, indicator, flip_l2, flip_l2_offset, flip_l2_set, flip_l1_key, flip_l2_key


def apply_swap(swap_ind, l1_d, l2_d, l1_key, l2_key, model, max_steps, improvement_threshold, verbose):
    l1_d, indicator, flip_l2, flip_l2_offset, flip_l2_set, flip_l1_key, flip_l2_key = prep_swap(swap_ind, l1_d, l2_d)
    if model.is_cuda():
        l1_d = l1_d.cuda()
        indicator = indicator.cuda()
        flip_l2 = flip_l2.cuda()
        flip_l2_offset = flip_l2_offset.cuda()
        flip_l2_set = flip_l2_set.cuda()

        flip_l1_key = flip_l1_key.cuda()
        flip_l2_key = flip_l2_key.cuda()

        l1_key = l1_key.cuda()
        l2_key = l2_key.cuda()

    var_batch = [l1_d.size(1)], l1_d, indicator
    model.init_optimizer(type='SGD')
    if verbose:
        init_score_vocabtype = model.score_embeddings(l2_key, l1_key)
        print('init score', init_score_vocabtype)
    #prev_step_score_vocabtype = init_score_vocabtype
    prev_loss = 100.
    num_steps = 0
    improvement = 1.
    while improvement >= improvement_threshold and num_steps < max_steps:
        loss, grad_norm = model.do_backprop(var_batch, seen=(flip_l2, flip_l2_offset, flip_l2_set))
        step_score_vocabtype = model.score_embeddings(l2_key, l1_key)
        #improvement = step_score_vocabtype - prev_step_score_vocabtype
        improvement = prev_loss - loss
        #prev_step_score_vocabtype = step_score_vocabtype
        prev_loss = loss
        num_steps += 1
        #if verbose:
        #    print(num_steps, 'step score', step_score_vocabtype, 'loss', loss)
    step_score_vocabtype = model.score_embeddings(l2_key, l1_key)
    if verbose:
        print('final score', step_score_vocabtype)
    return step_score_vocabtype, model


if __name__ == '__main__':
    print(sys.stdout.encoding)
    torch.manual_seed(1234)
    opt = argparse.ArgumentParser(description="write program description here")
    # insert options here
    opt.add_argument('--data_dir', action='store', dest='data_folder', required=True)
    opt.add_argument('--train_corpus', action='store', dest='train_corpus', required=True)
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
    opt.add_argument('--steps', action='store', dest='steps', required=False, default=10)
    opt.add_argument('--beam_size', action='store', dest='beam_size', default=10, type=int)
    opt.add_argument('--swap_limit', action='store', dest='swap_limit', default=0.3, type=float)
    opt.add_argument('--max_steps', action='store', dest='max_steps', default=100, type=int)
    opt.add_argument('--improvement', action='store', dest='improvement_threshold', default=0.01, type=float)
    opt.add_argument('--penalty', action='store', dest='penalty', default=0.2, type=float)
    opt.add_argument('--verbose', action='store_true', dest='verbose', default=False)
    opt.add_argument('--hillclimb', action='store', dest='hillclimb', type=int,  default=0,
                     choices=set([0, 1]), required=False)
    opt.add_argument('--hillclimb_iters', action='store', dest='hillclimb_iters', default=25, type=int, required=False)
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
    dataset = ParallelTextDataset(options.train_corpus, v2i, gv2i)
    dataloader = DataLoader(dataset, batch_size=options.batch_size,  shuffle=False, collate_fn=parallel_collate)
    total_batches = len(dataset)
    v_max_vocab = len(v2i)
    g_max_vocab = len(gv2i) if gv2i is not None else 0
    max_vocab = max(v_max_vocab, g_max_vocab)
    cbilstm = torch.load(options.trained_model, map_location=lambda storage, loc: storage)
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
        encoder = make_wl_encoder(max_vocab, we_size, learned_weights.data.clone())
        decoder = make_wl_decoder(max_vocab, we_size, encoder)
        cbilstm.encoder = encoder
        cbilstm.decoder = decoder
        cbilstm.g_encoder = g_cl_encoder
        cbilstm.g_decoder = g_cl_decoder
        cbilstm.init_param_freeze(CBiLSTM.L2_LEARNING)
    else:
        learned_weights = cbilstm.encoder.weight.data.clone()
        we_size = cbilstm.encoder.weight.size(1)
        max_vocab = cbilstm.encoder.weight.size(0)
        encoder = make_wl_encoder(max_vocab, we_size, learned_weights)
        decoder = make_wl_decoder(max_vocab, we_size, encoder)
        g_wl_encoder = make_wl_encoder(max_vocab, we_size)
        g_wl_decoder = make_wl_decoder(max_vocab, we_size, g_wl_encoder)
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
    penalty = options.penalty
    g_weights = cbilstm.g_encoder.weight.clone()
    score = cbilstm.score_embeddings(l2_key, l1_key)
    macaronic_sents = []
    for batch_idx, batch in enumerate(dataloader):
        lens, l1_data, l2_data = batch
        l1_str = ' '.join([str(_idx) + ':' + i2v[i.item()] for _idx, i in enumerate(l1_data[0, :])][1:-1])
        print('\n' + l1_str)

        # setup stack
        swaps = set([])
        remaining_swaps = set(range(1, l1_data[0, :].size(0) - 1))
        hyp_0 = Hyp(score, swaps, remaining_swaps, g_weights, l1_str)
        stack = [hyp_0]
        best_hyp = hyp_0
        swap_limit = int(float(l1_data[0, :].size(0)) * options.swap_limit)
        beam_size = options.beam_size
        print('perform beam search')
        while len(stack) > 0:
            curr_hyp = stack.pop(0)
            if options.verbose:
                print('curr_hyp', curr_hyp.sent_str, curr_hyp.score)
            if curr_hyp.score > best_hyp.score:
                best_hyp = curr_hyp
                if options.verbose:
                    print('best_hyp', best_hyp.sent_str, best_hyp.score)
            for rs in curr_hyp.remaining_swaps:
                new_swaps = copy.deepcopy(curr_hyp.swaps)
                new_swaps.add(rs)
                new_remaining_swaps = copy.deepcopy(curr_hyp.remaining_swaps)
                new_remaining_swaps.remove(rs)
                # optimistic_future_cost = (1. - penalty) * len(new_remaining_swaps)
                # optimistic_score = curr_hyp.score + optimistic_future_cost
                if len(new_swaps) <= swap_limit:  # and optimistic_score > best_hyp.score:
                    # make new encoder and decoder with curr_hyp weights
                    #g_wl_encoder = make_wl_encoder(max_vocab, we_size, curr_hyp.weights.data.clone())
                    g_wl_encoder = make_wl_encoder(max_vocab, we_size, g_weights.clone())
                    g_wl_decoder = make_wl_decoder(max_vocab, we_size, g_wl_encoder)
                    cbilstm.g_encoder = g_wl_encoder
                    cbilstm.g_decoder = g_wl_decoder
                    cbilstm.init_param_freeze(CBiLSTM.L2_LEARNING)
                    # compute score
                    swap_ind = torch.LongTensor([int(i) for i in sorted(list(new_swaps))])
                    l1_d = l1_data.clone()
                    l2_d = l2_data.clone()
                    indicator = torch.LongTensor([1] * l1_d.size(1)).unsqueeze(0)
                    indicator[:, swap_ind] = 2
                    new_macaronic = ' '.join([(i2v[l1_d[0, i].item()]
                                              if indicator[0, i].item() == 1
                                              else (text_effect.UNDERLINE + i2gv[l2_d[0, i].item()] + text_effect.END))
                                              for i in range(1, l1_d.size(1) - 1)])
                    new_score, cbilstm = apply_swap(swap_ind,
                                                    l1_d,
                                                    l2_d,
                                                    l1_key,
                                                    l2_key,
                                                    cbilstm,
                                                    options.max_steps,
                                                    options.improvement_threshold,
                                                    options.verbose)
                    # save new weights into new hyp
                    new_hyp = Hyp(score=new_score - (penalty * len(new_swaps)),
                                  swaps=new_swaps,
                                  remaining_swaps=new_remaining_swaps,
                                  weights=g_weights.clone(),  # cbilstm.g_encoder.weight.clone(),
                                  sent_str=new_macaronic)
                    stack.append(new_hyp)
                else:
                    pass
            stack.sort(key=attrgetter('score'), reverse=True)
            stack = stack[:beam_size]  # keep best successors
        if options.hillclimb == 1:
            print('perform hillclimb')
            iters = 0
            while iters < options.hillclimb_iters:
                if options.verbose:
                    print('hillclimb iter', iters)
                neighbor_hyps = []
                for neighbor_swaps, neighbor_remaining_swaps in get_stochastic_neighbors(best_hyp):
                    #g_wl_encoder = make_wl_encoder(max_vocab, we_size, best_hyp.weights.data.clone())
                    g_wl_encoder = make_wl_encoder(max_vocab, we_size, g_weights.clone())
                    g_wl_decoder = make_wl_decoder(max_vocab, we_size, g_wl_encoder)
                    cbilstm.g_encoder = g_wl_encoder
                    cbilstm.g_decoder = g_wl_decoder
                    cbilstm.init_param_freeze(CBiLSTM.L2_LEARNING)
                    swap_ind = torch.LongTensor([int(i) for i in sorted(list(neighbor_swaps))])
                    l1_d = l1_data.clone()
                    l2_d = l2_data.clone()
                    indicator = torch.LongTensor([1] * l1_d.size(1)).unsqueeze(0)
                    indicator[:, swap_ind] = 2
                    neighbor_macaronic = ' '.join([(i2v[l1_d[0, i].item()]
                                                  if indicator[0, i].item() == 1
                                                  else (text_effect.UNDERLINE + i2gv[l2_d[0, i].item()] + text_effect.END))
                                                  for i in range(1, l1_d.size(1) - 1)])
                    new_score, cbilstm = apply_swap(swap_ind,
                                                    l1_d,
                                                    l2_d,
                                                    l1_key,
                                                    l2_key,
                                                    cbilstm,
                                                    options.max_steps,
                                                    options.improvement_threshold,
                                                    options.verbose)
                    neighbor_score = new_score - (penalty * len(neighbor_swaps))
                    neighbor_hyp = Hyp(score=neighbor_score,
                                       swaps=neighbor_swaps,
                                       remaining_swaps=neighbor_remaining_swaps,
                                       weights=g_weights.clone(),  # cbilstm.g_encoder.weight.clone(),
                                       sent_str=neighbor_macaronic)
                    neighbor_hyps.append(neighbor_hyp)
                neighbor_hyps.sort(key=attrgetter('score'), reverse=True)
                if options.verbose:
                    print('neighbor_hyps scores', [h.score for h in neighbor_hyps])
                best_neighbor_hyp = neighbor_hyps[0]
                if best_neighbor_hyp.score > best_hyp.score:
                    best_hyp = best_neighbor_hyp
                    iters += 1
                    if options.verbose:
                        print('hillclimb found best_hyp', best_hyp.sent_str, best_hyp.score)
                else:
                    print('rejected neighbors')
                    iters = options.hillclimb_iters
        else:
            print('no hillclimb...')
            pass

        if options.verbose:
            print('final best_hyp',  best_hyp.sent_str, best_hyp.score)
        g_weights = best_hyp.weights.clone()
        score = best_hyp.score
        macaronic_sents.append((best_hyp.sent_str, best_hyp.score))

    print('search completed')
    for sent, score in macaronic_sents:
        print(sent + '\t' + str(score))
