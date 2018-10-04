#!/usr/bin/env python
import argparse
import collections
import copy
import numpy as np
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
from utils import TEXT_EFFECT
import pdb

global PAD, EOS, BOS, UNK
PAD = '<PAD>'
UNK = '<UNK>'
BOS = '<BOS>'
EOS = '<EOS>'

class Hyp(object):
    def __init__(self, score, weights, macaronic_sentence):
        self.score = score
        self.weights = weights
        self.macaronic_sentence = macaronic_sentence

    def get_actions(self,):
        return list(self.macaronic_sentence.swappable)

    def __str__(self,):
        return str(self.macaronic_sentence)


class MacaronicSentence(object):
    def __init__(self, tokens_l1, tokens_l2, int_l1, int_l2, swapped, swappable):
        self.__tokens_l1 = tokens_l1
        self.__tokens_l2 = tokens_l2
        self.__int_l1 = int_l1
        self.__int_l2 = int_l2
        #self.__int_l1.flags.writeable = False
        #self.__int_l2.flags.writeable = False
        assert type(self.__tokens_l1) == list
        assert len(self.__tokens_l1) == len(self.__tokens_l2)
        self.len = len(self.__tokens_l2)
        self.swapped = swapped
        self.swappable = swappable  # set([idx for idx, tl2 in enumerate(tokens_l2)][1:-1])

    @property
    def int_l1(self,):
        return self.__int_l1

    @property
    def int_l2(self,):
        return self.__int_l2

    @property
    def tokens_l2(self,):
        return self.__tokens_l2

    @property
    def tokens_l1(self,):
        return self.__tokens_l1

    def update_config(self, action):
        self.swapped.add(action)
        self.swappable.remove(action)
        return self

    def copy(self,):
        macaronic_copy = MacaronicSentence(self.tokens_l1,  # this does not change so need not deep copy
                                           self.tokens_l2,  # this also does not change
                                           self.int_l1.clone(),  # same
                                           self.int_l2.clone(),  # same
                                           copy.deepcopy(self.swapped),  # this  does change so we deepcopy
                                           copy.deepcopy(self.swappable))
        return macaronic_copy

    def display_macaronic(self,):
        s = [tl1 if idx not in self.swapped else TEXT_EFFECT.CYAN + tl2 + TEXT_EFFECT.END
             for idx, tl1, tl2 in zip(range(self.len), self.tokens_l1, self.tokens_l2)]
        return ' '.join(s)

    def __str__(self,):
        return self.display_macaronic()


def prep_swap(macaronic_config):
    macaronic_d1 = torch.LongTensor(macaronic_config.int_l1)
    macaronic_d2 = torch.LongTensor(macaronic_config.int_l2)
    swap_ind = torch.LongTensor(sorted(list(macaronic_config.swapped)))
    indicator = torch.LongTensor([1] * macaronic_d1.size(1)).unsqueeze(0)
    indicator[:, swap_ind] = 2
    # flip_l1 = l1_d[indicator == 2]  # .numpy().tolist()
    flip_l2 = macaronic_d2[indicator == 2]  # .numpy().tolist()
    # flip_set = set([(i, j) for i, j in zip(flip_l1.numpy().tolist(), flip_l2.numpy().tolist())])
    flip_l2_set, flip_l2_offset = torch.unique(flip_l2, sorted=True, return_inverse=True)
    flip_l2_set = flip_l2_set.long()
    flip_l2_offset = flip_l2_offset.long()

    # for reward_computation
    # flip_l1_key, flip_l2_key = zip(*flip_set)
    # flip_l1_key = torch.Tensor(list(flip_l1_key)).long()
    # flip_l2_key = torch.Tensor(list(flip_l2_key)).long()

    macaronic_d1[indicator == 2] = macaronic_d2[indicator == 2]
    return macaronic_d1, indicator, flip_l2, flip_l2_offset, flip_l2_set


def apply_swap(macaronic_config, model, weights, options):
    macaronic_d, indicator, flip_l2, flip_l2_offset, flip_l2_set = prep_swap(macaronic_config)
    if model.is_cuda():
        macaronic_d = macaronic_d.cuda()
        indicator = indicator.cuda()
        flip_l2 = flip_l2.cuda()
        flip_l2_offset = flip_l2_offset.cuda()
        flip_l2_set = flip_l2_set.cuda()

        # flip_l1_key = flip_l1_key.cuda()
        # flip_l2_key = flip_l2_key.cuda()

        #l1_key = l1_key.cuda()
        #l2_key = l2_key.cuda()

    var_batch = [macaronic_d.size(1)], macaronic_d, indicator
    #g_wl_encoder = make_wl_encoder(max_vocab, we_size, weights)
    #g_wl_decoder = make_wl_decoder(max_vocab, we_size, g_wl_encoder)
    #model.g_encoder = g_wl_encoder
    #model.g_decoder = g_wl_decoder
    #print('old weights', model.g_encoder.weight.sum(), model.g_decoder.weight.sum())
    model.update_g_weights(weights)
    #print('set weights', model.g_encoder.weight.sum(), model.g_decoder.weight.sum())
    #model.init_param_freeze(CBiLSTM.L2_LEARNING)
    #model.init_optimizer(type='SGD')
    model.init_param_freeze(CBiLSTM.L2_LEARNING)
    model.init_optimizer(type='SGD')
    if options.verbose:
        init_score_vocabtype = model.score_embeddings()
        print('init score', init_score_vocabtype)
    prev_loss = 100.
    num_steps = 0
    improvement = 1.
    while improvement >= options.improvement_threshold and num_steps < options.max_steps:
        loss, grad_norm = model.do_backprop(var_batch, seen=(flip_l2, flip_l2_offset, flip_l2_set))
        step_score_vocabtype = model.score_embeddings()
        #improvement = step_score_vocabtype - prev_step_score_vocabtype
        improvement = prev_loss - loss
        #prev_step_score_vocabtype = step_score_vocabtype
        prev_loss = loss
        num_steps += 1
        #if verbose:
        #    print(num_steps, 'step score', step_score_vocabtype, 'loss', loss)
    #step_score_vocabtype = model.score_embeddings(l2_key, l1_key)
    new_weights = model.g_encoder.weight.clone().detach().cpu()
    if options.verbose:
        print('final score', step_score_vocabtype)
        #print('new weights', model.g_encoder.weight.sum(), model.g_decoder.weight.sum())
    return step_score_vocabtype, new_weights


def beam_search(init_config, init_weights, model, options):
    score_0, _ = apply_swap(init_config,
                            model,
                            init_weights,
                            options)
    hyp_0 = Hyp(score_0, init_weights, init_config)
    best_hyp = hyp_0
    stack = [hyp_0]
    swap_limit = int(len(init_config.tokens_l1) * options.swap_limit)
    beam_size = options.beam_size
    while len(stack) > 0:
        curr_hyp = stack.pop(0)
        if options.verbose:
            print('curr_hyp', str(curr_hyp), curr_hyp.score)
        if curr_hyp.score > best_hyp.score:
            best_hyp = curr_hyp
            if options.verbose:
                print('best_hyp', str(best_hyp), best_hyp.score)

        for a in curr_hyp.macaronic_sentence.swappable:
            new_macaronic = curr_hyp.macaronic_sentence.copy()
            new_macaronic.swapped.add(a)
            new_macaronic.swappable.remove(a)
            if len(new_macaronic.swapped) <= swap_limit:  # and optimistic_score > best_hyp.score:
                # make new encoder and decoder with curr_hyp weights
                # compute score
                new_score, new_weights = apply_swap(new_macaronic,
                                                    model,
                                                    init_weights,
                                                    options)
                # save new weights into new hyp
                new_hyp = Hyp(score=new_score - (options.penalty * len(new_macaronic.swapped)),
                              weights=new_weights,
                              macaronic_sentence=new_macaronic)
                stack.append(new_hyp)
            else:
                pass
        stack.sort(key=attrgetter('score'), reverse=True)
        stack = stack[:beam_size]  # keep best successors
    if options.verbose:
        print('stack empty...')
    if options.verbose:
        print('final best_hyp',  str(best_hyp), best_hyp.score)
    final_weights = best_hyp.weights
    final_score = best_hyp.score
    final_config = best_hyp.macaronic_sentence
    return final_config, final_weights, final_score


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
    opt.add_argument('--gpuid', action='store', type=int, dest='gpuid', default=-1)
    opt.add_argument('--trained_model', action='store', dest='trained_model', required=True)
    opt.add_argument('--key', action='store', dest='key', required=True)
    opt.add_argument('--beam_size', action='store', dest='beam_size', default=10, type=int)
    opt.add_argument('--swap_limit', action='store', dest='swap_limit', default=0.3, type=float)
    opt.add_argument('--max_steps', action='store', dest='max_steps', default=100, type=int)
    opt.add_argument('--improvement', action='store', dest='improvement_threshold', default=0.01, type=float)
    opt.add_argument('--penalty', action='store', dest='penalty', default=0.0, type=float)
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
    dataset = ParallelTextDataset(options.train_corpus, v2i, gv2i)
    dataloader = DataLoader(dataset, batch_size=1,  shuffle=False, collate_fn=parallel_collate)
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
    cbilstm.set_key(l1_key, l2_key)
    cbilstm.init_key()
    print(cbilstm)
    macaronic_sents = []
    if cbilstm.is_cuda:
        weights = cbilstm.g_encoder.weight.clone().detach().cpu()
    else:
        weights = cbilstm.g_encoder.weight.clone().detach()

    for sent_idx, sent in enumerate(dataloader):
        lens, l1_data, l2_data = sent
        # setup stack
        swapped = set([])
        swappable = set(range(1, l1_data[0, :].size(0) - 1))
        l1_tokens = [i2v[i.item()] for i in l1_data[0, :]]
        l2_tokens = [i2gv[i.item()] for i in l2_data[0, :]]
        macaronic_0 = MacaronicSentence(l1_tokens, l2_tokens, l1_data.clone(), l2_data.clone(), swapped, swappable)
        config, weights, score = beam_search(macaronic_0, weights, cbilstm, options)
        macaronic_sents.append((config, score))

    print('search completed')
    for sent, score in macaronic_sents:
        print(str(sent) + '\t' + str(score))
