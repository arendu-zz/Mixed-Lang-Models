#!/usr/bin/env python
import argparse
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
import time


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

class PriorityQ(object):
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.__q = []

    def append(self, item):
        self.__q.append(item)
        self.__q.sort(key=attrgetter('score'), reverse=True)
        self.__q = self.__q[:self.maxsize]  # keep best successors

    def pop(self, random=False):
        if random:
            idx = np.random.choice(len(self.__q))
            return self.__q.pop(idx)
        else:
            return self.__q.pop(0)


    def length(self,):
        return len(self.__q)

class MacaronicState(object):
    def __init__(self,
                 macaronic_sentences,
                 displayed_sentence_idx,
                 model):
        self.macaronic_sentences = macaronic_sentences
        self.displayed_sentence_idx = displayed_sentence_idx
        self.model = model
        self.weights = None
        self.score = 0.
        self.terminal = False

    def __str__(self,):
        s = []
        for sent_id, sent in enumerate(self.macaronic_sentences):
            s.append(str(sent))
            if sent_id == self.displayed_sentence_idx:
                break
        s.append('weight id:' + str(id(self.weights)))
        s.append('terminal:' + str(self.terminal))
        s.append('score:' + str(self.score))
        return '\n'.join(s)

    def current_sentence(self,):
        if self.displayed_sentence_idx > -1:
            return self.macaronic_sentences[self.displayed_sentence_idx]
        else:
            return None

    def swap_counts(self,):
        c = 0
        u = set()
        for i in range(self.displayed_sentence_idx + 1):
            c += len(self.macaronic_sentences[i].l2_swapped_tokens)
            u.update(self.macaronic_sentences[i].l2_swapped_types)
        return c, len(u)

    def possible_actions(self,):
        s = self.current_sentence()
        if s is None:
            return []
        elif self.terminal:
            return []
        else:
            return [-1] + s.possible_swaps()

    def copy(self,):
        new_sentences = []
        for ms in self.macaronic_sentences:
            new_sentences.append(ms.copy())
        c = MacaronicState(new_sentences,
                           self.displayed_sentence_idx,
                           self.model)
        c.weights = self.weights
        c.score = self.score
        return c

    def next_state(self, action, model_config_func, **kwargs):
        c = self.copy()
        current_displayed_config = c.current_sentence()
        if action == -1:
            new_score, new_weights = model_config_func(current_displayed_config,
                                                       c.model,
                                                       c.weights,
                                                       kwargs['max_steps'],
                                                       kwargs['improvement_threshold'])
            c.weights = new_weights
            swap_token_count, swap_type_count = c.swap_counts()
            c.score = new_score - (kwargs['penalty'] * swap_type_count)
            if c.displayed_sentence_idx + 1 < len(c.macaronic_sentences):
                c.displayed_sentence_idx = self.displayed_sentence_idx + 1
            else:
                c.terminal = True
            return c
        else:
            current_displayed_config.update_config(action)
            new_score, _ = model_config_func(current_displayed_config,
                                             c.model,
                                             c.weights,
                                             kwargs['max_steps'],
                                             kwargs['improvement_threshold'])
            c.weights = c.weights  # should not give new_weights here!
            swap_token_count, swap_type_count = c.swap_counts()
            c.score = new_score - (kwargs['penalty'] * swap_type_count)
            return c


class MacaronicSentence(object):
    def __init__(self,
                 tokens_l1, tokens_l2,
                 int_l1, int_l2,
                 swapped, swappable,
                 l2_swapped_types, l2_swapped_tokens,
                 swap_limit):
        self.__tokens_l1 = tokens_l1
        self.__tokens_l2 = tokens_l2
        self.__int_l1 = int_l1
        self.__int_l2 = int_l2
        self.__swapped = swapped
        self.__swappable = swappable  # set([idx for idx, tl2 in enumerate(tokens_l2)][1:-1])
        self.__l2_swapped_types = l2_swapped_types
        self.__l2_swapped_tokens = l2_swapped_tokens
        self.swap_limit = swap_limit
        assert type(self.__tokens_l1) == list
        assert len(self.__tokens_l1) == len(self.__tokens_l2)
        self.len = len(self.__tokens_l2)

    @property
    def l2_swapped_types(self,):
        return self.__l2_swapped_types

    @property
    def l2_swapped_tokens(self,):
        return self.__l2_swapped_tokens

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

    @property
    def swapped(self,):
        return self.__swapped

    @property
    def swappable(self,):
        return self.__swappable
    
    def possible_swaps(self,):
        if len(self.swapped) < self.len * self.swap_limit:
            return list(self.swappable)
        else:
            return []

    def update_config(self, action):
        self.__swapped.add(action)
        self.__swappable.remove(action)
        l2_int_item = self.int_l2[:, action].item()
        assert isinstance(l2_int_item, int)
        self.__l2_swapped_types.add(l2_int_item)
        self.__l2_swapped_tokens.append(l2_int_item)
        return self

    def copy(self,):
        macaronic_copy = MacaronicSentence(self.tokens_l1,  # this does not change so need not deep copy
                                           self.tokens_l2,  # this also does not change
                                           self.int_l1, #.clone(),  # same
                                           self.int_l2, #.clone(),  # same
                                           copy.deepcopy(self.swapped),  # this  does change so we deepcopy
                                           copy.deepcopy(self.swappable),
                                           copy.deepcopy(self.__l2_swapped_types),  # this  does change so we deepcopy
                                           copy.deepcopy(self.__l2_swapped_tokens),
                                           self.swap_limit)
        return macaronic_copy

    def display_macaronic(self,):
        s = [tl1 if idx not in self.swapped else TEXT_EFFECT.CYAN + tl2 + TEXT_EFFECT.END
             for idx, tl1, tl2 in zip(range(self.len), self.tokens_l1, self.tokens_l2)]
        return ' '.join(s)

    def __str__(self,):
        return self.display_macaronic()

def prep_swap(macaronic_config):
    macaronic_d1 = macaronic_config.int_l1.clone()
    macaronic_d2 = macaronic_config.int_l2.clone()
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

def apply_swap(macaronic_config, model, weights, max_steps=1, improvement_threshold=0.01):
    macaronic_d, indicator, flip_l2, flip_l2_offset, flip_l2_set = prep_swap(macaronic_config)
    if model.is_cuda():
        macaronic_d = macaronic_d.cuda()
        indicator = indicator.cuda()
        flip_l2 = flip_l2.cuda()
        flip_l2_offset = flip_l2_offset.cuda()
        flip_l2_set = flip_l2_set.cuda()

    var_batch = [macaronic_d.size(1)], macaronic_d, indicator
    model.update_g_weights(weights)
    model.init_param_freeze(CBiLSTM.L2_LEARNING)
    model.init_optimizer(type='Adam')
    prev_loss = 100.
    num_steps = 0
    improvement = 1.
    while num_steps < max_steps and improvement > improvement_threshold:
        loss, grad_norm = model.do_backprop(var_batch, seen=(flip_l2, flip_l2_offset, flip_l2_set))
        step_score_vocabtype = model.score_embeddings()
        num_steps += 1
        improvement = prev_loss - loss
        prev_loss = loss
        # print('loss', loss, 'num_steps', num_steps, 'improvement', improvement, 'grad_norm', grad_norm, 'weights', model.g_encoder.weight.sum().item())
    #    print(num_steps, 'step score', step_score_vocabtype, 'loss', loss)
    #step_score_vocabtype = model.score_embeddings(l2_key, l1_key)
    new_weights = model.g_encoder.weight.clone().detach().cpu()
    return step_score_vocabtype, new_weights


def make_start_state(i2v, i2gv, init_weights, model, dl, **kwargs):
    score_0 = model.score_embeddings()
    macaronic_sentences = []
    for sent_idx, sent in enumerate(dl):
        lens, l1_data, l2_data = sent
        swapped = set([])
        swappable = set(range(1, l1_data[0, :].size(0) - 1))
        l1_tokens = [i2v[i.item()] for i in l1_data[0, :]]
        l2_tokens = [i2gv[i.item()] for i in l2_data[0, :]]
        ms = MacaronicSentence(l1_tokens, l2_tokens,
                               l1_data, l2_data,
                               swapped, swappable,
                               set([]), [],
                               kwargs['swap_limit'])
        macaronic_sentences.append(ms)
    state = MacaronicState(macaronic_sentences, 0, model)
    state.weights = init_weights
    state.score = score_0
    return state


def beam_search(init_state, **kwargs):
    beam_size = kwargs['beam_size']
    best_state = init_state

    q = PriorityQ(beam_size)
    q.append(init_state)
    while q.length() > 0:
        curr_state = q.pop(kwargs['stochastic'] == 1)
        if kwargs['verbose']:
            print('curr_state\n', str(curr_state))
        if curr_state.score > best_state.score:
            best_state = curr_state
            if kwargs['verbose']:
                print('best_state\n', str(best_state))
        if curr_state.terminal:
            return curr_state
        for action in curr_state.possible_actions():
            new_state = curr_state.next_state(action, apply_swap, **kwargs)
            q.append(new_state)

    return best_state


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
    opt.add_argument('--stochastic', action='store', dest='stochastic', default=1, type=int, choices=[0,1])
    opt.add_argument('--beam_size', action='store', dest='beam_size', default=10, type=int)
    opt.add_argument('--swap_limit', action='store', dest='swap_limit', default=0.3, type=float)
    opt.add_argument('--max_steps', action='store', dest='max_steps', default=10, type=int)
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
    kwargs = vars(options)
    start_state = make_start_state(i2v, i2gv, weights, cbilstm, dataloader, **kwargs)
    best_state = beam_search(start_state, **kwargs)
    print('search completed')
    print(str(best_state))
