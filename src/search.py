#!/usr/bin/env python
import argparse
import copy
import numpy as np
from operator import attrgetter
import pickle
import sys
import torch
import random

from model import CBiLSTM
from model import VarEmbedding
from model import WordRepresenter
from train import make_cl_decoder
from train import make_cl_encoder
from train import make_wl_decoder
from train import make_wl_encoder
from train import make_random_mask
import time

from utils.utils import ParallelTextDataset
from utils.utils import SPECIAL_TOKENS, TEXT_EFFECT

import pdb

global NEXT_SENT
NEXT_SENT = (-1, None)


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
        self.queue = []

    def append(self, item):
        self.queue.append(item)
        self.queue.sort(key=attrgetter('score'), reverse=True)
        self.queue = self.queue[:self.maxsize]  # keep best successors

    def pop(self, stochastic=False):
        if stochastic:
            idx = random.choice(range(len(self.__q)))
            return self.queue.pop(idx)
        else:
            return self.queue.pop(0)

    def __str__(self,):
        s = []
        s += ['--------------------------------------------']
        for idx, item in enumerate(self.queue):
            s += ['idx:' + str(idx)]
            s += [str(item)]
        s += ['--------------------------------------------']
        return '\n'.join(s)

    def length(self,):
        return len(self.queue)

class MacaronicState(object):
    def __init__(self,
                 macaronic_sentences,
                 displayed_sentence_idx,
                 model,
                 start_score,
                 binary_branching=0):
        self.macaronic_sentences = macaronic_sentences
        self.displayed_sentence_idx = displayed_sentence_idx
        self.model = model
        self.binary_branching = binary_branching
        self.weights = None
        self.score = 0.
        self.start_score = start_score
        self.terminal = False


    def __str__(self,):
        s = []
        for sent_id, sent in enumerate(self.macaronic_sentences):
            s.append(str(sent))
            if sent_id == self.displayed_sentence_idx:
                break
        actions, _ = self.possible_actions()
        s.append('possible_children:' + str(len(actions)))
        s.append('weightid         :' + str(id(self.weights)))
        s.append('is_terminal      :' + str(self.terminal))
        s.append('root_score       :' + str(self.start_score))
        s.append('score            :' + str(self.score))
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
            return [], []
        elif self.terminal:
            return [], []
        else:
            if self.binary_branching == 1 or self.binary_branching == 2:
                if len(s.possible_swaps()) > 0:
                    min_swappable = min(s.possible_swaps())
                    # next_sent_weight = 1.0 / (len(s.swappable) + 1.0)
                    # weights = [0.5 * (1.0 - next_sent_weight)] * 2
                    # weights += [next_sent_weight]
                    if self.binary_branching == 1:
                        return [(min_swappable, True), (min_swappable, False)], [0.5, 0.5]
                    else:
                        return [(min_swappable, True), (min_swappable, False), NEXT_SENT], [0.34, 0.33, 0.33]

                else:
                    return [NEXT_SENT], [1.0]
            else:
                weights = [1.0 / (len(s.swappable) + 1.0)] * (len(s.swappable) + 1)
                assert len(weights) == len(s.possible_swaps()) + 1
                return [(i, True) for i in s.possible_swaps()] + [NEXT_SENT], weights

    def copy(self,):
        new_sentences = []
        for ms in self.macaronic_sentences:
            new_sentences.append(ms.copy())
        c = MacaronicState(new_sentences,
                           self.displayed_sentence_idx,
                           self.model,
                           self.start_score,
                           self.binary_branching)
        c.weights = self.weights
        c.score = self.score
        return c

    def random_next_state(self, model_config_func, **kwargs):
        actions, action_weights = self.possible_actions()
        action = random.choices(population=actions, weights=action_weights, k=1)[0]
        c = self.copy()
        current_displayed_config = c.current_sentence()
        if action == NEXT_SENT:
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
            c.weights = c.weights  # should not give new_weights here!
            swap_token_count, swap_type_count = c.swap_counts()
            return c

    def next_state(self, action, model_config_func, **kwargs):
        assert isinstance(action, tuple)
        c = self.copy()
        current_displayed_config = c.current_sentence()
        if action == NEXT_SENT:
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
                 swapped, not_swapped, swappable,
                 l2_swapped_types, l2_swapped_tokens,
                 swap_limit):
        self.__tokens_l1 = tokens_l1
        self.__tokens_l2 = tokens_l2
        self.__int_l1 = int_l1
        self.__int_l2 = int_l2
        self.__swapped = swapped
        self.__not_swapped = not_swapped
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
    def not_swapped(self,):
        return self.__not_swapped

    @property
    def swappable(self,):
        return self.__swappable

    def possible_swaps(self,):
        if len(self.swapped) < self.len * self.swap_limit:
            return list(self.swappable)
        else:
            return []

    def update_config(self, action):
        assert isinstance(action, tuple)
        token_idx, is_swap = action
        self.__swappable.remove(token_idx)
        if is_swap:
            self.__swapped.add(token_idx)
            l2_int_item = self.int_l2[:, token_idx].item()
            assert isinstance(l2_int_item, int)
            self.__l2_swapped_types.add(l2_int_item)
            self.__l2_swapped_tokens.append(l2_int_item)
        else:
            self.__not_swapped.add(token_idx)
        return self

    def copy(self,):
        macaronic_copy = MacaronicSentence(self.tokens_l1,  # this does not change so need not deep copy
                                           self.tokens_l2,  # this also does not change
                                           self.int_l1, #.clone(),  # same
                                           self.int_l2, #.clone(),  # same
                                           copy.deepcopy(self.swapped),  # this  does change so we deepcopy
                                           copy.deepcopy(self.not_swapped),
                                           copy.deepcopy(self.swappable),
                                           copy.deepcopy(self.__l2_swapped_types),  # this  does change so we deepcopy
                                           copy.deepcopy(self.__l2_swapped_tokens),
                                           self.swap_limit)
        return macaronic_copy

    def color_it(self, w1, w2, w_idx):
        if w_idx in self.swapped:
            return TEXT_EFFECT.CYAN + w2 + TEXT_EFFECT.END
        elif w_idx in self.not_swapped:
            return TEXT_EFFECT.YELLOW + w1 + TEXT_EFFECT.END
        else:
            return w1

    def display_macaronic(self,):
        s = [self.color_it(tl1, tl2, idx)
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
    l1_data = macaronic_config.int_l1.clone()
    l2_data = macaronic_config.int_l2.clone()
    swap_ind = torch.LongTensor(sorted(list(macaronic_config.swapped)))
    flip_l2_set = torch.LongTensor(list(macaronic_config.l2_swapped_types))
    indicator = torch.LongTensor([1] * l1_data.size(1)).unsqueeze(0)
    indicator[:, swap_ind] = 2
    if model.is_cuda():
        l1_data = l1_data.cuda()
        l2_data = l2_data.cuda()
        indicator = indicator.cuda()
        mask = make_random_mask(l1_data, [l1_data.size(1)], 0.0, 0)
        flip_l2_set = flip_l2_set.cuda()
    var_batch = [l1_data.size(1)], l1_data, l2_data, indicator, mask
    model.update_g_weights(weights)
    model.init_param_freeze(CBiLSTM.L2_LEARNING)
    model.init_optimizer(type='SGD')

    #prev_loss = 100.
    num_steps = 0
    #improvement = 1.
    while num_steps < max_steps:  # and improvement > improvement_threshold:
        loss, acc, grad_norm = model.do_backprop(var_batch, l2_seen=flip_l2_set)  # (flip_l2, flip_l2_offset, flip_l2_set))
        step_score_vocabtype = model.score_embeddings()
        num_steps += 1
        #improvement = prev_loss - loss
        #prev_loss = loss
        #print(num_steps, 'step score', step_score_vocabtype, 'loss', loss)
    new_weights = model.l2_encoder.weight.clone().detach().cpu()
    return step_score_vocabtype, new_weights


def make_start_state(i2v, i2gv, init_weights, model, dl, **kwargs):
    score_0 = model.score_embeddings()
    macaronic_sentences = []
    for sent_idx, sent in enumerate(dl):
        lens, l1_data, l2_data, l1_text_data, l2_text_data = sent
        swapped = set([])
        not_swapped = set([])
        swappable = set(range(1, l1_data[0, :].size(0) - 1))
        l1_tokens = [SPECIAL_TOKENS.BOS] + l1_text_data[0].strip().split() + [SPECIAL_TOKENS.EOS] # [i2v[i.item()] for i in l1_data[0, :]]
        l2_tokens = [SPECIAL_TOKENS.BOS] + l2_text_data[0].strip().split() + [SPECIAL_TOKENS.EOS] # [i2gv[i.item()] for i in l2_data[0, :]]
        ms = MacaronicSentence(l1_tokens, l2_tokens,
                               l1_data, l2_data,
                               swapped, not_swapped, swappable,
                               set([]), [],
                               kwargs['swap_limit'])
        macaronic_sentences.append(ms)
        #if len(macaronic_sentences) > 0:
        #    break
    state = MacaronicState(macaronic_sentences, 0, model, score_0, kwargs['binary_branching'])
    state.weights = init_weights
    state.score = score_0
    return state


def beam_search(init_state, **kwargs):
    beam_size = kwargs['beam_size']
    best_state = init_state

    q = PriorityQ(beam_size)
    q.append(init_state)
    while q.length() > 0:
        print(q.length())
        curr_state = q.pop(kwargs['stochastic'] == 1)
        if 'verbose' in kwargs and kwargs['verbose']:
            #print('curr_state\n', str(curr_state))
            pass
        if curr_state.score >= best_state.score:
            best_state = curr_state
        actions, action_weights = curr_state.possible_actions()
        for action in actions:  # sorted(zip(action_weights, actions), reverse=True):
            new_state = curr_state.next_state(action, apply_swap, **kwargs)
            if new_state.displayed_sentence_idx - init_state.displayed_sentence_idx < kwargs['max_search_depth']:
                q.append(new_state)
    return best_state


def random_walk(init_state, **kwargs):
    state = init_state
    while not state.terminal and state.displayed_sentence_idx - init_state.displayed_sentence_idx < kwargs['max_search_depth']:
        state = state.random_next_state(apply_swap, **kwargs)
    return state


if __name__ == '__main__':
    print(sys.stdout.encoding)
    opt = argparse.ArgumentParser(description="write program description here")
    # insert options here
    opt.add_argument('--parallel_corpus', action='store', dest='parallel_corpus', required=True)
    opt.add_argument('--v2i', action='store', dest='v2i', required=True,
                     help='vocab to index pickle obj')
    opt.add_argument('--v2spell', action='store', dest='v2spell', required=False, default=None,
                     help='vocab to spelling pickle obj')
    opt.add_argument('--c2i', action='store', dest='c2i', required=False, default=None,
                     help='character (corpus and gloss)  to index pickle obj')
    opt.add_argument('--gv2i', action='store', dest='gv2i', required=True, default=None,
                     help='gloss vocab to index pickle obj')
    opt.add_argument('--gv2spell', action='store', dest='gv2spell', required=False, default=None,
                     help='gloss vocab to index pickle obj')
    opt.add_argument('--gpuid', action='store', type=int, dest='gpuid', default=-1)
    opt.add_argument('--cloze_model', action='store', dest='cloze_model', required=True)
    opt.add_argument('--key', action='store', dest='key', required=True)
    opt.add_argument('--stochastic', action='store', dest='stochastic', default=0, type=int, choices=[0, 1])
    opt.add_argument('--beam_size', action='store', dest='beam_size', default=10, type=int)
    opt.add_argument('--swap_limit', action='store', dest='swap_limit', default=0.3, type=float)
    opt.add_argument('--max_search_depth', action='store', dest='max_search_depth', default=10000, type=int)
    opt.add_argument('--random_walk', action='store', dest='random_walk', default=0, type=int, choices=[0, 1])
    opt.add_argument('--binary_branching', action='store', dest='binary_branching',
                     default=0, type=int, choices=[0, 1, 2])
    opt.add_argument('--max_steps', action='store', dest='max_steps', default=10, type=int)
    opt.add_argument('--improvement', action='store', dest='improvement_threshold', default=0.01, type=float)
    opt.add_argument('--penalty', action='store', dest='penalty', default=0.0, type=float)
    opt.add_argument('--verbose', action='store_true', dest='verbose', default=False)
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
    i2v = dict((v, k) for k, v in v2i.items())
    gv2i = pickle.load(open(options.gv2i, 'rb')) if options.gv2i is not None else None
    i2gv = dict((v, k) for k, v in gv2i.items())
    l1_key, l2_key = zip(*pickle.load(open(options.key, 'rb')))
    l2_key = torch.LongTensor(list(l2_key))
    l1_key = torch.LongTensor(list(l1_key))
    dataset = ParallelTextDataset(options.parallel_corpus, v2i, gv2i)
    v_max_vocab = len(v2i)
    g_max_vocab = len(gv2i)
    ff = [(i2v[i], i2gv[j]) for i, j in zip(l1_key.tolist(), l2_key.tolist())]
    pdb.set_trace()
    cloze_model = torch.load(options.cloze_model, map_location=lambda storage, loc: storage)

    if isinstance(cloze_model.encoder, VarEmbedding):
        gv2c, c2i = None, None
        wr = cloze_model.encoder.word_representer
        we_size = wr.we_size
        learned_weights = cloze_model.encoder.word_representer()
        g_wr = WordRepresenter(gv2c, c2i, len(c2i), wr.ce_size,
                               c2i[SPECIAL_TOKENS.PAD], wr.cr_size, we_size,
                               bidirectional=wr.bidirectional, dropout=wr.dropout,
                               num_required_vocab=max(v_max_vocab, g_max_vocab))
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
        cloze_model.encoder = encoder
        cloze_model.decoder = decoder
        cloze_model.l2_encoder = g_cl_encoder
        cloze_model.l2_decoder = g_cl_decoder
        cloze_model.init_param_freeze(CBiLSTM.L2_LEARNING)
    else:
        learned_weights = cloze_model.encoder.weight.data.clone()
        we_size = cloze_model.encoder.weight.size(1)
        encoder = make_wl_encoder(None, None, learned_weights)
        decoder = make_wl_decoder(encoder)
        g_wl_encoder = make_wl_encoder(g_max_vocab, we_size, None)
        g_wl_decoder = make_wl_decoder(g_wl_encoder)
        cloze_model.encoder = encoder
        cloze_model.decoder = decoder
        cloze_model.l2_encoder = g_wl_encoder
        cloze_model.l2_decoder = g_wl_decoder
        cloze_model.init_param_freeze(CBiLSTM.L2_LEARNING)
    if options.gpuid > -1:
        cloze_model.init_cuda()
    cloze_model.set_key(l1_key, l2_key)
    cloze_model.init_key()
    cloze_model.l2_dict = gv2i
    if isinstance(cloze_model, CBiLSTM):
        cloze_model.train()
    print(cloze_model)
    macaronic_sents = []
    if cloze_model.is_cuda:
        weights = cloze_model.l2_encoder.weight.clone().detach().cpu()
    else:
        weights = cloze_model.l2_encoder.weight.clone().detach()
    init_weights = weights.clone()
    kwargs = vars(options)
    start_state = make_start_state(i2v, i2gv, init_weights, cloze_model, dataset, **kwargs)
    now = time.time()
    if options.random_walk:
        scores = []
        random_state = random_walk(start_state, **kwargs)
        print('random walk completed', time.time() - now)
        print(str(random_state))
    else:
        best_state = beam_search(start_state, **kwargs)
        print('beam search completed', time.time() - now)
        print(str(best_state))
