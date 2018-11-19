#!/usr/bin/env python
import argparse
import pickle
import sys
import torch
import random
import time
import pdb

from model import CBiLSTM
from model import VarEmbedding
from model import WordRepresenter
from search import apply_swap
from search import beam_search
from search import random_walk
from search import make_start_state
from search import MacaronicSentence
from search import MacaronicState
from torch.utils.data import DataLoader
from train import make_cl_decoder
from train import make_cl_encoder
from train import make_wl_decoder
from train import make_wl_encoder
from utils import ParallelTextDataset
from utils import parallel_collate
from utils import TEXT_EFFECT

import numpy as np

global PAD, EOS, BOS, UNK, EPS
PAD = '<PAD>'
UNK = '<UNK>'
BOS = '<BOS>'
EOS = '<EOS>'
EPS = 1e-4

class SearchTree(object):
    def __init__(self, game, rollout_func, rollout_params, state_update_func):
        self.game = game
        self.rollout_func = rollout_func
        self.rollout_params = rollout_params
        self.state_update_func = state_update_func
        self.iters = 300
        self.nodes_seen = {}
        self.gamma = 1.0
        self.max_backup = 0
        self.C = 1.0
        self.D = 10.
        self.consecutive_action_threshold = 1

    def recursive_search(self, root_node):
        node = root_node
        while not node.state.terminal:
            self.nodes_seen = {}
            node = self.search(node)
            #node.reset()
        return node

    def search(self, root_node):
        self.nodes_seen[str(root_node.state)] = root_node
        best_action_count = 0
        prev_best_action = None
        for _ in range(self.iters):
            print('************************ search iter' + str(_) + '*************************')
            print('======root node=====\n', root_node)
            if root_node.completed_expansion:
                self.display_child_values(root_node, self.C)
                best_action, best_node = self.select_child(root_node, self.C)
                best_action_count = best_action_count + 1 if best_action == prev_best_action else 0
                prev_best_action = best_action
                if best_action_count > self.consecutive_action_threshold:
                    return best_node

            node, search_sequence = self.selection(root_node)
            print('===selection node===\n', node)
            if not node.state.terminal:
                node, search_sequence = self.expansion(node, search_sequence)
                print('===expansion node===\n', node)
                rollout_state = self.rollout(node)
                print('===rollout state===\n', rollout_state)
                reward = rollout_state.score - root_node.state.score
            else:
                reward = node.state.score - root_node.state.score
            self.backup(reward, 0, search_sequence)
            print('********************** end search iter' + str(_) + '*******************')
        best_action, best_node = self.select_child(root_node, 0.0)
        return best_node

    def selection(self, node):
        search_sequence = [node]
        while node.completed_expansion:
            assert not node.state.terminal
            action, node = self.select_child(node, self.C)
            search_sequence.append(node)
        return node, search_sequence

    def expansion(self, node, search_sequence):
        assert not node.completed_expansion
        if not node.state.terminal:
            node = self.expand_child(node)
            search_sequence.append(node)
        return node, search_sequence

    def rollout(self, node):
        assert self.rollout_func is not None
        now = time.time()
        rollout_start_state = node.state.copy()
        rollout_start_state.binary_branching = self.rollout_params['binary_branching']
        state = self.rollout_func(rollout_start_state, **self.rollout_params)
        print('rollout time', time.time() - now)
        return state

    def backup(self, reward, winner, search_sequence):
        # print('------------------backup-----------------')
        for node in search_sequence:
            node.visits += 1.0
            node.prev_value[winner] = node.value[winner]  # value at previous time-step
            if self.max_backup == 1:
                node.value[winner] = reward if reward > node.value[winner] else node.value[winner]
            else:
                node.value[winner] += ((reward - node.value[winner])/(node.visits))
            node.prod_std_deviation[winner] += (node.value[winner] - node.prev_value[winner]) * (node.value[winner] - node.value[winner])
        # print('-----------------------------------------')

        return True

    def display_child_values(self, node, exp_param):
        combined, values, ucb, variance_terms, actions, visits = self.child_scores(node, exp_param)
        f = [(a, c, v, u, vrt, vs) for a, c, v, u, vrt, vs in zip(actions, combined, values, ucb, variance_terms, visits)]
        s = '\n'.join(['combined %0.2f' % c +
                       ' value: %0.2f' % v +
                       ' ucb: %0.2f' % u +
                       ' var_term:' + str(vrt) +
                       ' action:' + str(a) +
                       ' visits:' + str(vs) for
                       a, c, v, u, vrt, vs in sorted(f)])
        print(s)
        return np.mean(variance_terms) < 1.0

    def child_scores(self, node, exp_param):
        n_visits = float(node.visits)
        cn_player = 0 #3 - node.state.player  # make selection from current node to child node
        values = np.zeros(len(node.expanded_actions))
        variance_terms = np.zeros(len(node.expanded_actions))
        ucb = np.zeros(len(node.expanded_actions))
        actions = [None] * len(node.expanded_actions)
        visits = []
        for idx, (a, cn) in enumerate(node.expanded_actions.items()):
            cn_visits = cn.visits
            cn_value = cn.value[cn_player] #(float(cn.rewards[cn_player]) / float(cn_visits))
            cn_ucb = np.sqrt(np.log2(n_visits) / cn_visits)
            cn_variance = (cn.prod_std_deviation[cn_player]/cn_visits) ** 2
            assert not np.isnan(cn_value)
            assert not np.isnan(cn_ucb)
            values[idx] = cn_value
            variance_terms[idx] = np.sqrt(cn_variance + (self.D / cn_visits))
            ucb[idx] = cn_ucb
            actions[idx] = a
            visits.append(cn.visits)
        combined = values + (exp_param * ucb) + variance_terms
        return combined, values, ucb, variance_terms, actions, visits

    def select_child(self, node, exp_param):
        combined, values, ucb, var_terms, actions, visits = self.child_scores(node, exp_param)
        flat_nonzero = np.flatnonzero(combined == combined.max())
        max_idx = np.random.choice(flat_nonzero)
        best_action = actions[max_idx]
        return best_action, node.expanded_actions[best_action]

    def expand_child(self, node):
        assert len(node.unexpanded_actions) > 0
        pa = list(node.unexpanded_actions.keys())
        action = random.choice(pa)
        _ = node.unexpanded_actions.pop(action)
        new_state = self.game.next_state(node.state, action)
        __actions, __action_weights = self.game.possible_actions(new_state)
        new_state_unexpanded_actions = {a: None for a in __actions}
        new_state_expanded_actions = {}
        if str(new_state) in self.nodes_seen:
            expanded_node = self.nodes_seen[str(new_state)]
        else:
            expanded_node = SearchNode(new_state, new_state_unexpanded_actions, new_state_expanded_actions)
            self.nodes_seen[str(new_state)] = expanded_node

        node.update_expansion(action, expanded_node)
        return expanded_node


class SearchNode(object):
    def __init__(self, state, unexpanded_actions, expanded_actions):
        self.state = state
        self.unexpanded_actions = unexpanded_actions
        self.expanded_actions = expanded_actions
        self.visits = 0
        self.completed_expansion = False
        self.value = [EPS]
        self.prev_value = [EPS]
        self.prod_std_deviation = [1.0]

    def reset(self, ):
        possible_actions, possible_action_weights = self.state.possible_actions()
        self.unexpanded_actions = {a: None for a in possible_actions}
        self.expanded_actions = {}
        self.visits = 0
        self.completed_expansion = False
        self.value = [EPS]
        self.prev_value = [EPS]
        self.prod_std_deviation = [1.0]


    def update_expansion(self, action, child_node):
        assert action not in self.expanded_actions
        self.expanded_actions[action] = child_node
        self.completed_expansion = len(self.unexpanded_actions) == 0
        return True

    def __str__(self,):
        s = str(self.state) + '\n'
        s += 'visits:' + str(self.visits) + '\n'
        s += 'value:' + ' '.join(['%.2f' % i for i in self.value]) + '\n'
        s += 'u:' + str(len(self.unexpanded_actions)) + ' e:' + str(len(self.expanded_actions)) + '\n'
        s += 'exp complete:' + str(self.completed_expansion) + '\n'
        return s


class Game(object):
    def __init__(self, dataloader, model, model_config_func, opt):
        self.dataloader = dataloader
        self.model = model
        self.model_config_func = model_config_func
        self.opt = opt

    def possible_actions(self, state):
        if state is not None:
            actions, action_weights = state.possible_actions()
            return actions, action_weights
        else:
            return [],[]

    def next_state(self, state, action):
        return state.next_state(action, self.model_config_func, **self.opt)


if __name__ == '__main__':
    print(sys.stdout.encoding)
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
    opt.add_argument('--stochastic', action='store', dest='stochastic', default=0, type=int, choices=[0, 1])
    opt.add_argument('--swap_limit', action='store', dest='swap_limit', default=0.3, type=float)
    opt.add_argument('--key', action='store', dest='key', required=True)
    opt.add_argument('--penalty', action='store', dest='penalty', default=0.2, type=float)
    opt.add_argument('--binary_branching', action='store', dest='binary_branching', default=0, type=int, choices=[0, 1])
    opt.add_argument('--beam_size', action='store', dest='beam_size', default=10, type=int)
    opt.add_argument('--max_steps', action='store', dest='max_steps', default=100, type=int)
    opt.add_argument('--improvement', action='store', dest='improvement_threshold', default=0.01, type=float)
    opt.add_argument('--rollout_function', action='store', dest='rollout_function', default='random_walk', type=str,
                     choices=['beam_search', 'random_walk'])
    opt.add_argument('--rollout_binary_branching', action='store', dest='rollout_binary_branching', default=1, type=int,
                     choices=[0, 1])
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
        g_wl_encoder = make_wl_encoder(max_vocab, we_size)  # randomly initialized
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
    if cbilstm.is_cuda:
        weights = cbilstm.g_encoder.weight.clone().detach().cpu()
    else:
        weights = cbilstm.g_encoder.weight.clone().detach()
    kwargs = vars(options)
    game = Game(dataloader, cbilstm, apply_swap, opt=kwargs)
    start_state = make_start_state(i2v, i2gv, weights, cbilstm, dataloader, **kwargs)
    print(str(start_state))
    rollout_params = {'stochastic': 0,
                      'verbose': False,  # options.verbose,
                      'improvement_threshold': options.improvement_threshold,
                      'penalty': 0.2, # options.penalty,
                      'beam_size': 1,  # options.beam_size,
                      'max_steps': 1, # options.max_steps,
                      'swap_limit': 1.0, #options.swap_limit,
                      'binary_branching': options.rollout_binary_branching,  # options.binary_branching,
                      'max_search_depth': 99}
    if options.rollout_function == 'random_walk':
        search_tree = SearchTree(game, random_walk, rollout_params, apply_swap)
    else:
        search_tree = SearchTree(game, beam_search, rollout_params, apply_swap)
    possible_actions, possible_action_weights = start_state.possible_actions()
    unexpanded_actions = {a: None for a in possible_actions}
    expanded_actions = {}
    root_node = SearchNode(start_state, unexpanded_actions, expanded_actions)
    terminal_node = search_tree.recursive_search(root_node)
    print('completed mcts')
    print(str(terminal_node))
