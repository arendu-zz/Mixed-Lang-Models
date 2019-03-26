#!/usr/bin/env python
import argparse
import numpy as np
import pickle
import sys
import torch
import random

from src.models.mse_model import MSE_CLOZE
from src.models.mse_model import L2_MSE_CLOZE
from src.models.model_untils import make_wl_encoder
from src.states.states import MacaronicState
from src.states.states import PriorityQ
from src.states.states import MacaronicSentence
from src.states.mcts_states import Game
from src.states.mcts_states import SearchNode
from src.states.mcts_states import SearchTree
from search_mse import apply_swap
from search_mse import beam_search
from search_mse import make_start_state
from train import make_random_mask


from src.utils.utils import ParallelTextDataset
from src.utils.utils import SPECIAL_TOKENS


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
    opt.add_argument('--use_orthographic_model', action='store', type=int, dest='use_orthographic_model',
                     choices=[0, 1, 2], default=0)
    opt.add_argument('--l2_init_weights', action='store', dest='l2_init_weights', required=False, default="")
    opt.add_argument('--key', action='store', dest='key', required=True)
    opt.add_argument('--per_line_key', action='store', dest='per_line_key', required=True)
    opt.add_argument('--stochastic', action='store', dest='stochastic', default=0, type=int, choices=[0, 1])
    opt.add_argument('--beam_size', action='store', dest='beam_size', default=10, type=int)
    opt.add_argument('--swap_limit', action='store', dest='swap_limit', default=0.3, type=float)
    opt.add_argument('--max_search_depth', action='store', dest='max_search_depth', default=3, type=int)
    opt.add_argument('--max_sentences', action='store', dest='max_sentences', default=10000, type=int)
    opt.add_argument('--binary_branching', action='store', dest='binary_branching',
                     default=0, type=int, choices=[0, 1, 2])
    opt.add_argument('--iters', action='store', dest='iters', type=int, required=True)
    opt.add_argument('--penalty', action='store', dest='penalty', default=0.0, type=float)
    opt.add_argument('--rank_threshold', action='store', dest='rank_threshold', default=10, type=int, required=True)
    opt.add_argument('--verbose', action='store_true', dest='verbose', default=False)
    opt.add_argument('--reward', action='store', dest='reward_type', type=str,
                     choices=['type_mrr_assist_check', 'token_mrr', 'token_ranking', 'mrr', 'ranking', 'cs'])
    opt.add_argument('--use_per_line_key', action='store', dest='use_per_line_key', type=int, choices=[1, 0])
    opt.add_argument('--accumulate_seen_l2', action='store', dest='accumulate_seen_l2', type=int,
                     choices=[0, 1], required=True)
    opt.add_argument('--debug_print', action='store_true', dest='debug_print', default=False)
    opt.add_argument('--search_output_prefix', action='store', dest='search_output_prefix',
                     default=None, required=False)
    opt.add_argument('--seed', action='store', dest='seed', default=1234, type=int)
    opt.add_argument('--backup_type', action='store', dest='backup_type', default='ave',
                     type=str, choices=['ave', 'max'])
    opt.add_argument('--rollout_function', action='store', dest='rollout_function', default='beam_search', type=str,
                     choices=['beam_search', 'random_walk', 'beam_search_per_sentence'])
    opt.add_argument('--rollout_binary_branching', action='store', dest='rollout_binary_branching', default=1, type=int,
                     choices=[0, 1])
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

    if options.search_output_prefix is not None:
        search_output = open(options.search_output_prefix + '.macaronic', 'w', encoding='utf-8')
        search_output_guesses = open(options.search_output_prefix + '.guesses', 'w', encoding='utf-8')
        search_output_json = open(options.search_output_prefix + '.json', 'w', encoding='utf-8')
    else:
        search_output = None
        search_output_guesses = None
        search_output_json = None

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
    l1_cloze_model = torch.load(options.cloze_model, map_location=lambda storage, loc: storage)
    we_size = l1_cloze_model.encoder.weight.size(1)
    if l1_cloze_model.ortho_mode == 0:
        print('using random l2_init_weights')
        l2_encoder = make_wl_encoder(g_max_vocab, we_size, None)
        l2_encoder.weight.data[:] = 0.0
    else:
        print('using FastText l2_init_weights')
        l2_init_weights = torch.load(options.l2_init_weights)
        l2_encoder = make_wl_encoder(None, None, l2_init_weights)
    cloze_model = L2_MSE_CLOZE(encoder=l1_cloze_model.encoder,
                               context_encoder=l1_cloze_model.context_encoder,
                               highway_ff=l1_cloze_model.highway_ff,
                               l1_dict=l1_cloze_model.l1_dict,
                               l2_encoder=l2_encoder,
                               l2_dict=gv2i,
                               l1_key=l1_key,
                               l2_key=l2_key,
                               iters=options.iters,
                               ##loss_type=options.training_loss_type,
                               ortho_mode=l1_cloze_model.ortho_mode)
    if options.gpuid > -1:
        cloze_model.init_cuda()
    cloze_model.init_key()
    cloze_model.eval()
    print(cloze_model)

    init_weights = cloze_model.get_weight().clone()

    kwargs = vars(options)
    kwargs['verbose'] = False
    mcts_kwargs = {'backup_type': options.backup_type, 'const_C': 0.01, 'const_D': 10.0, 'gamma': 1.0, 'mcts_iters': 10}
    game = Game(cloze_model, apply_swap, opt=kwargs)
    search_tree = SearchTree(game, mcts_kwargs, beam_search, kwargs, apply_swap)
    start_state = make_start_state(v2i, gv2i, i2v, i2gv, init_weights, cloze_model, dataset, **kwargs)
    possible_actions, possible_action_weights = start_state.possible_actions()
    unexpanded_actions = {a: None for a in possible_actions}
    expanded_actions = {}
    root_node = SearchNode(start_state, unexpanded_actions, expanded_actions)
    terminal_node = search_tree.recursive_search(root_node)
    print('completed mcts')
    print(str(terminal_node))
