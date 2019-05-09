#!/usr/bin/env python
import argparse
import numpy as np
from operator import attrgetter
import pickle
import sys
import torch
import random


from src.models.ce_model import CE_CLOZE
from src.models.ce_model import L2_CE_CLOZE
from src.models.ce_model import TiedEncoderDecoder, CharTiedEncoderDecoder
from src.models.ce_model import make_l2_tied_encoder_decoder
from src.models.model_untils import make_wl_encoder
import time

from src.utils.utils import ParallelTextDataset
from src.utils.utils import SPECIAL_TOKENS

from search_mse import beam_search_per_sentence, make_start_state, nearest_neighbors

import pdb


def remove_special_tokens(t2i):
    _t2i = sorted([(i, t) for t, i in t2i.items()])
    new_t2i = {}
    st = [SPECIAL_TOKENS.UNK, SPECIAL_TOKENS.BOS, SPECIAL_TOKENS.EOS,
          SPECIAL_TOKENS.PAD, SPECIAL_TOKENS.BOW, SPECIAL_TOKENS.EOW]
    for i, t in _t2i:
        if t not in st:
            new_t2i[t] = len(new_t2i)
    return new_t2i


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
    opt.add_argument('--gc2i', action='store', dest='gc2i', required=False, default=None,
                     help='character (corpus and gloss)  to index pickle obj')
    opt.add_argument('--char_aware', action='store', required=True, choices=[0, 1],
                     dest='char_aware', type=int, default=0)
    opt.add_argument('--gpuid', action='store', type=int, dest='gpuid', default=-1)
    opt.add_argument('--cloze_model', action='store', dest='cloze_model', required=True)
    opt.add_argument('--key', action='store', dest='key', required=True)
    opt.add_argument('--key_wt', action='store', dest='key_wt', required=True)
    opt.add_argument('--use_key_wt', action='store', dest='use_key_wt', required=True, type=int, choices=[0, 1])
    opt.add_argument('--learn_step_reg', action='store', dest='learn_step_reg', required=True, type=float)
    opt.add_argument('--per_line_key', action='store', dest='per_line_key', required=True)
    opt.add_argument('--stochastic', action='store', dest='stochastic', default=0, type=int, choices=[0, 1])
    opt.add_argument('--beam_size', action='store', dest='beam_size', default=10, type=int)
    opt.add_argument('--swap_limit', action='store', dest='swap_limit', default=0.3, type=float)
    opt.add_argument('--max_search_depth', action='store', dest='max_search_depth', default=10000, type=int)
    opt.add_argument('--max_sentences', action='store', dest='max_sentences', default=10000, type=int, required=False)
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
        search_output_latex = open(options.search_output_prefix + '.tex', 'w', encoding='utf8')
    else:
        search_output = None
        search_output_guesses = None
        search_output_json = None
        search_output_latex = None

    v2i = pickle.load(open(options.v2i, 'rb'))
    i2v = dict((v, k) for k, v in v2i.items())
    c2i = pickle.load(open(options.c2i, 'rb'))
    i2c = {v: k for k, v in c2i.items()}
    gv2i = pickle.load(open(options.gv2i, 'rb')) if options.gv2i is not None else None
    i2gv = dict((v, k) for k, v in gv2i.items())
    gc2i = pickle.load(open(options.gc2i, 'rb'))
    i2gc = {v: k for k, v in gc2i.items()}

    l1_key, l2_key = zip(*pickle.load(open(options.key, 'rb')))
    l2_key_wt = pickle.load(open(options.key_wt, 'rb'))
    l2_key_wt = torch.FloatTensor(l2_key_wt)
    if options.use_key_wt == 0:
        l2_key_wt.fill_(1.0)
    l2_key = torch.LongTensor(list(l2_key))
    l1_key = torch.LongTensor(list(l1_key))
    dataset = ParallelTextDataset(options.parallel_corpus, v2i, gv2i)
    v_max_vocab = len(v2i)
    g_max_vocab = len(gv2i)
    l1_cloze_model = torch.load(options.cloze_model, map_location=lambda storage, loc: storage)
    l2_tied_encoder_decoder = make_l2_tied_encoder_decoder(l1_cloze_model.tied_encoder_decoder,
                                                           v2i, c2i, options.v2spell,
                                                           gv2i, gc2i, options.gv2spell)
    l1_tied_encoder_decoder = l1_cloze_model.tied_encoder_decoder
    l1_tied_encoder_decoder.param_type = 'l1'
    l1_tied_encoder_decoder.mode = 'l2'
    if isinstance(l1_tied_encoder_decoder, CharTiedEncoderDecoder):
        l1_tied_encoder_decoder.init_cache()
    l1_context_encoder = l1_cloze_model.context_encoder
    #we_size = l1_cloze_model.encoder.weight.size(1)
    #l2_encoder = make_wl_encoder(g_max_vocab, we_size, None)
    cloze_model = L2_CE_CLOZE(context_encoder=l1_context_encoder,
                              l1_tied_encoder_decoder=l1_tied_encoder_decoder,
                              l2_tied_encoder_decoder=l2_tied_encoder_decoder,
                              l1_dict=l1_cloze_model.l1_dict,
                              l2_dict=gv2i,
                              l1_key=l1_key,
                              l2_key=l2_key,
                              l2_key_wt=l2_key_wt,
                              learn_step_regularization=options.learn_step_reg,
                              learning_steps=options.iters)
    if options.gpuid > -1:
        cloze_model.init_cuda()
    cloze_model.init_key()
    cloze_model.train()
    print(cloze_model)
    print('total params', sum([p.numel() for p in cloze_model.parameters()]))
    print('trainable params', sum([p.numel() for p in cloze_model.parameters() if p.requires_grad]))
    macaronic_sents = []
    init_weights = cloze_model.get_l2_state_dict()
    kwargs = vars(options)
    start_state = make_start_state(v2i, gv2i, i2v, i2gv, init_weights, cloze_model, dataset, **kwargs)
    pdb.set_trace()
    now = time.time()
    best_state = beam_search_per_sentence(search_output,
                                          search_output_guesses,
                                          search_output_json,
                                          search_output_latex,
                                          cloze_model, start_state, **kwargs)
    #best_state = beam_search(start_state, **kwargs)
    print('beam search completed', time.time() - now)
    print(str(best_state))
    _, all_swapped_types = best_state.swap_counts()
    print('num_exposed', len(all_swapped_types))
    l1_weights = cloze_model.get_l1_word_vecs() #encoder.weight.data.detach().clone()
    l2_weights = best_state.model.get_l2_word_vecs() #.weights.detach().clone()
    nn, nn_dict = nearest_neighbors(l1_weights, l2_weights,
                                    all_swapped_types,
                                    cloze_model.l1_dict_idx, cloze_model.l2_dict_idx, kwargs['rank_threshold'])
    print(nn)
    search_output.close()
    search_output_guesses.close()
    search_output_json.close()
    search_output_latex.close()
