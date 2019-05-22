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
from src.models.ce_model import spell2mat
from src.models.ce_model import TiedEncoderDecoder, CharTiedEncoderDecoder, CGramTiedEncoderDecoder
from src.models.ce_model import make_l2_tied_encoder_decoder
from src.models.model_untils import make_wl_encoder
import time

from src.utils.utils import ParallelTextDataset
from src.utils.utils import SPECIAL_TOKENS

from search_mse import beam_search_per_sentence, make_start_state, nearest_neighbors


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
    opt.add_argument('--l1_v2cgramspell', action='store', dest='l1_v2cgramspell', required=True,
                     help='vocab to spelling pickle obj')
    opt.add_argument('--l1_cgram2i', action='store', dest='l1_cgram2i', required=True,
                     help='character to index pickle obj comma sepeated list of files')
    opt.add_argument('--gv2i', action='store', dest='gv2i', required=True, default=None,
                     help='gloss vocab to index pickle obj')
    opt.add_argument('--l2_v2cgramspell', action='store', dest='l2_v2cgramspell', required=True,
                     help='vocab to spelling pickle obj')
    opt.add_argument('--l2_v2cgramspell_by_l1', action='store', dest='l2_v2cgramspell_by_l1', required=True,
                     help='vocab to spelling pickle obj')
    opt.add_argument('--l2_cgram2i', action='store', dest='l2_cgram2i', required=True,
                     help='character to index pickle obj comma sepeated list of files')
    opt.add_argument('--char_aware', action='store', required=True, choices=[0, 1, 2],
                     dest='char_aware', type=int, default=0)
    opt.add_argument('--gpuid', action='store', type=int, dest='gpuid', default=-1)
    opt.add_argument('--cloze_model', action='store', dest='cloze_model', required=True)
    opt.add_argument('--key', action='store', dest='key', required=True)
    opt.add_argument('--key_wt', action='store', dest='key_wt', required=True)
    opt.add_argument('--use_key_wt', action='store', dest='use_key_wt', required=True, type=int, choices=[0, 1])
    opt.add_argument('--learn_step_reg', action='store', dest='learn_step_reg', required=True, type=float)
    opt.add_argument('--zero_reg', action='store', dest='zero_reg', required=True, type=float)
    opt.add_argument('--reg_type', action='store', dest='reg_type', required=True, type=str, choices=['mse', 'huber', 'l1'])
    opt.add_argument('--per_line_key', action='store', dest='per_line_key', required=True)
    opt.add_argument('--stochastic', action='store', dest='stochastic', default=0, type=int, choices=[0, 1])
    opt.add_argument('--beam_size', action='store', dest='beam_size', default=10, type=int)
    opt.add_argument('--swap_limit', action='store', dest='swap_limit', default=0.3, type=float)
    opt.add_argument('--max_search_depth', action='store', dest='max_search_depth', default=10000, type=int)
    opt.add_argument('--max_sentences', action='store', dest='max_sentences', default=60, type=int, required=False)
    opt.add_argument('--binary_branching', action='store', dest='binary_branching',
                     default=0, type=int, choices=[0, 1, 2])
    opt.add_argument('--iters', action='store', dest='iters', type=int, required=True)
    opt.add_argument('--penalty', action='store', dest='penalty', default=0.0, type=float)
    opt.add_argument('--init_l2_with_l1_scale', action='store', dest='init_l2_with_l1_scale', default=1.0, type=float)
    opt.add_argument('--grad_norm', action='store', dest='grad_norm', default=5.0, type=float)
    opt.add_argument('--init_l2_with_l1', action='store', dest='init_l2_with_l1', type=str,
                     choices=['init_main', 'init_subwords', 'no_init'])
    opt.add_argument('--rank_threshold', action='store', dest='rank_threshold', default=10, type=int, required=True)
    opt.add_argument('--verbose', action='store_true', dest='verbose', default=False)
    opt.add_argument('--reward', action='store', dest='reward_type', type=str,
                     choices=['type_mrr_assist_check', 'token_mrr', 'token_ranking', 'mrr', 'ranking', 'cs'])
    opt.add_argument('--use_per_line_key', action='store', dest='use_per_line_key', type=int, choices=[1, 0])
    opt.add_argument('--accumulate_seen_l2', action='store', dest='accumulate_seen_l2', type=int,
                     choices=[0, 1], required=True)
    opt.add_argument('--debug_print', action='store_true', dest='debug_print', default=False)
    opt.add_argument('--init_range', action='store', dest='init_range', type=float, required=True)
    opt.add_argument('--l2_pos_weighted', action='store', dest='l2_pos_weighted',
                     type=int, choices=[0, 1], required=True)
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
    #c2i = pickle.load(open(options.c2i, 'rb'))
    #i2c = {v: k for k, v in c2i.items()}
    gv2i = pickle.load(open(options.gv2i, 'rb')) if options.gv2i is not None else None
    i2gv = dict((v, k) for k, v in gv2i.items())
    #gc2i = pickle.load(open(options.gc2i, 'rb'))
    #i2gc = {v: k for k, v in gc2i.items()}
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
    #load cgrams
    l1_cgram_files = options.l1_cgram2i.split(',')
    l1_cgram2i_list = []
    l1_i2cgram_list = []
    for cgram_file in l1_cgram_files:
        cngram2i = pickle.load(open(cgram_file, 'rb'))
        i2cngram = {v: k for k, v in cngram2i.items()}
        l1_cgram2i_list.append(cngram2i)
        l1_i2cgram_list.append(i2cngram)
    l1_v2cgramspell_files = options.l1_v2cgramspell.split(',')
    l1_cgram_spelling_mat_list = []
    for v2cgramspell_file in l1_v2cgramspell_files:
        f = pickle.load(open(v2cgramspell_file, 'rb'))
        m = spell2mat(f)
        assert len(v2i) == m.shape[0]
        l1_cgram_spelling_mat_list.append(m)

    l2_cgram_files = options.l2_cgram2i.split(',')
    l2_cgram2i_list = []
    l2_i2cgram_list = []
    for cgram_file in l2_cgram_files:
        cngram2i = pickle.load(open(cgram_file, 'rb'))
        i2cngram = {v: k for k, v in cngram2i.items()}
        l2_cgram2i_list.append(cngram2i)
        l2_i2cgram_list.append(i2cngram)
    l2_v2cgramspell_files = options.l2_v2cgramspell.split(',')
    l2_cgram_spelling_mat_list = []
    for v2cgramspell_file in l2_v2cgramspell_files:
        f = pickle.load(open(v2cgramspell_file, 'rb'))
        m = spell2mat(f)
        assert len(gv2i) == m.shape[0]
        l2_cgram_spelling_mat_list.append(m)
    l2_v2cgramspell_by_l1_files = options.l2_v2cgramspell_by_l1.split(',')
    l2_cgram_by_l1_spelling_mat_list = []
    for v2cgramspell_file in l2_v2cgramspell_by_l1_files:
        f = pickle.load(open(v2cgramspell_file, 'rb'))
        m = spell2mat(f)
        assert len(gv2i) == m.shape[0]
        l2_cgram_by_l1_spelling_mat_list.append(m)

    l1_cloze_model = torch.load(options.cloze_model, map_location=lambda storage, loc: storage)
    l1_tied_encoder_decoder = l1_cloze_model.tied_encoder_decoder
    l1_tied_encoder_decoder.param_type = 'l1'
    l1_tied_encoder_decoder.mode = 'l2'
    min_cgram = l1_tied_encoder_decoder.min_cgram #int(options.pool_type.split(':')[0])
    max_cgram = l1_tied_encoder_decoder.max_cgram #int(options.pool_type.split(':')[1])
    l1_tied_encoder_decoder.min_cgram = min_cgram
    l1_tied_encoder_decoder.max_cgram = max_cgram
    l2_tied_encoder_decoder = make_l2_tied_encoder_decoder(l1_cloze_model.tied_encoder_decoder,
                                                           v2i, l1_cgram2i_list, l1_cgram_spelling_mat_list,
                                                           gv2i, l2_cgram2i_list,
                                                           l2_cgram_spelling_mat_list,
                                                           l2_cgram_by_l1_spelling_mat_list,
                                                           init_l2_prior=False, #options.init_l2_prior == 1,
                                                           init_l2_with_l1=options.init_l2_with_l1, #options.init_l2_with_l1 == 1,
                                                           scale=options.init_l2_with_l1_scale,
                                                           init_range=options.init_range,
                                                           l2_pos_weighted=options.l2_pos_weighted == 1)
    if isinstance(l1_tied_encoder_decoder, CharTiedEncoderDecoder):
        l1_tied_encoder_decoder.init_cache()
    if isinstance(l1_tied_encoder_decoder, CGramTiedEncoderDecoder):
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
                              zero_regularization=options.zero_reg,
                              regularization_type=options.reg_type,
                              learning_steps=options.iters,
                              max_grad_norm=options.grad_norm)
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
    l1_weights = cloze_model.get_l1_word_vecs() #encoder.weight.data.detach().clone()
    l2_weights = start_state.model.get_l2_word_vecs() #.weights.detach().clone()
    all_l2_types = set(list(range(l2_weights.shape[0])))
    nn, nn_dict = nearest_neighbors(l1_weights, l2_weights,
                                    all_l2_types,
                                    cloze_model.l1_dict_idx, cloze_model.l2_dict_idx, kwargs['rank_threshold'])
    print(nn)
    print('----------------------')
    print(l2_weights.mean().item(), l2_weights.std().item(), l2_weights.min().item(), l2_weights.max().item(), 'l2 weight stats')
    #l2_prior_weights = start_state.model.l2_tied_encoder_decoder.l2_word_vecs_prior
    #nn, nn_dict = nearest_neighbors(l1_weights, l2_prior_weights,
    #                                all_l2_types,
    #                                cloze_model.l1_dict_idx, cloze_model.l2_dict_idx, kwargs['rank_threshold'])
    #print(nn)
    now = time.time()
    best_state = beam_search_per_sentence(search_output,
                                          search_output_guesses,
                                          search_output_json,
                                          search_output_latex,
                                          cloze_model, start_state, **kwargs)
    #best_state = beam_search(start_state, **kwargs)
    print('beam search completed', time.time() - now)
    print('----------------------')
    _, all_swapped_types = best_state.swap_counts()
    l1_weights = cloze_model.get_l1_word_vecs() #encoder.weight.data.detach().clone()
    l2_weights = best_state.model.get_l2_word_vecs() #.weights.detach().clone()
    nn, nn_dict = nearest_neighbors(l1_weights, l2_weights,
                                    all_swapped_types,
                                    cloze_model.l1_dict_idx, cloze_model.l2_dict_idx, kwargs['rank_threshold'])
    print(nn)
    print('final l2_weights', l2_weights.min().item(), l2_weights.max().item(), l1_weights.min().item(), l1_weights.max().item())
    print('----------------------')
    print(str(best_state))
    print('num_exposed', len(all_swapped_types))
    search_output.close()
    search_output_guesses.close()
    search_output_json.close()
    search_output_latex.close()
