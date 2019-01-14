#!/usr/bin/env python
import argparse
import pickle
import sys
import torch

from src.models.model import CBiLSTM
from src.models.model import CBiLSTMFast
from src.models.map_model import CBiLSTMFastMap
from src.models.cloze_model import L1_Cloze_Model
from src.models.cloze_model import L2_Cloze_Model
from src.models.model import CTransformerEncoder
from src.models.model import VarEmbedding
from src.models.model import WordRepresenter
from src.states.states import MacaronicSentence
from src.models.model import make_cl_decoder
from src.models.model import make_cl_encoder
from src.models.model import make_wl_decoder
from src.models.model import make_wl_encoder
from search import apply_swap
from search import nearest_neighbors

from src.utils.utils import ParallelTextDataset
from src.utils.utils import SPECIAL_TOKENS
from src.utils.utils import TEXT_EFFECT


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
    opt.add_argument('--mask_unseen_l2', action='store', dest='mask_unseen_l2', default=1, type=int, choices=[0, 1])
    opt.add_argument('--joined_l2_l1', action='store', dest='joined_l2_l1', default=0, type=int, choices=[0, 1])
    opt.add_argument('--beam_size', action='store', dest='beam_size', default=10, type=int)
    opt.add_argument('--swap_limit', action='store', dest='swap_limit', default=0.3, type=float)
    opt.add_argument('--max_search_depth', action='store', dest='max_search_depth', default=10000, type=int)
    opt.add_argument('--random_walk', action='store', dest='random_walk', default=0, type=int, choices=[0, 1])
    opt.add_argument('--binary_branching', action='store', dest='binary_branching',
                     default=0, type=int, choices=[0, 1, 2])
    opt.add_argument('--max_steps', action='store', dest='max_steps', default=1, type=int)
    opt.add_argument('--l2_iters', action='store', dest='l2_iters', default=0, type=int)
    opt.add_argument('--improvement', action='store', dest='improvement_threshold', default=0.01, type=float)
    opt.add_argument('--penalty', action='store', dest='penalty', default=0.0, type=float)
    opt.add_argument('--verbose', action='store_true', dest='verbose', default=False)
    opt.add_argument('--reward', action='store', dest='reward_type',type=str, choices=['ranking', 'cs'])
    opt.add_argument('--debug_print', action='store_true', dest='debug_print', default=False)
    opt.add_argument('--seed', action='store', dest='seed', default=1234, type=int)
    options = opt.parse_args()
    print(options)
    torch.manual_seed(options.seed)
    if options.gpuid > -1:
        torch.cuda.set_device(options.gpuid)
        tmp = torch.ByteTensor([0])
        tmp.cuda()
        print("using GPU", options.gpuid)
    else:
        print("using CPU!")

    v2i = pickle.load(open(options.v2i, 'rb'))
    i2v = dict((v, k) for k, v in v2i.items())
    gv2i = pickle.load(open(options.gv2i, 'rb')) if options.gv2i is not None else None
    i2gv = dict((v, k) for k, v in gv2i.items())
    l1_key, l2_key = zip(*pickle.load(open(options.key, 'rb')))
    l2_key = torch.LongTensor(list(l2_key))
    l1_key = torch.LongTensor(list(l1_key))
    dataset = ParallelTextDataset(options.parallel_corpus, v2i, gv2i)
    v_max_vocab = len(v2i)
    g_max_vocab = len(gv2i) if gv2i is not None else 0
    l1_cloze_model = torch.load(options.cloze_model, map_location=lambda storage, loc: storage)
    we_size = l1_cloze_model.encoder.weight.size(1)
    l2_encoder = make_wl_encoder(g_max_vocab, we_size, None)
    l2_decoder = make_wl_decoder(l2_encoder)
    cloze_model = L2_Cloze_Model(l1_cloze_model.encoder,
                                 l1_cloze_model.decoder,
                                 l1_cloze_model.rnn,
                                 l1_cloze_model.linear,
                                 l1_cloze_model.l1_dict,
                                 l2_encoder,
                                 l2_decoder,
                                 gv2i,
                                 l1_key,
                                 l2_key,
                                 options.mask_unseen_l2)
    if options.joined_l2_l1:
        #TODO: joinin the new scheme
        cloze_model.join_l2_weights()

    if options.gpuid > -1:
        cloze_model.init_cuda()
    cloze_model.init_key()
    #en_en_sim = batch_cosine_sim(cloze_model.encoder.weight.data.clone(),
    #                             cloze_model.encoder.weight.data.clone())
    #_, en_en_neighbors = torch.topk(en_en_sim, 6, 1)
    #cloze_model.en_en_neighbors = en_en_neighbors
    cloze_model.train()

    kwargs = vars(options)
    print(cloze_model)
    cloze_model.l2_iters = options.l2_iters
    macaronic_sents = []
    sent_init_weights = cloze_model.get_weight()
    hist_flip_l2 = {}
    hist_limit = 1
    penalty = options.penalty  # * ( 1.0 / 8849.0)
    total_swap_types = set([])

    for batch_idx, batch in enumerate(dataset):
        lens, l1_data, l2_data, l1_text_data, l2_text_data = batch
        l1_tokens = [SPECIAL_TOKENS.BOS] + l1_text_data[0].strip().split() + [SPECIAL_TOKENS.EOS] # [i2v[i.item()] for i in l1_data[0, :]]
        l2_tokens = [SPECIAL_TOKENS.BOS] + l2_text_data[0].strip().split() + [SPECIAL_TOKENS.EOS] # [i2gv[i.item()] for i in l2_data[0, :]]
        swapped = set([])
        not_swapped = set([])
        #swappable = set(range(1, l1_data[0, :].size(0) - 1))
        swappable = set([idx for idx, i in enumerate(l2_data[0, :]) if i != gv2i[SPECIAL_TOKENS.UNK]][1:-1])
        macaronic_0 = MacaronicSentence(l1_tokens,
                                        l2_tokens,
                                        l1_data.clone(),
                                        l2_data.clone(),
                                        swapped, not_swapped, swappable,
                                        set([]), [],
                                        1.0)
        go_next = False
        while not go_next:
            l1_str = ' '.join([str(_idx) + ':' + i for _idx, i in enumerate(macaronic_0.tokens_l1) if _idx in macaronic_0.swappable])
            print('\n' + l1_str)
            swaps_selected = input('swqp (1,' + str(lens[0] - 2) + '): ').split(',')
            swaps_selected = set([int(i) for i in swaps_selected])
            swap_str = ' '.join([(i2v[l1_data[0, i].item()]
                                 if i not in swaps_selected else
                                 (TEXT_EFFECT.UNDERLINE + i2gv[l2_data[0, i].item()] + TEXT_EFFECT.END))
                                 for i in range(1, l1_data.size(1) - 1)])
            new_macaronic = macaronic_0.copy()
            for a in swaps_selected:
                new_macaronic.update_config((a, True))
            if options.verbose:
                print(new_macaronic)
            #swap_result = apply_swap(macaronic_0,
            #                         cloze_model,
            #                         sent_init_weights,
            #                         options.max_steps,
            #                         options.improvement_threshold,
            #                         options.reward_type,
            #                         macaronic_0.l2_swapped_types)
            #print('init score', swap_result['score'])
            swap_result = apply_swap(new_macaronic,
                                     cloze_model,
                                     sent_init_weights,
                                     new_macaronic.l2_swapped_types,
                                     **kwargs)
            print('swap score:', swap_result['score'])
            print('swap score + penalty:', swap_result['score'] - options.penalty * len(total_swap_types.union(new_macaronic.l2_swapped_types)))
            l1_weights = cloze_model.encoder.weight.data.detach().clone()
            l2_weights = swap_result['weights']
            print(nearest_neighbors(l1_weights, l2_weights, new_macaronic.l2_swapped_types,
                                    cloze_model.l1_dict_idx, cloze_model.l2_dict_idx))
            go_next = input('next line or retry (n/r):')
            go_next = go_next == 'n'
            if go_next:
                print('going to next...')
                total_swap_types = total_swap_types.union(new_macaronic.l2_swapped_types)
                if cloze_model.is_cuda:
                    sent_init_weights = cloze_model.l2_encoder.weight.clone().detach().cpu()
                else:
                    sent_init_weights = cloze_model.l2_encoder.weight.clone().detach()
            else:
                pass
