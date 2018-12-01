#!/usr/bin/env python
import argparse
import numpy as np
from operator import attrgetter
import pickle
import sys
import torch
import random
import pdb

from src.models.model import CBiLSTM
from src.models.model import CBiLSTMFast
from src.models.map_model import CBiLSTMFastMap
from src.models.model import CTransformerEncoder
from src.models.model import VarEmbedding
from src.models.model import WordRepresenter
from src.states.states import MacaronicState
from src.states.states import PriorityQ
from src.states.states import MacaronicSentence
from src.states.states import NEXT_SENT
from src.models.model import make_cl_decoder
from src.models.model import make_cl_encoder
from src.models.model import make_wl_decoder
from src.models.model import make_wl_encoder
from train import make_random_mask
import time

from src.utils.utils import ParallelTextDataset
from src.utils.utils import SPECIAL_TOKENS


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


def apply_swap_bp(macaronic_config, model, weights, max_steps=1, improvement_threshold=0.01, previous_seen_l2=None):
    assert isinstance(macaronic_config, MacaronicSentence)
    l1_data = macaronic_config.int_l1.clone()
    l2_data = macaronic_config.int_l2.clone()
    swap_ind = torch.LongTensor(sorted(list(macaronic_config.swapped)))
    if previous_seen_l2 is not None:
        seen_l2 = macaronic_config.l2_swapped_types.union(previous_seen_l2)
    else:
        seen_l2 = macaronic_config.l2_swapped_types
    flip_l2_set = torch.LongTensor(list(seen_l2))
    indicator = torch.LongTensor([1] * l1_data.size(1)).unsqueeze(0)
    indicator[:, swap_ind] = 2
    mask = make_random_mask(l1_data, [l1_data.size(1)], 0.0, 0)
    if model.is_cuda():
        l1_data = l1_data.cuda()
        l2_data = l2_data.cuda()
        indicator = indicator.cuda()
        mask = mask.cuda() #make_random_mask(l1_data, [l1_data.size(1)], 0.0, 0)
        flip_l2_set = flip_l2_set.cuda()
    var_batch = [l1_data.size(1)], l1_data, l2_data, indicator, mask
    #model.update_g_weights(weights)
    #model.init_param_freeze(CBiLSTM.L2_LEARNING)
    #model.init_optimizer(type='SGD')
    new_weights = model.do_bp_forward(var_batch, weights)
    step_score_vocabtype = model.score_embeddings(new_weights)
    return step_score_vocabtype, new_weights


def apply_swap(macaronic_config, model, weights, max_steps=1, improvement_threshold=0.01, previous_seen_l2=None):
    assert isinstance(macaronic_config, MacaronicSentence)
    l1_data = macaronic_config.int_l1.clone()
    l2_data = macaronic_config.int_l2.clone()
    swap_ind = torch.LongTensor(sorted(list(macaronic_config.swapped)))
    if previous_seen_l2 is not None:
        seen_l2 = macaronic_config.l2_swapped_types.union(previous_seen_l2)
    else:
        seen_l2 = macaronic_config.l2_swapped_types
    flip_l2_set = torch.LongTensor(list(seen_l2))
    indicator = torch.LongTensor([1] * l1_data.size(1)).unsqueeze(0)
    indicator[:, swap_ind] = 2
    mask = make_random_mask(l1_data, [l1_data.size(1)], 0.0, 0)
    if model.is_cuda():
        l1_data = l1_data.cuda()
        l2_data = l2_data.cuda()
        indicator = indicator.cuda()
        mask = mask.cuda() #make_random_mask(l1_data, [l1_data.size(1)], 0.0, 0)
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
        step_score_vocabtype = model.score_embeddings(model.l2_encoder.weight.data.clone())
        num_steps += 1
        #improvement = prev_loss - loss
        #prev_loss = loss
        #print(num_steps, 'step score', step_score_vocabtype, 'loss', loss)
    #new_weights = model.l2_encoder.weight.clone().detach().cpu()
    new_weights = model.get_weight()
    return step_score_vocabtype, new_weights


def make_start_state(i2v, i2gv, init_weights, model, dl, **kwargs):
    score_0 = model.score_embeddings(model.l2_encoder.weight.data.clone())
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
        if len(macaronic_sentences) > 1000:
            break
    state = MacaronicState(macaronic_sentences, 0, model, score_0, kwargs['binary_branching'])
    state.weights = init_weights
    state.score = score_0
    return state


def beam_search_per_sentence(init_state, **kwargs):
    beam_size = kwargs['beam_size']
    best_state = init_state
    q = PriorityQ(beam_size)
    q.append(init_state)
    while not best_state.terminal:
        while q.length() > 0:
            curr_state = q.pop(kwargs['stochastic'] == 1)
            if 'verbose' in kwargs and kwargs['verbose']:
                print('curr_state\n', str(curr_state))
                pass
            if curr_state.score >= best_state.score:
                best_state = curr_state
            actions, action_weights = curr_state.possible_actions()
            for action in actions:  # sorted(zip(action_weights, actions), reverse=True):
                if action == NEXT_SENT:
                    pass
                else:
                    new_state = curr_state.next_state(action, apply_swap, **kwargs)
                    if new_state.displayed_sentence_idx - init_state.displayed_sentence_idx < kwargs['max_search_depth']:
                        q.append(new_state)
        if best_state.terminal:
            pass
        else:
            assert q.length() == 0, "q should be empty at this point!"
            actions, action_weights = best_state.possible_actions()
            assert NEXT_SENT in actions
            init_next_sentence_state = best_state.next_state(NEXT_SENT, apply_swap, **kwargs)
            q.append(init_next_sentence_state)
    return best_state


def beam_search(init_state, **kwargs):
    beam_size = kwargs['beam_size']
    best_state = init_state

    q = PriorityQ(beam_size)
    q.append(init_state)
    while q.length() > 0:
        curr_state = q.pop(kwargs['stochastic'] == 1)
        if 'verbose' in kwargs and kwargs['verbose']:
            print('curr_state\n', str(curr_state))
            pass
        if curr_state.score >= best_state.score and curr_state.terminal:
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
    opt.add_argument('--joined_l2_l1', action='store', dest='joined_l2_l1', default=0, type=int, choices=[0, 1])
    opt.add_argument('--mask_unseen_l2', action='store', dest='mask_unseen_l2', default=1, type=int, choices=[0, 1])
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
    cloze_model = torch.load(options.cloze_model, map_location=lambda storage, loc: storage)

    if isinstance(cloze_model.encoder, VarEmbedding):
        gv2c, c2i = None, None
        wr = cloze_model.encoder.word_representer
        we_size = wr.we_size
        learned_l1_weights = cloze_model.encoder.word_representer()
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
        encoder = make_wl_encoder(None, None, learned_l1_weights.data.clone())
        decoder = make_wl_decoder(encoder)
        cloze_model.encoder = encoder
        cloze_model.decoder = decoder
        cloze_model.l2_encoder = g_cl_encoder
        cloze_model.l2_decoder = g_cl_decoder
        cloze_model.init_param_freeze(CBiLSTM.L2_LEARNING)
    elif isinstance(cloze_model, CBiLSTM) or isinstance(cloze_model, CBiLSTMFast):
        learned_l1_weights = cloze_model.encoder.weight.data.clone()
        we_size = cloze_model.encoder.weight.size(1)
        encoder = make_wl_encoder(None, None, learned_l1_weights)
        decoder = make_wl_decoder(encoder)
        g_wl_encoder = make_wl_encoder(g_max_vocab, we_size, None)
        g_wl_decoder = make_wl_decoder(g_wl_encoder)
        cloze_model.encoder = encoder
        cloze_model.decoder = decoder
        cloze_model.l2_encoder = g_wl_encoder
        cloze_model.l2_decoder = g_wl_decoder
        cloze_model.init_param_freeze(CBiLSTM.L2_LEARNING)
    elif isinstance(cloze_model, CBiLSTMFastMap):
        learned_l1_weights = cloze_model.encoder.weight.data.clone()
        we_size = cloze_model.encoder.weight.size(1)
        encoder = make_wl_encoder(None, None, learned_l1_weights)
        decoder = make_wl_decoder(encoder)
        g_wl_encoder = make_wl_encoder(g_max_vocab, we_size, None)
        g_wl_decoder = make_wl_decoder(g_wl_encoder)
        cloze_model.encoder = encoder
        cloze_model.decoder = decoder
        map_weights = torch.FloatTensor(g_max_vocab, v_max_vocab).uniform_(-0.01, 0.01)
        cloze_model.init_l2_weights(map_weights)
        cloze_model.init_param_freeze(CBiLSTM.L2_LEARNING)
    elif isinstance(cloze_model, CTransformerEncoder):
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
    else:
        raise NotImplementedError("unknown cloze_model" + str(type(cloze_model)))

    if options.joined_l2_l1:
        assert not isinstance(cloze_model, CBiLSTMFastMap)
        cloze_model.join_l2_weights()
    cloze_model.mask_unseen_l2 = options.mask_unseen_l2

    if options.gpuid > -1:
        cloze_model.init_cuda()
    cloze_model.set_key(l1_key, l2_key)
    cloze_model.init_key()
    cloze_model.l2_dict = gv2i
    if isinstance(cloze_model, CBiLSTM) or \
       isinstance(cloze_model, CBiLSTMFast) or \
       isinstance(cloze_model, CBiLSTMFastMap):
        cloze_model.train()

    print(cloze_model)
    macaronic_sents = []
    weights = cloze_model.get_weight()
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
