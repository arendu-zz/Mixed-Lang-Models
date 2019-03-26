#!/usr/bin/env python
__author__ = 'arenduchintala'
import copy
import random
import json

from src.utils.utils import TEXT_EFFECT


NEXT_SENT = (-1, None)


class PriorityQ(object):
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.queue = []

    def append(self, item):
        self.queue.append(item)
        self.queue.sort(key=lambda x: (x.score, x.swap_token_counts), reverse=True)
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
        self.swap_token_counts = 0

    def tie_break_score(self,):
        return (self.score, -self.swap_token_counts)  # to break ties

    def to_json(self,):
        j = []
        for sent_id, sent in enumerate(self.macaronic_sentences):
            s = sent.to_json()
            j.append(s)
        return json.dumps(j)

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
        s.append('start_score       :' + str(self.start_score))
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
        return c, u

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
        c.swap_token_counts = self.swap_token_counts
        return c

    def next_state(self, action, apply_swap_func, **kwargs):
        assert isinstance(action, tuple)
        assert kwargs['reward_type'] == 'type_mrr_assist_check'
        c = self.copy()
        current_displayed_config = c.current_sentence()
        #print(c)
        if action == NEXT_SENT:
            #TODO: why do we need to use apply_swap_func???
            swap_token_count, swap_types = c.swap_counts()
            swap_result = apply_swap_func(current_displayed_config,
                                          c.model,
                                          c.weights,
                                          swap_types,
                                          **kwargs)
            c.weights = swap_result['weights']
            c.swap_token_counts = self.swap_token_counts
            assert swap_token_count == c.swap_token_counts
            #if c.score != c.start_score + (swap_result['score'] - (kwargs['penalty'] * swap_type_count)):
            #if c.score != c.start_score + swap_result['score']:
            if c.score != swap_result['score']:
                raise BaseException("c.score may not correct!")

            #c.score = c.start_score + (swap_result['score'] - (kwargs['penalty'] * swap_type_count))
            #c.score = c.start_score + swap_result['score']
            c.model.update_l2_exposure(current_displayed_config.l2_swapped_types)
            c.score = swap_result['score']
            if c.displayed_sentence_idx + 1 < len(c.macaronic_sentences):
                c.displayed_sentence_idx = self.displayed_sentence_idx + 1
                c.start_score = c.score
                #TODO:# should the score be set to 0??
            else:
                c.start_score = c.start_score
                #TODO:# should the score be set to 0??
                c.terminal = True
            next_state = c
        else:
            #print(action)
            #print('before', current_displayed_config)
            current_displayed_config.update_config(action)
            #print('after', current_displayed_config)
            if action[1] or action[0] == 1:
                swap_token_count, swap_types = c.swap_counts()
                #swap_type_count = len(swap_types)
                c.swap_token_counts = self.swap_token_counts + (1 if action[1] else 0)
                swap_result = apply_swap_func(current_displayed_config,
                                              c.model,
                                              c.weights,
                                              swap_types,
                                              **kwargs)
                #c.weights = c.weights  # should not give new_weights here!
                #print('here!',  swap_token_count, c.swap_token_counts, swap_token_count == c.swap_token_counts)
                #c.score = c.start_score + (swap_result['score'] - (kwargs['penalty'] * swap_type_count))
                #c.score = c.start_score + swap_result['score']
                c.score = swap_result['score']
                c.swap_token_counts = self.swap_token_counts + (1 if action[1] else 0)
                assert swap_token_count == c.swap_token_counts
            else:
                #assert self.score == swap_result['score'] - (kwargs['penalty'] * swap_type_count)
                c.score = self.score
                c.swap_token_counts = self.swap_token_counts
            next_state = c
        return next_state


class MacaronicSentence(object):
    def __init__(self,
                 tokens_l1, tokens_l2,
                 int_l1, int_l2,
                 swapped, not_swapped, swappable,
                 l2_swapped_types, l2_swapped_tokens,
                 swap_limit,
                 l1_key,
                 l2_key):
        self.__tokens_l1 = tokens_l1
        self.__tokens_l2 = tokens_l2
        self.__int_l1 = int_l1
        self.__int_l2 = int_l2
        self.__swapped = swapped
        self.__not_swapped = not_swapped
        self.__swappable = swappable  # set([idx for idx, tl2 in enumerate(tokens_l2)][1:-1])
        self.__l2_swapped_types = l2_swapped_types
        self.__l2_swapped_tokens = l2_swapped_tokens
        self.__l1_key = l1_key
        self.__l2_key = l2_key
        self.swap_limit = swap_limit
        assert type(self.__tokens_l1) == list
        assert len(self.__tokens_l1) == len(self.__tokens_l2)
        self.len = len(self.__tokens_l2)

    @property
    def l2_key(self,):
        return self.__l2_key

    @property
    def l1_key(self,):
        return self.__l1_key

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
                                           self.swap_limit,
                                           self.l1_key, # also does not change do no need to clone
                                           self.l2_key)
        return macaronic_copy

    def color_it(self, w1, w2, w_idx):
        if w_idx in self.swapped:
            return TEXT_EFFECT.CYAN + w2 + TEXT_EFFECT.END
        elif w_idx in self.not_swapped:
            return TEXT_EFFECT.YELLOW + w1 + TEXT_EFFECT.END
        else:
            return w1 #TEXT_EFFECT.YELLOW + w1 + TEXT_EFFECT.END

    def to_obj_list(self):
        j = []
        for idx, tl1, tl2, il1, il2 in zip(range(self.len), self.tokens_l1, self.tokens_l2, self.int_l1.view(-1), self.int_l2.view(-1)):
            w = {'idx': idx,
                 'display': 'l2' if idx in self.swapped else 'l1',
                 'l1_token': tl1,
                 'l2_token': tl2,
                 'l1_vid': il1.item(),
                 'l2_vid': il2.item(),
                 'nearest_neighbors': []}
            j.append(w)
        return j

    def to_json(self,):
        return json.dumps(self.to_obj_list())

    def display_macaronic(self,):
        s = [self.color_it(tl1, tl2, idx)
             for idx, tl1, tl2 in zip(range(self.len), self.tokens_l1, self.tokens_l2)]
        return ' '.join(s[1:-1])

    def __str__(self,):
        return self.display_macaronic()
