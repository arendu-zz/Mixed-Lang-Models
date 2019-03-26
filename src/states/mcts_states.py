#!/usr/bin/env python
__author__ = 'arenduchintala'
import numpy as np
import time
import random
EPS = 1e-4


class SearchTree(object):
    def __init__(self, game, mcts_params, rollout_func, rollout_params, state_update_func):
        self.game = game
        self.rollout_func = rollout_func
        self.rollout_params = rollout_params
        self.state_update_func = state_update_func
        self.iters = mcts_params['mcts_iters']
        self.nodes_seen = {}
        self.gamma = mcts_params.get('gamma', 1.0)
        self.backup_type = mcts_params['backup_type']
        self.C = mcts_params.get('const_C', 0.01)
        self.D = mcts_params.get('const_D', 10.0)
        self.consecutive_action_threshold = 1

    def recursive_search(self, root_node):
        node = root_node
        while not node.state.terminal:
            self.nodes_seen = {}
            node = self.search(node)
            #node.reset() #TODO:
        return node

    def search(self, root_node):
        self.nodes_seen[str(root_node.state)] = root_node
        #best_action_count = 0
        #prev_best_action = None
        for _ in range(self.iters):
            print('************************ search iter' + str(_) + '*************************')
            print('======root node=====\n', root_node)
            if root_node.completed_expansion:
                _c, _v, _u, _vt, _a, _vis = self.display_child_values(root_node, self.C)
            else:
                print('not completed_expansion')
                #rank = tuple(np.array(_vis).argsort().argsort().tolist())
                #if rank == prev_rank:
                #    same_rank += 1
                #else:
                #    same_rank = 0
                #prev_rank = rank
                #print('same_rank', same_rank, rank)
                #if max(_vis) - min(_vis) >= 3:
                #    break
            #    best_action, best_node = self.select_child(root_node, self.C)
            #    best_action_count = best_action_count + 1 if best_action == prev_best_action else 0
            #    prev_best_action = best_action
            #    if best_action_count > self.consecutive_action_threshold:
            #        return best_node

            node, search_sequence = self.selection(root_node)
            print('===selection node===\n', node)
            if not node.state.terminal:
                node, search_sequence = self.expansion(node, search_sequence)
                print('===expansion node===\n', node)
                if not node.state.terminal:
                    rollout_state = self.rollout(node)
                    print('===rollout state===\n', rollout_state)
                    reward = rollout_state.score  # - root_node.state.score
            else:
                reward = node.state.score # - root_node.state.score
            self.backup(reward, 0, search_sequence)
            print('********************** end search iter' + str(_) + '*******************')
        best_action, best_node = self.select_child(root_node, 0.0)
        return best_node

    def selection(self, node):
        search_sequence = [node]
        while node.completed_expansion:
            assert not node.state.terminal
            action, child_node = self.select_child(node, self.C)
            assert child_node != node
            search_sequence.append(child_node)
            node = child_node
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
        rollout_start_state.binary_branching = self.rollout_params['rollout_binary_branching']
        state = self.rollout_func(rollout_start_state, **self.rollout_params)
        print('rollout time', time.time() - now)
        if rollout_start_state.score == float('-inf'):
            state.score = 0.0
            print('forcing score to 0.0 because rollout_start_state is bad')
        return state

    def backup(self, reward, winner, search_sequence):
        # print('------------------backup-----------------')
        for node in search_sequence:
            node.visits += 1.0
            node.prev_value[winner] = node.value[winner]  # value at previous time-step
            if self.backup_type == 'max':
                node.value[winner] = reward if reward > node.value[winner] else node.value[winner]
            elif self.backup_type == 'ave':
                node.value[winner] += ((reward - node.value[winner])/(node.visits))
            else:
                raise NotImplementedError("unknown backup_type")
            node.prod_std_deviation[winner] += (node.value[winner] - node.prev_value[winner]) * (node.value[winner] - node.value[winner])
        # print('-----------------------------------------')

        return True

    def display_child_values(self, node, exp_param):
        combined, values, ucb, variance_terms, actions, visits = self.child_scores(node, exp_param)
        f = [(a, c, v, u, vrt, vs) for a, c, v, u, vrt, vs in zip(actions, combined, values, ucb, variance_terms, visits)]
        s = '\n'.join(['combined %0.4f' % c +
                       ' value: %0.4f' % v +
                       ' ucb: %0.4f' % u +
                       ' var_term:' + str(vrt) +
                       ' action:' + str(a) +
                       ' visits:' + str(vs) for
                       a, c, v, u, vrt, vs in sorted(f)])
        print(s)
        return combined, values, ucb, variance_terms, actions, visits

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
        combined = (1.0 - exp_param) * values + (exp_param * ucb)  #+ variance_terms
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
            assert expanded_node != node
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
    def __init__(self, model, model_config_func, opt):
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



