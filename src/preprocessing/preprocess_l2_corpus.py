#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle
import argparse
import fastText
import torch
from src.utils.utils import SPECIAL_TOKENS

import editdistance as ed
import math


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--l1_data_dir', type=str, required=True,
                        help="l1 data directory path with all l1 pkl objects")
    parser.add_argument('--l2_data_dir', type=str, required=True,
                        help="l2 data directory path with parallel_file")
    parser.add_argument('--l2_save_dir', type=str, required=True,
                        help="l2 save dir")
    parser.add_argument('--max_word_len', type=int, default=20, help='cut off words longer than this')
    parser.add_argument('--wordvec_bin', action='store', dest='word_vec_file', required=True)
    return parser.parse_args()


def to_str(lst):
    return ','.join([str(i) for i in lst])


class Preprocess(object):
    def __init__(self):
        self.spl_words = set([SPECIAL_TOKENS.PAD,
                              SPECIAL_TOKENS.BOS,
                              SPECIAL_TOKENS.EOS,
                              SPECIAL_TOKENS.UNK])


    def build(self, l1_data_dir, l2_data_dir, l2_save_dir, ft_model, max_word_len):
        l1_vocab = pickle.load(open(os.path.join(l1_data_dir, 'l1.vocab.pkl'), 'rb'))
        l1_idx2v = pickle.load(open(os.path.join(l1_data_dir, 'l1.idx2v.pkl'), 'rb'))
        l1_v2idx = pickle.load(open(os.path.join(l1_data_dir, 'l1.v2idx.pkl'), 'rb'))
        l1_mat = torch.load(l1_data_dir + '/l1.mat.pt')
        l1_vidx2spelling = pickle.load(open(os.path.join(l1_data_dir, 'l1.vidx2spelling.pkl'), 'rb'))
        #l1_vidx2unigram_prob = pickle.load(open(os.path.join(l1_data_dir, 'l1.vidx2unigram_prob.pkl'), 'rb'))
        l1_idx2c = pickle.load(open(os.path.join(l1_data_dir, 'l1.idx2c.pkl'), 'rb'))
        l1_c2idx = pickle.load(open(os.path.join(l1_data_dir, 'l1.c2idx.pkl'), 'rb'))

        l2_v2idx = {SPECIAL_TOKENS.PAD: 0, SPECIAL_TOKENS.BOS: 1, SPECIAL_TOKENS.EOS: 2, SPECIAL_TOKENS.UNK: 3}
        l2_idx2v = {0: SPECIAL_TOKENS.PAD, 1: SPECIAL_TOKENS.BOS, 2: SPECIAL_TOKENS.EOS, 3: SPECIAL_TOKENS.UNK}
        l2_vidx2count = {0: 0, 1: 0, 2: 0, 3: 0}
        l2_vidx2line_nums = {0: set([]), 1: set([]), 2: set([]), 3: set([])}
        full_data_key = set([])
        line_keys = []
        print('loaded l1 data...')
        l2_line_idx = 0
        l2_line_term_count = {}
        with open(os.path.join(l2_data_dir, 'parallel_corpus'), 'r', encoding='utf-8') as f:
            for line in f:
                print(line)
                l2_line_idx += 1
                line_key = set()
                l1_line, l2_line = line.split('|||')
                l1_line_txt = l1_line.strip().split()
                l2_line_txt = l2_line.strip().split()
                assert len(l1_line_txt) == len(l2_line_txt)
                for l1_w, l2_w in zip(l1_line_txt, l2_line_txt):
                    l2_w = l2_w.lower()
                    l1_w = l1_w.lower()
                    if l2_w != SPECIAL_TOKENS.NULL:
                        l2_w_idx = l2_v2idx.get(l2_w, len(l2_v2idx))
                        l2_v2idx[l2_w] = l2_w_idx
                        l2_idx2v[l2_w_idx] = l2_w
                        l2_vidx2count[l2_w_idx] = l2_vidx2count.get(l2_w_idx, 0) + 1
                        l2_vidx2line_nums[l2_w_idx] = l2_vidx2line_nums.get(l2_w_idx, set([])).union(set([l2_line_idx]))
                        l2_line_term_count[l2_line_idx, l2_w_idx] = l2_line_term_count.get((l2_line_idx, l2_w_idx), 0) + 1

                    if l1_w in l1_v2idx and l2_w in l2_v2idx and \
                        l1_w not in [SPECIAL_TOKENS.PAD, SPECIAL_TOKENS.BOS, SPECIAL_TOKENS.EOS, SPECIAL_TOKENS.UNK] and \
                            l2_w not in [SPECIAL_TOKENS.PAD, SPECIAL_TOKENS.BOS, SPECIAL_TOKENS.EOS, SPECIAL_TOKENS.UNK]:
                        full_data_key.add((l1_v2idx[l1_w], l2_v2idx[l2_w]))
                        line_key.add((l1_v2idx[l1_w], l2_v2idx[l2_w]))
                        print(l1_w, l2_w, l1_v2idx[l1_w], l2_v2idx[l2_w])
                line_keys.append(list(sorted(line_key)))
        full_data_key = list(sorted(full_data_key))
        assert len(l2_v2idx) == len(l2_idx2v)
        pickle.dump(l2_v2idx, open(os.path.join(l2_save_dir, 'l2.v2idx.pkl'), 'wb'))
        pickle.dump(l2_idx2v, open(os.path.join(l2_save_dir, 'l2.idx2v.pkl'), 'wb'))
        l2_max_count = max(l2_vidx2count.values())
        l2_vidx2tf = {}
        for l2_vidx, l2_vidx_c in l2_vidx2count.items():
            l2_vidx2tf[l2_vidx] = float(l2_vidx_c) / float(l2_max_count)
        for l2_vidx, l2_d in l2_vidx2line_nums.items():
            l2_vidx2line_nums[l2_vidx] = len(l2_d)
        l2_vidx2idf = {}
        for l2_vidx, l2_dlen in l2_vidx2line_nums.items():
            if l2_dlen == 0:
                l2_vidx2idf[l2_vidx] = 0.0
            else:
                l2_vidx2idf[l2_vidx] = math.log(float(l2_line_idx) / (float(l2_dlen))) + 1.0

        l2_key_wt = []
        for l1k, l2k in full_data_key:
            l2_key_wt.append(l2_vidx2idf[l2k])
        #min_tfidf = 0
        #max_tfidf = 0
        #l2_vidx2tfidf = {}
        #for l2_vidx in l2_vidx2tf:
        #    l2_vidx2tfidf[l2_vidx] = l2_vidx2tf[l2_vidx] * l2_vidx2idf[l2_vidx]
        #    min_tfidf = min_tfidf if min_tfidf < l2_vidx2tfidf[l2_vidx] else l2_vidx2tfidf[l2_vidx]
        #    max_tfidf = max_tfidf if max_tfidf > l2_vidx2tfidf[l2_vidx] else l2_vidx2tfidf[l2_vidx]
        #for l2_vidx in l2_vidx2tfidf:
        #    l2_vidx2tfidf[l2_vidx] = (l2_vidx2tfidf[l2_vidx] - min_tfidf) / (max_tfidf - min_tfidf)

        l2_c2idx = {c: i for c, i in l1_c2idx.items()}
        l2_vidx2spelling = {}  # TODO
        for l2_v, l2_idx in l2_v2idx.items():
            print('spelling', l2_v, l2_idx)
            if l2_v not in [SPECIAL_TOKENS.PAD, SPECIAL_TOKENS.NULL, SPECIAL_TOKENS.BOS, SPECIAL_TOKENS.EOS]:
                l2_chars = [SPECIAL_TOKENS.BOW] + [c for c in l2_v] + [SPECIAL_TOKENS.EOW]
            else:
                l2_chars = [SPECIAL_TOKENS.BOW] + [l2_v] + [SPECIAL_TOKENS.EOW]
            l2_chars = l2_chars[:max_word_len]
            l2_chars_len = len(l2_chars)
            for l2c in l2_chars:
                l2_c2idx[l2c] = l2_c2idx.get(l2c, len(l2_c2idx))
            l2_chars = l2_chars + ([SPECIAL_TOKENS.PAD] * (max_word_len - l2_chars_len))
            l2_chars_idx = [l2_c2idx[l2c] for l2c in l2_chars]
            l2_chars_idx[-1] = l2_chars_len
            l2_vidx2spelling[l2_idx] = l2_chars_idx
        l2_idx2c = {i: c for c, i in l2_c2idx.items()}
        pickle.dump(l2_c2idx, open(os.path.join(l2_save_dir, 'l2.c2idx.pkl'), 'wb'))
        pickle.dump(l2_idx2c, open(os.path.join(l2_save_dir, 'l2.idx2c.pkl'), 'wb'))
        pickle.dump(l2_vidx2spelling, open(os.path.join(l2_save_dir, 'l2.vidx2spelling.pkl'), 'wb'))
        if ft_model is not None:
            mat = torch.FloatTensor(len(l2_idx2v), 300).uniform_(-1.0, 1.0)
            for i, v in l2_idx2v.items():
                if i < 4:
                    mat[i, :] = l1_mat[i, :]  # special symbols have the same embedding
                else:
                    v_vec = ft_model.get_word_vector(v)
                    mat[i, :] = torch.tensor(v_vec)
            torch.save(mat, l2_save_dir + '/l2.mat.pt')
        else:
            print('not creating l2.mat.pt')
        l2_l1_min_ed = {}
        for l2_i, l2_v in l2_idx2v.items():
            print(l2_i, len(l2_idx2v))
            for l1_i, l1_v in l1_idx2v.items():
                if l1_i < 10000:
                    e = ed.eval(l1_v, l2_v) / max(len(l2_v), len(l1_v))
                    if e <= 0.334:
                        prev_e = l2_l1_min_ed.get((l2_i, l2_v), (1.0, None, None))
                        if prev_e[0] > e:
                            l2_l1_min_ed[(l2_i, l2_v)] = (e, (l1_i, l1_v))
        mat_ed = torch.FloatTensor(len(l2_idx2v), 300).fill_(0.0)
        for k, v in l2_l1_min_ed.items():
            l2_i, l2_v = k
            _, (l1_i, l1_v) = v
            mat_ed[l2_i, :] = l1_mat[l1_i, :]
            print(l2_v, 'closest', l1_v)
        mat_ed[[0, 1, 2, 3], :] = l1_mat[[0, 1, 2, 3], :]
        torch.save(mat_ed, l2_save_dir + '/l2_ed.mat.pt')
        pickle.dump(full_data_key, open(os.path.join(l2_save_dir, 'l1.l2.key.pkl'), 'wb'))
        pickle.dump(l2_key_wt, open(os.path.join(l2_save_dir, 'l2.key.wt.pkl'), 'wb'))
        pickle.dump(line_keys, open(os.path.join(l2_save_dir, 'per_line.l1.l2.key.pkl'), 'wb'))

        mat_key = torch.FloatTensor(len(l2_idx2v), 300).fill_(0.0)
        mat_key[[0, 1, 2, 3], :] = l1_mat[[0, 1, 2, 3], :]
        txt_key = open(os.path.join(l2_save_dir, 'full_data_key.txt'), 'w', encoding='utf-8')
        for l1idx, l2idx in full_data_key:
            txt = str(l1idx) + ' ' + l1_idx2v[l1idx] + ' ' + ' ' + str(l2idx) + ' ' + l2_idx2v[l2idx] + '\n'
            txt_key.write(txt)
            mat_key[l2idx, :] = l1_mat[l1idx, :]
        txt_key.close()
        torch.save(mat_key, l2_save_dir + '/l2_key.mat.pt')
        info = open(os.path.join(l2_save_dir, 'INFO.FILE'), 'w')
        info.write("the l2*pkl files and l1.l2.key.pkl file was created using the l1 vocabulary from:" + l1_data_dir)
        info.close()
        return l2_v2idx


if __name__ == '__main__':
    args = parse_args()
    preprocess = Preprocess()
    ft_model = fastText.load_model(args.word_vec_file)
    preprocess.build(l1_data_dir=args.l1_data_dir,
                     l2_data_dir=args.l2_data_dir,
                     l2_save_dir=args.l2_save_dir,
                     ft_model=ft_model,
                     max_word_len=args.max_word_len)
