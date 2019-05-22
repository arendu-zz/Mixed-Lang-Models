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
import pdb

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
        #l1_vocab = pickle.load(open(os.path.join(l1_data_dir, 'l1.vocab.pkl'), 'rb'))
        l1_idx2v = pickle.load(open(os.path.join(l1_data_dir, 'l1.idx2v.pkl'), 'rb'))
        l1_v2idx = pickle.load(open(os.path.join(l1_data_dir, 'l1.v2idx.pkl'), 'rb'))
        l1_mat = torch.load(l1_data_dir + '/l1.mat.pt')
        #l1_vidx2c1gram_spelling = pickle.load(open(os.path.join(l1_data_dir, 'l1.vidx2c1gram_spelling.pkl'), 'rb'))
        #l1_vidx2unigram_prob = pickle.load(open(os.path.join(l1_data_dir, 'l1.vidx2unigram_prob.pkl'), 'rb'))
        #l1_idx2c1gram = pickle.load(open(os.path.join(l1_data_dir, 'l1.idx2c1gram.pkl'), 'rb'))
        l1_c1gram2idx = pickle.load(open(os.path.join(l1_data_dir, 'l1.c1gram2idx.pkl'), 'rb'))
        l1_c2gram2idx = pickle.load(open(os.path.join(l1_data_dir, 'l1.c2gram2idx.pkl'), 'rb'))
        l1_c3gram2idx = pickle.load(open(os.path.join(l1_data_dir, 'l1.c3gram2idx.pkl'), 'rb'))
        l1_c4gram2idx = pickle.load(open(os.path.join(l1_data_dir, 'l1.c4gram2idx.pkl'), 'rb'))

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
                #assert len(line_key) > 0, "this line has issues" + str(line)
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
        l2_c1gram2idx = {}
        l2_c2gram2idx = {}
        l2_c3gram2idx = {}
        l2_c4gram2idx = {}
        #l2_c1gram2idx = {c: i for c, i in l1_c1gram2idx.items()}
        #l2_c2gram2idx = {c: i for c, i in l1_c2gram2idx.items()}
        #l2_c3gram2idx = {c: i for c, i in l1_c3gram2idx.items()}
        #l2_c4gram2idx = {c: i for c, i in l1_c4gram2idx.items()}
        max_vl1 = 0
        max_vl2 = 0
        max_vl3 = 0
        max_vl4 = 0
        vocab_info = []
        max_vl1_by_l1 = 0
        max_vl2_by_l1 = 0
        max_vl3_by_l1 = 0
        max_vl4_by_l1 = 0
        vocab_info_by_l1 = []
        for l2_v, l2_idx in l2_v2idx.items():
            print('spelling', l2_v, l2_idx)
            if l2_v not in [SPECIAL_TOKENS.PAD, SPECIAL_TOKENS.NULL, SPECIAL_TOKENS.BOS, SPECIAL_TOKENS.EOS, SPECIAL_TOKENS.UNK]:
                l2_c1grams = [SPECIAL_TOKENS.BOW] + [c for c in l2_v] + [SPECIAL_TOKENS.EOW]
                l2_c2grams = [''.join(z) for z in zip(*[l2_c1grams[i:] for i in range(2)])]
                l2_c3grams = [''.join(z) for z in zip(*[l2_c1grams[i:] for i in range(3)])]
                l2_c4grams = [''.join(z) for z in zip(*[l2_c1grams[i:] for i in range(4)])]
            else:
                l2_c1grams = [l2_v]
                l2_c2grams = [l2_v]
                l2_c3grams = [l2_v]
                l2_c4grams = [l2_v]

            c1_len = len(l2_c1grams)
            for l2c in l2_c1grams:
                l2_c1gram2idx[l2c] = l2_c1gram2idx.get(l2c, len(l2_c1gram2idx))
            cs1 = [l2_c1gram2idx[c] for c in l2_c1grams]
            max_vl1 = max_vl1 if c1_len < max_vl1 else c1_len

            cs1_by_l1 = [l1_c1gram2idx.get(c, l1_c1gram2idx[SPECIAL_TOKENS.PAD]) for c in l2_c1grams]
            max_vl1_by_l1 = max_vl1_by_l1 if c1_len < max_vl1_by_l1 else c1_len

            c2_len = len(l2_c2grams)
            for l2c in l2_c2grams:
                l2_c2gram2idx[l2c] = l2_c2gram2idx.get(l2c, len(l2_c2gram2idx))
            cs2 = [l2_c2gram2idx[c] for c in l2_c2grams]
            max_vl2 = max_vl2 if c2_len < max_vl2 else c2_len

            cs2_by_l1 = [l1_c2gram2idx.get(c, l1_c2gram2idx[SPECIAL_TOKENS.PAD]) for c in l2_c2grams]
            max_vl2_by_l1 = max_vl2_by_l1 if c2_len < max_vl2_by_l1 else c2_len

            c3_len = len(l2_c3grams)
            for l2c in l2_c3grams:
                l2_c3gram2idx[l2c] = l2_c3gram2idx.get(l2c, len(l2_c3gram2idx))
            cs3 = [l2_c3gram2idx[c] for c in l2_c3grams]
            max_vl3 = max_vl3 if c3_len < max_vl3 else c3_len

            cs3_by_l1 = [l1_c3gram2idx.get(c, l1_c3gram2idx[SPECIAL_TOKENS.PAD]) for c in l2_c3grams]
            max_vl3_by_l1 = max_vl3_by_l1 if c3_len < max_vl3_by_l1 else c3_len

            c4_len = len(l2_c4grams)
            for l2c in l2_c4grams:
                l2_c4gram2idx[l2c] = l2_c4gram2idx.get(l2c, len(l2_c4gram2idx))
            cs4 = [l2_c4gram2idx[c] for c in l2_c4grams]
            max_vl4 = max_vl4 if c4_len < max_vl4 else c4_len

            cs4_by_l1 = [l1_c4gram2idx.get(c, l1_c4gram2idx[SPECIAL_TOKENS.PAD]) for c in l2_c4grams]
            max_vl4_by_l1 = max_vl4_by_l1 if c4_len < max_vl4_by_l1 else c4_len

            vocab_info.append((l2_idx, l2_v, c1_len, cs1, c2_len, cs2, c3_len, cs3, c4_len, cs4))
            vocab_info_by_l1.append((l2_idx, l2_v, c1_len, cs1_by_l1, c2_len, cs2_by_l1, c3_len, cs3_by_l1, c4_len, cs4_by_l1))

        l2_vidx2c1gram_spelling = {}
        l2_vidx2c2gram_spelling = {}
        l2_vidx2c3gram_spelling = {}
        l2_vidx2c4gram_spelling = {}
        for l2_idx, l2v, c1l, cs1, c2l, cs2, c3l, cs3, c4l, cs4 in vocab_info:
            #cs1 = l2_c1grams[:max_word_len]
            padder1 = [0] * (max_vl1 - c1l)
            l2_vidx2c1gram_spelling[l2_idx] = cs1 + padder1 + [c1l]

            padder2 = [0] * (max_vl2 - c2l)
            l2_vidx2c2gram_spelling[l2_idx] = cs2 + padder2 + [c2l]

            padder3 = [0] * (max_vl3 - c3l)
            l2_vidx2c3gram_spelling[l2_idx] = cs3 + padder3 + [c3l]

            padder4 = [0] * (max_vl4 - c4l)
            l2_vidx2c4gram_spelling[l2_idx] = cs4 + padder4 + [c4l]

        pickle.dump(l2_vidx2c1gram_spelling, open(os.path.join(l2_save_dir, 'l2.vidx2c1gram_spelling.pkl'), 'wb'))
        pickle.dump(l2_vidx2c2gram_spelling, open(os.path.join(l2_save_dir, 'l2.vidx2c2gram_spelling.pkl'), 'wb'))
        pickle.dump(l2_vidx2c3gram_spelling, open(os.path.join(l2_save_dir, 'l2.vidx2c3gram_spelling.pkl'), 'wb'))
        pickle.dump(l2_vidx2c4gram_spelling, open(os.path.join(l2_save_dir, 'l2.vidx2c4gram_spelling.pkl'), 'wb'))

        l2_vidx2c1gram_by_l1_spelling = {}
        l2_vidx2c2gram_by_l1_spelling = {}
        l2_vidx2c3gram_by_l1_spelling = {}
        l2_vidx2c4gram_by_l1_spelling = {}
        for l2_idx, l2v, c1l, cs1_by_l1, c2l, cs2_by_l1, c3l, cs3_by_l1, c4l, cs4_by_l1 in vocab_info_by_l1:
            #cs1 = l2_c1grams[:max_word_len]
            padder1 = [0] * (max_vl1_by_l1 - c1l)
            l2_vidx2c1gram_by_l1_spelling[l2_idx] = cs1_by_l1 + padder1 + [c1l]

            padder2 = [0] * (max_vl2_by_l1 - c2l)
            l2_vidx2c2gram_by_l1_spelling[l2_idx] = cs2_by_l1 + padder2 + [c2l]

            padder3 = [0] * (max_vl3_by_l1 - c3l)
            l2_vidx2c3gram_by_l1_spelling[l2_idx] = cs3_by_l1 + padder3 + [c3l]

            padder4 = [0] * (max_vl4_by_l1 - c4l)
            l2_vidx2c4gram_by_l1_spelling[l2_idx] = cs4_by_l1 + padder4 + [c4l]

        pickle.dump(l2_vidx2c1gram_by_l1_spelling, open(os.path.join(l2_save_dir, 'l2.vidx2c1gram_by_l1_spelling.pkl'), 'wb'))
        pickle.dump(l2_vidx2c2gram_by_l1_spelling, open(os.path.join(l2_save_dir, 'l2.vidx2c2gram_by_l1_spelling.pkl'), 'wb'))
        pickle.dump(l2_vidx2c3gram_by_l1_spelling, open(os.path.join(l2_save_dir, 'l2.vidx2c3gram_by_l1_spelling.pkl'), 'wb'))
        pickle.dump(l2_vidx2c4gram_by_l1_spelling, open(os.path.join(l2_save_dir, 'l2.vidx2c4gram_by_l1_spelling.pkl'), 'wb'))

        l2_idx2c = {i: c for c, i in l2_c1gram2idx.items()}
        pickle.dump(l2_c1gram2idx, open(os.path.join(l2_save_dir, 'l2.c2idx.pkl'), 'wb'))
        pickle.dump(l2_idx2c, open(os.path.join(l2_save_dir, 'l2.idx2c.pkl'), 'wb'))
        l2_idx2c1gram = {i: c for c, i in l2_c1gram2idx.items()}
        l2_idx2c2gram = {i: c for c, i in l2_c2gram2idx.items()}
        l2_idx2c3gram = {i: c for c, i in l2_c3gram2idx.items()}
        l2_idx2c4gram = {i: c for c, i in l2_c4gram2idx.items()}

        print('l2 char vocab size', len(l2_c1gram2idx), len(l2_idx2c1gram))
        assert len(l2_idx2c1gram) == len(l2_c1gram2idx)
        pickle.dump(l2_idx2c1gram, open(os.path.join(l2_save_dir, 'l2.idx2c1gram.pkl'), 'wb'))
        pickle.dump(l2_c1gram2idx, open(os.path.join(l2_save_dir, 'l2.c1gram2idx.pkl'), 'wb'))
        print('l2 char2 vocab size', len(l2_c2gram2idx), len(l2_idx2c2gram))
        assert len(l2_idx2c2gram) == len(l2_c2gram2idx)
        pickle.dump(l2_idx2c2gram, open(os.path.join(l2_save_dir, 'l2.idx2c2gram.pkl'), 'wb'))
        pickle.dump(l2_c2gram2idx, open(os.path.join(l2_save_dir, 'l2.c2gram2idx.pkl'), 'wb'))
        print('l2 char3 vocab size', len(l2_c3gram2idx), len(l2_idx2c3gram))
        assert len(l2_idx2c3gram) == len(l2_c3gram2idx)
        pickle.dump(l2_idx2c3gram, open(os.path.join(l2_save_dir, 'l2.idx2c3gram.pkl'), 'wb'))
        pickle.dump(l2_c3gram2idx, open(os.path.join(l2_save_dir, 'l2.c3gram2idx.pkl'), 'wb'))
        print('l2 char4 vocab size', len(l2_c4gram2idx), len(l2_idx2c4gram))
        assert len(l2_idx2c4gram) == len(l2_c4gram2idx)
        pickle.dump(l2_idx2c4gram, open(os.path.join(l2_save_dir, 'l2.idx2c4gram.pkl'), 'wb'))
        pickle.dump(l2_c4gram2idx, open(os.path.join(l2_save_dir, 'l2.c4gram2idx.pkl'), 'wb'))

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
