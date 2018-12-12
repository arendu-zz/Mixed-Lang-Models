# -*- coding: utf-8 -*-
import os
import pickle
import argparse
from utils.utils import SPECIAL_TOKENS

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--l1_data_dir', type=str, required=True,
                        help="l1 data directory path with all l1 pkl objects")
    parser.add_argument('--l2_data_dir', type=str, required=True,
                        help="l2 data directory path with parallel_file")
    parser.add_argument('--l2_save_dir', type=str, required=True,
                        help="l2 save dir")
    return parser.parse_args()


def to_str(lst):
    return ','.join([str(i) for i in lst])


class Preprocess(object):
    def __init__(self):
        self.spl_words = set([SPECIAL_TOKENS.PAD,
                              SPECIAL_TOKENS.BOS,
                              SPECIAL_TOKENS.EOS,
                              SPECIAL_TOKENS.UNK])


    def build(self, l1_data_dir, l2_data_dir, l2_save_dir):
        l1_vocab = pickle.load(open(os.path.join(l1_data_dir, 'l1.vocab.pkl'), 'rb'))
        l1_idx2v = pickle.load(open(os.path.join(l1_data_dir, 'l1.idx2v.pkl'), 'rb'))
        l1_v2idx = pickle.load(open(os.path.join(l1_data_dir, 'l1.v2idx.pkl'), 'rb'))
        #l1_vidx2spelling = pickle.load(open(os.path.join(l1_data_dir, 'l1.vidx2spelling.pkl'), 'rb'))
        #l1_vidx2unigram_prob = pickle.load(open(os.path.join(l1_data_dir, 'l1.vidx2unigram_prob.pkl'), 'rb'))
        #l1_idx2c = pickle.load(open(os.path.join(l1_data_dir, 'l1.idx2c.pkl'), 'rb'))
        #l1_c2idx = pickle.load(open(os.path.join(l1_data_dir, 'l1.c2idx.pkl'), 'rb'))

        l2_v2idx = {SPECIAL_TOKENS.PAD: 0, SPECIAL_TOKENS.BOS: 1, SPECIAL_TOKENS.EOS: 2, SPECIAL_TOKENS.UNK: 3}
        l2_idx2v = {0: SPECIAL_TOKENS.PAD, 1: SPECIAL_TOKENS.BOS, 2: SPECIAL_TOKENS.EOS, 3: SPECIAL_TOKENS.UNK}
        l2_vidx2spelling = {}  # TODO
        l2_c2idx = {}  # TODO
        l2_idx2c = {}  # TODO
        key = set([])
        with open(os.path.join(l2_data_dir, 'parallel_corpus'), 'r', encoding='utf-8') as f:
            for line in f:
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

                    if l1_w in l1_v2idx and l2_w in l2_v2idx and \
                        l1_w not in [SPECIAL_TOKENS.PAD, SPECIAL_TOKENS.BOS, SPECIAL_TOKENS.EOS, SPECIAL_TOKENS.UNK] and \
                            l2_w not in [SPECIAL_TOKENS.PAD, SPECIAL_TOKENS.BOS, SPECIAL_TOKENS.EOS, SPECIAL_TOKENS.UNK]:
                        key.add((l1_v2idx[l1_w], l2_v2idx[l2_w]))
                        print(l1_w, l2_w, l1_v2idx[l1_w], l2_v2idx[l2_w])
        key = list(sorted(key))
        assert len(l2_v2idx) == len(l2_idx2v)
        pickle.dump(l2_v2idx, open(os.path.join(l2_save_dir, 'l2.v2idx.pkl'), 'wb'))
        pickle.dump(l2_idx2v, open(os.path.join(l2_save_dir, 'l2.idx2v.pkl'), 'wb'))
        pickle.dump(key, open(os.path.join(l2_save_dir, 'l1.l2.key.pkl'), 'wb'))
        txt_key = open(os.path.join(l2_save_dir, 'key.txt'), 'w', encoding='utf-8')
        for l1idx, l2idx in key:
            txt = str(l1idx) + ' ' + l1_idx2v[l1idx] + ' ' + ' ' + str(l2idx) + ' ' + l2_idx2v[l2idx] + '\n'
            txt_key.write(txt)
        txt_key.close()
        info = open(os.path.join(l2_save_dir, 'INFO.FILE'), 'w')
        info.write("the l2*pkl files and l1.l2.key.pkl file was created using the l1 vocabulary from:" + l1_data_dir)
        info.close()
        return True


if __name__ == '__main__':
    args = parse_args()
    preprocess = Preprocess()
    preprocess.build(l1_data_dir=args.l1_data_dir,
                     l2_data_dir=args.l2_data_dir,
                     l2_save_dir=args.l2_save_dir)
