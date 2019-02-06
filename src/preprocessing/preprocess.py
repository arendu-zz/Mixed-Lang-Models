# -*- coding: utf-8 -*-
import os
import pickle
import argparse

import pdb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help="data directory path, in this folder a corpus.txt file is expected")
    parser.add_argument('--max_word_len', type=int, default=20, help='ignore words longer than this')
    parser.add_argument('--max_vocab', type=int, default=50000, help='only keep most frequent words')
    return parser.parse_args()


def to_str(lst):
    return ','.join([str(i) for i in lst])


class Preprocess(object):
    def __init__(self):
        # spl sym for words
        self.unk = '<UNK>'
        self.unk_c = '<UNK_C>'
        self.bos = '<BOS>'
        self.eos = '<EOS>'
        # spl sym for chars
        self.bow = '<BOW>'
        self.eow = '<EOW>'
        self.pad = '<PAD>'
        self.spl_words = set([self.pad, self.bos, self.eos, self.unk])

    def build(self, data_dir, corpus_file, gloss_file,
              max_word_len=30, max_vocab=50000):
        c2idx = {self.pad: 0, self.bow: 1, self.eow: 3, self.unk_c: 4}
        if os.path.exists(gloss_file):
            flist = [(corpus_file, 'corpus'), (gloss_file, 'gloss')]
        else:
            flist = [(corpus_file, 'corpus')]
        for _file, _name in flist:
            print("building vocab...", _file)
            line_num = 0
            self.wc = {}
            total_words = 0
            with open(_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line_num += 1
                    if not line_num % 1000:
                        print("working on {}kth line".format(line_num // 1000))
                    words = line.strip().split()
                    assert len(words) > 0
                    for word in words:
                        if word in self.spl_words:
                            pass
                        elif len(word) + 2 < max_word_len:
                            word = word.lower()
                            self.wc[word] = self.wc.get(word, 0) + 1
                            total_words += 1
                        else:
                            pass
            print("")
            print("total word types", len(self.wc))
            self.vocab = [self.pad, self.bos, self.eos, self.unk] + \
                sorted(self.wc, key=self.wc.get, reverse=True)[:max_vocab]  # all word types frequency sorted
            total_count = sum([self.wc.get(v, 0) for v in self.vocab])
            vocab_info = []
            max_vl = 0
            for v in self.vocab:
                if v not in self.spl_words:
                    for c in list(v):
                        if ord(c) < 256:
                            c2idx[c] = c2idx.get(c, len(c2idx))
                        else:
                            c2idx[self.unk_c] = c2idx.get(self.unk_c, len(c2idx))
                else:
                    c2idx[v] = c2idx.get(v, len(c2idx))
                vs = [self.bow] + list(v) + [self.eow]
                cs = [c2idx.get(c, c2idx[self.unk_c]) for c in vs]
                vl = len(cs)
                vocab_info.append((vl, v, cs))
                max_vl = max_vl if max_vl > vl else vl

            # vocab_info = sorted(vocab_info, reverse=True)
            vidx2spelling = {}
            vidx2unigram_prob = {}
            v2idx = {}
            for v_idx, (vl, v, cs) in enumerate(vocab_info):
                v2idx[v] = v_idx
                padder = [0] * (max_vl - vl)
                vidx2spelling[v_idx] = cs + padder + [vl]
                vidx2unigram_prob[v_idx] = float(self.wc.get(v, 0)) / total_count
            idx2c = {idx: c for c, idx in c2idx.items()}
            idx2v = {idx: v for v, idx in v2idx.items()}
            print("build done")
            print("saving files...")
            pickle.dump(self.vocab, open(os.path.join(data_dir, _name + '.vocab.pkl'), 'wb'))
            pickle.dump(idx2v, open(os.path.join(data_dir, _name + '.idx2v.pkl'), 'wb'))
            pickle.dump(v2idx, open(os.path.join(data_dir, _name + '.v2idx.pkl'), 'wb'))
            pickle.dump(vidx2spelling, open(os.path.join(data_dir, _name + '.vidx2spelling.pkl'), 'wb'))
            pickle.dump(vidx2unigram_prob, open(os.path.join(data_dir, _name + '.vidx2unigram_prob.pkl'), 'wb'))

        pickle.dump(idx2c, open(os.path.join(data_dir, 'idx2c.pkl'), 'wb'))
        pickle.dump(c2idx, open(os.path.join(data_dir, 'c2idx.pkl'), 'wb'))

        # making key
        if os.path.exists(os.path.join(data_dir, 'gloss' + '.v2idx.pkl')):
            key = set([])
            v2idx = pickle.load(open(os.path.join(data_dir, 'corpus' + '.v2idx.pkl'), 'rb'))
            gv2idx = pickle.load(open(os.path.join(data_dir, 'gloss' + '.v2idx.pkl'), 'rb'))
            c_file = open(corpus_file, 'r', encoding='utf-8').readlines()
            g_file = open(gloss_file, 'r', encoding='utf-8').readlines()
            for c_line, g_line in zip(c_file, g_file):
                c_words = c_line.strip().split()
                g_words = g_line.strip().split()
                assert len(c_words) == len(g_words)
                for c, g in zip(c_words, g_words):
                    if c in v2idx and g in gv2idx:
                        key.add((gv2idx[g], v2idx[c]))

            pickle.dump(key, open(os.path.join(data_dir, 'key.pkl'), 'wb'))
        else:
            print('Can not make "key.pkl" file, no gloss found')


if __name__ == '__main__':
    args = parse_args()
    preprocess = Preprocess()
    corpus_file = os.path.join(args.data_dir, 'corpus.en')
    gloss_file = os.path.join(args.data_dir, 'gloss.txt')
    preprocess.build(data_dir=args.data_dir,
                     corpus_file=corpus_file,
                     gloss_file=gloss_file,
                     max_word_len=args.max_word_len,
                     max_vocab=args.max_vocab)
