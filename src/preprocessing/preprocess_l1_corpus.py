# -*- coding: utf-8 -*-
import os
import pickle
import argparse
import torch
import fastText
from ..utils.utils import SPECIAL_TOKENS
from src.rewards import get_nearest_neighbors_simple


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help="data directory path, in this folder a corpus.txt file is expected")
    parser.add_argument('--wordvec_bin', action='store', dest='word_vec_file', required=True)
    parser.add_argument('--max_word_len', type=int, default=20, help='ignore words longer than this')
    parser.add_argument('--max_vocab', type=int, default=100000, help='only keep most frequent words')
    return parser.parse_args()


def load_word_vec(_file, voc2i):
    ft_model = fastText.load_model(_file)
    #wv = {}
    #for l in open(_file, 'r', encoding='utf8').readlines():
    #    i = l.strip().split()
    #    if len(i) == 301:
    #        w = i[0][0].lower() + i[0][1:]
    #        v = torch.Tensor([float(f) for f in i[1:]]).unsqueeze(0)
    #        if w in voc2i:  # in the vocab list that we have already created
    #            if w in wv:  # in the set of vocab that we have seen another form i.e. different capitalization
    #                wv[w] = torch.cat((wv[w], v), dim=0)
    #            else:
    #                wv[w] = v
    #        else:
    #            pass
    #print(len(wv), 'normalized vocab from word_vec_file')
    #mat = torch.FloatTensor(len(voc2i), 300).uniform_(-1.0, 1.0)
    #missing = []
    #for v, i in voc2i.items():
    #    if v in wv:
    #        mat[i] = wv[v].mean(dim=0)
    #    else:
    #        missing.append(v)
    #print(len(missing), 'got random vec')
    mat = torch.FloatTensor(len(voc2i), 300).uniform_(-1.0, 1.0)
    for v, i in voc2i.items():
        mat[i, :] = torch.tensor(ft_model.get_word_vector(v))
    return mat


def to_str(lst):
    return ','.join([str(i) for i in lst])




class Preprocess(object):
    def __init__(self,):
        # spl sym for words
        self.spl_words = set([SPECIAL_TOKENS.PAD,
                              SPECIAL_TOKENS.BOS,
                              SPECIAL_TOKENS.EOS,
                              SPECIAL_TOKENS.UNK])

    def build(self, data_dir, corpus_file, dev_file,
              max_word_len=30, max_vocab=50000):
        c2idx = {}
        for t in [SPECIAL_TOKENS.PAD,
                  SPECIAL_TOKENS.BOW,
                  SPECIAL_TOKENS.EOW,
                  SPECIAL_TOKENS.UNK_C,
                  SPECIAL_TOKENS.BOS,
                  SPECIAL_TOKENS.EOS,
                  SPECIAL_TOKENS.UNK]:
            c2idx[t] = len(c2idx)

        print("building vocab...", corpus_file)
        line_num = 0
        self.wc = {}
        total_words = 0
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                line_num += 1
                if not line_num % 1000:
                    print("working on {}kth line".format(line_num // 1000))
                words = line.strip().split()
                assert len(words) > 0, str(line_num) + ' is empty'
                for word in words:
                    word = word.lower()
                    if word in self.spl_words:
                        pass
                    elif len(word) + 2 < max_word_len:
                        self.wc[word] = self.wc.get(word, 0) + 1
                        total_words += 1
                    else:
                        pass
        print("")
        print("total word types", len(self.wc))
        self.vocab = [SPECIAL_TOKENS.PAD, SPECIAL_TOKENS.BOS, SPECIAL_TOKENS.EOS, SPECIAL_TOKENS.UNK] + \
            sorted(self.wc, key=self.wc.get, reverse=True)[:max_vocab]  # all word types frequency sorted
        print("word types", len(self.vocab))
        total_count = sum([self.wc.get(v, 0) for v in self.vocab])
        vocab_info = []
        max_vl = 0
        for v in self.vocab:
            if v not in self.spl_words:
                for c in list(v):
                    if ord(c) < 256:
                        c2idx[c] = c2idx.get(c, len(c2idx))
                    else:
                        pass
            else:
                c2idx[v] = c2idx.get(v, len(c2idx))
            vs = [SPECIAL_TOKENS.BOW] + list(v) + [SPECIAL_TOKENS.EOW]
            cs = [c2idx[c] if c in c2idx else c2idx[SPECIAL_TOKENS.UNK_C] for c in vs]
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
        idx2v = {idx: v for v, idx in v2idx.items()}
        idx2c = {idx: c for c, idx in c2idx.items()}
        print("build done")
        print("saving files...")
        print('vocab size', len(self.vocab))
        pickle.dump(self.vocab, open(os.path.join(data_dir, 'l1.vocab.pkl'), 'wb'))
        pickle.dump(idx2v, open(os.path.join(data_dir, 'l1.idx2v.pkl'), 'wb'))
        print('v2idx size', len(v2idx))
        assert len(idx2v) == len(v2idx) == len(vidx2spelling)
        pickle.dump(v2idx, open(os.path.join(data_dir, 'l1.v2idx.pkl'), 'wb'))
        pickle.dump(vidx2spelling, open(os.path.join(data_dir, 'l1.vidx2spelling.pkl'), 'wb'))
        pickle.dump(vidx2unigram_prob, open(os.path.join(data_dir, 'l1.vidx2unigram_prob.pkl'), 'wb'))
        print('char vocab size', len(c2idx), len(idx2c))
        assert len(idx2c) == len(c2idx)
        pickle.dump(idx2c, open(os.path.join(data_dir, 'l1.idx2c.pkl'), 'wb'))
        pickle.dump(c2idx, open(os.path.join(data_dir, 'l1.c2idx.pkl'), 'wb'))
        return v2idx


if __name__ == '__main__':
    args = parse_args()
    preprocess = Preprocess()
    corpus_file = os.path.join(args.data_dir, 'corpus.en')
    dev_file = os.path.join(args.data_dir, 'dev.en')
    v2idx = preprocess.build(data_dir=args.data_dir,
                             corpus_file=corpus_file,
                             dev_file=dev_file,
                             max_word_len=args.max_word_len,
                             max_vocab=args.max_vocab)
    embedding = load_word_vec(args.word_vec_file, v2idx)
    torch.save(embedding, os.path.join(args.data_dir, 'l1.mat.pt'))
