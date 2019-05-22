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
        c1gram2idx = {}
        c2gram2idx = {}
        c3gram2idx = {}
        c4gram2idx = {}
        for t in [SPECIAL_TOKENS.PAD,
                  SPECIAL_TOKENS.BOW,
                  SPECIAL_TOKENS.EOW,
                  SPECIAL_TOKENS.UNK_C,
                  SPECIAL_TOKENS.BOS,
                  SPECIAL_TOKENS.EOS,
                  SPECIAL_TOKENS.UNK]:
            c1gram2idx[t] = len(c1gram2idx)
            c2gram2idx[t] = len(c2gram2idx)
            c3gram2idx[t] = len(c3gram2idx)
            c4gram2idx[t] = len(c4gram2idx)

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
        max_vl1 = 0
        max_vl2 = 0
        max_vl3 = 0
        max_vl4 = 0
        for v in self.vocab:
            if v not in self.spl_words:
                for c in list(v):
                    if ord(c) < 256:
                        c1gram2idx[c] = c1gram2idx.get(c, len(c1gram2idx))
                    else:
                        pass
                vs = [SPECIAL_TOKENS.BOW] + list(v) + [SPECIAL_TOKENS.EOW]
                vs_2grams = [''.join(z) for z in zip(*[vs[i:] for i in range(2)])]
                vs_3grams = [''.join(z) for z in zip(*[vs[i:] for i in range(3)])]
                vs_4grams = [''.join(z) for z in zip(*[vs[i:] for i in range(4)])]
            else:
                c1gram2idx[v] = c1gram2idx.get(v, len(c1gram2idx))
                vs = [v]
                vs_2grams = [v]
                vs_3grams = [v]
                vs_4grams = [v]
            cs = [c1gram2idx[c] if c in c1gram2idx else c1gram2idx[SPECIAL_TOKENS.UNK_C] for c in vs]
            vl1 = len(cs)
            for c2 in vs_2grams:
                c2gram2idx[c2] = c2gram2idx.get(c2, len(c2gram2idx))
            cs2 = [c2gram2idx[c] for c in vs_2grams]
            vl2 = len(cs2)
            for c3 in vs_3grams:
                c3gram2idx[c3] = c3gram2idx.get(c3, len(c3gram2idx))
            cs3 = [c3gram2idx[c] for c in vs_3grams]
            vl3 = len(cs3)
            for c4 in vs_4grams:
                c4gram2idx[c4] = c4gram2idx.get(c4, len(c4gram2idx))
            cs4 = [c4gram2idx[c] for c in vs_4grams]
            vl4 = len(cs4)
            vocab_info.append((v, vl1, cs, vl2, cs2, vl3, cs3, vl4, cs4))
            max_vl1 = max_vl1 if max_vl1 > vl1 else vl1
            max_vl2 = max_vl2 if max_vl2 > vl2 else vl2
            max_vl3 = max_vl3 if max_vl3 > vl3 else vl3
            max_vl4 = max_vl4 if max_vl4 > vl4 else vl4

        # vocab_info = sorted(vocab_info, reverse=True)
        vidx2c1gram_spelling = {}
        vidx2c2gram_spelling = {}
        vidx2c3gram_spelling = {}
        vidx2c4gram_spelling = {}
        vidx2unigram_prob = {}
        v2idx = {}
        for v_idx, (v, vl1, cs1, vl2, cs2, vl3, cs3, vl4, cs4) in enumerate(vocab_info):
            v2idx[v] = v_idx
            padder1 = [0] * (max_vl1 - vl1)
            vidx2c1gram_spelling[v_idx] = cs1 + padder1 + [vl1]
            padder2 = [0] * (max_vl2 - vl2)
            vidx2c2gram_spelling[v_idx] = cs2 + padder2 + [vl2]
            padder3 = [0] * (max_vl3 - vl3)
            vidx2c3gram_spelling[v_idx] = cs3 + padder3 + [vl3]
            padder4 = [0] * (max_vl4 - vl4)
            vidx2c4gram_spelling[v_idx] = cs4 + padder4 + [vl4]
            vidx2unigram_prob[v_idx] = float(self.wc.get(v, 0)) / total_count
        idx2v = {idx: v for v, idx in v2idx.items()}
        idx2c1gram = {idx: c for c, idx in c1gram2idx.items()}
        idx2c2gram = {idx: c for c, idx in c2gram2idx.items()}
        idx2c3gram = {idx: c for c, idx in c3gram2idx.items()}
        idx2c4gram = {idx: c for c, idx in c4gram2idx.items()}
        print("build done")
        print("saving files...")
        print('vocab size', len(self.vocab))
        pickle.dump(self.vocab, open(os.path.join(data_dir, 'l1.vocab.pkl'), 'wb'))
        pickle.dump(idx2v, open(os.path.join(data_dir, 'l1.idx2v.pkl'), 'wb'))
        print('v2idx size', len(v2idx))
        assert len(idx2v) == len(v2idx) == len(vidx2c1gram_spelling)
        pickle.dump(v2idx, open(os.path.join(data_dir, 'l1.v2idx.pkl'), 'wb'))
        pickle.dump(vidx2c1gram_spelling, open(os.path.join(data_dir, 'l1.vidx2c1gram_spelling.pkl'), 'wb'))
        pickle.dump(vidx2c2gram_spelling, open(os.path.join(data_dir, 'l1.vidx2c2gram_spelling.pkl'), 'wb'))
        pickle.dump(vidx2c3gram_spelling, open(os.path.join(data_dir, 'l1.vidx2c3gram_spelling.pkl'), 'wb'))
        pickle.dump(vidx2c4gram_spelling, open(os.path.join(data_dir, 'l1.vidx2c4gram_spelling.pkl'), 'wb'))
        pickle.dump(vidx2unigram_prob, open(os.path.join(data_dir, 'l1.vidx2unigram_prob.pkl'), 'wb'))
        print('char vocab size', len(c1gram2idx), len(idx2c1gram))
        assert len(idx2c1gram) == len(c1gram2idx)
        pickle.dump(idx2c1gram, open(os.path.join(data_dir, 'l1.idx2c1gram.pkl'), 'wb'))
        pickle.dump(c1gram2idx, open(os.path.join(data_dir, 'l1.c1gram2idx.pkl'), 'wb'))
        print('char2 vocab size', len(c2gram2idx), len(idx2c2gram))
        assert len(idx2c2gram) == len(c2gram2idx)
        pickle.dump(idx2c2gram, open(os.path.join(data_dir, 'l1.idx2c2gram.pkl'), 'wb'))
        pickle.dump(c2gram2idx, open(os.path.join(data_dir, 'l1.c2gram2idx.pkl'), 'wb'))
        print('char3 vocab size', len(c3gram2idx), len(idx2c3gram))
        assert len(idx2c3gram) == len(c3gram2idx)
        pickle.dump(idx2c3gram, open(os.path.join(data_dir, 'l1.idx2c3gram.pkl'), 'wb'))
        pickle.dump(c3gram2idx, open(os.path.join(data_dir, 'l1.c3gram2idx.pkl'), 'wb'))
        print('char4 vocab size', len(c4gram2idx), len(idx2c4gram))
        assert len(idx2c4gram) == len(c4gram2idx)
        pickle.dump(idx2c4gram, open(os.path.join(data_dir, 'l1.idx2c4gram.pkl'), 'wb'))
        pickle.dump(c4gram2idx, open(os.path.join(data_dir, 'l1.c4gram2idx.pkl'), 'wb'))
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
