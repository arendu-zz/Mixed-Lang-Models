#!/usr/bin/env python
__author__ = 'arenduchintala'
import argparse
import os
import pickle
from src.preprocessing.preprocess_l1_corpus import Preprocess as Preprocess_L1
from src.preprocessing.get_nn import save_nn_mat
import torch

if __name__ == '__main__':
    opt = argparse.ArgumentParser(description="write program description here")

    # insert options here
    opt.add_argument('--l1_data_name', action='store', dest='l1_data_name', required=True)
    opt.add_argument('--lmdata', action='store', dest='lmdata', required=True)
    opt.add_argument('--max_vocab', action='store', dest='max_vocab',
                     default=60000, type=int, required=False)
    opt.add_argument('--max_word_len', action='store', dest='max_word_len',
                     default=20, type=int, required=False)
    args = opt.parse_args()
    print(args)

    preprocess_l1 = Preprocess_L1()
    corpus_file = os.path.join(args.lmdata, args.l1_data_name, 'corpus.en')
    dev_file = os.path.join(args.lmdata, args.l1_data_name, 'dev.en')
    l1_v2idx = preprocess_l1.build(data_dir=os.path.join(args.lmdata, args.l1_data_name),
                                   corpus_file=corpus_file,
                                   dev_file=dev_file,
                                   max_word_len=args.max_word_len,
                                   max_vocab=args.max_vocab)
