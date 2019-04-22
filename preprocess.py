#!/usr/bin/env python
__author__ = 'arenduchintala'
import argparse
import os
import fastText
import pickle
from src.preprocessing.preprocess_l1_corpus import Preprocess as Preprocess_L1
from src.preprocessing.preprocess_l2_corpus import Preprocess as Preprocess_L2
from src.preprocessing.preprocess_l1_corpus import load_word_vec
from src.preprocessing.dump_nn import save_nn_mat
import torch
import editdistance

if __name__ == '__main__':
    opt = argparse.ArgumentParser(description="write program description here")

    # insert options here
    opt.add_argument('--l1_data_name', action='store', dest='l1_data_name', required=True)
    opt.add_argument('--l2_data_name', action='store', dest='l2_data_name', required=True)
    opt.add_argument('--do_l1', action='store_true', dest='do_l1', required=False, default=False)
    opt.add_argument('--do_l2', action='store_true', dest='do_l2', required=False, default=False)
    opt.add_argument('--word_vec_bin', action='store', dest='word_vec_bin',
                     default='/export/b07/arenduc1/fast-text-vecs/crawl-300d-2M-subword.bin', required=False)
    opt.add_argument('--max_vocab', action='store', dest='max_vocab',
                     default=60000, type=int, required=False)
    opt.add_argument('--max_word_len', action='store', dest='max_word_len',
                     default=20, type=int, required=False)
    opt.add_argument('--lmdata', action='store', dest='lmdata',
                     default='/export/b07/arenduc1/macaronic-multi-agent/lmdata', required=False)
    opt.add_argument('--aligned_data', action='store', dest='aligned_data',
                     default='/export/b07/arenduc1/macaronic-multi-agent/aligned_data', required=False)
    args = opt.parse_args()
    print(args)

    if args.do_l1:
        print("l1 preprocessing...")
        preprocess_l1 = Preprocess_L1()
        corpus_file = os.path.join(args.lmdata, args.l1_data_name, 'corpus.en')
        dev_file = os.path.join(args.lmdata, args.l1_data_name, 'dev.en')
        l1_v2idx = preprocess_l1.build(data_dir=os.path.join(args.lmdata, args.l1_data_name),
                                       corpus_file=corpus_file,
                                       dev_file=dev_file,
                                       max_word_len=args.max_word_len,
                                       max_vocab=args.max_vocab)
        print("loading word_vec_bin...")
        embedding = load_word_vec(args.word_vec_bin, l1_v2idx)
        torch.save(embedding, os.path.join(args.lmdata, args.l1_data_name, 'l1.mat.pt'))

        # getting nearest neighbors for l1 word embeddings and saving them
        l1_mat = torch.load(os.path.join(args.lmdata, args.l1_data_name, 'l1.mat.pt'))
        idx2v = pickle.load(open(os.path.join(args.lmdata, args.l1_data_name, 'l1.idx2v.pkl'), 'rb'))
        save_nn_mat(l1_mat, idx2v, os.path.join(args.lmdata, args.l1_data_name))
    else:
        print("skip_l1 preprocessing...")

    if args.do_l2:
        print("l2 preprocessing...")
        save_dir = os.path.join(args.aligned_data, args.l2_data_name, args.l1_data_name)
        preprocess_l2 = Preprocess_L2()
        print("loading word_vec_bin...")
        ft_model = None #fastText.load_model(args.word_vec_bin)
        print("building...")
        l2_v2idx = preprocess_l2.build(l1_data_dir=os.path.join(args.lmdata, args.l1_data_name),
                                       l2_data_dir=os.path.join(args.aligned_data, args.l2_data_name),
                                       l2_save_dir=save_dir,
                                       ft_model=ft_model,
                                       max_word_len=args.max_word_len)
        l1_v2idx = pickle.load(open(os.path.join(args.lmdata, args.l1_data_name, 'l1.v2idx.pkl'), 'rb'))
        #ed_mat = torch.zeros(len(l2_v2idx), len(l1_v2idx))
        #i = 0
        #for l2_v, l2_idx in l2_v2idx.items():
        #    print(i, len(l2_v2idx))
        #    i += 1
        #    for l1_v, l1_idx in l1_v2idx.items():
        #        ed = float(editdistance.eval(l2_v, l1_v)) / float(len(l2_v))
        #        ed_mat[l2_idx, l1_idx] = ed
        #torch.save(ed_mat, os.path.join(args.aligned_data, args.l2_data_name, 'l2.l1.ed.mat.pt'))
    else:
        print("skip_l2 preprocessing...")
