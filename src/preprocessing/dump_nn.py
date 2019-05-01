#!/usr/bin/env python
import torch
import pickle
from src.rewards import get_nearest_neighbors_simple
import sys
import os


def save_nn_mat(vmat, idx2v, data_dir):
    r = 5000
    nn_list = []
    nn_txt = open(os.path.join(data_dir + 'l1.nn.txt'), 'w')
    for i in range(int(vmat.shape[0] // r) + 1):
        min_ = r * i
        max_ = r * (i + 1)
        max_ = vmat.shape[0] if max_ > vmat.shape[0] else max_
        print(min_, max_)
        f = get_nearest_neighbors_simple(vmat[min_:max_], vmat, 51)
        for row in range(f.shape[0]):
            t = [idx2v[i_] for i_ in f[row, :].tolist()]
            #print(' '.join(t))
            nn_txt.write(' '.join(t) + '\n')
        nn_txt.flush()
        f = f[:, 1:]
        nn_list.append(f)
    nn_txt.close()
    nn_mat = torch.cat(nn_list, dim=0)
    torch.save(nn_mat, os.path.join(data_dir, 'l1.nn.pt'))


if __name__ == '__main__':
    l1_data = "/export/b07/arenduc1/macaronic-multi-agent/lmdata/" + sys.argv[1]
    vmat = torch.load(l1_data + '/l1.mat.pt')
    idx2v = pickle.load(open(l1_data + '/l1.idx2v.pkl', 'rb'))
    save_nn_mat(vmat, idx2v, l1_data)
