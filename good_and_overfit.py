#!/usr/bin/env python
__author__ = 'arenduchintala'
import os
import glob
import sys

ppl_list = []
log = []
max_epoch = 0
for file in glob.glob(sys.argv[1] + "/e_*.model"):
    base_file = os.path.basename(file)
    epoch = int(base_file.split('_')[1])
    dev_loss = float(base_file.split('_')[7])
    acc = float(base_file.split('_')[-1].replace('.model', ''))
    ppl_list.append((-acc, dev_loss, file, epoch))
    log.append((epoch, -acc, dev_loss))
    max_epoch = epoch if epoch > max_epoch else max_epoch


log_file = open(sys.argv[1] + "/training.ppl.log", 'w', encoding='utf-8')
log.sort()
for e, a, dl in log:
    log_file.write("epoch:" + str(e) + " acc:" + str(-a) + " dev_los:" + str(dl) + '\n')
log_file.close()

print(sys.argv[1])
ppl_list.sort()
print(ppl_list[0])
for i in range(len(ppl_list)):
    if i == 0:
        os.symlink(ppl_list[i][2], sys.argv[1] + '/good_model')
    elif ppl_list[i][2] == max_epoch:
        os.symlink(ppl_list[i][2], sys.argv[1] + '/overfit_model')
    else:
        os.remove(ppl_list[i][2])
