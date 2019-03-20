#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -M adithya.renduchintala@jhu.edu
#$ -m eas
#$ -l ram_free=3G,mem_free=3G,h_rt=72:00:00
#$ -r no
set -e
if [ $# -ne 1 ]; then
  echo 1>&2 "Usage: $0 L1_DATA_NAME"
  exit 3
fi
echo $1 $2

source /home/arenduc1/anaconda3/bin/activate /home/arenduc1/anaconda3/envs/pytorch041env
PROJECT_DIR=/export/b07/arenduc1/zero-shot-code-switch
L1_DATA_NAME=$1
python $PROJECT_DIR/preprocess.py --l1_data_name $L1_DATA_NAME \
                                  --max_vocab 60000 \
                                  --max_word_len 20 \
                                  --lmdata $PROJECT_DIR/lmdata

