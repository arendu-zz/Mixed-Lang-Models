#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -M adithya.renduchintala@jhu.edu
#$ -m eas
#$ -l gpu=1,hostname=c*,ram_free=4G,mem_free=4G,h_rt=72:00:00
#$ -r no
if [ $# -ne 1 2 ]; then
  echo 1>&2 "Usage: $0 lmdir lam"
  exit 3
fi
echo $1 $2

set -e
device=-1 #`free-gpu`
source /home/arenduc1/anaconda3/bin/activate /home/arenduc1/anaconda3/envs/pytorch041env
PROJECT_DIR=/export/b07/arenduc1/zero-shot-code-switch
L1_DATA=$PROJECT_DIR/lmdata/$1
SAVE_DIR=$L1_DATA/lm_models
mkdir -p $SAVE_DIR
ADV_LAMBDA=$2
python $PROJECT_DIR/train.py \
  --gpuid $device \
  --save_dir $SAVE_DIR \
  --adv_lambda ${ADV_LAMBDA} \
  --train_corpus $L1_DATA/corpus.en \
  --dev_corpus $L1_DATA/dev.en \
  --v2i $L1_DATA/l1.v2idx.pkl \
  --adv_labels $L1_DATA/adv_labels.pkl > $PROJECT_DIR/$1.$2.log 2> $PROJECT_DIR/$1.$2.err
