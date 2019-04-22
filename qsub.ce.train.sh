#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -M adithya.renduchintala@jhu.edu
#$ -m eas
#$ -l gpu=1,hostname=c0*|c1[123457890]*|b1[123456789]*,ram_free=4G,mem_free=4G,h_rt=72:00:00
#$ -r no
set -e
if [ $# -ne 8 ]; then
  echo 1>&2 "Usage: $0 L1_DATA_NAME EMBEDDING_PRETRAIN EMBEDDING_SIZE MODEL_SIZE NUM_LAYERS CONTEXT_ENCODER CHKPT BS"
  exit 3
fi
source /home/arenduc1/anaconda3/bin/activate /home/arenduc1/anaconda3/envs/pytorch041env
PROJECT_DIR=/export/b07/arenduc1/macaronic-multi-agent
L1_DATA_NAME=$1 #wiki103 or grimm
L1_DATA=$PROJECT_DIR/lmdata/${L1_DATA_NAME}
EMBEDDING_PRETRAIN=$2
EMBEDDING_SIZE=$3
MODEL_SIZE=$4 #600 #300, 1000
NUM_LAYERS=$5
LOSS_TYPE="ce" #$3 #mse, huber, cs
CONTEXT_ENCODER=$6
CHKPT=$7
BATCH_SIZE=$8
EPOCHS=50
device=`free-gpu`
SEED=2000
USE_EARLY_STOP=1
LOSS_AT="all"
NAME=ce.$CONTEXT_ENCODER.la.$LOSS_AT.seed.$SEED.es.$EMBEDDING_SIZE.ep.$EMBEDDING_PRETRAIN.rs.$MODEL_SIZE.nl.$NUM_LAYERS
SAVE_DIR=$L1_DATA/l1_ce_models/$NAME
mkdir -p $SAVE_DIR
\rm -rf $PROJECT_DIR/__pycache__
echo "detailed log files are in:$SAVE_DIR/log"
python $PROJECT_DIR/train_ce.py \
  --gpuid $device \
  --save_dir $SAVE_DIR \
  --train_corpus $L1_DATA/corpus.en \
  --dev_corpus $L1_DATA/dev.en \
  --batch_size $BATCH_SIZE \
  --model_size $MODEL_SIZE \
  --embedding_size $EMBEDDING_SIZE \
  --embedding_pretrain $EMBEDDING_PRETRAIN \
  --num_layers $NUM_LAYERS \
  --loss_at $LOSS_AT \
  --epochs $EPOCHS \
  --v2i $L1_DATA/l1.v2idx.pkl \
  --vmat $L1_DATA/l1.mat.pt \
  --v2spell $L1_DATA/l1.vidx2spelling.pkl \
  --c2i $L1_DATA/l1.c2idx.pkl \
  --seed $SEED \
  --use_early_stop $USE_EARLY_STOP \
  --context_encoder $CONTEXT_ENCODER \
  --checkpoint_freq $CHKPT > ${SAVE_DIR}/log 2> ${SAVE_DIR}/err
#\rm -f $SAVE_DIR/good_model
#python $PROJECT_DIR/good_and_overfit.py $SAVE_DIR
#python $PROJECT_DIR/dump_embeddings.py --cloze_model $SAVE_DIR/good_model > $SAVE_DIR/l1_emb.txt
source /home/arenduc1/anaconda3/bin/deactivate
tail ${SAVE_DIR}/log | mail -s "ce-train: $L1_DATA $NAME" adithya.renduchintala@jhu.edu
