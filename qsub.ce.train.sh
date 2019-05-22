#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -M adithya.renduchintala@jhu.edu
#$ -m eas
#$ -l gpu=1,hostname=c*,ram_free=4G,mem_free=4G,h_rt=72:00:00
#$ -r no
set -e
if [ $# -ne 6 ]; then
  echo 1>&2 "Usage: $0 L1_DATA_NAME MODEL_SIZE CONTEXT_ENCODER{lm,cloze} CHARAWARE{0,1,2} POOL{Max,Ave,LP} LEARN_MAIN_EMBS{0,1}"
  exit 3
fi
source /home/arenduc1/anaconda3/bin/activate /home/arenduc1/anaconda3/envs/pytorch041env
PROJECT_DIR=/export/b07/arenduc1/macaronic-multi-agent
L1_DATA_NAME=$1 #wikisimple #or grimm
L1_DATA=$PROJECT_DIR/lmdata/${L1_DATA_NAME}
EMBEDDING_PRETRAIN=0
EMBEDDING_SIZE=$2
MODEL_SIZE=$2 #600 #300, 1000
LOSS_TYPE="ce" #$3 #mse, huber, cs
CONTEXT_ENCODER=$3
CHARAWARE=$4
POOL_TYPE=$5
LANG_BIT_RATIO=0
LEARN_MAIN_EMBS=$6
CHKPT=1000
if [ $CONTEXT_ENCODER == 'lm' ]; then
  NUM_LAYERS=2
elif [ $CONTEXT_ENCODER == 'cloze' ]; then
  NUM_LAYERS=1
fi
BATCH_SIZE=5000
EPOCHS=8
device=`free-gpu`
SEED=2000
USE_EARLY_STOP=1
LOSS_AT="all"
NAME=es.$USE_EARLY_STOP.ce.$CONTEXT_ENCODER.seed.$SEED.es.$EMBEDDING_SIZE.rs.$MODEL_SIZE.nl.$NUM_LAYERS.ca.$CHARAWARE.pt.$POOL_TYPE.lbr.$LANG_BIT_RATIO.lme.$LEARN_MAIN_EMBS.epoch.$EPOCHS
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
  --cgram2i $L1_DATA/l1.c1gram2idx.pkl,$L1_DATA/l1.c2gram2idx.pkl,$L1_DATA/l1.c3gram2idx.pkl,$L1_DATA/l1.c4gram2idx.pkl \
  --v2cgramspell $L1_DATA/l1.vidx2c1gram_spelling.pkl,$L1_DATA/l1.vidx2c2gram_spelling.pkl,$L1_DATA/l1.vidx2c3gram_spelling.pkl,$L1_DATA/l1.vidx2c4gram_spelling.pkl \
  --seed $SEED \
  --use_early_stop $USE_EARLY_STOP \
  --context_encoder $CONTEXT_ENCODER \
  --char_aware $CHARAWARE \
  --pool_type $POOL_TYPE \
  --lang_bit_ratio $LANG_BIT_RATIO \
  --learn_main_embs $LEARN_MAIN_EMBS \
  --checkpoint_freq $CHKPT 2>&1 | tee  ${SAVE_DIR}/log
#\rm -f $SAVE_DIR/good_model
#python $PROJECT_DIR/good_and_overfit.py $SAVE_DIR
#python $PROJECT_DIR/dump_embeddings.py --cloze_model $SAVE_DIR/good_model > $SAVE_DIR/l1_emb.txt
source /home/arenduc1/anaconda3/bin/deactivate
head -n50 ${SAVE_DIR}/log > ${SAVE_DIR}/msg
tail ${SAVE_DIR}/log >> ${SAVE_DIR}/msg
cat ${SAVE_DIR}/msg | mail -s "ce-train:$L1_DATA" adithya.renduchintala@jhu.edu
