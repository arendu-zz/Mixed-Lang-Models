#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -M adithya.renduchintala@jhu.edu
#$ -m eas
#$ -l gpu=1,hostname=c*,ram_free=4G,mem_free=4G,h_rt=72:00:00
#$ -r no
set -e
source /home/arenduc1/anaconda3/bin/activate /home/arenduc1/anaconda3/envs/pytorch041env
PROJECT_DIR=/export/b07/arenduc1/macaronic-multi-agent
DATA=$1
MODEL_SIZE=$2
LOSS_TYPE=$3
EPOCHS=$4
BATCH_SIZE=10000
MASK_VAL=0.1
device=`free-gpu`
SEED=2000
USE_EARLY_STOP=0
NAME=lt.$LOSS_TYPE.seed.$SEED.bs.$BATCH_SIZE.rnn_size.$MODEL_SIZE.wmask.$MASK_VAL
SAVE_DIR=$PROJECT_DIR/$DATA/l1_mse_models/$NAME
mkdir -p $SAVE_DIR
\rm -rf $PROJECT_DIR/__pycache__
echo "detailed log files are in:$SAVE_DIR/log"
python $PROJECT_DIR/train_mse.py \
  --gpuid $device \
  --save_dir $SAVE_DIR \
  --train_corpus $PROJECT_DIR/$DATA/corpus.en \
  --dev_corpus $PROJECT_DIR/$DATA/dev.en \
  --batch_size $BATCH_SIZE \
  --model_size $MODEL_SIZE \
  --loss_type $LOSS_TYPE \
  --epochs $EPOCHS \
  --v2i $PROJECT_DIR/$DATA/l1.v2idx.pkl \
  --vmat $PROJECT_DIR/$DATA/l1.mat.pt \
  --v2spell $PROJECT_DIR/$DATA/l1.vidx2spelling.pkl \
  --c2i $PROJECT_DIR/$DATA/l1.c2idx.pkl \
  --seed $SEED \
  --use_early_stop $USE_EARLY_STOP \
  --mask_val $MASK_VAL > ${SAVE_DIR}/log 2> ${SAVE_DIR}/err
\rm -f $SAVE_DIR/good_model
\rm -f $SAVE_DIR/overfit_model
python $PROJECT_DIR/good_and_overfit.py $SAVE_DIR
python $PROJECT_DIR/dump_embeddings.py --cloze_model $SAVE_DIR/good_model > $SAVE_DIR/l1_emb.txt
source /home/arenduc1/anaconda3/bin/deactivate
grep 'Ending' ${SAVE_DIR}/log | mail -s "mse-train: $DATA $NAME" adithya.renduchintala@jhu.edu
