#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -M adithya.renduchintala@jhu.edu
#$ -m eas
#$ -l gpu=1,hostname=c0*|c1[123457890]*|b1[123456789]*,ram_free=4G,mem_free=4G,h_rt=72:00:00
#$ -r no
set -e
if [ $# -ne 4 ]; then
  echo 1>&2 "Usage: $0 L1_DATA_NAME MODEL_SIZE USE_RAND_HIDDENS NOISE_PROFILE"
  exit 3
fi
source /home/arenduc1/anaconda3/bin/activate /home/arenduc1/anaconda3/envs/pytorch041env
PROJECT_DIR=/export/b07/arenduc1/macaronic-multi-agent
L1_DATA_NAME=$1 #wiki103 or grimm
L1_DATA=$PROJECT_DIR/lmdata/${L1_DATA_NAME}
MODEL_SIZE=$2 #600 #300, 1000
LOSS_TYPE=mse #$3 #mse, huber, cs
EPOCHS=15
BATCH_SIZE=10000
MASK_VAL=0.1
device=`free-gpu`
SEED=2000
USE_EARLY_STOP=0
USE_ORTHOGRAPHIC_MODEL=0
USE_RAND_HIDDENS=$3
NOISE_PROFILE=$4
LOSS_AT=all
NUM_HIGHWAYS=2
CONTEXT_ENCODER=RNN # or 'Attention' or RNN
NAME=lt.$LOSS_TYPE.la.$LOSS_AT.rs.$MODEL_SIZE.nh.$NUM_HIGHWAYS.ce.$CONTEXT_ENCODER.rh.$USE_RAND_HIDDENS.np.$NOISE_PROFILE
SAVE_DIR=$L1_DATA/l1_mse_models/$NAME
mkdir -p $SAVE_DIR
\rm -rf $PROJECT_DIR/__pycache__
echo "detailed log files are in:$SAVE_DIR/log"
python $PROJECT_DIR/train_mse.py \
  --gpuid $device \
  --save_dir $SAVE_DIR \
  --train_corpus $L1_DATA/corpus.en \
  --dev_corpus $L1_DATA/dev.en \
  --batch_size $BATCH_SIZE \
  --context_encoder_type $CONTEXT_ENCODER \
  --model_size $MODEL_SIZE \
  --loss_type $LOSS_TYPE \
  --loss_at $LOSS_AT \
  --epochs $EPOCHS \
  --v2i $L1_DATA/l1.v2idx.pkl \
  --vmat $L1_DATA/l1.mat.pt \
  --nn_mat $L1_DATA/l1.nn.pt \
  --v2spell $L1_DATA/l1.vidx2spelling.pkl \
  --c2i $L1_DATA/l1.c2idx.pkl \
  --seed $SEED \
  --use_early_stop $USE_EARLY_STOP \
  --use_rand_hiddens $USE_RAND_HIDDENS \
  --use_orthographic_model $USE_ORTHOGRAPHIC_MODEL \
  --num_highways $NUM_HIGHWAYS \
  --noise_profile $NOISE_PROFILE \
  --mask_val $MASK_VAL > ${SAVE_DIR}/log 2> ${SAVE_DIR}/err
\rm -f $SAVE_DIR/good_model
python $PROJECT_DIR/good_and_overfit.py $SAVE_DIR
#python $PROJECT_DIR/dump_embeddings.py --cloze_model $SAVE_DIR/good_model > $SAVE_DIR/l1_emb.txt
source /home/arenduc1/anaconda3/bin/deactivate
grep 'Ending' ${SAVE_DIR}/log | mail -s "mse-train: $L1_DATA $NAME" adithya.renduchintala@jhu.edu
