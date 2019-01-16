#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -M adithya.renduchintala@jhu.edu
#$ -m eas
#$ -l hostname=b1[123456789]*|c*|b2*,ram_free=3G,mem_free=3G,h_rt=72:00:00
#$ -r no
source /home/arenduc1/anaconda3/bin/activate /home/arenduc1/anaconda3/envs/pytorch041env
export LC_ALL="en_US.UTF-8"
PROJECT_DIR=/export/b07/arenduc1/macaronic-multi-agent
L1_DATA_NAME=$1
L1_DATA=$PROJECT_DIR/lmdata/${L1_DATA_NAME}
L2_DATA_NAME=$2
L2_DATA=$PROJECT_DIR/aligned_data/${L2_DATA_NAME}
USE_PER_LINE_KEY=0
MAX_STEPS=1
SWAP_LIMIT=1.0
PENALTY=0.0
BINARY_BRANCHING=1
BEAM_SIZE=1
RANDOM_WALK=0
STOCHASTIC=0
SEED=2000
REWARD_TYPE=type_mrr_assist_check
RANK_THRESHOLD=$3 #3, 10, 20
LOSS_TYPE=$4 #mse, huber , cs, cs_margin etc..
ACCUMLUATE_SEEN_L2=0  # only used for penalty
ITERS=3
SEED=2000
BATCH_SIZE=10000
MASK_VAL=0.1
MODEL_SIZE=600
INIT_MODEL_DIR=lt.$LOSS_TYPE.seed.$SEED.bs.$BATCH_SIZE.rnn_size.$MODEL_SIZE.wmask.$MASK_VAL
#TRAINED_MODEL=$PROJECT_DIR/data/grimm_stories_en_es/cbilstm_models/${INIT_MODEL_DIR}/${INIT_MODEL}_model
TRAINED_MODEL=$L1_DATA/l1_mse_models/${INIT_MODEL_DIR}/good_model
#SAVE_DIR=$PROJECT_DIR/data/$DATA
NAME=interactive.out
device=`free-gpu`
python $PROJECT_DIR/interactive_mse.py \
  --gpuid $device \
  --parallel_corpus ${L2_DATA}/parallel_corpus \
  --v2i ${L1_DATA}/l1.v2idx.pkl \
  --gv2i ${L2_DATA}/${L1_DATA_NAME}/l2.v2idx.pkl \
  --cloze_model $TRAINED_MODEL \
  --key ${L2_DATA}/${L1_DATA_NAME}/l1.l2.key.pkl \
  --per_line_key ${L2_DATA}/${L1_DATA_NAME}/per_line.l1.l2.key.pkl \
  --use_per_line_key $USE_PER_LINE_KEY \
  --beam_size $BEAM_SIZE \
  --swap_limit $SWAP_LIMIT \
  --penalty $PENALTY \
  --binary_branching $BINARY_BRANCHING \
  --training_loss_type $LOSS_TYPE \
  --seed $SEED \
  --iters $ITERS \
  --verbose $VERBOSE \
  --accumulate_seen_l2 $ACCUMLUATE_SEEN_L2 \
  --rank_threshold $RANK_THRESHOLD \
  --reward $REWARD_TYPE
#> ${SAVE_DIR}/log 2> ${SAVE_DIR}/err
