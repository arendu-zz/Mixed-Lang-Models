#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -M adithya.renduchintala@jhu.edu
#$ -m eas
#$ -l gpu=1,hostname=b1[123456789]*|c0*|c1[123457890]*|b2*,mem_free=3G
#$ -r no
set -e
if [ $# -ne 10 ]; then
  echo 1>&2 "Usage: $0 MODEL_SIZE NUM_LAYER CONTEXT_ENCODER CHARAWARE{0, 1} POOL{Max,Ace,LP,RNN} LANG_BIT_RATIO{0,0.01,0.5} L2_DATA_NAME SEED RANK LSR"
  exit 3
fi
source /home/arenduc1/anaconda3/bin/activate /home/arenduc1/anaconda3/envs/pytorch041env
export LC_ALL="en_US.UTF-8"
PROJECT_DIR=/export/b07/arenduc1/macaronic-multi-agent
L1_DATA_NAME=wiki103
L1_DATA=$PROJECT_DIR/lmdata/${L1_DATA_NAME}
L2_DATA_NAME=$7
L2_DATA=$PROJECT_DIR/aligned_data/${L2_DATA_NAME}
EMBEDDING_PRETRAIN=0
EMBEDDING_SIZE=$1
MODEL_SIZE=$1
NUM_LAYERS=$2
CONTEXT_ENCODER=$3
CHARAWARE=$4
POOL_TYPE=$5
LANG_BIT_RATIO=$6
MAX_STEPS=1
SWAP_LIMIT=1.0
PENALTY=0
USE_PER_LINE_KEY=0
ACCUMLUATE_SEEN_L2=0  # only used for penalty
BINARY_BRANCHING=1
BEAM_SIZE=1
REWARD_TYPE=type_mrr_assist_check
NOISE_PROFILE=1
SEED=$8 #2000
RANK_THRESHOLD=$9
LEARN_STEP_REG=${10}
ITERS=50
LOSS_TYPE="ce"
LOSS_AT="all"
\rm -rf ${PROJECT_DIR}/__pycache__
INIT_MODEL_DIR=es.1.ce.$CONTEXT_ENCODER.seed.2000.es.$EMBEDDING_SIZE.rs.$MODEL_SIZE.nl.$NUM_LAYERS.ca.$CHARAWARE.pt.$POOL_TYPE.lbr.$LANG_BIT_RATIO.epoch.15
#INIT_MODEL_DIR=es.$USE_EARLY_STOP.ce.$CONTEXT_ENCODER.la.$LOSS_AT.seed.2000.es.$EMBEDDING_SIZE.ep.$EMBEDDING_PRETRAIN.rs.$MODEL_SIZE.nl.$NUM_LAYERS.ca.$CHARAWARE
#INIT_MODEL_DIR=ce.$CONTEXT_ENCODER.la.$LOSS_AT.seed.2000.es.$EMBEDDING_SIZE.ep.$EMBEDDING_PRETRAIN.rs.$MODEL_SIZE.nl.$NUM_LAYERS.ca.$CHARAWARE
TRAINED_MODEL=$L1_DATA/l1_ce_models/${INIT_MODEL_DIR}/best.model
NAME=$INIT_MODEL_DIR.$REWARD_TYPE.branching.$BINARY_BRANCHING.beam.$BEAM_SIZE.pen.$PENALTY.plk.$USE_PER_LINE_KEY.acc.$ACCUMLUATE_SEEN_L2.rank.$RANK_THRESHOLD.it.$ITERS.lsr.$LEARN_STEP_REG.ss.$SEED
echo "full log file in"
SAVE_DIR=${L2_DATA}/${L1_DATA_NAME}/ce_search_outputs_tied
mkdir -p $SAVE_DIR
echo "${SAVE_DIR}/${NAME}"
device=`free-gpu`
python $PROJECT_DIR/search_ce.py \
  --gpuid $device \
  --parallel_corpus ${L2_DATA}/parallel_corpus \
  --v2i ${L1_DATA}/l1.v2idx.pkl \
  --v2spell $L1_DATA/l1.vidx2spelling.pkl \
  --c2i $L1_DATA/l1.c2idx.pkl \
  --gv2i ${L2_DATA}/${L1_DATA_NAME}/l2.v2idx.pkl \
  --gv2spell ${L2_DATA}/${L1_DATA_NAME}/l2.vidx2spelling.pkl \
  --gc2i ${L2_DATA}/${L1_DATA_NAME}/l2.c2idx.pkl \
  --cloze_model $TRAINED_MODEL \
  --key ${L2_DATA}/${L1_DATA_NAME}/l1.l2.key.pkl \
  --key_wt ${L2_DATA}/${L1_DATA_NAME}/l2.key.wt.pkl \
  --use_key_wt 1 \
  --per_line_key ${L2_DATA}/${L1_DATA_NAME}/per_line.l1.l2.key.pkl \
  --use_per_line_key $USE_PER_LINE_KEY \
  --beam_size $BEAM_SIZE \
  --swap_limit $SWAP_LIMIT \
  --penalty $PENALTY \
  --iters $ITERS \
  --learn_step_reg $LEARN_STEP_REG \
  --binary_branching $BINARY_BRANCHING \
  --reward $REWARD_TYPE \
  --rank_threshold $RANK_THRESHOLD \
  --char_aware $CHARAWARE \
  --seed $SEED \
  --accumulate_seen_l2 $ACCUMLUATE_SEEN_L2 \
  --search_output_prefix  ${SAVE_DIR}/$NAME \
  --verbose $VERBOSE #> ${SAVE_DIR}/$NAME.log 2> ${SAVE_DIR}/$NAME.err
#source /home/arenduc1/anaconda3/bin/deactivate
#echo "file:${SAVE_DIR}/${NAME}.log" | mail -s "ce-macaronic-search: $NAME" adithya.renduchintala@jhu.edu
