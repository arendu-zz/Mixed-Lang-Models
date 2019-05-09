#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -M adithya.renduchintala@jhu.edu
#$ -m eas
#$ -l gpu=1,hostname=b1[123456789]*|c0*|c1[123457890]*|b2*,mem_free=3G
#$ -r no
set -e
if [ $# -ne 4 ]; then
  echo 1>&2 "Usage: $0 L1_DATA_NAME MODEL_SIZE L2_DATA_NAME RANK_THRESHOLD"
  exit 3
fi
source /home/arenduc1/anaconda3/bin/activate /home/arenduc1/anaconda3/envs/pytorch041env
export LC_ALL="en_US.UTF-8"
PROJECT_DIR=/export/b07/arenduc1/macaronic-multi-agent
L1_DATA_NAME=$1
L1_DATA=$PROJECT_DIR/lmdata/${L1_DATA_NAME}
L2_DATA_NAME=$3
L2_DATA=$PROJECT_DIR/aligned_data/${L2_DATA_NAME}
MODEL_SIZE=$2
MAX_STEPS=1
SWAP_LIMIT=1.0
PENALTY=0
USE_PER_LINE_KEY=0
ACCUMLUATE_SEEN_L2=0  # only used for penalty
BINARY_BRANCHING=1
BEAM_SIZE=1
MAX_SENT=5000
REWARD_TYPE=type_mrr_assist_check
USE_ORTHOGRAPHIC_MODEL=0
USE_RAND_HIDDENS=0
NOISE_PROFILE=1
RANK_THRESHOLD=$4
ITERS=5
NUM_HIGHWAYS=2
CONTEXT_ENCODER=RNN #'Attention' # or 'Attention'

LOSS_TYPE="mse" ##"huber" #mse, huber , cs, cs_margin etc..
LOSS_AT=all
SEED=2000
BATCH_SIZE=10000
MASK_VAL=0.1
L2_INIT_WEIGHTS="zero"
INIT_MODEL_DIR=lt.$LOSS_TYPE.la.$LOSS_AT.rs.$MODEL_SIZE.nh.$NUM_HIGHWAYS.ce.$CONTEXT_ENCODER.rh.$USE_RAND_HIDDENS.np.$NOISE_PROFILE
#INIT_MODEL_DIR=lt.$LOSS_TYPE.la.$LOSS_AT.seed.$SEED.bs.$BATCH_SIZE.rs.$MODEL_SIZE.cmask.$MASK_VAL.nh.$NUM_HIGHWAYS.ce.$CONTEXT_ENCODER.rh.$USE_RAND_HIDDENS.np.$NOISE_PROFILE
#TRAINED_MODEL=$PROJECT_DIR/data/grimm_stories_en_es/cbilstm_models/${INIT_MODEL_DIR}/${INIT_MODEL}_model
TRAINED_MODEL=$L1_DATA/l1_mse_models/${INIT_MODEL_DIR}/good_model
#SAVE_DIR=$PROJECT_DIR/data/$DATA
NAME=$INIT_MODEL_DIR.$REWARD_TYPE.branching.$BINARY_BRANCHING.beam.$BEAM_SIZE.pen.$PENALTY.plk.$USE_PER_LINE_KEY.acc.$ACCUMLUATE_SEEN_L2.rank.$RANK_THRESHOLD.it.$ITERS
echo "full log file in"
SAVE_DIR=${L2_DATA}/${L1_DATA_NAME}/mse_search_outputs
mkdir -p $SAVE_DIR
echo "${L2_DATA}/${L1_DATA_NAME}/mse_search_outputs/${NAME}"
device=`free-gpu`
python $PROJECT_DIR/search_mse.py \
  --gpuid $device \
  --parallel_corpus ${L2_DATA}/parallel_corpus \
  --v2i ${L1_DATA}/l1.v2idx.pkl \
  --gv2i ${L2_DATA}/${L1_DATA_NAME}/l2.v2idx.pkl \
  --cloze_model $TRAINED_MODEL \
  --l2_init_weights ${L2_DATA}/${L1_DATA_NAME}/${L2_INIT_WEIGHTS} \
  --key ${L2_DATA}/${L1_DATA_NAME}/l1.l2.key.pkl \
  --key_wt ${L2_DATA}/${L1_DATA_NAME}/l2.key.wt.pkl \
  --use_key_wt 1 \
  --per_line_key ${L2_DATA}/${L1_DATA_NAME}/per_line.l1.l2.key.pkl \
  --use_per_line_key $USE_PER_LINE_KEY \
  --beam_size $BEAM_SIZE \
  --swap_limit $SWAP_LIMIT \
  --penalty $PENALTY \
  --iters $ITERS \
  --binary_branching $BINARY_BRANCHING \
  --reward $REWARD_TYPE \
  --rank_threshold $RANK_THRESHOLD \
  --seed $SEED \
  --max_sentences $MAX_SENT \
  --accumulate_seen_l2 $ACCUMLUATE_SEEN_L2 \
  --search_output_prefix  ${SAVE_DIR}/$NAME \
  --verbose $VERBOSE #> ${SAVE_DIR}/$NAME.log 2> ${SAVE_DIR}/$NAME.err
#source /home/arenduc1/anaconda3/bin/deactivate
#echo "file:${SAVE_DIR}/${NAME}.log" | mail -s "mse-macaronic-search: $NAME" adithya.renduchintala@jhu.edu
