#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -M adithya.renduchintala@jhu.edu
#$ -m eas
#$ -l gpu=1,hostname=b1[123456789]*|c*|b2*,mem_free=3G
#$ -r no
source /home/arenduc1/anaconda3/bin/activate /home/arenduc1/anaconda3/envs/pytorch041env
export LC_ALL="en_US.UTF-8"
PROJECT_DIR=/export/b07/arenduc1/macaronic-multi-agent
L1_DATA_NAME=$1
L1_DATA=$PROJECT_DIR/lmdata/${L1_DATA_NAME}
L2_DATA_NAME=$2
L2_DATA=$PROJECT_DIR/aligned_data/${L2_DATA_NAME}
MODEL_SIZE=$3
MAX_STEPS=1
SWAP_LIMIT=1.0
PENALTY=0
USE_PER_LINE_KEY=0
ACCUMLUATE_SEEN_L2=0  # only used for penalty
BINARY_BRANCHING=1
BEAM_SIZE=1
MAX_SENT=5000
REWARD_TYPE=type_mrr_assist_check
RANK_THRESHOLD=3 #, 20
ITERS=3
USE_ORTHOGRAPHIC_MODEL=$4
USE_RAND_HIDDENS=$5
NUM_HIGHWAYS=$6
CONTEXT_ENCODER=$7 #'Attention' # or 'Attention'

LOSS_TYPE="huber" #mse, huber , cs, cs_margin etc..
SEED=2000
BATCH_SIZE=10000
MASK_VAL=0.1
INIT_MODEL_DIR=lt.$LOSS_TYPE.seed.$SEED.bs.$BATCH_SIZE.rs.$MODEL_SIZE.cmask.$MASK_VAL.o.$USE_ORTHOGRAPHIC_MODEL.rh.$USE_RAND_HIDDENS.nh.$NUM_HIGHWAYS.ce.$CONTEXT_ENCODER
#TRAINED_MODEL=$PROJECT_DIR/data/grimm_stories_en_es/cbilstm_models/${INIT_MODEL_DIR}/${INIT_MODEL}_model
TRAINED_MODEL=$L1_DATA/l1_mse_models/${INIT_MODEL_DIR}/good_model
#SAVE_DIR=$PROJECT_DIR/data/$DATA
NAME=$INIT_MODEL_DIR.$REWARD_TYPE.branching.$BINARY_BRANCHING.beam.$BEAM_SIZE.pen.$PENALTY.plk.$USE_PER_LINE_KEY.acc.$ACCUMLUATE_SEEN_L2.rank.$RANK_THRESHOLD
echo "full log file in"
echo "${L2_DATA}/${L1_DATA_NAME}/search_outputs/$NAME"
mkdir -p ${L2_DATA}/${L1_DATA_NAME}/search_outputs
device=`free-gpu`
python $PROJECT_DIR/search_mse.py \
  --gpuid $device \
  --parallel_corpus ${L2_DATA}/parallel_corpus \
  --v2i ${L1_DATA}/l1.v2idx.pkl \
  --gv2i ${L2_DATA}/${L1_DATA_NAME}/l2.v2idx.pkl \
  --cloze_model $TRAINED_MODEL \
  --l2_init_weights ${L2_DATA}/${L1_DATA_NAME}/l2.mat.pt \
  --key ${L2_DATA}/${L1_DATA_NAME}/l1.l2.key.pkl \
  --per_line_key ${L2_DATA}/${L1_DATA_NAME}/per_line.l1.l2.key.pkl \
  --use_per_line_key $USE_PER_LINE_KEY \
  --beam_size $BEAM_SIZE \
  --swap_limit $SWAP_LIMIT \
  --penalty $PENALTY \
  --iters $ITERS \
  --training_loss_type $LOSS_TYPE \
  --binary_branching $BINARY_BRANCHING \
  --reward $REWARD_TYPE \
  --rank_threshold $RANK_THRESHOLD \
  --seed $SEED \
  --max_sentences $MAX_SENT \
  --accumulate_seen_l2 $ACCUMLUATE_SEEN_L2 \
  --search_output_prefix  ${L2_DATA}/${L1_DATA_NAME}/search_outputs/$NAME \
  --verbose $VERBOSE #> ${L2_DATA}/${L1_DATA_NAME}/search_outputs/$NAME.log 2> ${L2_DATA}/${L1_DATA_NAME}/search_outputs/$NAME.err
#source /home/arenduc1/anaconda3/bin/deactivate
#echo "file:${L2_DATA}/${L1_DATA_NAME}/${NAME}.log" | mail -s "mse-macaronic-search: $NAME" adithya.renduchintala@jhu.edu
