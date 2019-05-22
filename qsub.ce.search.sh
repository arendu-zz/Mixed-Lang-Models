#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -M adithya.renduchintala@jhu.edu
#$ -m eas
#$ -l gpu=1,hostname=b1[123456789]*|c0*|c1[123457890]*|b2*,mem_free=3G
#$ -r no
set -e
if [ $# -ne 7 ]; then
  echo 1>&2 "Usage: $0 POOL{Max,Ace,LP,RNN} LEARN_MAIN_EMBS L2_DATA_NAME INIT_L2_WITH_L1_SCALE SEED INIT_L2_WITH_L1_TYPE BEAM"
  exit 3
fi
source /home/arenduc1/anaconda3/bin/activate /home/arenduc1/anaconda3/envs/pytorch041env
export LC_ALL="en_US.UTF-8"
PROJECT_DIR=/export/b07/arenduc1/macaronic-multi-agent
L1_DATA_NAME=wiki103_large_voc
L1_DATA=$PROJECT_DIR/lmdata/${L1_DATA_NAME}
L2_DATA_NAME=$3
L2_DATA=$PROJECT_DIR/aligned_data/${L2_DATA_NAME}
EMBEDDING_PRETRAIN=0
EMBEDDING_SIZE=300
MODEL_SIZE=300
CONTEXT_ENCODER="cloze"
CHARAWARE=2
if [ $CONTEXT_ENCODER == 'lm' ]; then
  NUM_LAYERS=2
elif [ $CONTEXT_ENCODER == 'cloze' ]; then
  NUM_LAYERS=1
fi
POOL_TYPE=$1
LANG_BIT_RATIO=0
LEARN_MAIN_EMBS=$2
MAX_STEPS=1
SWAP_LIMIT=1.0
PENALTY=0
USE_PER_LINE_KEY=0
ACCUMLUATE_SEEN_L2=0  # only used for penalty
BINARY_BRANCHING=1
REWARD_TYPE=type_mrr_assist_check
NOISE_PROFILE=1
INIT_L2_WITH_L1_SCALE=$4
SEED=$5
RANK_THRESHOLD=4
LEARN_STEP_REG=1.0
ZERO_REG=0.0
REG_TYPE=mse
ITERS=10
GRAD_NORM=5.0
INIT_L2_WITH_L1_TYPE=$6 #init_subwords
BEAM_SIZE=$7
IDF=0
PW=0
IR=0
LOSS_TYPE="ce"
LOSS_AT="all"
#NAME=es.$USE_EARLY_STOP.ce.$CONTEXT_ENCODER.seed.$SEED.es.$EMBEDDING_SIZE.rs.$MODEL_SIZE.nl.$NUM_LAYERS.ca.$CHARAWARE.pt.$POOL_TYPE.lbr.$LANG_BIT_RATIO.lme.$LEARN_MAIN_EMBS.epoch.$EPOCHS
INIT_MODEL_DIR=es.1.ce.$CONTEXT_ENCODER.seed.2000.es.$EMBEDDING_SIZE.rs.$MODEL_SIZE.nl.$NUM_LAYERS.ca.$CHARAWARE.pt.$POOL_TYPE.lbr.0.lme.$LEARN_MAIN_EMBS.epoch.8
#INIT_MODEL_DIR=es.$USE_EARLY_STOP.ce.$CONTEXT_ENCODER.la.$LOSS_AT.seed.2000.es.$EMBEDDING_SIZE.ep.$EMBEDDING_PRETRAIN.rs.$MODEL_SIZE.nl.$NUM_LAYERS.ca.$CHARAWARE
#INIT_MODEL_DIR=ce.$CONTEXT_ENCODER.la.$LOSS_AT.seed.2000.es.$EMBEDDING_SIZE.ep.$EMBEDDING_PRETRAIN.rs.$MODEL_SIZE.nl.$NUM_LAYERS.ca.$CHARAWARE
TRAINED_MODEL=$L1_DATA/l1_ce_models/${INIT_MODEL_DIR}/best.model
NAME=$INIT_MODEL_DIR.$REWARD_TYPE.branching.$BINARY_BRANCHING.beam.$BEAM_SIZE.pen.$PENALTY.plk.$USE_PER_LINE_KEY.acc.$ACCUMLUATE_SEEN_L2.rank.$RANK_THRESHOLD.it.$ITERS.lsr.$LEARN_STEP_REG.zr.$ZERO_REG.rt.$REG_TYPE.ss.$SEED.lme.$LEARN_MAIN_EMBS.is.${INIT_L2_WITH_L1_SCALE}.${INIT_L2_WITH_L1_TYPE}.idf.$IDF.ir.$IR.pw.$PW
echo "full log file in"
SAVE_DIR=${L2_DATA}/${L1_DATA_NAME}/ce_search_outputs_tied
mkdir -p $SAVE_DIR
echo "${SAVE_DIR}/${NAME}"
device=`free-gpu`
\rm -rf ${PROJECT_DIR}/__pycache__
python $PROJECT_DIR/search_ce.py \
  --gpuid $device \
  --parallel_corpus ${L2_DATA}/parallel_corpus \
  --v2i ${L1_DATA}/l1.v2idx.pkl \
  --l1_cgram2i $L1_DATA/l1.c1gram2idx.pkl,$L1_DATA/l1.c2gram2idx.pkl,$L1_DATA/l1.c3gram2idx.pkl,$L1_DATA/l1.c4gram2idx.pkl \
  --l1_v2cgramspell $L1_DATA/l1.vidx2c1gram_spelling.pkl,$L1_DATA/l1.vidx2c2gram_spelling.pkl,$L1_DATA/l1.vidx2c3gram_spelling.pkl,$L1_DATA/l1.vidx2c4gram_spelling.pkl \
  --gv2i ${L2_DATA}/${L1_DATA_NAME}/l2.v2idx.pkl \
  --l2_cgram2i ${L2_DATA}/${L1_DATA_NAME}/l2.c1gram2idx.pkl,${L2_DATA}/${L1_DATA_NAME}/l2.c2gram2idx.pkl,${L2_DATA}/${L1_DATA_NAME}/l2.c3gram2idx.pkl,${L2_DATA}/${L1_DATA_NAME}/l2.c4gram2idx.pkl \
  --l2_v2cgramspell ${L2_DATA}/${L1_DATA_NAME}/l2.vidx2c1gram_spelling.pkl,${L2_DATA}/${L1_DATA_NAME}/l2.vidx2c2gram_spelling.pkl,${L2_DATA}/${L1_DATA_NAME}/l2.vidx2c3gram_spelling.pkl,${L2_DATA}/${L1_DATA_NAME}/l2.vidx2c4gram_spelling.pkl \
  --l2_v2cgramspell_by_l1 ${L2_DATA}/${L1_DATA_NAME}/l2.vidx2c1gram_by_l1_spelling.pkl,${L2_DATA}/${L1_DATA_NAME}/l2.vidx2c2gram_by_l1_spelling.pkl,${L2_DATA}/${L1_DATA_NAME}/l2.vidx2c3gram_by_l1_spelling.pkl,${L2_DATA}/${L1_DATA_NAME}/l2.vidx2c4gram_by_l1_spelling.pkl \
  --cloze_model $TRAINED_MODEL \
  --key ${L2_DATA}/${L1_DATA_NAME}/l1.l2.key.pkl \
  --key_wt ${L2_DATA}/${L1_DATA_NAME}/l2.key.wt.pkl \
  --use_key_wt $IDF \
  --per_line_key ${L2_DATA}/${L1_DATA_NAME}/per_line.l1.l2.key.pkl \
  --use_per_line_key $USE_PER_LINE_KEY \
  --beam_size $BEAM_SIZE \
  --swap_limit $SWAP_LIMIT \
  --penalty $PENALTY \
  --init_l2_with_l1_scale $INIT_L2_WITH_L1_SCALE \
  --init_l2_with_l1 $INIT_L2_WITH_L1_TYPE \
  --iters $ITERS \
  --learn_step_reg $LEARN_STEP_REG \
  --zero_reg $ZERO_REG \
  --reg_type $REG_TYPE \
  --binary_branching $BINARY_BRANCHING \
  --reward $REWARD_TYPE \
  --rank_threshold $RANK_THRESHOLD \
  --char_aware $CHARAWARE \
  --grad_norm $GRAD_NORM \
  --init_range $IR \
  --l2_pos_weighted $PW \
  --seed $SEED \
  --accumulate_seen_l2 $ACCUMLUATE_SEEN_L2 \
  --search_output_prefix  ${SAVE_DIR}/$NAME \
  --verbose $VERBOSE > ${SAVE_DIR}/$NAME.log 2> ${SAVE_DIR}/$NAME.err
source /home/arenduc1/anaconda3/bin/deactivate
echo "file:${SAVE_DIR}/${NAME}.log"  > ${SAVE_DIR}/msg
#tail -n100 ${SAVE_DIR}/${NAME}.log >> ${SAVE_DIR}/msg
SEARCHSCORE=`grep ^score ${SAVE_DIR}/${NAME}.log`
SWAPTYPES=`grep ^swap_types ${SAVE_DIR}/${NAME}.log`
SWAPTOKENS=`grep ^swap_token_count ${SAVE_DIR}/${NAME}.log`
echo $SEARCHSCORE >>  ${SAVE_DIR}/msg
echo $SWAPTYPES >>  ${SAVE_DIR}/msg
echo $SWAPTOKENS >>  ${SAVE_DIR}/msg
echo "--------------------" >> ${SAVE_DIR}/msg
#head -n 50 ${SAVE_DIR}/${NAME}.log >> ${SAVE_DIR}/msg
cat ${SAVE_DIR}/msg | mail -s "new 0 init ce-macaronic-search result" adithya.renduchintala@jhu.edu
