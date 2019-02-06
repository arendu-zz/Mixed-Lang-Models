#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -M adithya.renduchintala@jhu.edu
#$ -m eas
#$ -l ram_free=3G,mem_free=3G,h_rt=72:00:00
#$ -r no
set -e
source /home/arenduc1/anaconda3/bin/activate /home/arenduc1/anaconda3/envs/pytorch041env
PROJECT_DIR=/export/b07/arenduc1/macaronic-multi-agent
REM="" #--do_l1 or --do_l2 or both
N=0
for n in $(seq 1 $#); do
  if [ $N == 0 ]; then
    L1_DATA_NAME=$1
  elif [ $N == 1 ]; then
    L2_DATA_NAME=$1
  else
    REM="$REM $1"
  fi
  N=$((N + 1))
  shift
done
L2_DATA=$PROJECT_DIR/aligned_data/$L2_DATA_NAME
SAVE_DIR=$L2_DATA/$L1_DATA_NAME
mkdir -p $SAVE_DIR
python $PROJECT_DIR/preprocess.py --l1_data_name $L1_DATA_NAME \
                                  --l2_data_name $L2_DATA_NAME \
                                  --word_vec_bin /export/b07/arenduc1/fast-text-vecs/crawl-300d-2M-subword.bin \
                                  --max_vocab 60000 \
                                  --max_word_len 20 \
                                  --lmdata /export/b07/arenduc1/macaronic-multi-agent/lmdata \
                                  --aligned_data /export/b07/arenduc1/macaronic-multi-agent/aligned_data $REM

##if [ $SKIP_L1 == "skip" ]; then
##  echo "reusing preprocessing of $L1_DATA_NAME"
##else
##  L1_DATA=$PROJECT_DIR/lmdata/${L1_DATA_NAME}
##  echo $L1_DATA
##  python $PROJECT_DIR/src/preprocessing/preprocess_l1_corpus.py \
##    --data_dir ${L1_DATA} \
##    --max_word_len 20 \
##    --wordvec_bin  /export/b07/arenduc1/fast-text-vecs/crawl-300d-2M-subword.bin \
##    --max_vocab 60000 > ${L1_DATA}/preprocess.log 2> ${L1_DATA}/preprocess.err
##  python $PROJECT_DIR/src/preprocessing/get_nn.py $L1_DATA_NAME
##fi
##L2_DATA_NAME=$2
##L2_DATA=$PROJECT_DIR/aligned_data/$L2_DATA_NAME
##SAVE_DIR=$L2_DATA/$L1_DATA_NAME
##echo $L2_DATA
##mkdir -p $SAVE_DIR
##python $PROJECT_DIR/src/preprocessing/preprocess_l2_corpus.py \
##  --l1_data_dir ${L1_DATA} \
##  --l2_data_dir ${L2_DATA} \
##  --wordvec_bin /export/b07/arenduc1/fast-text-vecs/crawl-300d-2M-subword.bin \
##  --l2_save_dir $SAVE_DIR > ${SAVE_DIR}/preprocess.log 2> ${SAVE_DIR}/preprocess.err
##
