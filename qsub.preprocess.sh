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
L1_DATA=$1
L1_DATA_NAME=`basename "$L1_DATA"`
echo $L1_DATA
echo $L1_DATA_NAME
#python $PROJECT_DIR/src/preprocess_l1_corpus_mse.py \
#  --data_dir ${L1_DATA} \
#  --max_word_len 20 \
#  --wordvecs  /export/b07/arenduc1/fast-text-vecs/crawl-300d-2M-subword.vec \
#  --max_vocab 60000 > ${L1_DATA}/preprocess.log 2> ${L1_DATA}/preprocess.err
#for L2_DATA in "./aligned_data/grimm_en_es/story1" "./aligned_data/metamorphosis" "./aligned_data/metamorphosis_2" "./aligned_data/sense_and_sensibility" "./aligned_data/sense_and_sensibility_2"; do
L2_DATA=$2
echo $L2_DATA
L2_SAVE_FILE=${L2_DATA}/${L1_DATA_NAME}
mkdir -p $L2_SAVE_FILE
python $PROJECT_DIR/src/preprocess_l2_corpus.py \
  --l1_data_dir ${L1_DATA} \
  --l2_data_dir ${L2_DATA} \
  --l2_save_dir ${L2_SAVE_FILE} > ${L2_SAVE_FILE}/preprocess.log 2> ${L2_SAVE_FILE}/preprocess.err
#done
