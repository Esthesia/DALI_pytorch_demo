#!/bin/bash

CUR_DIR="."
LOG_DIR="$CUR_DIR/logging"

WORKER=$1
EPOCHS=$2
BATCH=$3
DATA=$4
MODEL='resnet50'

DATA_DIR="$CUR_DIR/$DATA"

# python3 main.py -j $WORKER --epochs $EPOCHS ./size2 --arch='resnet50' --use-dali --dali-cpu --batch-size 1 > CPU1Batch.log
python3 main.py -j ${WORKER} --epochs ${EPOCHS} ${DATA_DIR} --arch=${MODEL} --use-dali --batch-size ${BATCH} > ${LOG_DIR}/CPU${BATCH}Batch.log &
python_pid=$!
python3 gpu_util_logger.py ${LOG_DIR}/Util${BATCH}Batch.csv &
wait $python_pid

pkill -9 -ef "logger"
