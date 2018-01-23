#!/bin/bash
set -x

if [ -z "$1" ]
  then
    echo "No argument supplied"
    exit
fi

ROOT_DIR="/ifp/users/haichao/projects/camid"
TRAIN_ID=$1
USE_PRETRAINED=False
MODEL_PATH="$ROOT_DIR/results/exp1"
TRAIN_LIST="$ROOT_DIR/data/dresden/filelist_train"
VAL_LIST="$ROOT_DIR/data/dresden/filelist_valid"
NUM_VAL_DATA=500
LEARNING_RATE=0.015
EPOCHS=100
PY_SCRIPT="train.py"

RESULT_DIR="${ROOT_DIR}/results/$TRAIN_ID"
mkdir "$RESULT_DIR/"
rm -rf $RESULT_DIR/*
LOG_FILE="$RESULT_DIR/log.txt"
CONFIG="$RESULT_DIR/config"
mkdir "$CONFIG"
cp -r ./* "$CONFIG/"

ARGS="--result_dir=$RESULT_DIR --use_pretrained=$USE_PRETRAINED --model_path=$MODEL_PATH --train_list=$TRAIN_LIST --val_list=$VAL_LIST --num_val_data=$NUM_VAL_DATA --num_epochs=$EPOCHS --learning_rate=$LEARNING_RATE"
srun --gres=gpu:1 python -u "$CONFIG/$PY_SCRIPT" $ARGS 2>&1| tee $LOG_FILE
