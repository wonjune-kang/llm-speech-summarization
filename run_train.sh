#!/bin/bash

RUN_NAME="llama3_hubert_full"
CONFIG_FILE="config/llama3_hubert.yaml"
GPU_IDX=0

python -u train.py -c $CONFIG_FILE -g $GPU_IDX -n $RUN_NAME
