#!/bin/bash

RUN_NAME="full_training_run"
CONFIG_FILE="config/config_full.yaml"
GPU_IDX=0

python -u train.py -c $CONFIG_FILE -g $GPU_IDX -n $RUN_NAME
