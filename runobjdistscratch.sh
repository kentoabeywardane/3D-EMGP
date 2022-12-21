#!/bin/bash

export PYTHON=/data/people/kabeywar/envs/anaconda3/envs/pyg/bin/python
export CONFIG=config/finetune_chiral_true_dist_scratch.yml
export SCRIPT=script/finetune_chiral_true.py
export EPOCHS=150

$PYTHON -u $SCRIPT --config_path $CONFIG --epochs $EPOCHS
