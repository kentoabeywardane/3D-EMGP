#!/bin/bash

export PYTHON=/data/people/kabeywar/envs/anaconda3/envs/pyg/bin/python
export CONFIG=config/finetune_drugs.yml
export SCRIPT=script/finetune_drugs.py
export RESTORE=checkpoints/checkpoint-3dmgp
export EPOCHS=1000

$PYTHON -u $SCRIPT --config_path $CONFIG --restore_path $RESTORE --epochs $EPOCHS
