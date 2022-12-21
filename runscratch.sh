#!/bin/bash

export PYTHON=/data/people/kabeywar/envs/anaconda3/envs/pyg/bin/python
export CONFIG=config/finetune_drugs_scratch.yml
export SCRIPT=script/finetune_drugs.py
export TESTSCRIPT=script/test_drugs.py
export EPOCHS=1000

$PYTHON -u $SCRIPT --config_path $CONFIG --epochs $EPOCHS

# $PYTHON -u $TESTSCRIPT --config_path $CONFIG
