#!/bin/bash

export PYTHON=/data/people/kabeywar/envs/anaconda3/envs/pyg/bin/python
export CONFIG=config/finetune_drugs.yml
export SCRIPT=script/test_drugs.py

$PYTHON -u $SCRIPT --config_path $CONFIG 
