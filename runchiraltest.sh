#!/bin/bash

export PYTHON=/data/people/kabeywar/envs/anaconda3/envs/pyg/bin/python
export CONFIG=config/test_chiral.yml
export SCRIPT=script/test_chiral.py

$PYTHON -u $SCRIPT --config_path $CONFIG
