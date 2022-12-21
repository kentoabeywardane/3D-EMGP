#!/bin/bash

export PYTHON=/data/people/kabeywar/envs/anaconda3/envs/pyg/bin/python
export CONFIG=config/finetune_md22.yml
export SCRIPT=script/finetune_md17.py
export RESTORE=checkpoints/checkpoint-3dmgp
export DATASET=MD22

MOLECULE=(docosahexaenoic_acid stachyose at_at_cg_cg)
NUM_TRAIN=(9500 9500 5000)
BATCH_SZ=(24 24 4)
for ((i=0; i<${#MOLECULE[@]}; i++))
do 
    echo "${MOLECULE[$i]} ${NUM_TRAIN[$i]} ${BATCH_SZ[$i]}" 
    $PYTHON -u $SCRIPT --config_path $CONFIG --restore_path $RESTORE --dataset $DATASET --molecule ${MOLECULE[$i]} --num_train ${NUM_TRAIN[$i]} --batch_size ${BATCH_SZ[$i]}
done

