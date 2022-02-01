#!/bin/bash

python experiments/bit_vector-vae/train.py \
    --mode binary-concrete \
    --lr 0.002 \
    --batch_size 64 \
    --n_epochs 100 \
    --latent_size 128 \
    --weight_decay 0. \
    --temperature_decay 1e-5 \
    --temperature_update_freq 1000 \
    --random_seed 42
