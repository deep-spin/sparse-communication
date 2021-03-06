#!/bin/bash

for i in {1..10}
do
    ((seed=$i + 321))
    python experiments/signal-game/train.py \
        --mode gs \
        --lr 0.001 \
        --entropy_coeff 0.00 \
        --batch_size 64 \
        --n_epochs 500 \
        --game_size 16 \
        --latent_size 256 \
        --embedding_size 256 \
        --hidden_size 512 \
        --weight_decay 0. \
        --temperature_decay 1e-5 \
        --temperature_update_freq 1000 \
        --random_seed ${seed}
done
