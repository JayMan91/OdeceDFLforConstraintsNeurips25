#!/bin/bash

for s in 11 12 13 14 15; do

    for alpha in 0.8 0.7 0.6 0.5 0.4 0.3 0.2; do
    	python3 KnapsackCapaExp.py --num_items 50 --seed $s --model_name odece --infeasibility_aversion_coeff $alpha --max_epochs 20 --lr 0.01 --deg 6 --noise_width 0.25
    done
    python3 KnapsackCapaExp.py --num_items 50 --seed $s --model_name mse --lr 0.01 --deg 6 --noise_width 0.25 --max_epochs 20 
    python3 KnapsackCapaExp.py --num_items 50 --seed $s --model_name comboptnet --lr 0.01 --deg 6 --noise_width 0.25 --max_epochs 20 --tau 0.1
    python3 KnapsackCapaExp.py --num_items 50 --seed $s --model_name sfl --lr 0.01 --deg 6 --noise_width 0.25 --max_epochs 20  --temp 0.1
done