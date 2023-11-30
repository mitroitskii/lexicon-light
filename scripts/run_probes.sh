#!/bin/bash

# for l in {-1..32}; do
#     python train_gptj_probe.py --layer $l --target_idx -1 --train_data ../data/train_tiny_1000.csv --val_data ../data/val_tiny_500.csv --test_data ../data/test_tiny_500.csv
# done

CUDA_VISIBLE_DEVICES=1 python train_probe.py --layer 0 --target_idx -1 --train_data ../data/train_tiny_1000.csv --val_data ../data/val_tiny_500.csv --test_data ../data/test_tiny_500.csv