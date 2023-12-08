#!/bin/bash
w
python train_probe.py --layer 0 --target_idx -1 --train_data ../data/train_tiny_1000.csv --val_data ../data/val_tiny_500.csv --test_data ../data/test_tiny_500.csv