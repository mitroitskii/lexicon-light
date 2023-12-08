#!/bin/bash
# change probe_bsz depending on how much memory you have (one batch use ~3GB GPU memory)
# best results were achieved with batch size 6

python train_probe.py --layer 0 --target_idx -1 --probe_bsz 6