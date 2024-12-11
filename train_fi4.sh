#!/usr/bin/env bash
python -m t3.train --train-code="fi4" --dataset-scale=0.1 --d-model=256 --num-heads=8 --dff=1024 --num-layers=8 --seq-len=2048 --batch-size=128