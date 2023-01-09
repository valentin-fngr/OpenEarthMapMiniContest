#!/usr/bin/env bash
python train.py --device=cuda:3 --decode_channels=64 --dropout=0.24 --window_size=8 --lr=0.01
python train.py --device=cuda:3 --decode_channels=64 --dropout=0.16 --window_size=4 --lr=0.01

