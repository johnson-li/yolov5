#!/usr/bin/zsh
source ~/.zshrc
conda activate dev

concurrency=8
exp_name=webrtc_exp3
weight=weights/yolov5s.pt
ls ~/Data/$exp_name | xargs -P $concurrency -I FILE bash -c 'python -m stream_local -p ~/Data/'$exp_name'/FILE/dump -w '$weight

