#!/usr/bin/zsh
source ~/.zshrc
conda activate dev8

concurrency=1 # max = 3
exp_name=webrtc_exp5
weight=weights/yolov5x.pt
ls /mnt/wd/$exp_name | xargs -P $concurrency -I FILE bash -c 'python -m stream_local -p /mnt/wd/'$exp_name'/FILE/dump -w '$weight

