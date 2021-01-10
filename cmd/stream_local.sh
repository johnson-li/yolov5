#!/usr/bin/zsh
source ~/.zshrc
conda activate dev

concurrency=1
exp_name=webrtc_exp3
weight=weights/yolov5s.pt
ls ~/Data/$exp_name | xargs -P $concurrency -I FILE bash -c 'python -m stream_local -p ~/Data/'$exp_name'/FILE/dump -c 0.1 -i 0.2 -d 1 -w '$weight
#ls -d ~/Data/$exp_name/baseline_* | xargs -P $concurrency -I FILE bash -c 'python -m stream_local -p FILE/dump -w '$weight

