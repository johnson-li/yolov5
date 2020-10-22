#!/bin/bash


concurrency=1 # max = 3
exp_name=webrtc-exp2
ls ~/Data/$exp_name | xargs -P $concurrency -I FILE bash -c 'python -m stream_local -p ~/Data/'$exp_name'/FILE/dump'

