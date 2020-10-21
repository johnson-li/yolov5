#!/bin/bash


concurrency=1 # max = 3
ls ~/Data/webrtc-exp1 | xargs -P $concurrency -I FILE bash -c 'python -m stream_local -p ~/Data/webrtc-exp1/FILE/dump'

