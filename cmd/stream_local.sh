#!/bin/bash

for f in ~/Data/webrtc/*
do
    python -m stream_local -p $f/dump
done

