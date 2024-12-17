#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file=configs/cfg_2gpus.yaml --main_process_port 7234 main.py
