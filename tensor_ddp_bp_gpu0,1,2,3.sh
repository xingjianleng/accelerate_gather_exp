#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file=configs/cfg_4gpus.yaml --main_process_port 7234 tensor_ddp_bp.py
