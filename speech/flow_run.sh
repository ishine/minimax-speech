#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.

pretrained_model_dir=./pretrained_models/CosyVoice2-0.5B

# train llm
export CUDA_VISIBLE_DEVICES="0,1,2,3"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1986
dist_backend="nccl"
num_workers=24
prefetch=100
train_engine=torch_ddp
model=flow

torchrun --nnodes=1 --nproc_per_node=$num_gpus --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
  train.py \
  --train_engine $train_engine \
  --config config.yaml \
  --train_data ./data.list \
  --cv_data ./data.list \
  --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
  --model $model \
  --model_dir /data/checkpoint/$model/ \
  --num_workers ${num_workers} \
  --prefetch ${prefetch} \
  --use_amp \
  --checkpoint /data/checkpoint/flow/epoch_5_step_174001.pt

# torchrun --nproc_per_node=4 --nnodes=1 --rdzv_id=2024 --rdzv_backend="c10d" --rdzv_endpoint="localhost:0" `which s3tokenizer` --root_path /data/dataset/ \
#                 --model speech_tokenizer_v2_25hz \
#                 --device "cuda" \
#                 --batch_size 64 \
#                 --file_list /data/learnable-speech/speech/files.txt \
#                 --skip_existing
