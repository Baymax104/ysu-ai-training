#!/usr/bin/env bash
base_dir=$(pwd)
echo "当前工作目录: $base_dir"

model_path=$MODEL_PATH
if [ -z "$model_path" ]; then
  echo "未设置MODEL_PATH环境变量"
  return 0
fi
echo "模型所在位置: $MODEL_PATH"

python finetune_hf.py ../preprocessing/data "$MODEL_PATH" configs/lora.yaml
