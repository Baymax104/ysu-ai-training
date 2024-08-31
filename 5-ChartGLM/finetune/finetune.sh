#!/usr/bin/env bash
if [ -n "$MODEL_PATH" ]; then
  echo "模型所在路径: $MODEL_PATH"
  python src/finetune.py ../preprocessing/data "$MODEL_PATH" configs/lora.yaml
fi
