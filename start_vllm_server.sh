#!/bin/bash
# Starts the vLLM OpenAI-compatible server for Qwen3-VL-8B.
# Run this from WSL before starting run_vllm.bat.
# Model loads in ~10 min from /mnt/c. Leave this window open while processing.

source ~/vllm-env/bin/activate
export HF_HOME=/mnt/c/models/huggingface

echo "Starting vLLM server for Qwen3-VL-8B-Instruct..."
echo "Model will load in ~10 minutes. Keep this window open."
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --gpu-memory-utilization 0.88 \
    --max-model-len 8192 \
    --max-num-seqs 36 \
    --limit-mm-per-prompt image=1 \
    --host 0.0.0.0 \
    --port 8000
