#!/bin/bash

# PYTHONPATH=/home/yuzhu/chaoyang/projects/Delinquency/delinquency/github/v2/

# Choose model (-hf) from:
# - unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_XL
# - unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF:Q5_K_XL

# Key settings:
# - --ctx-size 262144: Total context (256k) divided by -np slots (262144/8 = 32768 per slot)
# - -np 8: 8 parallel slots
# - Each slot: 32768 tokens (ctx-size / np)
# - -n 8192: Max output tokens per request
# - -b/-ub: High values for continuous batching efficiency

CUDA_VISIBLE_DEVICES=0 ~/App/llama.cpp/build/bin/llama-server \
    -hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_XL \
    --jinja -ngl -1 --threads -1 --ctx-size 524288 \
    --flash-attn on \
    -b 32768 -ub 16384 \
    -n 8192 \
    -ctk q8_0 -ctv q8_0 \
    -np 16 \
    --host 0.0.0.0 --port 8080 \
    --temp 0.7 --min-p 0.0 --top-p 0.80 --top-k 20 --presence-penalty 1.0


CUDA_VISIBLE_DEVICES=0 ~/App/llama.cpp/build/bin/llama-server \
    -hf unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF:Q4_K_XL \
    --jinja -ngl -1 --threads -1 --ctx-size 524288 \
    --flash-attn on \
    -b 32768 -ub 16384 \
    -n 8192 \
    -ctk q8_0 -ctv q8_0 \
    -np 16 \
    --host 0.0.0.0 --port 8080 \
    --temp 0.6 --min-p 0.0 --top-p 0.95 --top-k 20 --presence-penalty 1.0

