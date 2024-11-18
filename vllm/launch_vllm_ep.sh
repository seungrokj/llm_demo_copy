#!/bin/bash
python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --swap-space 16 \
    --disable-log-requests \
    --dtype float16 \
    --tensor-parallel-size $TP \
    --host 0.0.0.0 \
    --port $PORT \
    --num-scheduler-steps 1 \
    --enforce-eager \
    --max-model-len 8192 \
    --distributed-executor-backend mp
