#!/bin/bash

# Miaoda Agent K8s Example Runner Script
# Run Miaoda agent tasks with debug output

# Configuration
INPUT_JSONL=miaoda-dataset/test_samples.jsonl
CONCURRENT=3
MODEL_NAME=miaoda
OUTPUT_DIR=miaoda-output/$MODEL_NAME"-test"
TOOL_TIMEOUT=600
LLM_TIMEOUT=600
TOTAL_TIMEOUT=3600
CPU_REQUEST=0.3
MEMORY_REQUEST=1Gi

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Run Miaoda K8s example
python3 miaoda_k8s_example.py \
    --jsonl $INPUT_JSONL \
    --concurrent $CONCURRENT \
    --timeline \
    --output-dir $OUTPUT_DIR \
    --tool-timeout $TOOL_TIMEOUT \
    --llm-timeout $LLM_TIMEOUT \
    --max-execution-time $TOTAL_TIMEOUT \
    --cpu $CPU_REQUEST \
    --memory $MEMORY_REQUEST
