#!/bin/bash

# DeepSeek V3.1 K8s Example Debug Runner Script
# Single instance test with debug output

# Configuration
INPUT_JSONL=swe-bench-verified-workspace/test-00000-of-00001-with-images.jsonl
CONCURRENT=1
MODEL_NAME=dsv31-terminus
OUTPUT_DIR=swe-bench-verified-workspace/output_root/$MODEL_NAME"-v31-test"
TOOL_TIMEOUT=600
LLM_TIMEOUT=600
TOTAL_TIMEOUT=3600

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Run DeepSeek V3.1 K8s example with debug mode
python3 ds_k8s_example.py \
    --jsonl $INPUT_JSONL \
    --concurrent $CONCURRENT \
    --timeline \
    --output-dir $OUTPUT_DIR \
    --tool-timeout $TOOL_TIMEOUT \
    --llm-timeout $LLM_TIMEOUT \
    --max-execution-time $TOTAL_TIMEOUT
