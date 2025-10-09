
INPUT_JSONL=test-swe-bench-verified-50.jsonl
#test-00000-of-00001-with-images.jsonl
CONCURRENT=100
MODEL_NAME=dsv3-terminus
OUTPUT_DIR=$MODEL_NAME"-output-v16"
TOOL_TIMEOUT=600
LLM_TIMEOUT=600
TOTAL_TIMEOUT=1800


#python3 r2e_k8s_example.py --jsonl $INPUT_JSONL --concurrent $CONCURRENT --timeline --output-dir $OUTPUT_DIR --tool-timeout $TOOL_TIMEOUT --llm-timeout $LLM_TIMEOUT --max-execution-time $TOTAL_TIMEOUT
#python3 generate_swe_verified.py --jsonl $INPUT_JSONL --output-dir $OUTPUT_DIR --model-name $MODEL_NAME --output $OUTPUT_DIR/$MODEL_NAME.jsonl
export SWEBENCH_API_KEY=swb_PauP11Z3PoZIZzQRRN2gM3BzKl3U7Yz3L8y6NaJkOR8_68ac0515
sb-cli submit swe-bench_verified test --predictions_path $OUTPUT_DIR/$MODEL_NAME.jsonl --run_id $OUTPUT_DIR
