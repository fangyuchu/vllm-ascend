#!/bin/bash

HOST="0.0.0.0"
PORT=8006
DATA_PARALLEL_SIZE=4
REDUNDANT_EXPERTS=0
LOCAL_MODEL_PATH="/AIdata/JW/Qwen3-30B-A3B-W8A8"
MODEL_NAME="Qwen3-30B-A3B-W8A8"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dp)
            DATA_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --re)
            REDUNDANT_EXPERTS="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --local-model)
            MODEL_NAME=$LOCAL_MODEL_PATH
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --dp SIZE                    Set data parallel size (default: 4)"
            echo "  --re SIZE                    Set redundant experts (default: 0)"
            echo "  --host HOST                  Set host address (default: 0.0.0.0)"
            echo "  --port PORT                  Set port number (default: 8006)"
            echo "  --model MODEL_NAME           Set model name or path"
            echo "  -h, --help                   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

echo "Starting vLLM server for $MODEL_NAME with data parallel size: $DATA_PARALLEL_SIZE and redundant experts: $REDUNDANT_EXPERTS"

export DYNAMIC_EPLB="true"

vllm serve $LOCAL_MODEL_PATH \
    --data-parallel-size $DATA_PARALLEL_SIZE \
    --data-parallel-size-local $DATA_PARALLEL_SIZE \
    --enable-expert-parallel \
    --enable-fault-tolerance \
    --api-server-count 1 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --additional-config '{\"eplb_config\":{\"dynamic_eplb\": true, \"num_redundant_experts\":${REDUNDANT_EXPERTS}}}' \
    --quantization ascend \
    --host $HOST \
    --port $PORT
