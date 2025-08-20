if [ $# -ne 1 ]; then
    echo "Usage: $0 <target_cv>"
    exit 1
fi

TARGET_CV=$1
PYTHON_SCRIPT="/mnt/n0/sslm/ServerlessLLM/tools/trace/benchmark_trace.py"


nohup python $PYTHON_SCRIPT --action trace_locality --target_cv $TARGET_CV --sllm_model_config_file_path /mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/bd-use-l40.json > log/0-locality.log 2>&1 &
