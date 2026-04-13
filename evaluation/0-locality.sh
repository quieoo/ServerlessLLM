
cd ../tools/trace/
python benchmark_trace.py --target_cv 1 --sllm_model_config_file_path ../mock_allocation/configs/L40-large.json --mapping round_robin --days 2 --action trace_locality
cd ../../evaluation