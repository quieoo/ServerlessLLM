BinaryPath="/mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/build/Allocateion"
CONFIG_PATH_1="/mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/L40-small.1.json"
REQ_FILE_PATH_1="/mnt/n0/sslm/ServerlessLLM/tools/trace/outputs/l40_cv0.5.small.txt"

CONFIG_PATH_2="/mnt/n0/sslm/ServerlessLLM/tools/mock_allocation/configs/L40-large.1.json"
REQ_FILE_PATH_2="/mnt/n0/sslm/ServerlessLLM/tools/trace/outputs/l40_cv0.5.large.txt"


echo "Starting experimental group (allocation)..."
$BinaryPath  -g 43 -m 100 -r guas -s 40 -p 1 -f 0 --affinity --gpu 0 --req_file_path $REQ_FILE_PATH_1 --config $CONFIG_PATH_1
$BinaryPath  -g 43 -m 100 -r guas -s 40 -p 1 -f 1 --affinity --gpu 0 --req_file_path $REQ_FILE_PATH_1 --config $CONFIG_PATH_1
$BinaryPath  -g 43 -m 100 -r guas -s 40 -p 4 -f 1 --affinity --gpu 0 --req_file_path $REQ_FILE_PATH_1 --config $CONFIG_PATH_1

$BinaryPath  -g 43 -m 100 -r guas -s 40 -p 1 -f 0 --affinity --gpu 0 --req_file_path $REQ_FILE_PATH_2 --config $CONFIG_PATH_2
$BinaryPath  -g 43 -m 100 -r guas -s 40 -p 1 -f 1 --affinity --gpu 0 --req_file_path $REQ_FILE_PATH_2 --config $CONFIG_PATH_2
$BinaryPath  -g 43 -m 100 -r guas -s 40 -p 4 -f 1 --affinity --gpu 0 --req_file_path $REQ_FILE_PATH_2 --config $CONFIG_PATH_2

echo "Finished experimental group (allocation)..."