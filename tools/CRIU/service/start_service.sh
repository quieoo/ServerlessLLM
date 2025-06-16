sudo su
conda activate /mnt/n0/.conda/envs/sllm-worker-0.6
criu service --address criu_service.socket
