python3 example.py &
PID=$!
echo "Python进程PID: $PID"

sudo /mnt/n0/sslm/criu-4.1/criu/criu/criu dump -t $PID -D ./checkpoint --shell-job

