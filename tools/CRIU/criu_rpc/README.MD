成功跑起来的一个demo：
    运行一个RPC Server，完成初始化之后调用criu service dump。在另一个窗口调用criu service restore恢复Server，并使用RPC Client连接。
````bash
#  0. 所有指令都需要在sudo下执行
sudo su
conda activate /mnt/n0/.conda/envs/sllm-worker-0.6

# 0.1 启动criu service
criu service --address criu_service.socket

# 1. 转储
setsid python server.py < /dev/null &>> server.log

# 2. 恢复 （在一个新的窗口给，执行步骤0）
python test_2.py /mnt/n0/sslm/ServerlessLLM/tools/CRIU/service/criu_service.socket /mnt/n0/sslm/ServerlessLLM/tools/CRIU/criu_rpc/imgs/

# 3. 访问RPC Server
python client.py

````

渐进探索最晚的checkpoint点
````python
import xformers
````
这个库是无法转储，但是导入时延又大

