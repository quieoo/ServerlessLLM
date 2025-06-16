首先需启动CRIU服务，通过命令行指定服务套接字地址,此服务负责处理后续的RPC请求
````
sudo su
criu service --address /mnt/n0/sslm/ServerlessLLM/tools/CRIU/service_dump/criu_service.socket 
````

编译
````
gcc client.c -o cli -lcriu
````

