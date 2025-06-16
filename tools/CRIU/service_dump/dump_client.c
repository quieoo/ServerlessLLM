#include <criu/criu.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>  // 引入 string.h 以使用 strerror 函数
#include <unistd.h>  // 用于 sleep 函数

int main() {


  // 初始化 CRIU 选项
  if (criu_init_opts() < 0) {
    fprintf(stderr, "CRIU 初始化选项失败: %s\n", strerror(errno));
    return -1;
  }

  // 设置 CRIU service 地址（Unix Domain Socket）
  criu_set_service_address(
      "/mnt/n0/sslm/ServerlessLLM/tools/CRIU/service_dump/criu_service.socket");

  // 设置镜像存储目录（修正原代码中 fd 参数错误，直接使用路径字符串）
  // 由于 'criu_set_images_dir' 未声明，使用建议的 'criu_set_images_dir_fd' 替代
  // 打开镜像存储目录并获取文件描述符
  // 为了解决 'O_DIRECTORY' 未声明的问题，包含 <fcntl.h> 头文件

  int dir_fd =
      open("/mnt/n0/sslm/ServerlessLLM/tools/CRIU/service_dump/checkpoint_dir",
           O_DIRECTORY);
  if (dir_fd == -1) {
    fprintf(stderr, "打开镜像存储目录失败: %s\n", strerror(errno));
    return -1;
  }
  criu_set_images_dir_fd(dir_fd);



  // 启用对控制终端进程（shell job）和已建立 TCP 连接的支持
  criu_set_shell_job(true);
  criu_set_tcp_established(true);

  printf("程序启动，开始执行...\n");

  for(int i=0;i<10;i++)
  {
    if(i==5){
      printf("开始转储\n");
      // 执行 CRIU 转储（通过 service 方式）
      int ret=criu_dump();
      if (ret < 0) {
        // fprintf(stderr, "转储失败: %s\n", strerror(errno));
        printf("转储失败: %d\n",ret);
        
        return -1;
      }
      printf("转储成功\n");

    }
    printf("i=%d\n",i);
    sleep(1);
  }
  return 0;
}