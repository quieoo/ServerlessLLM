#include <criu/criu.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

int main() {
  // 初始化 CRIU 选项（与转储时一致）
  if (criu_init_opts() < 0) {
    fprintf(stderr, "CRIU 初始化失败: %s\n", strerror(errno));
    return -1;
  }

  // 设置服务地址（与转储时一致）
  criu_set_service_address(
      "/mnt/n0/sslm/ServerlessLLM/tools/CRIU/service_dump/criu_service.socket");

  // 设置镜像目录（与转储时一致）
  int dir_fd =
      open("/mnt/n0/sslm/ServerlessLLM/tools/CRIU/service_dump/checkpoint_dir",
           O_DIRECTORY);
  if (dir_fd == -1) {
    fprintf(stderr, "打开镜像目录失败: %s\n", strerror(errno));
    return -1;
  }
  criu_set_images_dir_fd(dir_fd);

  criu_set_log_file("restore.log");
  criu_set_log_level(4);
  // 启用对控制终端进程（shell job）和已建立 TCP 连接的支持
  criu_set_shell_job(true);
  criu_set_tcp_established(true);
  // 执行恢复
  int ret = criu_restore();
  if (ret < 0) {
    fprintf(stderr, "恢复失败: %d\n", ret);
    return -1;
  }

  printf("CRIU 恢复成功，进程已继续执行\n");
  return 0;
}