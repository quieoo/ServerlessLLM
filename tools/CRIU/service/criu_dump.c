#include <criu/criu.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>

int main(int argc, char *argv[]) {
    // 参数解析
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <socket_path> <images_dir>\n", argv[0]);
        return -1;
    }

    // 初始化CRIU配置
    if (criu_init_opts() < 0) {
        perror("criu_init_opts failed");
        return -1;
    }

    // 设置服务socket路径
    // criu_set_service_address(argv[1]);

    // 创建/打开镜像目录
    int dir_fd = open(argv[2], O_DIRECTORY);
    if (dir_fd == -1) {
        perror("open images_dir failed");
        return -1;
    }
    criu_set_images_dir_fd(dir_fd);

    // 设置转储选项
    criu_set_shell_job(true);
    criu_set_log_level(4);
    criu_set_leave_running(true);
    criu_set_tcp_established(true);

    // 执行转储
    if (criu_dump() < 0) {
        fprintf(stderr, "CRIU dump failed: %s\n", strerror(errno));
        close(dir_fd);
        return -1;
    }

    printf("转储成功！镜像存储于：%s\n", argv[2]);
    close(dir_fd);
    return 0;
}