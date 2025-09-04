#!/usr/bin/env python3
import os
import time
import shutil

# 配置路径
disk_path = "./testfile.bin"
tmpfs_dir = "/mnt/tmpfs"
tmpfs_path = os.path.join(tmpfs_dir, "testfile.bin")

# 确保 tmpfs 已挂载： mount -t tmpfs -o size=1G tmpfs /mnt/tmpfs
# 若 testfile 不存在，可以生成一个 ~300MB 的文件
def generate_file(path, size_mb=300):
    with open(path, "wb") as f:
        f.write(os.urandom(size_mb * 1024 * 1024))

if not os.path.exists(disk_path):
    print(f"Generating {disk_path} ...")
    generate_file(disk_path, 300)

# 拷贝一份到 tmpfs
print(f"Copying {disk_path} -> {tmpfs_path}")
shutil.copyfile(disk_path, tmpfs_path)

def read_test(path):
    start = time.time()
    with open(path, "rb") as f:
        while f.read(1024 * 1024):  # 1MB per read
            pass
    end = time.time()
    return end - start

# 读两遍：第一次可能包含 IO，第二次走缓存可对比
print("Testing disk file ...")
t1 = read_test(disk_path)
t2 = read_test(disk_path)

print("Testing tmpfs file ...")
t3 = read_test(tmpfs_path)
t4 = read_test(tmpfs_path)

print(f"Disk file first read:  {t1:.3f} s")
print(f"Disk file second read: {t2:.3f} s")
print(f"Tmpfs first read:      {t3:.3f} s")
print(f"Tmpfs second read:     {t4:.3f} s")


# 测试结果：两种方式没有太大的区别，甚至SSD要快一些。300MB文件大概读取时间都在100ms左右