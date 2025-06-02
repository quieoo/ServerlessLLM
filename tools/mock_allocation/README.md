## 模拟运行显存池分配器，测试分配性能

### 运行前提
build文件夹下存在文件”model_requests.seq“,记录了模型请求的信息。
如果不存在的话需要带“--regenerate”参数运行，生成新的请求文件。
参数“-s"表示请求规模，越大请求规模越大。
参数“--affinity"表示基于代码设定的模型亲和性设置生成的模型请求中模型比例，否则使用”-r"参数指定随机生成("-r random")还是按照正态分布的比例生成("-r guas")。

### 1. W/O Reuse
基础版本，没有显存池，没有数据重用
运行：
```bash
./Allocateion -g 20 -m 100 -r guas -s 40 -p 0 --affinity --gpu 1
```
"-g"指定GPU显存池大小，以GB为单位。
"-m"指定CPU内存池大小，以GB为单位。
“-gpu"指定运行GPU。
“-p"指定内存分配策略

### 2. GlobalMerge
数据重用，使用全局合并策略
运行：
```bash
./Allocateion -g 20 -r guas -s 40 -p 2 --affinity --gpu 1
```

### 3. WBPM+GreedyMerge
数据重用，使用WBPM+贪心合并策略
设置“vram_manager_v4.h"头文件中的配置如下：
```c++
#define BipartMatchEnable
// #define RecursiveSplitEnable
```
重新编译
运行：
```bash
./Allocateion -g 20 -r guas -s 40 -p 4 --affinity --gpu 1
```

### 4.Partition+BinPack
数据重用，使用分治装箱策略
设置“vram_manager_v4.h"头文件中的配置如下：
```c++
// #define BipartMatchEnable
#define RecursiveSplitEnable
```
重新编译
运行：
```bash
./Allocateion -g 20 -r guas -s 40 -p 4 --affinity --gpu 1
```
