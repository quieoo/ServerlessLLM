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

### VMM 后端（参数与 KV 统一页池）

默认仍使用上述 legacy 连续显存池。添加 `--memory_backend vmm` 后，参数
tensor group 和 KV block 从每张 GPU 的同一个 CUDA VMM physical-page budget
按需映射；VMM 页大小由 `cuMemGetAllocationGranularity` 查询。由于物理碎片
不再要求连续地址，`-p/--model-pool` 的碎片整理策略在该模式下不启用。

VMM 后端保留 tensor group/fingerprint 的参数缓存语义：一个 tensor group
对上层表现为连续的虚拟地址，但其底层可以由任意空闲物理页组成。模型每次
`LoadModel` 后重新获得参数地址，因此权重被驱逐时可以释放 VA 与物理页，
下次加载时重新建立映射。

当前实现采用了以下性能优化：

1. **CUDA Runtime 预热**：初始化每张 GPU 时执行 `cudaSetDevice` 和
   `cudaFree(nullptr)`，提前创建 primary context，避免首个模型请求承担
   CUDA Runtime 初始化时间。
2. **预创建物理页池**：启动时按照 `-g/--gpu-pool` 指定的容量，通过
   `cuMemCreate` 预创建全部 physical pages。默认页大小使用
   `cuMemGetAllocationGranularity(..., CU_MEM_ALLOC_GRANULARITY_MINIMUM)` 的
   返回值，也可以通过 `--vmm_page_size_mb` 设置为原生粒度的 2 次幂倍数；
   请求热路径不再执行 `cuMemCreate/cuMemRelease`。
3. **O(1) 物理页分配**：空闲 physical handle 保存在 free-list 中，权重和
   KV 都通过栈式 pop/push 获取和归还页面，避免扫描显存 region 或运行
   bin-packing/碎片合并算法。
4. **连续 VA、离散物理页**：每个 tensor group/KV allocation 只预留一段
   连续 VA，再将多个独立 physical pages 映射到其中。只要总空闲页足够，
   就不会因为缺少连续物理区间而失败，也不需要搬移已有参数来整理碎片。
5. **批量设置访问权限**：一个 allocation 的物理页映射完成后，只对整个
   连续 VA 区间调用一次 `cuMemSetAccess`，而不是逐页设置权限。
6. **只映射 cache miss**：命中的 tensor group 直接复用已有 VA，完全命中
   的模型请求不会产生新的 `cuMemMap`；日志中的 `cached_bytes`、
   `to_load_bytes` 和 `map_pages` 可用于核对命中与映射开销。
7. **cache-affinity 多 GPU 调度**：优先选择目标模型缓存字节数最多的 GPU；
   缓存量相同时选择空闲页更多的 GPU，再以 GPU ID 保证确定性，减少跨卡
   重复加载和不必要的权重驱逐。
8. **成本感知权重驱逐**：显存不足时，根据历史访问次数和模型加载敏感度
   选择冷权重释放，同时保护当前正在加载/运行模型的 tensor groups。
9. **权重/KV 统一容量与 KV 优先**：模型切换时先回收旧模型 KV；为当前
   模型分配 KV 时，可以回收非当前模型的缓存权重，使 KV 不受固定分区比例
   限制。权重和 KV 的占用都计入同一个 physical-page pool utilization。

VMM 模式会输出 `cached_bytes`、`to_load_bytes`、`evicted_weight_bytes`、
`reclaimed_kv_bytes` 和 `map_pages`；结束时还会输出 `Reduced IO`（累计权重
cache hit 字节 / 累计请求权重字节）和 `VMM weight H2D bytes`（累计实际加载的
逻辑权重字节）。因此可以直接比较不同页大小日志中的总缓存复用率；该指标不含
KV 分配，也不计入 page 向上取整产生的物理容量开销。需要注意，当前平台的原生页为 2 MB，
不同 physical handle 不能通过一次 `cuMemMap` 批量映射，因此冷加载大模型
仍会产生数千到数万次逐页 `cuMemMap/cuMemUnmap`。预创建消除了热路径中的
物理页创建开销，但逐页映射仍是当前 VMM 后端相对 legacy 的主要性能成本。

可以通过 `--vmm_page_size_mb` 扫描 physical-page size。`0` 表示使用设备
原生最小粒度；显式值必须是原生粒度的 2 次幂倍数，例如当前 2 MB 设备可以
测试 `2/4/8/16/32/64`。更大的页会减少 `cuMemMap/cuMemUnmap` 次数，但每个
tensor group 和 KV allocation 都要向上对齐到该页大小，因此会增加内部碎片。

```bash
nohup ./docs/1.1-overall_vmm.sh > ./docs/1.1-overall_vmm.log 2>&1 &
nohup bash -c "VMM_PAGE_SIZE_MB=64 ./1.1-overall_vmm.sh" > 1.1-overall_vmm_page64.log 2>&1 &


for page_mb in 0 2 4 8 16 32 64; do
  VMM_PAGE_SIZE_MB="$page_mb" ./docs/1.1-overall_vmm.sh
done
```

VMM 当前要求真实 CUDA 映射和拷贝，不能与 `--mock_copy` 一起使用。请求模型
切换时会回收旧模型 KV；当 KV 需要容量时，优先驱逐非当前模型的缓存权重。
