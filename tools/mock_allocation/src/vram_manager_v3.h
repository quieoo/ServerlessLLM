/*
第三版
- 通过每次只处理一个新数据段将问题降级成多选择的背包问题, 在装载当前数据段时,
提前分配下一个数据段的空间
- 正在装载的数据段无法被移动, 因此它将整个空间分成了独立的两部分
- 根据正在装载的数据段大小, 结合CPU->GPU带宽可以估计一个覆盖时间上限: T_overlap
- 当移动成本小于T_overlap时, 可以认为移动成本为0
- 贪心的方法寻找最小的移动成本+丢弃成本之和


使用的类
GPUMemoryRegion_V3:
    其他和GPUMemoryRegion_V2一样
    添加一个成员变量enum status, 取代原本的is_allocated. status有三种取值:
0-free, 1-loading, 2-allocated

GPUTensorPool_V3:
成员变量:
    memory_regions: 记录张量池的使用情况
    allocated_regions: 支持快速访问已分配的TG
    model_access: 记录历史的模型访问, 用来估计模型的访问概率
    total_access: 总访问次数
    GPUBandwidth: GPU-GPU数据移动带宽
    CPUBandwidth: CPU-当前GPU 移动带宽
    drop_costs: 记录当前已经分配区域的丢弃成本的有序容器(set),
在每次装载阶段开始时更新

成员方法:
    AllocateFreeRegion(free_region, allocate_size):
        1. 检查free_region状态和size是否正确
        2. 如果free_region.size > allocate_size, 则创建新的区域并更新链表
        3. 更新free_region和新区域的状态
    UpdateDropCost():
        - 根据当前model_access, 遍历区域链表, 收集所有已分配区域,
重新构建drop_costs. 丢弃成本的计算方式: 区域大小 * 模型访问概率 * 模型装载带宽 *
模型装载敏感度
        - 当total_access为0时, 所有模型的访问概率都为1
    UseModel(model_path):
        更新model_access和total_access

    AllocateASAP(request_size, T_overlap):
        使用贪心的方法找到丢弃成本+移动成本最低的方案分配下一个数据段:
        1. 扫描区域链表, 如果遇到足够大的空闲区域,
直接调用AllocateFreeRegion()分配空闲区域并返回
        2. 否则, 确定左右子空间,左子空间从区域链表头开始到loading状态的区域结束,
右子空间从loading状态区域的下一个区域开始, 到链表结束. 注意, 左右子空间可能为空,
比如, 为第一个段分配空间,此时没有状态为loading的区域;
或者loading区域在链表头或者链表尾.
        3. 判断左右子空间各自的空闲区域之和是否足够, 如果足够,
计算各自的最低合并成本: 3.1 从第一个空闲区域开始,
寻找在其之后的第一个空闲区域,他们之间的空闲区域大小之和超过request_size,
计算这个范围内的非空闲区域大小之和, 结合GPUBandwidth, 计算T_merge 3.2
空闲区域起始位置加一, 继续寻找到它的结束空闲区域, 计算新的T_merge,
更新最小的T_merge_min 3.3 持续遍历, 直到遇到一个空闲区域起始位置,
它后面的所有空闲区域加起来都不够满足分配要求, 结束遍历. 3.4 返回T_merge_min,
以及空闲区域起始位置, 终止位置
        4. 取左右子空间中最小的T_merge_min, 如果T_merge_min低于T_overlap,
则按照对应的起始/终止空闲区域合并, 调用AllocateFreeRegion()分配空闲区域并返回
        5. 否则, 计算额外成本: Cost_extra = (T_merge_min - T_overlap) * S, S
是当前模型的装载时延敏感度. 此时的总成本Cost_All[0]=Cost_extra,
是不丢弃任何段的情况下的最小成本.
        6. 根据drop_costs找到候选丢弃区域组的所有可能G:
            - 只包含一个区域的候选丢弃组(G1): 从低到高遍历drop_costs,
获得小于Cost_extra的区域加入G1
            - 包含两个区域的候选丢弃组(G2): 从G1中找到区域组合(a, b),
要求两者的丢弃成本之和小于Cost_extra, 并且属于同一个子空间, 加入G2
            - 包含三个区域的候选丢弃组(G3): 从G2找到区域组合(a, b),
从G1中找到区域c, 三者的丢弃成本之和小于Cost_extra, 并且属于同一个子空间
            - 包含四个区域的候选丢弃组(G4): 从G3找到区域组合(a, b, c),
从G1中找到区域d, 四者的丢弃成本之和小于Cost_extra, 并且属于同一个子空间
            - 持续寻找可能的候选丢弃组, 直到遇到某一个Gi, 找不到任何的可能选择,
则退出循环
            - 将所有G1, G2, ..., Gi合并, 形成最终的候选丢弃区域组G
        7. 对于G中的每一个选择, 模拟丢弃, 然后重复步骤1-5,
计算当前丢弃选择下移动的额外成本 Cost_extra',
加上丢弃成本获得最终成本Cost_All[i], 记录最小的Cost_All
        8. 按照最小的Cost_All丢弃已分配区域, 合并空闲区域,
调用AllocateFreeRegion()分配空闲区域并返回


VRAMManager_V3:
成员变量:
    和VRAMManager_V2一样

成员方法:
    VRAMManager_V3: 和VRAMManager_V2一样
    RegisterModel: 和VRAMManager_V2一样
    MemoryUsage: 和VRAMManager_V2一样
    LoadModel(model_path, device_id):
        1. 检查模型是否注册, 检查device_id是否在gpu_tensor_pools_中
        2. 访问registered_models_中的模型, 获得所有TG的fingerprint
        3. 遍历所有TG, 检查是否在allocated_regions中, 如果不在则加入待装载列表
        5. 遍历待装载列表:
            5.1 如果是第一个TG, 则直接调用AllocateASAP(request_size,
0)分配空闲区域 5.2 根据TG的大小和CPUBandwidth计算T_Overlap 5.3
使用异步cudamemcpy从CPU装载TG 5.4 调用AllocateASAP(request_size,
T_overlap)分配下一个TG的空间 5.5 调用cudaStreamSynchronize确定CPU装载结束
        6. 调用UseModel(model_path), 调用UpdateDropCost()
*/

#pragma once

#include <cuda_runtime.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "logger.h"
#include "registered_model.h"
#include "vram_manager_base.h"

// 定义区域状态枚举
#define UNVALID_COST 1e9

class GPUMemoryRegion_V3
    : public std::enable_shared_from_this<GPUMemoryRegion_V3> {
 public:
  RegionStatus status;  // 0: free, 1: loading, 2: allocated
  char* addr;
  size_t size;
  std::string fingerprint;
  std::shared_ptr<RegisteredModel> model_ref;
  std::shared_ptr<GPUMemoryRegion_V3> prev;
  std::shared_ptr<GPUMemoryRegion_V3> next;

  GPUMemoryRegion_V3()
      : status(FREE), addr(nullptr), size(0), prev(nullptr), next(nullptr) {}

  GPUMemoryRegion_V3(void* addr_, size_t size_)
      : status(FREE),
        addr(static_cast<char*>(addr_)),
        size(size_),
        prev(nullptr),
        next(nullptr) {}

  bool isAdjacent(std::shared_ptr<GPUMemoryRegion_V3> other) {
    return (this->addr + this->size == other->addr) ||
           (other->addr + other->size == this->addr);
  }

  void merge(std::shared_ptr<GPUMemoryRegion_V3> other) {
    if (this->isAdjacent(other)) {
      // 判断other在右侧还是左侧
      if (this->addr + this->size == other->addr) {  // other在右侧
        this->size += other->size;
        this->next = other->next;
        if (other->next) {
          other->next->prev = shared_from_this();
        }
      } else {  // other在左侧
        this->addr = other->addr;
        this->size += other->size;
        this->prev = other->prev;
        if (other->prev) {
          other->prev->next = shared_from_this();
        }
      }
    }
  }

  std::string toString() {
    return "[GPUMemoryRegion_V3: addr=" +
           std::to_string(reinterpret_cast<size_t>(addr)) +
           ", size=" + std::to_string(size) + ", fingerprint=" + fingerprint +
           ", status=" + std::to_string(status) +
           ", model_ref=" + (model_ref ? model_ref->model_path() : "nullptr") +
           "]";
  }

  bool isSame(std::shared_ptr<GPUMemoryRegion_V3> other) {
    return this->addr == other->addr;
  }
  // 在当前区域是空闲区域, 相邻区域是已分配区域的情况下,
  // 交换当前区域和相邻区域的数据以及状态
  void moveSwapAdjacent(std::shared_ptr<GPUMemoryRegion_V3> other) {
    if (!isAdjacent(other)) {
      LOG(ERROR) << "Regions are not adjacent.";
      return;
    }
    if (status != FREE || other->status != ALLOCATED) {
      LOG(ERROR) << "Invalid region statuses.";
      return;
    }

    // 检查合并后的总大小是否足够容纳数据
    const size_t total_merged_size = size + other->size;
    if (total_merged_size < other->size) {
      LOG(ERROR) << "Merged region is smaller than allocated region.";
      return;
    }
    // 保存原始信息
    std::string original_fingerprint = other->fingerprint;
    size_t original_size = other->size;

    // 执行数据移动
    cudaError_t err = cuda_safe_move(addr, other->addr, other->size);
    if (err != cudaSuccess) {
      LOG(ERROR) << "cuda_safe_move failed: " << cudaGetErrorString(err);
      return;
    }

    // 更新当前区域状态
    status = ALLOCATED;
    fingerprint = original_fingerprint;
    size = original_size;
    model_ref = other->model_ref;  // 更新模型路径

    // 更新other区域状态
    other->status = FREE;
    other->addr = addr + size;
    other->size = total_merged_size - size;
    other->fingerprint.clear();
    other->model_ref.reset();  // 重置模型引用
  }
};
struct DropCostEntry_V3 {
  double cost;
  std::shared_ptr<GPUMemoryRegion_V3> region;
  bool operator<(const DropCostEntry_V3& other) const {
    // 添加地址比较保证唯一性
    if (cost != other.cost) {
      return cost < other.cost;
    }
    return reinterpret_cast<uintptr_t>(region->addr) <
           reinterpret_cast<uintptr_t>(other.region->addr);
  }
};

struct DropCostGroup {
  double total_cost;
  std::vector<std::shared_ptr<GPUMemoryRegion_V3>> regions;
  bool operator<(const DropCostGroup& other) const {
    if (total_cost != other.total_cost) {
      return total_cost < other.total_cost;
    }
    // 当总成本相同时，按区域地址排序
    for (size_t i = 0; i < regions.size() && i < other.regions.size(); ++i) {
      auto addr1 = reinterpret_cast<uintptr_t>(regions[i]->addr);
      auto addr2 = reinterpret_cast<uintptr_t>(other.regions[i]->addr);
      if (addr1 != addr2) {
        return addr1 < addr2;
      }
    }
    return regions.size() < other.regions.size();
  }
};

class GPUTensorPool_V3 {
 public:
  std::shared_ptr<GPUMemoryRegion_V3> memory_regions;  // 内存区域链表头
  std::unordered_map<std::string, std::shared_ptr<GPUMemoryRegion_V3>>
      allocated_regions;
  std::unordered_map<std::string, size_t> model_access;
  size_t total_access{0};

  // 新增成员
  double GPUBandwidth = 400.0 * 1024 * 1024;  // GPU-GPU数据移动带宽: 400MB/ms
  double CPUBandwidth = 20.0 * 1024 * 1024;   // CPU->GPU数据拷贝带宽: 20MB/ms
  std::set<DropCostEntry_V3> drop_costs;  // 有序容器，存储已分配区域的丢弃成本

  int device_id;
  cudaStream_t stream_;

  std::unordered_map<std::string, std::pair<long long, int>>
      model_allocation_time;

  GPUTensorPool_V3(int device_id_, size_t total_size, double gpu_bw,
                   double cpu_bw)
      : device_id(device_id_), GPUBandwidth(gpu_bw), CPUBandwidth(cpu_bw) {
    cudaSetDevice(device_id_);
    void* gpu_memory;
    cudaError_t err = cudaMalloc(&gpu_memory, total_size);
    if (err != cudaSuccess) {
      LOG(ERROR) << "cudaMalloc error: " << cudaGetErrorString(err);
      exit(1);
    }
    err = cudaStreamCreate(&stream_);
    if (err != cudaSuccess) {
      LOG(ERROR) << "cudaStreamCreate error: " << cudaGetErrorString(err);
    }
    // 初始化内存链表
    memory_regions =
        std::make_shared<GPUMemoryRegion_V3>(gpu_memory, total_size);
  }

  char* GetBaseAddr() { return memory_regions->addr; }

  void FreeRegion(std::shared_ptr<GPUMemoryRegion_V3> region) {
    region->status = FREE;
    allocated_regions.erase(region->fingerprint);

    // 合并前驱
    if (region->prev && region->prev->status == FREE) {
      region->prev->merge(region);
      region = region->prev;  // 指向合并后的区域
    }

    // 合并后继
    if (region->next && region->next->status == FREE) {
      region->merge(region->next);
    }

    // 更新链表指针
    if (region->prev) {
      region->prev->next = region;
    }
    if (region->next) {
      region->next->prev = region;
    }
  }
  // AllocateFreeRegion: 分配空闲区域
  std::shared_ptr<GPUMemoryRegion_V3> AllocateFreeRegion(
      std::shared_ptr<GPUMemoryRegion_V3> free_region, size_t allocate_size,
      std::string fingerprint) {
    // 1. 检查free_region状态是否为FREE且尺寸足够
    if (free_region->status != FREE || free_region->size < allocate_size) {
      LOG(ERROR) << "AllocateFreeRegion: invalid free_region status or size";
      return nullptr;
    }
    // 2. 如果区域尺寸大于分配要求，则分割区域
    if (free_region->size > allocate_size) {
      auto new_region = std::make_shared<GPUMemoryRegion_V3>(
          free_region->addr + allocate_size, free_region->size - allocate_size);
      new_region->status = FREE;
      new_region->prev = free_region;
      new_region->next = free_region->next;
      if (free_region->next) {
        free_region->next->prev = new_region;
      }
      free_region->next = new_region;
      free_region->size = allocate_size;
    }
    // 3. 更新分配区域状态
    free_region->status = ALLOCATED;
    free_region->fingerprint = fingerprint;

    // 更新allocated_regions
    allocated_regions[fingerprint] = free_region;

    return free_region;
  }

  // UpdateDropCost:
  // 根据model_access计算每个已分配区域的丢弃成本并更新drop_costs
  void UpdateDropCost() {
    drop_costs.clear();
    // LOG(INFO) << "allocated_regions: ";
    // for (auto& pair : allocated_regions) {
    //   LOG(INFO) << pair.first << ": " << pair.second->toString();
    // }
    for (auto& pair : allocated_regions) {
      auto region = pair.second;
      if (!region->model_ref) {
        LOG(ERROR) << "region->model_ref is nullptr";
        return;
      }
      double access_prob =
          (total_access == 0
               ? 1.0
               : (double)model_access[region->model_ref->model_path()] /
                     total_access);
      // 丢弃成本计算公式：区域大小 * 模型访问概率 * 模型装载带宽 *
      // 模型装载敏感度
      double cost = region->size / CPUBandwidth * access_prob *
                    region->model_ref->GetLoadSensitive();
      drop_costs.insert({cost, region});
    }

    // 打印drop_costs
    // LOG(INFO) << "drop_costs: ";
    // for (auto& entry : drop_costs) {
    //   LOG(INFO) << "cost: " << entry.cost
    //             << ", region: " << entry.region->toString();
    // }
  }

  // UseModel: 更新模型访问记录
  void UseModel(const std::string& model_path) {
    model_access[model_path]++;
    total_access++;
  }
  // 遍历区域链表, 收集所有已分配区域,
  // 将已分配区域移动到region_start开始的位置,从而将空闲区域合并在已分配区域的后面,
  // 返回空闲区域
  std::shared_ptr<GPUMemoryRegion_V3> MergeRegions(
      std::shared_ptr<GPUMemoryRegion_V3> region_start,
      std::shared_ptr<GPUMemoryRegion_V3> region_end) {
    if (!region_start || !region_end) {
      LOG(ERROR) << "MergeRegions: Invalid input regions";
      return nullptr;
    }
    if (region_start == region_end) {
      return region_start;
    }
    // 保存边界指针
    auto prev_before = region_start->prev;
    auto next_after = region_end->next;

    bool swapped;
    do {
      swapped = false;
      auto current = region_start;
      while (current && current != region_end->next) {
        if (current->status == FREE) {
          // 合并后重置current指针
          if (current->next && current->next->status == FREE) {
            auto old_next = current->next;
            current->merge(current->next);
            swapped = true;

            // 更新region_end如果被合并的是末尾区域
            if (old_next == region_end) {
              region_end = current;
            }
            continue;  // 合并后重新检查当前节点
          }

          // 增加指针有效性检查
          if (current->next && current->next->status == ALLOCATED) {
            // 原子化更新映射
            if (allocated_regions.count(current->next->fingerprint)) {
              allocated_regions[current->next->fingerprint] = current;
            }
            current->moveSwapAdjacent(current->next);
            swapped = true;

            // 移动后重置current指针
            current = current->prev ? current->prev : region_start;
            continue;
          }
        }
        current = current->next;
      }

      // 增加region_end有效性检查
      if (!region_end || region_end->next == nullptr) {
        break;
      }

    } while (swapped);

    // 优化空闲区域查找逻辑
    auto free_region = region_start;
    size_t total_free = 0;
    while (free_region && free_region != region_end->next) {
      if (free_region->status == FREE) {
        total_free += free_region->size;
        // 合并连续空闲区域
        while (free_region->next && free_region->next->status == FREE) {
          free_region->merge(free_region->next);
        }
      }
      free_region = free_region->next;
    }

    // 返回最大的连续空闲区域
    free_region = region_start;
    std::shared_ptr<GPUMemoryRegion_V3> max_free = nullptr;
    while (free_region && free_region != region_end->next) {
      if (free_region->status == FREE &&
          (!max_free || free_region->size > max_free->size)) {
        max_free = free_region;
      }
      free_region = free_region->next;
    }
    if (!max_free) {
      LOG(ERROR) << "MergeRegions: got null max_free";
      return nullptr;
    }
    LOG(INFO) << "MergeRegions: max_free: " << max_free->toString();
    return max_free;
  }

  std::shared_ptr<GPUMemoryRegion_V3> MergeRegions_V2(
      std::shared_ptr<GPUMemoryRegion_V3> region_start,
      std::shared_ptr<GPUMemoryRegion_V3> region_end) {
    if (!region_start || !region_end) {
      LOG(ERROR) << "MergeRegions: Invalid input regions";
      return nullptr;
    }
    // 保存边界指针
    auto prev_before = region_start->prev;
    auto next_after = region_end->next;

    // 收集范围内所有已分配区域
    // 收集空闲区域大小
    size_t free_region_size = 0;
    std::vector<std::shared_ptr<GPUMemoryRegion_V3>> allocated_regions_tmp;
    for (auto cur = region_start; cur && cur != next_after; cur = cur->next) {
      if (cur->status == ALLOCATED) {
        allocated_regions_tmp.push_back(cur);
      } else if (cur->status == LOADING) {
        LOG(ERROR) << "MergeRegions: CANNOT MERGE LOADING REGION";
        return nullptr;
      } else {
        free_region_size += cur->size;
      }
    }

    if (free_region_size == 0) {
      LOG(ERROR) << "MergeRegions: NO FREE REGION TO MERGE";
      return nullptr;
    }

    // 移动数据并重新排列
    char* current_addr = region_start->addr;
    for (auto& region : allocated_regions_tmp) {
      if (region->addr != current_addr) {
        cudaError_t err =
            cuda_safe_move(current_addr, region->addr, region->size);
        if (err != cudaSuccess) {
          LOG(ERROR) << "cuda_safe_move failed: " << cudaGetErrorString(err);
          return nullptr;
        }
      }
      region->addr = current_addr;
      current_addr += region->size;
    }

    // 重建链表连接
    for (size_t i = 0; i < allocated_regions_tmp.size(); ++i) {
      allocated_regions_tmp[i]->prev =
          (i > 0) ? allocated_regions_tmp[i - 1] : prev_before;
      allocated_regions_tmp[i]->next = (i < allocated_regions_tmp.size() - 1)
                                           ? allocated_regions_tmp[i + 1]
                                           : next_after;
    }

    // 创建新空闲区域
    std::shared_ptr<GPUMemoryRegion_V3> free_region =
        std::make_shared<GPUMemoryRegion_V3>(current_addr, free_region_size);

    // 连接前后指针
    if (!allocated_regions_tmp.empty()) {
      allocated_regions_tmp.back()->next = free_region;
      free_region->prev = allocated_regions_tmp.back();
    } else {
      free_region->prev = prev_before;
    }

    free_region->next = next_after;

    // 更新边界指针
    if (next_after) {
      next_after->prev = free_region;
    }

    if (prev_before) {
      prev_before->next = allocated_regions_tmp.empty()
                              ? free_region
                              : allocated_regions_tmp.front();
    } else {
      // 当前节点是头节点
      memory_regions = allocated_regions_tmp.empty()
                           ? free_region
                           : allocated_regions_tmp.front();
    }
    return free_region;
  }

  struct MergeInfo {
    double T_merge;
    std::shared_ptr<GPUMemoryRegion_V3> start;
    std::shared_ptr<GPUMemoryRegion_V3> end;

    void print() {
      LOG(INFO) << "T_merge: " << T_merge << ", start: " << start->toString()
                << ", end: " << end->toString();
    }
  };

  MergeInfo GetMinMergeCost(std::shared_ptr<GPUMemoryRegion_V3> memory_regions,
                            size_t request_size) {
    // 1. 扫描区域链表
    std::shared_ptr<GPUMemoryRegion_V3> cur = memory_regions;
    while (cur) {
      if (cur->status == FREE && cur->size >= request_size) {
        return MergeInfo{0, cur, cur};
      }
      cur = cur->next;
    }

    // 2. 查找划分点(LOADING状态的区域)和计算左右子空间
    std::shared_ptr<GPUMemoryRegion_V3> loading_region = nullptr;
    std::shared_ptr<GPUMemoryRegion_V3> cur_left = memory_regions;
    size_t left_free_size = 0;
    size_t right_free_size = 0;

    // 查找LOADING状态的区域，同时统计左侧空闲空间
    while (cur_left) {
      if (cur_left->status == FREE) {
        left_free_size += cur_left->size;
      }
      if (cur_left->status == LOADING) {
        loading_region = cur_left;
        break;
      }
      cur_left = cur_left->next;
    }

    // 统计右侧空闲空间
    auto cur_right = loading_region ? loading_region->next : nullptr;
    while (cur_right) {
      if (cur_right->status == FREE) {
        right_free_size += cur_right->size;
      } else if (cur_right->status == LOADING) {
        LOG(ERROR) << "Multiple LOADING regions found";
        return MergeInfo{UNVALID_COST, nullptr, nullptr};
      }
      cur_right = cur_right->next;
    }

    // 3. 计算左右子空间的最小合并时间

    auto calculate_min_merge =
        [this](std::shared_ptr<GPUMemoryRegion_V3> start,
               std::shared_ptr<GPUMemoryRegion_V3> space_end,
               size_t request_size) -> MergeInfo {
      MergeInfo result = {UNVALID_COST, nullptr, nullptr};

      for (auto cur = start; cur && cur != space_end; cur = cur->next) {
        if (cur->status != FREE) continue;

        size_t total_free = cur->size;
        size_t move_size = 0;
        auto next = cur->next;
        auto free_end = cur;

        // 寻找足够的空闲空间组合
        while (next && next != space_end && total_free < request_size) {
          if (next->status == FREE) {
            total_free += next->size;
            free_end = next;
          } else {
            move_size += next->size;
          }
          next = next->next;
        }

        if (total_free >= request_size) {
          double T_merge = move_size / GPUBandwidth;
          if (T_merge < result.T_merge) {
            result = {T_merge, cur, free_end};
          }
        } else {
          break;
        }
      }
      return result;
    };

    MergeInfo left_merge = {UNVALID_COST, nullptr, nullptr};
    MergeInfo right_merge = {UNVALID_COST, nullptr, nullptr};

    if (left_free_size >= request_size) {
      left_merge =
          calculate_min_merge(memory_regions, loading_region, request_size);
    }
    if (right_free_size >= request_size && loading_region) {
      right_merge =
          calculate_min_merge(loading_region->next, nullptr, request_size);
    }
    // 如果左右子空间都无法找到足够的空闲空间组合，返回无效结果
    if (left_merge.T_merge == UNVALID_COST &&
        right_merge.T_merge == UNVALID_COST) {
      return {UNVALID_COST, nullptr, nullptr};
    }
    // 4. 选择最小合并时间方案
    MergeInfo best_merge =
        left_merge.T_merge <= right_merge.T_merge ? left_merge : right_merge;
    return best_merge;
  }

  bool isSameSpace(std::shared_ptr<GPUMemoryRegion_V3> region1,
                   std::shared_ptr<GPUMemoryRegion_V3> region2,
                   const std::shared_ptr<GPUMemoryRegion_V3> loading_region) {
    if (!loading_region) {
      // 如果没有 loading_region，说明整个空间是一个子空间
      return true;
    }

    char* loading_addr = loading_region->addr;
    bool region1_left = region1->addr < loading_addr;
    bool region2_left = region2->addr < loading_addr;

    return region1_left == region2_left;
  }

  // AllocateASAP: 通过贪心方法选择移动成本+丢弃成本最小的方案分配区域
  // 参数:
  //   request_size: 分配请求的大小
  //   T_overlap: 重叠时间阈值
  //   S: 模型装载敏感度
  // 返回:
  //   分配成功返回分配的区域, 失败返回nullptr
  // 存在一个问题: 候选丢弃组的数量可能会爆炸
  std::shared_ptr<GPUMemoryRegion_V3> AllocateASAP(
      size_t request_size, std::string fingerprint, double T_overlap, double S,
      const std::shared_ptr<GPUMemoryRegion_V3>& loading_region) {
    // 计算剩余空间是否足够
    size_t left_free_size = 0;
    size_t right_free_size = 0;
    MergeInfo left_best_merge = {UNVALID_COST, nullptr, nullptr};
    MergeInfo right_best_merge = {UNVALID_COST, nullptr, nullptr};
    std::shared_ptr<GPUMemoryRegion_V3> cur_left = memory_regions;
    while (cur_left) {
      if (cur_left->status == FREE) {
        left_free_size += cur_left->size;
      }
      if (cur_left->status == LOADING) {
        break;
      }
      cur_left = cur_left->next;
    }
    auto cur_right = loading_region ? loading_region->next : nullptr;
    while (cur_right) {
      if (cur_right->status == FREE) {
        right_free_size += cur_right->size;
      } else if (cur_right->status == LOADING) {
        LOG(ERROR) << "Multiple LOADING regions found";
        return nullptr;
      }
      cur_right = cur_right->next;
    }
    // 如果左右子空间都不够, 则需要丢弃已分配区域,
    // 保证具有Move-only的可行解作为初始解
    if (left_free_size < request_size && right_free_size < request_size) {
      LOG(INFO) << "free space is not enough, need to drop regions";

      // 空闲空间不够
      // 按照drop cost的升序选择已分配区域丢弃, 直到选择区域满足请求大小要求
      // 从左右子空间中选择最小成本的丢弃方式
      DropCostGroup left_group;
      DropCostGroup right_group;
      // 遍历drop_costs, 选择丢弃成本最小的区域
      bool is_left_full = false, is_right_full = false;
      for (const auto& entry : drop_costs) {
        if (entry.region->status != ALLOCATED ||
            entry.region == loading_region) {
          continue;
        }
        // 判断属于左侧还是右侧
        if (!loading_region || entry.region->addr < loading_region->addr) {
          if (!is_left_full) {
            left_group.regions.push_back(entry.region);
            left_group.total_cost = entry.cost;
            left_free_size += entry.region->size;
          }
        } else if (entry.region->addr > loading_region->addr) {
          if (!is_right_full) {
            right_group.regions.push_back(entry.region);
            right_group.total_cost = entry.cost;
            right_free_size += entry.region->size;
          }
        }
        if (left_free_size >= request_size) {
          is_left_full = true;
        }
        if (right_free_size >= request_size) {
          is_right_full = true;
        }
        if (is_left_full && is_right_full) {
          break;
        }
      }
      // 选择最小成本的丢弃方式
      DropCostGroup best_drop;
      if (!is_right_full && !is_left_full) {
        LOG(ERROR) << "No enough free space to allocate";
        return nullptr;
      } else if (!is_left_full) {
        best_drop = right_group;
      } else if (!is_right_full) {
        best_drop = left_group;
      } else {
        best_drop = left_group.total_cost < right_group.total_cost
                        ? left_group
                        : right_group;
      }
      // LOG(INFO) << "Request_size: " << request_size;
      // LOG(INFO) << "Loading Region: "
      //           << (loading_region ? loading_region->toString() : "nullptr");
      // LOG(INFO) << "best_drop: ";
      // for (const auto& region : best_drop.regions) {
      //   LOG(INFO) << "region: " << region->toString();
      // }
      // 丢弃best drop的区域
      for (auto& region : best_drop.regions) {
        FreeRegion(region);
      }
    }
    // 首先尝试不丢弃任何区域, 计算最小的合并成本
    MergeInfo best_merge = GetMinMergeCost(memory_regions, request_size);
    double T_merge_min = best_merge.T_merge;
    if (T_merge_min == UNVALID_COST) {
      LOG(ERROR) << "Move-only solution is not feasible even after greedy drop";
      return nullptr;
    }

    // LOG(INFO) << "T_merge_min: " << T_merge_min << ", T_overlap: " <<
    // T_overlap;
    if (T_merge_min <= T_overlap) {
      // best_merge.print();
      // 如果最小合并成本小于T_overlap, 则合并空闲区域并分配
      auto merged_free_region =
          MergeRegions_V2(best_merge.start, best_merge.end);
      return AllocateFreeRegion(merged_free_region, request_size, fingerprint);
    }
    // LOG(INFO) << "T_merge_min: " << T_merge_min
    //           << " > T_overlap: " << T_overlap;
    double Cost_extra = (T_merge_min - T_overlap) * S;
    // LOG(INFO) << "Has a Move Cost with Cost_extra: " << Cost_extra;

    // 构建候选丢弃组
    std::vector<std::vector<DropCostGroup>> candidate_groups;
    std::vector<DropCostGroup> g1;
    double current_cost = 0;
    // 构建G1
    for (const auto& entry : drop_costs) {
      if (entry.cost > Cost_extra) break;
      g1.push_back({entry.cost, {entry.region}});
    }
    if (!g1.empty()) {
      candidate_groups.push_back(g1);
    }
    // 输出G1的内容
    // LOG(INFO) << "G1: ";
    // for (const auto& group : g1) {
    //   LOG(INFO) << "total_cost: " << group.total_cost;
    //   for (const auto& region : group.regions) {
    //     LOG(INFO) << "region: " << region->toString();
    //   }
    // }
    // LOG(INFO) << "G1 size: " << g1.size();
    // 构建所有可能的候选组
    auto build_next_group =
        [&](const std::vector<DropCostGroup>& prev_group,
            const std::vector<DropCostGroup>& g1, double cost_limit,
            const std::shared_ptr<GPUMemoryRegion_V3> loading_region) {
          std::vector<DropCostGroup> next_group;
          for (const auto& prev_comb : prev_group) {
            for (const auto& g1_comb : g1) {
              double new_cost = prev_comb.total_cost + g1_comb.total_cost;
              // 检查是否超过成本限制
              if (new_cost >= cost_limit) break;
              // 检查是否在同一子空间
              if (!isSameSpace(g1_comb.regions.front(),
                               prev_comb.regions.front(), loading_region)) {
                continue;
              }
              // 检查是否有重叠
              auto g1_region = g1_comb.regions.front();
              bool has_overlap = false;
              for (const auto& region : prev_comb.regions) {
                if (g1_region->isSame(region)) {
                  has_overlap = true;
                  break;
                }
              }
              if (has_overlap) continue;

              // 构建新的组合
              DropCostGroup new_comb = prev_comb;
              new_comb.regions.insert(new_comb.regions.end(),
                                      g1_comb.regions.begin(),
                                      g1_comb.regions.end());
              new_comb.total_cost = new_cost;
              next_group.push_back(new_comb);
            }
          }
          return next_group;
        };

    // 构建所有可能的候选组
    if (!candidate_groups.empty()) {
      while (true) {
        auto next_groups = build_next_group(candidate_groups.back(), g1,
                                            Cost_extra, loading_region);
        if (next_groups.empty()) break;
        LOG(INFO) << "G" << candidate_groups.size()
                  << " size: " << next_groups.size();
        candidate_groups.push_back(std::move(next_groups));
      }
    }

    size_t total_drop_groups = 0;
    for (const auto& group : candidate_groups) {
      total_drop_groups += group.size();
    }
    LOG(INFO) << "Found Candidate Drop Groups: " << total_drop_groups;

    // 寻找最优方案
    double min_total_cost = Cost_extra;
    DropCostGroup best_drop_group;
    MergeInfo best_merge_info;

    for (const auto& group : candidate_groups) {
      for (const auto& comb : group) {
        //  暂时修改对应区域的状态为Free
        for (const auto& region : comb.regions) {
          if (region->status != ALLOCATED) {
            LOG(ERROR) << "region->status!= ALLOCATED";
            return nullptr;
          }
          region->status = FREE;
        }
        // 计算合并成本
        auto merge_info = GetMinMergeCost(memory_regions, request_size);
        double T_merge = merge_info.T_merge;

        if (T_merge - T_overlap + comb.total_cost < min_total_cost) {
          min_total_cost = T_merge - T_overlap + comb.total_cost;
          best_drop_group = comb;
          best_merge_info = merge_info;
        }
        // 恢复对应区域的状态
        for (const auto& region : comb.regions) {
          region->status = ALLOCATED;
        }
      }
    }

    // 执行最优方案
    // 丢弃区域
    if (!best_drop_group.regions.empty()) {
      for (const auto& region : best_drop_group.regions) {
        allocated_regions.erase(region->fingerprint);
        region->status = FREE;
      }
    }
    // 合并空闲区域
    auto merged_free_region =
        MergeRegions_V2(best_merge_info.start, best_merge_info.end);
    // 分配区域
    auto allocated_region =
        AllocateFreeRegion(merged_free_region, request_size, fingerprint);
    return allocated_region;
  }

  // 使用贪心的方法构建候选丢弃组
  std::shared_ptr<GPUMemoryRegion_V3> AllocateASAP_V2(
      size_t request_size, std::string fingerprint, std::string model,
      double T_overlap, double S,
      const std::shared_ptr<GPUMemoryRegion_V3>& loading_region) {
    auto start_time = std::chrono::high_resolution_clock::now();
    // 计算剩余空间是否足够
    size_t left_free_size = 0;
    size_t right_free_size = 0;
    MergeInfo left_best_merge = {UNVALID_COST, nullptr, nullptr};
    MergeInfo right_best_merge = {UNVALID_COST, nullptr, nullptr};
    std::shared_ptr<GPUMemoryRegion_V3> cur_left = memory_regions;
    while (cur_left) {
      if (cur_left->status == FREE) {
        left_free_size += cur_left->size;
      }
      if (cur_left->status == LOADING) {
        break;
      }
      cur_left = cur_left->next;
    }
    auto cur_right = loading_region ? loading_region->next : nullptr;
    while (cur_right) {
      if (cur_right->status == FREE) {
        right_free_size += cur_right->size;
      } else if (cur_right->status == LOADING) {
        LOG(ERROR) << "Multiple LOADING regions found";
        return nullptr;
      }
      cur_right = cur_right->next;
    }
    // 如果左右子空间都不够, 则需要丢弃已分配区域,
    // 保证具有Move-only的可行解作为初始解
    if (left_free_size < request_size && right_free_size < request_size) {
      // LOG(INFO) << "free space is not enough, need to drop regions";

      // 空闲空间不够
      // 按照drop cost的升序选择已分配区域丢弃, 直到选择区域满足请求大小要求
      // 从左右子空间中选择最小成本的丢弃方式
      DropCostGroup left_group;
      DropCostGroup right_group;
      // 遍历drop_costs, 选择丢弃成本最小的区域
      bool is_left_full = false, is_right_full = false;
      for (const auto& entry : drop_costs) {
        if (entry.region->status != ALLOCATED ||
            entry.region == loading_region) {
          continue;
        }
        // 判断属于左侧还是右侧
        if (!loading_region || entry.region->addr < loading_region->addr) {
          if (!is_left_full) {
            left_group.regions.push_back(entry.region);
            left_group.total_cost = entry.cost;
            left_free_size += entry.region->size;
          }
        } else if (entry.region->addr > loading_region->addr) {
          if (!is_right_full) {
            right_group.regions.push_back(entry.region);
            right_group.total_cost = entry.cost;
            right_free_size += entry.region->size;
          }
        }
        if (left_free_size >= request_size) {
          is_left_full = true;
        }
        if (right_free_size >= request_size) {
          is_right_full = true;
        }
        if (is_left_full && is_right_full) {
          break;
        }
      }
      // 选择最小成本的丢弃方式
      DropCostGroup best_drop;
      if (!is_right_full && !is_left_full) {
        LOG(ERROR) << "No enough free space to allocate";
        return nullptr;
      } else if (!is_left_full) {
        best_drop = right_group;
      } else if (!is_right_full) {
        best_drop = left_group;
      } else {
        best_drop = left_group.total_cost < right_group.total_cost
                        ? left_group
                        : right_group;
      }
      // LOG(INFO) << "Request_size: " << request_size;
      // LOG(INFO) << "Loading Region: "
      //           << (loading_region ? loading_region->toString() : "nullptr");
      // LOG(INFO) << "best_drop: ";
      // for (const auto& region : best_drop.regions) {
      //   LOG(INFO) << "region: " << region->toString();
      // }
      // 丢弃best drop的区域
      for (auto& region : best_drop.regions) {
        FreeRegion(region);
      }
    }
    // 首先尝试不丢弃任何区域, 计算最小的合并成本
    MergeInfo best_merge = GetMinMergeCost(memory_regions, request_size);
    double T_merge_min = best_merge.T_merge;
    if (T_merge_min == UNVALID_COST) {
      LOG(ERROR) << "Move-only solution is not feasible even after greedy drop";
      return nullptr;
    }

    // LOG(INFO) << "T_merge_min: " << T_merge_min << ", T_overlap: " <<
    // T_overlap;
    std::shared_ptr<GPUMemoryRegion_V3> free_region = nullptr;
    if (T_merge_min <= T_overlap) {
      // 如果最小合并成本小于T_overlap, 则合并空闲区域并分配
      free_region = MergeRegions_V2(best_merge.start, best_merge.end);
    } else {
      double Cost_extra = (T_merge_min - T_overlap) * S;
      // 构建候选丢弃组
      // 替换原有候选组构建逻辑为贪心算法
      std::vector<
          std::pair<double, std::vector<std::shared_ptr<GPUMemoryRegion_V3>>>>
          candidate_drops;

      // 按丢弃成本排序的已分配区域
      std::vector<std::shared_ptr<GPUMemoryRegion_V3>> sorted_regions;
      for (const auto& entry : drop_costs) {
        if (entry.region->status == ALLOCATED &&
            entry.region != loading_region) {
          sorted_regions.push_back(entry.region);
        }
      }

      // 贪心算法：逐步增加丢弃区域，记录每次的总成本
      double accumulated_cost = 0.0;
      std::vector<std::shared_ptr<GPUMemoryRegion_V3>> current_drop;

      for (const auto& region : sorted_regions) {
        // 模拟丢弃当前区域
        current_drop.push_back(region);
        accumulated_cost += region->size / CPUBandwidth *
                            (model_access[region->model_ref->model_path()] /
                             (double)total_access) *
                            region->model_ref->GetLoadSensitive();

        // 临时释放区域并计算合并成本
        region->status = FREE;
        auto merge_info = GetMinMergeCost(memory_regions, request_size);

        // 计算总成本
        double total_cost = accumulated_cost +
                            std::max(0.0, merge_info.T_merge - T_overlap) * S;

        // 记录候选方案
        candidate_drops.emplace_back(total_cost, current_drop);

        // 检查是否满足空间需求
        if (merge_info.T_merge <= T_overlap) {
          // 此时继续丢弃新的区域不可能会带来更好的结果，停止搜索
          break;
        }
      }
      // 恢复已分配区域的状态
      for (const auto& region : current_drop) {
        region->status = ALLOCATED;
      }

      // 找到最小总成本的方案
      auto best_solution = std::min_element(
          candidate_drops.begin(), candidate_drops.end(),
          [](const auto& a, const auto& b) { return a.first < b.first; });

      // 执行最佳丢弃方案
      for (const auto& region : best_solution->second) {
        FreeRegion(region);
      }
      auto merge_info = GetMinMergeCost(memory_regions, request_size);
      free_region = MergeRegions_V2(merge_info.start, merge_info.end);
    }
    auto ret = AllocateFreeRegion(free_region, request_size, fingerprint);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - start_time)
                        .count();

    model_allocation_time[model].first += duration;
    model_allocation_time[model].second += 1;
    return ret;
  }
  void MemoryRegionView() {
    auto region = memory_regions;
    while (region != nullptr) {
      LOG(INFO) << region->toString();
      region = region->next;
    }
  }

  void PrintAverageAllocationTime() {
    for (const auto& pair : model_allocation_time) {
      const std::string& model_path = pair.first;
      long long total_time = pair.second.first;
      int call_count = pair.second.second;
      double average_time = static_cast<double>(total_time) / call_count;
      std::cout << "Model: " << model_path
                << ", Average Allocation Time: " << average_time << " ms"
                << std::endl;
    }
  }
};

class VRAMManager_V3 : public VRAMManagerBase {
 private:
  std::unordered_map<std::string, std::shared_ptr<RegisteredModel>>
      registered_models_;
  std::unordered_map<int, std::shared_ptr<GPUTensorPool_V3>> gpu_tensor_pools_;
  std::mutex mutex_;

  size_t total_tg_access{0};
  size_t tg_hit_count{0};
  size_t total_tg_access_volume{0};
  size_t tg_hit_volume{0};


 public:
  // 构造函数与VRAMManager_V2类似，只不过创建的tensor
  // pool为V3版本，需要传入带宽参数
  VRAMManager_V3(size_t gpu_tensor_pool_size, const std::vector<int>& gpu_ids,
                 double gpu_bw, double cpu_bw) {
    for (int gpu_id : gpu_ids) {
      LOG(INFO) << "Creating GPUTensorPool_V3 for device " << gpu_id;
      gpu_tensor_pools_[gpu_id] = std::make_shared<GPUTensorPool_V3>(
          gpu_id, gpu_tensor_pool_size, gpu_bw, cpu_bw);
    }
  }

  ~VRAMManager_V3() {
    LOG(INFO) << "===== VRAMManager_V2 Destructor =====";
    LOG(INFO) << "Total Tensor Group Accesses: " << total_tg_access;
    LOG(INFO) << "Tensor Group Hit Count: " << tg_hit_count;
    LOG(INFO) << "Tensor Group Hit Rate: "
              << double(tg_hit_count) / total_tg_access;

    LOG(INFO) << "Total Tensor Group Access Volume: "
              << double(total_tg_access_volume) / 1024.0 / 1024.0 / 1024.0
              << " GB";
    LOG(INFO) << "Tensor Group Hit Volume: "
              << double(tg_hit_volume) / 1024.0 / 1024.0 / 1024.0 << " GB";
    LOG(INFO) << "Tensor Group Hit Volume Rate: "
              << double(tg_hit_volume) / total_tg_access_volume;

    LOG(INFO) << "Model Average Allocation Time:";
    for (auto& pair : gpu_tensor_pools_) {
      pair.second->PrintAverageAllocationTime();
    }
  }

  int64_t RegisterModel(const std::string& model_path, int sensitive = 1) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (registered_models_.find(model_path) != registered_models_.end()) {
      LOG(WARNING) << "Model already registered: " << model_path;
      return registered_models_[model_path]->model_size();
    }
    auto model = std::make_shared<RegisteredModel>(model_path, sensitive);
    model->MergeTGs(100LL * 1024 * 1024);

    if (model->LoadModelFromDisk(8) != 0) {
      return -1;
    }
    registered_models_[model_path] = model;
    LOG(INFO) << "Model " << model_path
              << " registered, size: " << model->model_size();
    return model->model_size();
  }

  // MemoryUsage 与 VRAMManager_V2 保持一致
  void MemoryUsage() {
    for (auto& pair : gpu_tensor_pools_) {
      LOG(INFO) << "Device ID: " << pair.first;
      pair.second->MemoryRegionView();
    }
  }

  // LoadModel 实现第三版逻辑
  std::string LoadModel(const std::string& model_path, int device_id) {
    std::unique_lock<std::mutex> lock(mutex_);
    auto model_it = registered_models_.find(model_path);
    if (model_it == registered_models_.end()) {
      LOG(ERROR) << "Model not registered: " << model_path;
      return "ERROR";
    }

    auto pool_it = gpu_tensor_pools_.find(device_id);
    if (pool_it == gpu_tensor_pools_.end()) {
      LOG(ERROR) << "Invalid device_id: " << device_id;
      return "ERROR";
    }

    auto& pool = pool_it->second;
    auto& model = model_it->second;

    // 通过模型获取所有TensorGroup的fingerprint
    const auto& tg_index = model->GetTensorGroupIndexes();
    std::vector<TensorGroupIndex> tg_to_allocate;
    std::vector<int> tg_to_load;
    std::vector<char*> allocated_regions(tg_index.size(), nullptr);
    for (int i = 0; i < tg_index.size(); i++) {
      auto mem_region = pool->allocated_regions.count(tg_index[i].fingerprint)
                            ? pool->allocated_regions[tg_index[i].fingerprint]
                            : nullptr;
      if (!mem_region) {
        tg_to_allocate.push_back(tg_index[i]);
        tg_to_load.push_back(i);
      } else {
        allocated_regions[i] = mem_region->addr;
        tg_hit_count++;
        tg_hit_volume+= tg_index[i].size;
      }
      total_tg_access++;
      total_tg_access_volume+= tg_index[i].size;
    }
    LOG(INFO) << "LoadModel: " << model_path
              << ", groups to load: " << tg_to_load.size();

    // 5. 遍历待装载列表
    std::shared_ptr<GPUMemoryRegion_V3> loading_region = nullptr;
    auto host_ptrs = model->GetTensorGroupHostPtr();
    for (int idx = 0; idx < tg_to_allocate.size(); idx++) {
      auto tg = tg_to_allocate[idx];
      auto tg_id = tg_to_load[idx];
      if (idx == 0) {
        // 5.1 第一个TG，直接分配，不考虑重叠延时
        auto region =
            pool->AllocateASAP_V2(tg.size, tg.fingerprint, model_path, 0,
                                  model->GetLoadSensitive(), nullptr);
        if (!region) {
          LOG(ERROR) << "Allocation failed for TG index " << idx;
          pool->MemoryRegionView();
          return "ERROR";
        }
        allocated_regions[tg_id] = region->addr;
        region->model_ref = model;
        loading_region = region;
      }

      // 5.2 根据TG大小和CPUBandwidth计算T_overlap
      double T_overlap = tg.size / pool->CPUBandwidth;
      auto prev_loading_region = loading_region;
      prev_loading_region->status = LOADING;
      // 5.3 异步拷贝数据，从CPU内存到GPU内存
      cudaError_t err =
          cudaMemcpyAsync(loading_region->addr, host_ptrs->get(tg_id), tg.size,
                          cudaMemcpyHostToDevice, pool->stream_);
      if (err != cudaSuccess) {
        LOG(ERROR) << "cudaMemcpyAsync failed: " << cudaGetErrorString(err)
                   << " TG index: " << idx;
        return "ERROR";
      }
      // 5.4 为下一个TG分配空间
      if (idx < tg_to_allocate.size() - 1) {
        auto region = pool->AllocateASAP_V2(
            tg_to_allocate[idx + 1].size, tg_to_allocate[idx + 1].fingerprint,
            model_path, T_overlap, model->GetLoadSensitive(), loading_region);
        if (!region) {
          LOG(ERROR) << "Allocation failed for TG index " << idx + 1
                     << " Size: " << tg_to_allocate[idx + 1].size;
          return "ERROR";
        }
        allocated_regions[tg_to_load[idx + 1]] = region->addr;
        region->model_ref = model;
        loading_region = region;
      }

      // 5.5 同步流确保数据拷贝结束
      err = cudaStreamSynchronize(pool->stream_);
      if (err != cudaSuccess) {
        LOG(ERROR) << "cudaStreamSynchronize failed: "
                   << cudaGetErrorString(err) << " TG index: " << idx;
      }
      prev_loading_region->status = ALLOCATED;
    }
    // 6. 更新模型使用情况及更新drop_costs
    pool->UseModel(model_path);
    pool->UpdateDropCost();

    // 生成返回字符串（简化示例）
    std::string ret;
    cudaIpcMemHandle_t handle;

    // 设置设备，并检查错误
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
      LOG(ERROR) << "cudaSetDevice error: " << cudaGetErrorString(err);
      return "ERROR";
    }

    // 获取 IPC 内存句柄，并检查错误
    err = cudaIpcGetMemHandle(&handle, pool->GetBaseAddr());
    if (err != cudaSuccess) {
      LOG(ERROR) << "cudaIpcGetMemHandle error: " << cudaGetErrorString(err);
      return "ERROR";
    }

    std::string handle_str(reinterpret_cast<const char*>(&handle),
                           sizeof(cudaIpcMemHandle_t));
    ret = toHex(std::vector<uint8_t>(handle_str.begin(), handle_str.end()));

    std::vector<size_t> response;
    response.push_back(device_id);

    // 检查 allocated_regions 与 tg_index 大小是否匹配
    if (allocated_regions.size() != tg_index.size()) {
      LOG(ERROR)
          << "Mismatch between allocated regions and tensor group indexes";
      return "ERROR";
    }

    for (int i = 0; i < allocated_regions.size(); i++) {
      // 检查当前分配区域是否有效
      if (!allocated_regions[i]) {
        LOG(ERROR) << "Allocated region for tensor group " << i << " is null";
        return "ERROR";
      }
      for (auto& tensor : tg_index[i].tensor_indexes) {
        size_t offset =
            allocated_regions[i] - pool->GetBaseAddr() + tensor.offset;
        response.push_back(offset);
      }
    }

    std::string response_str(reinterpret_cast<const char*>(response.data()),
                             response.size() * sizeof(size_t));
    ret +=
        toHex(std::vector<uint8_t>(response_str.begin(), response_str.end()));

    return ret;
  }
};