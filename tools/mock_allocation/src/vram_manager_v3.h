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
enum RegionStatus { FREE = 0, LOADING = 1, ALLOCATED = 2 };
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
           ", status=" + std::to_string(status) + "]";
  }

  bool isSame(std::shared_ptr<GPUMemoryRegion_V3> other) {
    return this->addr == other->addr;
  }
};
struct DropCostEntry {
  double cost;  // 丢弃成本计算值
  std::shared_ptr<GPUMemoryRegion_V3> region;
  bool operator<(const DropCostEntry& other) const {
    // 按cost升序排列
    return cost < other.cost;
  }
};

struct DropCostGroup {
  double total_cost;  // 该组区域的总丢弃成本
  std::vector<std::shared_ptr<GPUMemoryRegion_V3>> regions;  // 该组区域
  bool operator<(const DropCostGroup& other) const {
    // 按total_cost升序排列
    return total_cost < other.total_cost;
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
  double GPUBandwidth;                 // GPU-GPU数据移动带宽
  double CPUBandwidth;                 // CPU->GPU数据拷贝带宽
  std::set<DropCostEntry> drop_costs;  // 有序容器，存储已分配区域的丢弃成本

  int device_id;
  cudaStream_t stream_;

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

  // AllocateFreeRegion: 分配空闲区域
  std::shared_ptr<GPUMemoryRegion_V3> AllocateFreeRegion(
      std::shared_ptr<GPUMemoryRegion_V3> free_region, size_t allocate_size) {
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
    return free_region;
  }

  // UpdateDropCost:
  // 根据model_access计算每个已分配区域的丢弃成本并更新drop_costs
  void UpdateDropCost() {
    drop_costs.clear();
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
      double cost = region->size * access_prob *
                    region->model_ref->GetLoadPenalty() *
                    region->model_ref->GetLoadSensitive();
      drop_costs.insert({cost, region});
    }
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
      LOG(ERROR) << "MergeRegions: region_start or region_end is nullptr";
      return nullptr;
    }

    std::vector<std::shared_ptr<GPUMemoryRegion_V3>> allocated_regions;
    // 收集所有已分配区域
    for (auto cur = region_start; cur && cur != region_end->next;
         cur = cur->next) {
      if (cur->status == ALLOCATED) {
        allocated_regions.push_back(cur);
      }
    }

    char* current_addr = region_start->addr;
    std::shared_ptr<GPUMemoryRegion_V3> prev_allocated = nullptr;

    // 移动已分配区域到 region_start 开始的位置
    for (auto& region : allocated_regions) {
      if (region->addr != current_addr) {
        cudaError_t err =
            cuda_safe_move(current_addr, region->addr, region->size);
        if (err != cudaSuccess) {
          LOG(ERROR) << "cuda_safe_move failed: " << cudaGetErrorString(err);
          return nullptr;
        }
      }

      // 更新区域信息
      region->addr = current_addr;
      region->prev = prev_allocated;
      if (prev_allocated) {
        prev_allocated->next = region;
      }
      prev_allocated = region;
      current_addr += region->size;
    }

    // 创建一个新的空闲区域
    std::shared_ptr<GPUMemoryRegion_V3> free_region = nullptr;
    if (current_addr < region_end->addr + region_end->size) {
      free_region = std::make_shared<GPUMemoryRegion_V3>(
          current_addr, (region_end->addr + region_end->size) - current_addr);
      free_region->status = FREE;
      free_region->prev = prev_allocated;
      if (prev_allocated) {
        prev_allocated->next = free_region;
      }
      free_region->next = region_end->next;
      if (region_end->next) {
        region_end->next->prev = free_region;
      }
    } else {
      if (prev_allocated) {
        prev_allocated->next = region_end->next;
      }
      if (region_end->next) {
        region_end->next->prev = prev_allocated;
      }
    }

    return free_region;
  }

  struct MergeInfo {
    double T_merge;
    std::shared_ptr<GPUMemoryRegion_V3> start;
    std::shared_ptr<GPUMemoryRegion_V3> end;
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
  std::shared_ptr<GPUMemoryRegion_V3> AllocateASAP(
      size_t request_size, double T_overlap, double S,
      const std::shared_ptr<GPUMemoryRegion_V3>& loading_region) {
    // 首先尝试不丢弃任何区域, 计算最小的合并成本
    MergeInfo best_merge = GetMinMergeCost(memory_regions, request_size);
    double T_merge_min = best_merge.T_merge;

    if (T_merge_min <= T_overlap) {
      // 如果最小合并成本小于T_overlap, 则合并空闲区域并分配
      auto merged_free_region = MergeRegions(best_merge.start, best_merge.end);
      return AllocateFreeRegion(merged_free_region, request_size);
    }

    // 需要丢弃
    // 当前模型的装载时延敏感度
    double Cost_extra = (T_merge_min - T_overlap) * S;

    // 构建候选丢弃组
    std::vector<std::vector<DropCostGroup>> candidate_groups;
    std::vector<DropCostGroup> g1;
    double current_cost = 0;
    // 构建G1
    for (const auto& entry : drop_costs) {
      if (entry.cost >= Cost_extra) break;
      current_cost += entry.cost;
      g1.push_back({current_cost, {entry.region}});
    }
    if (!g1.empty()) {
      candidate_groups.push_back(g1);
    }
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
                if (g1_region->isSame(region)) has_overlap = true;
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
        MergeRegions(best_merge_info.start, best_merge_info.end);
    // 分配区域
    auto allocated_region =
        AllocateFreeRegion(merged_free_region, request_size);
    return allocated_region;
  }
};

class VRAMManager_V3 : public VRAMManagerBase {
 private:
  std::unordered_map<std::string, std::shared_ptr<RegisteredModel>>
      registered_models_;
  std::unordered_map<int, std::shared_ptr<GPUTensorPool_V3>> gpu_tensor_pools_;
  std::mutex mutex_;

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

  ~VRAMManager_V3() { LOG(INFO) << "VRAMManager_V3 Destructor"; }

  int64_t RegisterModel(const std::string& model_path, int sensitive = 1) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (registered_models_.find(model_path) != registered_models_.end()) {
      LOG(WARNING) << "Model already registered: " << model_path;
      return registered_models_[model_path]->model_size();
    }
    auto model = std::make_shared<RegisteredModel>(model_path, sensitive);
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
      // 此处可以添加遍历显示 tensor pool 内区域信息的逻辑
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
      }
    }
    LOG(INFO) << "LoadModel: " << model_path
              << ", groups to load: " << tg_to_load.size();

    // 5. 遍历待装载列表
    for (int idx = 0; idx < tg_to_allocate.size(); idx++) {
      size_t request_size = tg_to_allocate[idx].size;
      if (idx == 0) {
        // 5.1 第一个TG，直接分配，不考虑重叠延时
        auto region = pool->AllocateASAP(request_size, 0,
                                         model->GetLoadSensitive(), nullptr);
        if (!region) {
          LOG(ERROR) << "Allocation failed for TG index " << idx;
          return "ERROR";
        }
        allocated_regions[tg_to_load[idx]] = region->addr;
      }

      // 5.2 根据TG大小和CPUBandwidth计算T_overlap
      double T_overlap = request_size / pool->CPUBandwidth;
      // 5.3 异步拷贝数据，从CPU内存到GPU内存
      std::vector<int> load0{idx};
      if (model->LoadModelFromMem(allocated_regions, load0, device_id)) {
        LOG(ERROR) << "LoadModelFromMem failed";
        return "ERROR";
      }

      // 5.4 为下一个TG分配空间
      auto region = pool->AllocateASAP(request_size, T_overlap,
                                       model->GetLoadSensitive(), );
      if (!region) {
        LOG(ERROR) << "Allocation failed for TG index " << idx;
        return "ERROR";
      }
      allocated_regions[tg_to_load[idx]] = region->addr;
      // 5.5 同步流确保数据拷贝结束
      cudaStreamSynchronize(pool->stream_);
    }
    // 6. 更新模型使用情况及更新drop_costs
    pool->UseModel(model_path);
    pool->UpdateDropCost();

    // 生成返回字符串（简化示例）
    std::string ret = "OK";
    return ret;
  }
};