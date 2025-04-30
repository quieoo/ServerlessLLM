/*
*********************************************************
第二版的VRAM Manger("Global")
主要特征:
- Cost Model指导的内存分配(主要是Drop Cost)
- 支持数据拷贝前的去碎片化, 通过移动已分配的内存区域来实现
- 所有模型都已缓存在主机内存中
*********************************************************

GPUMemoryRegion_V2:
在V1的基础上额外记录TG所在的模型引用, 支持快速访问区域的成本

GPUTensorPool_V2:
成员变量:
    memory_regions: 记录张量池的使用情况
    allocated_regions: 支持快速访问已分配的TG
    model_access: 记录历史的模型访问, 用来估计模型的访问概率
成员方法:
    GetTGByFingerprint(fingerprint):
        1. 根据fingerprint找到对应的TG
        2. 如果TG不在allocated_regions中, 返回空指针
        3. 如果TG在allocated_regions中, 返回TG引用
    CostAwareDrop(size_t requested_size):
        1. 判断空闲空间是否足够, 若是足够可以直接退出
        2. 如果空闲空间不够, 计算需要释放的空间
        3. 计算当前每个已分配区域的释放收益(区域大小)和成本, 成本计算方式为
            成本 = 区域大小 * 模型的装载惩罚(RegisteredModel.GetLoadPenalty()) *
模型访问概率(模型访问次数/总访问次数) *
模型的敏感度(RegisteredModel.GetLoadSensitive())
        4. 选择需要释放的区域, 要求释放大小大于等于需要释放的空间,
并且所有释放区域的成本之和最小
        5. 释放区域, 更新allocated_regions和memory_regions
    GlobalDeFrag():
        1. 遍历所有的已分配区域, 将所有的Allocated区域移动到左侧
        2. 合并所有的Free区域
        3. 更新allocated_regions和memory_regions
    BatchedAllocate(vector<size_t> requested_sizes):
        1. 确保空闲区域大小足够
        2. 遍历所有的请求, 分配空间
        3. 更新allocated_regions和memory_regions, 更新model_access
        4. 返回分配的区域

VRAMManager_V2:
成员变量:
    registered_models_: 所有注册的模型, 维护所有模型在CPU内存中的空间
    gpu_tensor_pools_: 所有GPU张量池

成员方法:
    VRAMManager_V2(size_t gpu_tensor_pool_size, vector<int> gpu_ids):
        根据gpu_ids创建GPU张量池, 每个池子的大小为gpu_tensor_pool_size
    RegisterModel(model_path, sensitivite):
        初始化RegisteredModel对象并保存,
调用RegisteredModel的LoadModelFromDisk方法将模型参数装入主机内存(默认主机内存足够大)
    LoadModel(model_path, device_id):
        1. 检查模型是否注册, 检查device_id是否在gpu_tensor_pools_中
        2. 访问registered_models_中的模型, 获得所有TG的fingerprint
        3. 遍历所有TG, 检查是否在allocated_regions中, 如果不在则加入待装载列表
        4. 调用CostAwareDrop释放空间
        5. 如果需要去碎片化, 调用GlobalDeFrag
        6. 调用BatchedAllocate分配空间
        7. 调用LoadModelFromMem将模型参数从主机内存复制到GPU内存
        8. 创建返回字符串, 包括每个TG的偏移量
*/
#pragma once

#include <cuda_runtime.h>

#include <chrono>
#include <unordered_set>

#include "logger.h"
#include "registered_model.h"
#include "vram_manager_base.h"
using namespace std;
class GPUMemoryRegion_V2
    : public std::enable_shared_from_this<GPUMemoryRegion_V2> {
 public:
  bool is_allocated;
  char* addr;
  size_t size;
  std::string fingerprint;
  std::shared_ptr<RegisteredModel> model_ref;  // 新增:指向所属模型的引用
  std::shared_ptr<GPUMemoryRegion_V2> prev;
  std::shared_ptr<GPUMemoryRegion_V2> next;

  GPUMemoryRegion_V2()
      : is_allocated(false), size(0), prev(nullptr), next(nullptr) {}

  GPUMemoryRegion_V2(void* addr_, size_t size)
      : is_allocated(false), size(size), prev(nullptr), next(nullptr) {
    this->addr = static_cast<char*>(addr_);
  }

  void SetAddr(void* addr) { this->addr = static_cast<char*>(addr); }

  bool isAdjacent(std::shared_ptr<GPUMemoryRegion_V2> other) {
    return (static_cast<char*>(this->addr) + this->size == other->addr) ||
           (static_cast<char*>(other->addr) + other->size == this->addr);
  }

  void merge(std::shared_ptr<GPUMemoryRegion_V2> other) {
    if (this->isAdjacent(other)) {
      // 判断other在右侧还是左侧
      if (static_cast<char*>(this->addr) + this->size ==
          other->addr) {  // other在右侧
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
    return "[GPUMemoryRegion_V2: addr=" +
           std::to_string(reinterpret_cast<size_t>(addr)) +
           ", size=" + std::to_string(size) + ", fingerprint=" + fingerprint +
           ", model_ref=" + (model_ref ? model_ref->model_path() : "null") +
           ", is_allocated=" + std::to_string(is_allocated) + "]";
  }
};

class GPUTensorPool_V2 {
 public:
  std::shared_ptr<GPUMemoryRegion_V2> memory_region_view;
  std::unordered_map<std::string, std::shared_ptr<GPUMemoryRegion_V2>>
      allocated_regions;
  std::unordered_map<std::string, size_t> model_access;
  size_t total_access{0};
  size_t total_size;
  int device_id;
  cudaStream_t stream_;

  // 存储每个模型的总时间（毫秒）和调用次数
  std::unordered_map<std::string, std::pair<long long, int>>
      model_allocation_time;

  GPUTensorPool_V2(int device_id_, size_t total_size_)
      : device_id(device_id_), total_size(total_size_) {
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

    auto initial_region =
        std::make_shared<GPUMemoryRegion_V2>(gpu_memory, total_size);
    memory_region_view = initial_region;
  }

  char* GetBaseAddr() { return memory_region_view->addr; }

  std::shared_ptr<GPUMemoryRegion_V2> GetTGByFingerprint(
      const std::string& fingerprint) {
    auto it = allocated_regions.find(fingerprint);
    if (it == allocated_regions.end()) {
      return nullptr;
    }
    return it->second;
  }

  size_t gcd(size_t a, size_t b) {
    while (b != 0) {
      size_t temp = b;
      b = a % b;
      a = temp;
    }
    return a;
  }
  // 计算多个数的最大公约数
  size_t findGCD(const std::vector<size_t>& nums) {
    if (nums.empty()) return 1;
    size_t result = nums[0];
    for (size_t i = 1; i < nums.size(); i++) {
      result = gcd(result, nums[i]);
      if (result == 1) break;
    }
    return result;
  }

  bool CostAwareDrop_DP(size_t request_size) {
    // 1. Calculate current free space
    size_t free_size = 0;
    for (auto region = memory_region_view; region != nullptr;
         region = region->next) {
      if (!region->is_allocated) {
        free_size += region->size;
      }
    }
    if (free_size >= request_size) {
      return true;
    }
    const size_t need_to_free = request_size - free_size;

    // 2. Collect candidate regions with cost
    vector<pair<size_t, double>> candidates;
    vector<size_t> sizes;  // 用于计算GCD
    for (auto& [_, region] : allocated_regions) {
      if (!region->model_ref) continue;

      double access_prob = 1.0;
      if (total_access > 0)
        access_prob =
            static_cast<double>(model_access[region->model_ref->model_path()]) /
            total_access;
      double cost = region->size * region->model_ref->GetLoadPenalty() *
                    access_prob * region->model_ref->GetLoadSensitive();
      candidates.emplace_back(region->size, cost);
      sizes.push_back(region->size);
    }
    if (candidates.empty()) return false;
    // 3. 计算所有size的GCD
    size_t block_size = sizes[0];
    for (size_t s : sizes) {
      block_size = gcd(block_size, s);  // C++17起支持std::gcd
    }
    LOG(INFO) << "block_size: " << block_size;

    // 4. 转换为block单位
    const size_t need_blocks =
        (need_to_free + block_size - 1) / block_size;  // 向上取整
    vector<int> weights;                               // 存储block数量
    vector<int> values;
    int max_block_weight = 0;

    for (auto& [size, cost] : candidates) {
      int blocks = static_cast<int>(size / block_size);
      weights.push_back(blocks);
      int value = static_cast<int>(cost * 100);  // 保持原精度
      if (value < 0) {
        LOG(ERROR) << "cost: " << cost << ", value: " << value;
      }
      values.push_back(static_cast<int>(cost * 100));  // 保持原精度
      max_block_weight = max(max_block_weight, blocks);
    }

    // 5. 动态规划数组优化
    const int dp_size =
        min(need_blocks + max_block_weight,
            static_cast<size_t>(need_blocks * 2));  // 限制范围避免过度扩展
    vector<int> dp(dp_size + 1, INT_MAX / 2);
    dp[0] = 0;
    vector<int> last_updated(dp_size + 1, -1);

    // 6. DP过程
    for (int i = 0; i < candidates.size(); ++i) {
      LOG(INFO) << "DP iteration: " << i << ", weight: " << weights[i]
                << ", value:" << values[i];
      const int w = weights[i];
      const int c = values[i];

      // 处理可以直接满足需求的区块
      if (w >= need_blocks) {
        if (c < dp[need_blocks]) {
          dp[need_blocks] = c;
          last_updated[need_blocks] = i;
        }
        continue;
      }

      // 逆向更新DP数组
      for (int j = min(dp_size - w, static_cast<int>(need_blocks)); j >= 0;
           --j) {
        if (dp[j] + c < dp[j + w]) {
          dp[j + w] = dp[j] + c;
          last_updated[j + w] = i;
        }
      }
    }

    // 7. 寻找最优解（实际需求可能被放大，需验证是否满足）
    int best_j = -1;
    int min_cost = INT_MAX;
    for (int j = need_blocks; j <= dp_size; ++j) {
      if (dp[j] < min_cost) {
        min_cost = dp[j];
        best_j = j;
      }
    }

    // 8. 验证实际释放空间是否足够
    if (best_j == -1 || (best_j * block_size) < need_to_free) {
      LOG(ERROR) << "GCD optimization failed, fallback to original method";
      // 这里可以添加回退到原始方法的逻辑
      return false;
    }

    // 7. Backtrack using last_updated array
    int remaining = need_to_free;
    while (remaining > 0) {
      // Find the closest reachable position
      int pos = remaining;
      while (pos > 0 && last_updated[pos] == -1) pos--;

      if (pos <= 0) break;  // No valid path

      const int i = last_updated[pos];
      const int s = weights[i];

      // Free the corresponding region
      auto it = allocated_regions.begin();
      advance(it, i);
      FreeRegion(it->second);

      // Update remaining capacity
      remaining -= s;
      if (remaining < 0) remaining = 0;
    }

    return true;
  }

  int64_t CostAwareDrop_Greedy(size_t requested_size) {
    // 1. 计算当前空闲空间
    size_t free_size = 0;
    size_t total_cost = 0;
    for (auto region = memory_region_view; region != nullptr;
         region = region->next) {
      if (!region->is_allocated) {
        free_size += region->size;
      }
    }

    if (free_size >= requested_size) {
      LOG(INFO) << "Free size is enough: " << free_size
                << ", requested size: " << requested_size;
      return 0;
    }

    size_t need_to_free = requested_size - free_size;

    // 2. 收集所有已分配区域的信息和成本
    std::vector<std::tuple<double, std::shared_ptr<GPUMemoryRegion_V2>, size_t>>
        candidates;
    size_t total_allocated_size = 0;
    for (auto& [_, region] : allocated_regions) {
      if (!region->model_ref) continue;
      double access_prob = 1.0;
      if (total_access > 0)
        access_prob =
            static_cast<double>(model_access[region->model_ref->model_path()]) /
            total_access;
      double cost = region->size * region->model_ref->GetLoadPenalty() *
                    access_prob * region->model_ref->GetLoadSensitive();
      candidates.emplace_back(cost, region, region->size);
      total_allocated_size += region->size;
    }

    if (total_allocated_size < need_to_free || candidates.empty()) {
      LOG(INFO) << "No enough allocated regions to free";
      return -1;
    }

    // 3. 按cost升序排序
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) {
                return std::get<0>(a) < std::get<0>(b);
              });

    // 4. 贪心释放
    size_t freed = 0;
    for (auto& [cost, region, size] : candidates) {
      // LOG(INFO) << "Freeing region: " << region->toString();
      FreeRegion(region);
      total_cost += cost;
      freed += size;
      if (freed >= need_to_free) break;
    }

    if (freed < need_to_free) {
      LOG(ERROR) << "No valid solution found for freeing memory";
      return -1;
    }

    LOG(INFO) << "CostAwareDrop: Freed " << freed << " bytes, requested "
              << requested_size << ", free size: " << free_size
              << ", total_cost: " << total_cost;
    return freed;
  }

  int GlobalDeFrag() {
    if (!memory_region_view) return 1;
    size_t total_move_data = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    // 1. 收集所有区域信息
    struct RegionInfo {
      bool is_allocated;
      size_t size;
      char* old_addr;
      std::string fingerprint;
      std::shared_ptr<RegisteredModel> model_ref;
    };
    std::vector<RegionInfo> regions;

    auto current = memory_region_view;
    while (current) {
      regions.push_back({current->is_allocated, current->size, current->addr,
                         current->fingerprint, current->model_ref});
      current = current->next;
    }

    // 2. 计算总空间和已分配空间
    char* base_addr = memory_region_view->addr;
    size_t total_allocated = 0;
    size_t total_free = 0;

    for (const auto& r : regions) {
      if (r.is_allocated) {
        total_allocated += r.size;
      } else {
        total_free += r.size;
      }
    }

    // 3. 重建链表 - 直接从第一个已分配区域开始
    char* current_addr = base_addr;
    std::shared_ptr<GPUMemoryRegion_V2> prev = nullptr;
    std::shared_ptr<GPUMemoryRegion_V2> head = nullptr;

    // 4. 首先处理已分配区域
    for (const auto& r : regions) {
      if (!r.is_allocated) continue;

      auto new_region =
          std::make_shared<GPUMemoryRegion_V2>(current_addr, r.size);
      new_region->is_allocated = true;
      new_region->fingerprint = r.fingerprint;
      new_region->model_ref = r.model_ref;

      if (!head) {
        head = new_region;
      } else {
        new_region->prev = prev;
        prev->next = new_region;
      }

      // 如果需要移动数据
      if (r.old_addr != current_addr) {
        total_move_data += r.size;
        cudaError_t err = cuda_safe_move(current_addr, r.old_addr, r.size);
        if (err != cudaSuccess) {
          LOG(ERROR) << "cuda_safe_move error: " << cudaGetErrorString(err);
          return 1;
        }
      }

      // 更新映射
      if (!r.fingerprint.empty()) {
        allocated_regions[r.fingerprint] = new_region;
      }

      prev = new_region;
      current_addr += r.size;
    }

    // 5. 如果有剩余空间，添加一个自由区域
    if (total_free > 0) {
      auto free_region =
          std::make_shared<GPUMemoryRegion_V2>(current_addr, total_free);
      free_region->is_allocated = false;
      if (!head) {
        // 如果没有已分配区域，这个自由区域就是头节点
        head = free_region;
      } else {
        free_region->prev = prev;
        prev->next = free_region;
      }
    }

    // 6. 更新视图指针
    memory_region_view = head;
    // LOG(INFO) << "GlobalDeFrag: Merged all free regions";
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                        end_time - start_time)
                        .count();
    LOG(INFO) << "GlobalDeFrag: Total move data: "
              << double(total_move_data) / 1024 / 1024 / 1024
              << "GB, Time: " << duration << "us";
    return 0;
  }

  std::vector<std::shared_ptr<GPUMemoryRegion_V2>> BatchedAllocate(
      const std::vector<TensorGroupIndex> request_tgs,
      std::shared_ptr<RegisteredModel> model) {
    auto start_time = std::chrono::high_resolution_clock::now();
    // 1. 确保空间足够
    size_t total_requested = 0;
    for (auto tg : request_tgs) {
      total_requested += tg.size;
    }

    auto ret = CostAwareDrop_Greedy(total_requested);
    if (ret < 0) {
      LOG(ERROR) << "CostAwareDrop_Greedy failed";
      return {};
    }
    if (GlobalDeFrag()) {
      LOG(ERROR) << "GlbalDe Frag Failed";
      return {};
    }
    // LOG(INFO)<<"See Memory";
    // MemoryRegionView();

    // 2. 分配空间
    std::vector<std::shared_ptr<GPUMemoryRegion_V2>> results;
    for (auto tg : request_tgs) {
      auto region = AllocateRegion(tg.size);
      if (!region) {
        // 回滚已分配的空间
        for (auto& r : results) {
          FreeRegion(r);
        }
        return {};
      }
      region->fingerprint = tg.fingerprint;
      region->model_ref = model;
      allocated_regions[region->fingerprint] = region;
      results.push_back(region);
    }

    // 3. 更新访问计数
    if (model) {
      model_access[model->model_path()]++;
    }
    total_access++;

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - start_time)
                        .count();

    model_allocation_time[model->model_path()].first += duration;
    model_allocation_time[model->model_path()].second += 1;
    return results;
  }
  std::shared_ptr<GPUMemoryRegion_V2> AllocateRegion(size_t size) {
    for (auto region = memory_region_view; region != nullptr;
         region = region->next) {
      if (!region->is_allocated && region->size >= size) {
        if (region->size > size) {
          auto new_region = std::make_shared<GPUMemoryRegion_V2>(
              region->addr + size, region->size - size);
          new_region->prev = region;
          new_region->next = region->next;
          if (region->next) {
            region->next->prev = new_region;
          }
          region->next = new_region;
          region->size = size;
        }
        region->is_allocated = true;
        return region;
      }
    }
    return nullptr;
  }

  void FreeRegion(std::shared_ptr<GPUMemoryRegion_V2> region) {
    region->is_allocated = false;
    allocated_regions.erase(region->fingerprint);

    // 合并前驱
    if (region->prev && !region->prev->is_allocated) {
      region->prev->merge(region);
      region = region->prev;  // 指向合并后的区域
    }

    // 合并后继
    if (region->next && !region->next->is_allocated) {
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

  void MemoryRegionView() {
    auto region = memory_region_view;
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

  void TestComplexAllocations() {
    LOG(INFO) << "\n\n===== Starting Complex Allocation Test =====";

    // 初始内存状态（完整空闲区域）
    LOG(INFO) << "=== Initial Memory (Full Free) ===";
    MemoryRegionView();

    // 测试1: 分配不同大小的区域
    const std::vector<size_t> alloc_sizes = {
        16 * 1024 * 1024,  // 16MB
        32 * 1024 * 1024,  // 32MB
        8 * 1024 * 1024,   // 8MB
        64 * 1024 * 1024   // 64MB
    };

    std::vector<std::shared_ptr<GPUMemoryRegion_V2>> regions;
    for (const auto& size : alloc_sizes) {
      auto region = AllocateRegion(size);
      if (!region) {
        LOG(ERROR) << "Allocation failed for size: " << size;
        return;
      }
      regions.push_back(region);
    }

    LOG(INFO) << "\n=== After Initial Allocations ===";
    MemoryRegionView();
    /* 预期状态：
       [Allocated 16MB] -> [Allocated 32MB] -> [Allocated 8MB] -> [Allocated
       64MB] （假设总内存足够大，尾部可能有剩余空闲空间）
    */

    // 测试2: 释放中间两个区域并验证合并
    LOG(INFO) << "\n=== Freeing Middle Regions ===";
    FreeRegion(regions[1]);  // 32MB
    FreeRegion(regions[2]);  // 8MB
    LOG(INFO) << "After Freeing 32MB + 8MB:";
    MemoryRegionView();
    /* 预期状态：
       [Allocated 16MB] -> [Free 40MB] -> [Allocated 64MB]
    */

    // 测试3: 从合并块中分配新区域
    LOG(INFO) << "\n=== Allocating from Merged Block ===";
    auto new_region1 = AllocateRegion(20 * 1024 * 1024);
    LOG(INFO) << "After Allocating 20MB:";
    MemoryRegionView();
    /* 预期状态：
       [Allocated 16MB] -> [Allocated 20MB] -> [Free 20MB] -> [Allocated 64MB]
    */

    // 测试4: 释放首尾区域并验证合并
    LOG(INFO) << "\n=== Freeing Edge Regions ===";
    FreeRegion(regions[0]);  // 16MB
    FreeRegion(regions[3]);  // 64MB
    LOG(INFO) << "After Freeing 16MB + 64MB:";
    MemoryRegionView();
    /* 预期状态：
       [Free 16MB] -> [Allocated 20MB] -> [Free 20MB] -> [Free 64MB]
       （若地址连续可能合并为更大的块）
    */

    // 测试5: 全释放后验证完整合并
    LOG(INFO) << "\n=== Freeing Remaining Regions ===";
    FreeRegion(new_region1);  // 20MB
    LOG(INFO) << "Final State After Full Release:";
    MemoryRegionView();
    /* 预期状态：
       [Single Free Block]
    */

    // 测试6: 分配精确匹配块
    LOG(INFO) << "\n=== Exact Fit Allocation Test ===";
    auto exact_region = AllocateRegion(total_size);  // 尝试分配整个内存
    if (exact_region) {
      LOG(INFO) << "Exact Allocation Success: " << exact_region->toString();
      FreeRegion(exact_region);
    }

    LOG(INFO) << "===== Test Completed =====";
  }
};

class VRAMManager_V2 : public VRAMManagerBase {
 private:
  std::unordered_map<std::string, std::shared_ptr<RegisteredModel>>
      registered_models_;
  std::unordered_map<int, std::shared_ptr<GPUTensorPool_V2>> gpu_tensor_pools_;
  std::mutex mutex_;

  size_t total_tg_access{0};
  size_t tg_hit_count{0};
  size_t total_tg_access_volume{0};
  size_t tg_hit_volume{0};

 public:
  VRAMManager_V2(size_t gpu_tensor_pool_size, const std::vector<int>& gpu_ids) {
    for (int gpu_id : gpu_ids) {
      LOG(INFO) << "Creating GPUTensorPool for device " << gpu_id << " with "
                << gpu_tensor_pool_size / 1024.0 / 1024.0 / 1024.0 << " GB";
      gpu_tensor_pools_[gpu_id] =
          std::make_shared<GPUTensorPool_V2>(gpu_id, gpu_tensor_pool_size);
    }
  }

  ~VRAMManager_V2() {
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
      LOG(WARNING) << "Model " << model_path << " already registered";
      return registered_models_[model_path]->model_size();
    }

    auto model = std::make_shared<RegisteredModel>(model_path, sensitive);
    if (model->LoadModelFromDisk(8) != 0) {  // 使用8个线程加载
      return -1;
    }

    registered_models_[model_path] = model;
    LOG(INFO) << "Model " << model_path << " registered with size "
              << model->model_size() / 1024.0 / 1024.0 / 1024.0 << " GB";
    return model->model_size();
  }

  std::string LoadModel(const std::string& model_path, int device_id) {
    std::unique_lock<std::mutex> lock(mutex_);

    // 1. 检查模型是否注册
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

    // 3. 准备加载TG
    const auto& tg_index = model->GetTensorGroupIndexes();
    std::vector<TensorGroupIndex> tg_to_allocate;

    // std::vector<size_t> sizes_to_allocate;
    std::vector<int> tg_to_load;
    std::vector<char*> allocated_regions(tg_index.size(), nullptr);

    for (int i = 0; i < tg_index.size(); i++) {
      auto mem_region = pool->GetTGByFingerprint(tg_index[i].fingerprint);
      if (!mem_region) {
        tg_to_allocate.push_back(tg_index[i]);
        tg_to_load.push_back(i);
        tg_hit_count++;
        tg_hit_volume += tg_index[i].size;
      } else {
        allocated_regions[i] = mem_region->addr;
      }
      total_tg_access_volume += tg_index[i].size;
    }
    total_tg_access += tg_index.size();
    LOG(INFO) << "Load Model: " << model_path
              << ", Tensor Groups to load: " << tg_to_load.size();

    // 4. 分配空间
    if (!tg_to_allocate.empty()) {
      auto new_regions = pool->BatchedAllocate(tg_to_allocate, model);
      if (new_regions.empty()) {
        LOG(ERROR) << "Failed to allocate GPU memory";
        return "ERROR";
      }

      // 更新allocated_regions
      for (int i = 0; i < tg_to_load.size(); i++) {
        allocated_regions[tg_to_load[i]] = new_regions[i]->addr;
      }
    }

    // 5. 加载数据到GPU
    if (!tg_to_load.empty()) {
      if (model->LoadModelFromMem(allocated_regions, tg_to_load, device_id) !=
          0) {
        LOG(ERROR) << "Failed to load model to GPU";
        return "ERROR";
      }
    }

    // 6. 创建返回字符串
    std::string ret;
    cudaIpcMemHandle_t handle;

    // 设置设备，并检查错误
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
      LOG(ERROR) << "cudaSetDevice error: " << cudaGetErrorString(err);
      return "ERROR";
    }

    // 获取 IPC 内存句柄，并检查错误
    err = cudaIpcGetMemHandle(&handle,
                              gpu_tensor_pools_[device_id]->GetBaseAddr());
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
        size_t offset = allocated_regions[i] -
                        gpu_tensor_pools_[device_id]->GetBaseAddr() +
                        tensor.offset;
        response.push_back(offset);
      }
    }

    std::string response_str(reinterpret_cast<const char*>(response.data()),
                             response.size() * sizeof(size_t));
    ret +=
        toHex(std::vector<uint8_t>(response_str.begin(), response_str.end()));

    return ret;
  }

  void MemoryUsage() {
    for (auto& [device_id, pool] : gpu_tensor_pools_) {
      LOG(INFO) << "Device ID: " << device_id;
      pool->MemoryRegionView();
    }
  }
};

void TestGPUTensorPoolAllocateAndFree() {
  // 假设测试用GPU 0，分配128MB
  int device_id = 0;
  size_t pool_size = 128 * 1024 * 1024;
  GPUTensorPool_V2 pool(device_id, pool_size);

  // 调用成员测试函数
  // pool.TestAllocateAndFreeRegion();
  pool.TestComplexAllocations();
}

void TestCostAwareDropWithModels() {
  // 初始化内存池（256MB）
  const size_t TOTAL_MEM = 256 * 1024 * 1024;
  GPUTensorPool_V2 pool(0, TOTAL_MEM);

  // 创建测试模型（不同敏感度）
  auto modelA = std::make_shared<RegisteredModel>();  // 高敏感度
  auto modelB = std::make_shared<RegisteredModel>();  // 低敏感度
  auto modelC = std::make_shared<RegisteredModel>();  // 未注册模型

  modelA->SetMeta("modelA", 3);
  modelB->SetMeta("modelB", 1);
  modelC->SetMeta("modelC", 5);
  // 设置模型访问频率（控制释放优先级）
  pool.model_access = {{"modelA", 8}, {"modelB", 2}};  // 总访问次数10
  pool.total_access = 10;

  /********************************************************************
   * 阶段1: 分配不同成本的区域
   ********************************************************************/
  auto alloc_test = [&](size_t size, auto model) {
    auto region = pool.AllocateRegion(size);
    region->model_ref = model;
    return region;
  };

  auto rA1 = alloc_test(64 * 1024 * 1024, modelA);   // 成本=64*3*(8/10)=153.6
  auto rB1 = alloc_test(32 * 1024 * 1024, modelB);   // 成本=32*1*(2/10)=6.4
  auto rC1 = alloc_test(128 * 1024 * 1024, modelC);  // 成本=128*1*1=128
  auto rA2 = alloc_test(32 * 1024 * 1024, modelA);   // 成本=32*3*(8/10)=76.8

  pool.allocated_regions["reg1"] = rA1;
  pool.allocated_regions["reg2"] = rB1;
  pool.allocated_regions["reg3"] = rC1;
  pool.allocated_regions["reg4"] = rA2;

  /* 当前内存布局：
     [A1:64M][B1:32M][C1:128M][A2:32M]
     总分配：64+32+128+32=256M（内存已满）
  */

  /********************************************************************
   * 阶段2: 触发内存回收（请求64MB空间）
   ********************************************************************/
  LOG(INFO) << "\n=== Before CostAwareDrop ===";
  pool.MemoryRegionView();

  // 预期释放顺序：C1(128) > A2(76.8) > A1(153.6) > B1(6.4)
  // 但实际只需要释放64MB，优先释放最高成本的C1即可满足
  pool.CostAwareDrop_Greedy(64 * 1024 * 1024);

  LOG(INFO) << "\n=== After Dropping C1 ===";
  pool.MemoryRegionView();
  /* 预期结果：
     [A1:64M][B1:32M][Free:128M][A2:32M]
     （由于C1被释放，释放空间128M满足需求）
  */
  pool.GlobalDeFrag();
  LOG(INFO) << "\n=== After Global DeFrag ===";
  pool.MemoryRegionView();
}
