/*
GPUTensorPool:
    - GetTensor(fingerprint)
    - GreedyDrop(size_t n): 
        检查空闲区域的总大小是否足够, 如果够直接返回
        否则, 计算需要释放的空间, 将已分配空间按照成本排序, 逐渐释放, 直到满足需求
        
    - BipartiteAllocate(vector<size_t> to_allocates):
        运行二分匹配算法:
            左节点: to_allocates
            右节点: 空闲区域
            当且仅当左节点小于空闲区域的size时左节点和右节点之间存在一条边
            寻找左节点的最大匹配
        返回最大匹配
    - GreedyMerge(size_t n):
        检查是否拥有足够大的空闲区域, 如果有直接返回
        通过滑动窗口确定一个空闲区域对(s, e), 要求s和e以及之间的空闲区域的size之和大于等于n, 并且s和e之间的已分配区域的size之和最小
        合并s到e范围内的空闲区域并返回
VRAMManager:
    LoadModel(model_path, device_id):
        1. 检查模型是否注册, 检查device_id是否在gpu_tensor_pools_中
        2. 访问registered_models_中的模型, 获得所有TG的fingerprint
        3. 遍历所有TG,调用GetTensor()检查是否在allocated_regions中, 如果不在则加入待装载列表, 统计需要总空间
        4. 调用GreedyDrop, 确保剩余空间足够
        5. while 循环:
            5.1 收集待装载的TG所需要的空间, 调用BipartiteAllocate
            5.2 如果返回的匹配数为0则退出循环, 否则按照返回结果空闲区域分配给对应的TG
        6. 如果还有未分配的TG:
            6.1 将TG按照大小降序排序
            6.2 遍历TG, 逐一调用GreedyMerge()获得足够大的空闲区域, 并执行分配
*/

/*
当前的BipartiteAllocate仅考虑空闲区域的大小是否足够，而没有考虑合并这些区域后的潜在成本。例如，选择多个不连续的小空闲区域进行分配，可能在后续合并时需要移动中间的已分配区域，从而增加开销。
修改二分图匹配的边权重，使其不仅考虑是否匹配，还考虑分配后的合并成本。例如，权重可以基于空闲区域的相邻区域情况，或者该空闲区域与其他空闲区域的接近程度。

具体实现步骤可能包括：
1. 在构建二分图时，为每个边计算权重，权重越低表示分配该区域后的合并成本越小。
2. 使用带权重的匈牙利算法寻找最大匹配，同时最小化总权重。
3. 调整权重计算方式，例如，空闲区域周围已分配区域的大小总和作为权重，这样优先选择周围已分配区域较少的空闲区域，减少后续合并时的移动。
*/

#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <algorithm>
#include <numeric>
#include <random>
#include "logger.h"
#include "registered_model.h"
#include "vram_manager_base.h"
#include "binary_utils.h"





class GPUMemoryRegion : public std::enable_shared_from_this<GPUMemoryRegion> {
public:
    RegionStatus status;  // 0: free, 1: loading, 2: allocated
    char* addr;
    size_t size;
    std::string fingerprint;
    std::shared_ptr<RegisteredModel> model_ref;
    std::shared_ptr<GPUMemoryRegion> prev;
    std::shared_ptr<GPUMemoryRegion> next;

    GPUMemoryRegion() 
        : status(FREE), addr(nullptr), size(0), prev(nullptr), next(nullptr) {}

    GPUMemoryRegion(void* addr_, size_t size_)
        : status(FREE), addr(static_cast<char*>(addr_)), size(size_), 
          prev(nullptr), next(nullptr) {}

    bool isAdjacent(const std::shared_ptr<GPUMemoryRegion>& other) const {
        return (this->addr + this->size == other->addr) || 
               (other->addr + other->size == this->addr);
    }

    void merge(const std::shared_ptr<GPUMemoryRegion>& other) {
        if (this->isAdjacent(other)) {
            if (this->addr + this->size == other->addr) { // other在右侧
                this->size += other->size;
                this->next = other->next;
                if (this->next){
                    this->next->prev = shared_from_this();
                }
            } else { // other在左侧
                this->addr = other->addr;
                this->size += other->size;
                this->prev = other->prev;
                if (other->prev) other->prev->next = shared_from_this();
            }
        }
    }

    std::string toString() {
        return "[GPUMemoryRegion: addr=" +
               std::to_string(reinterpret_cast<size_t>(addr)) +
               ", size=" + std::to_string(size) + ", fingerprint=" + fingerprint +
               ", status=" + std::to_string(status) +
               ", model_ref=" + (model_ref ? model_ref->model_path() : "nullptr") +
               "]";
      }
};

struct DropCostEntry {
    double cost;
    std::shared_ptr<GPUMemoryRegion> region;
    bool operator<(const DropCostEntry& other) const {
        if (cost != other.cost) return cost < other.cost;
        return reinterpret_cast<uintptr_t>(region->addr) < 
               reinterpret_cast<uintptr_t>(other.region->addr);
    }
};

struct TGNeedAllocates{
    int tg_id;
    TensorGroupIndex tg_index;
    TGNeedAllocates(int tg_id_, TensorGroupIndex tg_index_)
        : tg_id(tg_id_), tg_index(tg_index_) {}
};

class GPUTensorPool {
public:
    std::shared_ptr<GPUMemoryRegion> memory_regions;  // 内存区域链表头
    char* gpu_base_addr;
    std::unordered_map<std::string, std::shared_ptr<GPUMemoryRegion>> allocated_regions;
    std::unordered_map<std::string, size_t> model_access;  // 模型访问次数统计
    size_t total_access{0};
    std::vector<size_t> move_data_volume;


    int device_id;
    cudaStream_t stream_;
    double GPUBandwidth;  // GPU-GPU数据移动带宽
    double CPUBandwidth;  // CPU->GPU数据拷贝带宽

    GPUTensorPool(int device_id_, size_t total_size, double gpu_bw, double cpu_bw)
        : device_id(device_id_), GPUBandwidth(gpu_bw), CPUBandwidth(cpu_bw) {
        cudaSetDevice(device_id_);
        void* gpu_memory;
        cudaMalloc(&gpu_memory, total_size);
        cudaStreamCreate(&stream_);
        memory_regions = std::make_shared<GPUMemoryRegion>(gpu_memory, total_size);
        // LOG(INFO)<<"GPUTensorPool: device_id="<<device_id_<<", total_size="<<total_size<<", gpu_bw="<<gpu_bw<<", cpu_bw="<<cpu_bw;
    }

    ~GPUTensorPool(){
        // 输出移动的总数据量
        // size_t total_move_data=0;
        // for (size_t volume : move_data_volume) {
        //     total_move_data += volume;
        // }
        // LOG(INFO)<<"GPUTensorPool: device_id="<<device_id<<", total_move_data="<<total_move_data;
    }

    size_t GetTotalMove(){
        size_t total_move_data=0;
        for (size_t volume : move_data_volume) {
            total_move_data += volume;
        }
        return total_move_data;
    }
    void UseModel(const std::string& model_path) {
        model_access[model_path]++;
        total_access++;
    }

    void MemoryRegionView() {
        auto current = memory_regions;
        while (current) {
            double cost=0.0;
            if (current->status == ALLOCATED && current->model_ref) {
                double access_prob =
                  (total_access == 0
                       ? 1.0
                       : (double)model_access[current->model_ref->model_path()] /
                             total_access);
                cost = current->size / CPUBandwidth * access_prob * current->model_ref->GetLoadSensitive();
            }
            LOG(INFO) << "Region: addr=" << reinterpret_cast<void*>(current->addr)
                      << ", size=" << current->size 
                      << ", status=" << current->status
                      << ", fingerprint=" << current->fingerprint 
                      << ", model_ref=" << (current->model_ref? current->model_ref->model_path() : "nullptr")
                      << ", cost=" << cost;
            current = current->next;
        }
    }
    // 获取已分配的张量区域
    std::shared_ptr<GPUMemoryRegion> GetTensor(const std::string& fingerprint) {
        auto it = allocated_regions.find(fingerprint);
        return it != allocated_regions.end() ? it->second : nullptr;
    }

    size_t GetFreeSize(){
        size_t free_size = 0;
        auto current = memory_regions;
        while (current) {
            if (current->status == FREE) free_size += current->size;
            current = current->next;
        }
        return free_size;
    }

    int GlobalDeFrag() {
      if (!memory_regions) return 1;
      size_t total_move_data = 0;
      auto start_time = std::chrono::high_resolution_clock::now();
      // 1. 收集所有区域信息
      struct RegionInfo {
        RegionStatus status;
        size_t size;
        char* old_addr;
        std::string fingerprint;
        std::shared_ptr<RegisteredModel> model_ref;
      };
      std::vector<RegionInfo> regions;

      auto current = memory_regions;
      while (current) {
        regions.push_back({current->status, current->size, current->addr,
                           current->fingerprint, current->model_ref});
        current = current->next;
      }

      // 2. 计算总空间和已分配空间
      char* base_addr = memory_regions->addr;
      size_t total_allocated = 0;
      size_t total_free = 0;

      for (const auto& r : regions) {
        if (r.status==ALLOCATED) {
          total_allocated += r.size;
        } else {
          total_free += r.size;
        }
      }

      // 3. 重建链表 - 直接从第一个已分配区域开始
      char* current_addr = base_addr;
      std::shared_ptr<GPUMemoryRegion> prev = nullptr;
      std::shared_ptr<GPUMemoryRegion> head = nullptr;

      // 4. 首先处理已分配区域
      for (const auto& r : regions) {
        if (r.status!=ALLOCATED) continue;

        auto new_region =
            std::make_shared<GPUMemoryRegion>(current_addr, r.size);
        new_region->status = ALLOCATED;
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
            std::make_shared<GPUMemoryRegion>(current_addr, total_free);
        free_region->status = FREE;
        if (!head) {
          // 如果没有已分配区域，这个自由区域就是头节点
          head = free_region;
        } else {
          free_region->prev = prev;
          prev->next = free_region;
        }
      }

      // 6. 更新视图指针
      memory_regions = head;
      // LOG(INFO) << "GlobalDeFrag: Merged all free regions";
      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                          end_time - start_time)
                          .count();
      LOG(INFO) << "GlobalDeFrag: Total move data: " << double(total_move_data) / 1024 / 1024 / 1024 << "GB, Time: " << duration << "us" <<". Move / Allocated: "<<double(total_move_data)/total_allocated;
      move_data_volume.push_back(total_move_data);
      return 0;
    }

    size_t GetMergeCost(){
        // 计算总合并成本, 将所有空闲区域合并为一个大的空闲区域
        auto current = memory_regions;
        size_t total_merge_cost = 0;
        // 找到第一个空闲空间和最后一个空闲空间
        std::shared_ptr<GPUMemoryRegion> first_free = nullptr;
        std::shared_ptr<GPUMemoryRegion> last_free = nullptr;
        while (current) {
            if (current->status == FREE) {
                if (!first_free) first_free = current;
                last_free = current;
            }
            current = current->next;
        }
        // 空闲空间中间存在的已分配空间的大小之和
        size_t allocated_size = 0;
        current = first_free;
        while (current && current != last_free) {
            if (current->status == ALLOCATED) {
                allocated_size += current->size;
            }
            current = current->next;
        }
        // 计算总合并成本
        total_merge_cost = allocated_size;
        return total_merge_cost;
    }

    // 贪心释放策略
    int GreedyDrop(size_t n, std::string skip_model) {
      if(n==0) return 0;
      size_t free_size = 0;
      auto current = memory_regions;
      while (current) {
        if (current->status == FREE) free_size += current->size;
        current = current->next;
      }
      if (free_size >= n) return 0;
      size_t need_release = n - free_size;
      size_t released = 0;
      // LOG(INFO)<<"current free "<<free_size<<" need_release "<<need_release;

      // 统计已分配region的成本, 并按成本升序排序
      std::set<DropCostEntry> drop_costs;
      current = memory_regions;
      size_t total_allocated = 0;
      while (current) {
        if (current->status == ALLOCATED) {
          if (!current->model_ref) {
            LOG(ERROR) << "current->model_ref is nullptr";
            return 1;
          }
          if (current->model_ref->model_path() != skip_model) {
            double access_prob =
                (total_access == 0
                     ? 1.0
                     : (double)model_access[current->model_ref->model_path()] /
                           total_access);
            double cost = current->size / CPUBandwidth * access_prob *
                          current->model_ref->GetLoadSensitive();
            drop_costs.insert({cost, current});
            total_allocated += current->size;
          }
        }
        current = current->next;
      }

      // 按成本升序释放
      for (auto it = drop_costs.begin();
           it != drop_costs.end() && released < need_release; ++it) {
        auto region = it->region;
        // LOG(INFO)<<"GreedyDrop: release region="<<region->toString()<<"cost="<<it->cost <<" released="<<released<<"need_release="<<need_release;
        released += region->size;
        FreeRegion(region);
      }

      if (released < need_release) {
        LOG(ERROR) << "GreedyDrop: released=" << released
                   << " need_release=" << need_release
                   << " total_allocated=" << total_allocated;
        return 1;
      }
      // LOG(INFO)<<"GreedyDrop: released="<<released<<"
      // need_release="<<need_release;
      return 0;
    }

    void NPMWeightFunc4(
        std::vector<std::vector<BipartEdge>>& adj,
        std::vector<size_t>& to_allocates,
        std::vector<std::shared_ptr<GPUMemoryRegion>>& free_regions) {
      size_t min_request =
          *std::min_element(to_allocates.begin(), to_allocates.end());
          size_t max_request =
          *std::max_element(to_allocates.begin(), to_allocates.end());
      const int64_t PERFECT_MATCH_BONUS = max_request * 10;
      const int64_t FRAGMENTED_PENALTY = 10;
      const double MOVE_COST_FACTOR = 1;

      for (size_t i = 0; i < to_allocates.size(); ++i) {
        size_t req_size = to_allocates[i];
        for (size_t j = 0; j < free_regions.size(); ++j) {
          auto& region = free_regions[j];
          if (req_size > region->size) continue;
          // 移动成本计算（包含相邻已分配区域）
          int64_t move_cost = 0;
          int64_t left_move_cost = 0;
          int64_t right_move_cost = 0;
          auto left_region = region->prev;
          auto right_region = region->next;
          while (left_region && left_region->status == ALLOCATED) {
            left_move_cost += left_region->size;
            left_region = left_region->prev;
          }
          while (right_region && right_region->status == ALLOCATED) {
            right_move_cost += right_region->size;
            right_region = right_region->next;
          }

          move_cost = min(left_move_cost, right_move_cost) * MOVE_COST_FACTOR;

          // 完美匹配奖励
          size_t perfect_match =
              (region->size == req_size) ? PERFECT_MATCH_BONUS : 0;

          // 剩余空间碎片惩罚
          size_t remaining = region->size - req_size;
        //   LOG(INFO)<<"req_size: "<<req_size<<" region->size: "<<region->size<<" remaining: "<<remaining<<" min_request: "<<min_request;
          size_t frag_penalty =
              (remaining > 0 && remaining < min_request)
                  ? FRAGMENTED_PENALTY * (min_request - remaining)
                  : 0;
          

          // 最终权重 = 移动成本 - 连续性奖励 + 碎片惩罚 - 完美匹配奖励
        //   int64_t final_weight = move_cost + frag_penalty - perfect_match;
        //   int64_t final_weight = move_cost/min_request;
        // int64_t final_weight = perfect_match-frag_penalty;
        //   int64_t final_weight = frag_penalty;
          double m=1.0;
          double r=5.0;
        //   int64_t final_weight = -1 * m*((double)move_cost/min_request) - r*((double)remaining/min_request);
          int64_t final_weight =-1 * remaining;
          adj[i].push_back({j, final_weight});
        }
      }
    }

    std::vector<std::pair<size_t, std::shared_ptr<GPUMemoryRegion>>>
    WeightedBipartiteAllocate(std::vector<size_t>& to_allocates) {
        
      std::vector<std::shared_ptr<GPUMemoryRegion>> free_regions;
      auto current = memory_regions;
      while (current) {
        if (current->status == FREE) free_regions.push_back(current);
        current = current->next;
      }
    //   LOG(INFO)<<"to_allocate size: "<<to_allocates.size()<<" free_regions size: "<<free_regions.size();
      std::vector<std::vector<BipartEdge>> adj(to_allocates.size());

    //   权重函数1: 完全匹配为无穷大, 部分匹配为1
    //   NPMWeightFunc1(adj, to_allocates, free_regions);

    //   权重函数2: 考虑合并成本和连续性, 转换为最小权重匹配
    //   NPMWeightFunc2(adj, to_allocates, free_regions);

    //   权重函数3: 考虑剩余空间和碎片惩罚, 尽可能减少碎片 -> 目标是装入所有请求
    //   NPMWeightFunc3(adj, to_allocates, free_regions);


      NPMWeightFunc4(adj, to_allocates, free_regions);

    //   NPMWeightFunc5(adj, to_allocates, free_regions);

      // 转换为最小权重匹配（原算法是最大权重，取反后求最大）
    //   for (auto& edges : adj) {
    //     for (auto& edge : edges) {
    //       edge.weight = -edge.weight;  // 取反后，最大权重对应原最小权重
    //     }
    //   }

      auto match_idxs=MaxWeightBMatchingWithBoost(adj);
      
      std::vector<std::pair<size_t, std::shared_ptr<GPUMemoryRegion>>> matches;
      for (const auto& idx : match_idxs) {
        matches.emplace_back(idx.first, free_regions[idx.second]);
      }
      return matches;
    }

    // 二分图匹配分配
    std::vector<std::pair<size_t, std::shared_ptr<GPUMemoryRegion>>> 
    BipartiteAllocate(const std::vector<size_t>& to_allocates) {
        std::vector<std::shared_ptr<GPUMemoryRegion>> free_regions;
        auto current = memory_regions;
        while (current) {
            if (current->status == FREE) free_regions.push_back(current);
            current = current->next;
        }

        // 构建二分图邻接表
        std::vector<std::vector<int>> adj(to_allocates.size());
        for (size_t i = 0; i < to_allocates.size(); ++i) {
            for (size_t j = 0; j < free_regions.size(); ++j) {
                if (to_allocates[i] <= free_regions[j]->size) {
                    adj[i].push_back(j);
                }
            }
        }

        // 匈牙利算法找最大匹配
        std::vector<int> match(free_regions.size(), -1);
        std::vector<bool> visited;
        auto dfs = [&](auto&& self, int u) -> bool {
            for (int v : adj[u]) {
                if (!visited[v]) {
                    visited[v] = true;
                    if (match[v] == -1 || self(self, match[v])) {
                        match[v] = u;
                        return true;
                    }
                }
            }
            return false;
        };

        int result = 0;
        for (size_t u = 0; u < to_allocates.size(); ++u) {
            visited.assign(free_regions.size(), false);
            if (dfs(dfs, u)) result++;
        }

        // 收集匹配结果
        std::vector<std::pair<size_t, std::shared_ptr<GPUMemoryRegion>>> matches;
        for (size_t j = 0; j < free_regions.size(); ++j) {
            if (match[j] != -1) {
                matches.emplace_back(match[j], free_regions[j]);
            }
        }
        return matches;
    }


// 合并指定范围内的内存区域，将范围内的空闲区域合并为一个大的空闲区域
// 参数:
// region_start: 合并范围的起始内存区域指针
// region_end: 合并范围的结束内存区域指针
// 返回值:
// 合并后的空闲区域指针，如果合并失败则返回nullptr
    std::shared_ptr<GPUMemoryRegion> MergeRegions(
        std::shared_ptr<GPUMemoryRegion> region_start,
        std::shared_ptr<GPUMemoryRegion> region_end, bool is_do_move = 1) {
        size_t move_cost=0;
        // 检查输入的起始和结束区域指针是否有效
        if (!region_start || !region_end) {
            LOG(ERROR) << "MergeRegions: Invalid input regions";
            return nullptr;
        }
        // 检查起始和结束区域是否是同一个区域
        if (region_start == region_end) {
            move_data_volume.push_back(move_cost);
            return region_start;
        }
        // 保存边界指针
        auto prev_before = region_start->prev;
        auto next_after = region_end->next;
    
        // 收集范围内所有已分配区域
        // 收集空闲区域大小
        size_t free_region_size = 0;
        std::vector<std::shared_ptr<GPUMemoryRegion>> allocated_regions_tmp;
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
              if (is_do_move) {
                cudaError_t err =
                    cuda_safe_move(current_addr, region->addr, region->size);
                if (err != cudaSuccess) {
                  LOG(ERROR)
                      << "cuda_safe_move failed: " << cudaGetErrorString(err);
                  return nullptr;
                }
                move_cost += region->size;
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
        std::shared_ptr<GPUMemoryRegion> free_region =
            std::make_shared<GPUMemoryRegion>(current_addr, free_region_size);
    
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
        move_data_volume.push_back(move_cost);
        return free_region;
    }

    // 贪心合并策略
    std::shared_ptr<GPUMemoryRegion> GreedyMerge(size_t n, bool merge=1) {
        // 先检查是否有足够大的空闲区域
        auto current = memory_regions;
        size_t total_free = 0;
        while (current) {
            if (current->status == FREE && current->size >= n) return current;
            if (current->status == FREE) total_free += current->size;
            current = current->next;
        }

        if(total_free < n){
            LOG(ERROR)<<"GreedyMerge: NO ENOUGH FREE REGION";
            MemoryRegionView();
            return nullptr;
        }

        // 滑动窗口找最优合并区间
        size_t min_move_cost = SIZE_MAX;
        std::shared_ptr<GPUMemoryRegion> best_s, best_e;

        current = memory_regions;
        while (current) {
            if (current->status != FREE) { current = current->next; continue; }

            auto s = current;
            size_t free_sum = 0;
            size_t move_cost = 0;
            auto e = s;

            while (e) {
                if (e->status == LOADING) break; // s和e之间不能有正在加载的区域
                if (e->status == FREE) free_sum += e->size;
                else move_cost += e->size;

                if (free_sum >= n) {
                    if (move_cost < min_move_cost) {
                        min_move_cost = move_cost;
                        best_s = s;
                        best_e = e;
                    }
                    break;
                }
                e = e->next;
            }
            current = current->next;
        }

        if (!best_s || !best_e) return nullptr;
        LOG(INFO)<<"GreedyMerge: min_move_cost="<<(double)(min_move_cost)/(1024*1024*1024)<<" GB";
        // LOG(INFO)<<"GreedyMerge: best_s="<<best_s->toString()<<" best_e="<<best_e->toString() <<" Min move cost: "<<min_move_cost;
        return MergeRegions(best_s, best_e, merge);
    }

    void FreeRegion(std::shared_ptr<GPUMemoryRegion> region, bool debug=0) {
        if(!region){
            LOG(ERROR)<<"FreeRegion: region is nullptr";
            return;
        }
        if (region->status != ALLOCATED) {
            LOG(ERROR) << "FreeRegion: region " << region->addr << " status invalid (" << region->status << ")";
            return;
        }
        region->status = FREE;
        if(allocated_regions.find(region->fingerprint)!=allocated_regions.end()){
            allocated_regions.erase(region->fingerprint);
        }
        // allocated_regions.erase(region->fingerprint);
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
    

    std::shared_ptr<GPUMemoryRegion> AllocateRegionFromList(size_t size, std::string fp) {
        for (auto region = memory_regions; region != nullptr;
             region = region->next) {
          if (region->status==FREE && region->size >= size) {
            if (region->size > size) {
              auto new_region = std::make_shared<GPUMemoryRegion>(
                  region->addr + size, region->size - size);
              new_region->prev = region;
              new_region->next = region->next;
              if (region->next) {
                region->next->prev = new_region;
              }
              region->next = new_region;
              region->size = size;
            }
            region->status = ALLOCATED;
            region->fingerprint = fp;
            allocated_regions[fp] = region;
            return region;
          }
        }
        return nullptr;
      }

    std::shared_ptr<GPUMemoryRegion> AllocateFreeRegion(
        std::shared_ptr<GPUMemoryRegion> free_region, size_t allocate_size, 
        const std::string& fingerprint) {
        if (free_region->status != FREE || free_region->size < allocate_size) return nullptr;
        if (free_region->size > allocate_size) {
            auto new_region = std::make_shared<GPUMemoryRegion>(
                free_region->addr + allocate_size, free_region->size - allocate_size);
            new_region->prev = free_region;
            new_region->next = free_region->next;
            if (free_region->next) free_region->next->prev = new_region;
            free_region->next = new_region;
            free_region->size = allocate_size;
        }
        free_region->status = ALLOCATED;
        free_region->fingerprint = fingerprint;

        allocated_regions[fingerprint] = free_region;
        return free_region;
    }




// 分配内存区域，返回分配的内存区域和新的空闲区域（如果存在）
std::pair<std::shared_ptr<GPUMemoryRegion>, std::shared_ptr<GPUMemoryRegion>> AllocateNewRegion(
    std::shared_ptr<GPUMemoryRegion> old, size_t allocate_size,
    const std::string& fingerprint) {
  if (old->status != FREE || old->size < allocate_size) {
    LOG(ERROR) << "AllocateNewRegion: old region " << old->addr
               << " status invalid (" << old->status << ")";
    return {nullptr, nullptr};
  }

  std::shared_ptr<GPUMemoryRegion> new_region = nullptr;
  if (old->size > allocate_size) {
    new_region = std::make_shared<GPUMemoryRegion>(
        old->addr + allocate_size, old->size - allocate_size);
    new_region->prev = old;
    new_region->next = old->next;
    if (old->next) old->next->prev = new_region;
    old->next = new_region;
    old->size = allocate_size;
  }
  old->status = ALLOCATED;
  old->fingerprint = fingerprint;

  allocated_regions[fingerprint] = old;
  return {old, new_region};
}

    bool CheckLinkConsistency() {
        auto current = memory_regions;
        size_t node_count = 0;
        bool is_consistent = true;

        while (current) {
            node_count++;
            // 检查前驱指针的后向指向是否正确
            if (current->prev) {
                if (current->prev->next != current) {
                    LOG(ERROR) << "Link inconsistency: Node addr=" << reinterpret_cast<void*>(current->addr)
                               << " prev->next (addr=" << reinterpret_cast<void*>(current->prev->next ? current->prev->next->addr : nullptr)
                               << ") does not point back to current node";
                    is_consistent = false;
                }
            }

            // 检查后继指针的前向指向是否正确
            if (current->next) {
                if (current->next->prev != current) {
                    LOG(ERROR) << "Link inconsistency: Node addr=" << reinterpret_cast<void*>(current->addr)
                               << " next->prev (addr=" << reinterpret_cast<void*>(current->next->prev ? current->next->prev->addr : nullptr)
                               << ") does not point back to current node";
                    is_consistent = false;
                }
            }

            // 头节点的prev必须为nullptr（除非链表为空）
            if (node_count == 1 && current->prev != nullptr) {
                LOG(ERROR) << "Link inconsistency: Head node addr=" << reinterpret_cast<void*>(current->addr)
                           << " has non-null prev pointer (addr=" << reinterpret_cast<void*>(current->prev->addr) << ")";
                is_consistent = false;
            }

            current = current->next;
        }

        if (is_consistent) {
            LOG(INFO) << "Link consistency check passed. Total nodes: " << node_count;
        } else {
            LOG(ERROR) << "Link consistency check failed. Total nodes: " << node_count;
        }
        return is_consistent;
    }

    bool CheckAllocatedConsistency() {
        std::unordered_set<std::string> list_allocated;  // 链表中ALLOCATED节点的指纹集合
        std::unordered_set<std::string> map_allocated;   // allocated_regions中的指纹集合
        bool is_consistent = true;

        // 遍历链表收集ALLOCATED节点的指纹
        auto current = memory_regions;
        while (current) {
            if (current->status == ALLOCATED) {
                if (current->fingerprint.empty()) {
                    LOG(ERROR) << "Allocated node addr=" << reinterpret_cast<void*>(current->addr) 
                               << " has empty fingerprint";
                    is_consistent = false;
                } else {
                    list_allocated.insert(current->fingerprint);
                }
            }
            current = current->next;
        }

        // 收集allocated_regions中的指纹
        for (const auto& pair : allocated_regions) {
            map_allocated.insert(pair.first);
        }

        // 检查链表中存在但映射表中不存在的指纹
        for (const auto& fp : list_allocated) {
            if (!map_allocated.count(fp)) {
                LOG(ERROR) << "Fingerprint '" << fp << "' exists in list but not in allocated_regions";
                is_consistent = false;
            }
        }

        // 检查映射表中存在但链表中不存在的指纹
        for (const auto& fp : map_allocated) {
            if (!list_allocated.count(fp)) {
                LOG(ERROR) << "Fingerprint '" << fp << "' exists in allocated_regions but not in list";
                is_consistent = false;
            }
        }

        // 检查数量是否一致
        if (list_allocated.size() != map_allocated.size()) {
            LOG(ERROR) << "Allocated node count mismatch: list has " << list_allocated.size() 
                       << ", allocated_regions has " << map_allocated.size();
            is_consistent = false;
        }

        if (is_consistent) {
            LOG(INFO) << "Allocated consistency check passed. Total allocated nodes: " << list_allocated.size();
        } else {
            LOG(ERROR) << "Allocated consistency check failed";
        }
        return is_consistent;
    }

//  TEST USE

void CopyMemoryLayout(GPUTensorPool& other) {
    // 清空当前对象的内存区域和已分配区域
    memory_regions.reset();
    allocated_regions.clear();
    // model_access = other.model_access;
    // total_access = other.total_access;
    // move_data_volume = other.move_data_volume;

    // 复制GPU内存区域链表
    std::shared_ptr<GPUMemoryRegion> other_current = other.memory_regions;
    std::shared_ptr<GPUMemoryRegion> prev = nullptr;
    std::shared_ptr<GPUMemoryRegion> new_head = nullptr;

    while (other_current) {
        // 创建新的内存区域对象
        auto new_region = std::make_shared<GPUMemoryRegion>();
        new_region->status = other_current->status;
        new_region->addr = other_current->addr;
        new_region->size = other_current->size;
        new_region->fingerprint = other_current->fingerprint;
        new_region->model_ref = other_current->model_ref;
        new_region->prev = prev;

        if (prev) {
            prev->next = new_region;
        } else {
            new_head = new_region;
        }

        if (new_region->status == ALLOCATED) {
            allocated_regions[new_region->fingerprint] = new_region;
        }

        prev = new_region;
        other_current = other_current->next;
    }

    // 更新当前对象的内存区域链表头
    memory_regions = new_head;
}

size_t MockGenerateRegions(int num_regions, size_t min_size, size_t max_size, int is_regenerate) {
  const char* SAVE_FILE = "mock_regions.bin";
  size_t total_free = 0;
  // 从文件加载现有配置
  if (is_regenerate == 0) {
    std::ifstream ifs(SAVE_FILE, std::ios::binary);
    if (ifs) {
      // 清空现有状态
      memory_regions.reset();
      allocated_regions.clear();

      // 读取区域数量
      int saved_regions = 0;
      ifs.read(reinterpret_cast<char*>(&saved_regions), sizeof(int));

      // 重建内存区域链表
      std::shared_ptr<GPUMemoryRegion> prev = nullptr;
      for (int i = 0; i < saved_regions; ++i) {
        auto region = std::make_shared<GPUMemoryRegion>();

        // 读取基础数据
        size_t addr_val, size;
        ifs.read(reinterpret_cast<char*>(&addr_val), sizeof(size_t));
        ifs.read(reinterpret_cast<char*>(&size), sizeof(size_t));
        ifs.read(reinterpret_cast<char*>(&region->status),
                 sizeof(RegionStatus));

        region->addr = reinterpret_cast<char*>(addr_val);
        region->size = size;

        // 重建链表连接
        if (prev) {
          prev->next = region;
          region->prev = prev;
        } else {
          memory_regions = region;
        }
        prev = region;

        // 读取指纹信息
        if (region->status == ALLOCATED) {
          size_t fp_len;
          ifs.read(reinterpret_cast<char*>(&fp_len), sizeof(size_t));
          char* buffer = new char[fp_len + 1];
          ifs.read(buffer, fp_len);
          buffer[fp_len] = '\0';
          region->fingerprint = buffer;
          delete[] buffer;

          allocated_regions[region->fingerprint] = region;
        }

        if (region->status == FREE) total_free += region->size;
      }
      LOG(INFO) << "Loaded " << saved_regions << " regions from " << SAVE_FILE;
      return total_free;
    }
  }

  // 清空现有内存区域
  memory_regions.reset();
  allocated_regions.clear();

  // 生成随机内存区域链表
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> size_dist(min_size, max_size);  // 100B-1KB
  std::uniform_int_distribution<int> status_dist(0, 1);  // RegionStatus枚举值

  char* current_addr = reinterpret_cast<char*>(0x10000000);  // 虚拟起始地址
  std::shared_ptr<GPUMemoryRegion> prev = nullptr;

  for (int i = 0; i < num_regions; ++i) {
    size_t size = size_dist(gen);
    RegionStatus status = static_cast<RegionStatus>(status_dist(gen));
    if (status == LOADING) {
      status = ALLOCATED;  // 确保都是FREE或者ALLOCATED状态
    }

    if (status == FREE) {
      total_free += size;
    }

    auto region = std::make_shared<GPUMemoryRegion>();
    region->addr = current_addr;
    region->size = size;
    region->status = status;
    region->prev = prev;

    if (prev) {
      prev->next = region;
    } else {
      memory_regions = region;  // 设置链表头
    }

    current_addr += size;  // 模拟地址间隔
    prev = region;

    // 如果是已分配状态，生成随机指纹
    if (status == ALLOCATED) {
      region->fingerprint = "mock_fp_" + std::to_string(i);
      allocated_regions[region->fingerprint] = region;
    }
  }

  // 遍历链表, 合并相邻的FREE区域
  auto current = memory_regions;
  bool merged;
  do {
    merged = false;
    current = memory_regions;
    while (current && current->next) {
      if (current->status == FREE && current->next->status == FREE) {
        current->merge(current->next);
        merged = true;
        break;  // 合并后重新扫描
      } else {
        current = current->next;
      }
    }
  } while (merged);

  // 检查是否有相邻的空闲区域
  current = memory_regions;
  while (current && current->next) {
    if (current->status == FREE && current->next->status == FREE) {
      LOG(ERROR) << "Adjacent free regions found after merging";
      return 0;
    }
    current = current->next;
  }

  // 保存生成结果
  std::ofstream ofs(SAVE_FILE, std::ios::binary);
  if (ofs) {
    // 统计区域总数
    int region_count = 0;
    auto current = memory_regions;
    while (current) {
      region_count++;
      current = current->next;
    }

    // 写入区域数量
    ofs.write(reinterpret_cast<const char*>(&region_count), sizeof(int));

    // 写入每个区域数据
    current = memory_regions;
    while (current) {
      // 写入基础数据
      size_t addr_val = reinterpret_cast<size_t>(current->addr);
      ofs.write(reinterpret_cast<const char*>(&addr_val), sizeof(size_t));
      ofs.write(reinterpret_cast<const char*>(&current->size), sizeof(size_t));
      ofs.write(reinterpret_cast<const char*>(&current->status),
                sizeof(RegionStatus));

      // 写入指纹信息
      if (current->status == ALLOCATED) {
        size_t fp_len = current->fingerprint.size();
        ofs.write(reinterpret_cast<const char*>(&fp_len), sizeof(size_t));
        ofs.write(current->fingerprint.c_str(), fp_len);
      }

      current = current->next;
    }
    // LOG(INFO) << "Saved " << region_count << " regions to " << SAVE_FILE;
  }

  return total_free;
}


std::vector<size_t> MockGenerateRequests(size_t total_free, int num_requests, size_t min_size, 
                                         size_t max_request_size,
                                         int is_regenerate) {
  const char* SAVE_FILE = "mock_requests.bin";

  // 从文件加载现有请求
  if (is_regenerate == 0) {
    std::ifstream ifs(SAVE_FILE, std::ios::binary);
    if (ifs) {
      std::vector<size_t> requests;
      size_t request_count = 0;
      ifs.read(reinterpret_cast<char*>(&request_count), sizeof(size_t));

      requests.resize(request_count);
      ifs.read(reinterpret_cast<char*>(requests.data()),
               request_count * sizeof(size_t));

      LOG(INFO) << "Loaded " << requests.size() << " requests from " <<
      SAVE_FILE;
      return requests;
    }
  }

  // 生成新请求（原有逻辑）
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> size_dist(min_size, max_request_size);

  std::vector<size_t> requests;
  size_t remaining_free = total_free;
  for (int i = 0; i < num_requests; ++i) {
    size_t request_size = size_dist(gen);
    if (request_size <= remaining_free) {
      requests.push_back(request_size);
      remaining_free -= request_size;
    }
  }

  // 保存生成结果
  std::ofstream ofs(SAVE_FILE, std::ios::binary);
  if (ofs) {
    size_t request_count = requests.size();
    ofs.write(reinterpret_cast<const char*>(&request_count), sizeof(size_t));
    ofs.write(reinterpret_cast<const char*>(requests.data()),
              requests.size() * sizeof(size_t));
    // LOG(INFO) << "Saved " << requests.size() << " requests to " << SAVE_FILE;
  }

  // LOG(INFO) << "Generated " << requests.size() << " mock requests,
  // remainming_free: "<<remaining_free;
  return requests;
}

// ... existing code ...

std::vector<size_t> MockBartiteMatching(std::vector<size_t> requests,
                                        int matching_methods) {
  auto remaining = requests;
  size_t id = 0;
  while (!remaining.empty()) {
    // 输出remaining的内容
    // LOG(INFO)<<"Remaining TG to allocate: "<<remaining.size();
    // for (auto tg : remaining) {
    //     LOG(INFO)<<"TG: "<<tg.tg_id<<" - "<<tg.tg_index.toString();
    // }
    std::vector<std::pair<size_t, std::shared_ptr<GPUMemoryRegion>>> matches;
    switch (matching_methods) {
      case 0:
        matches = BipartiteAllocate(remaining);
        break;
      case 1:
        matches = WeightedBipartiteAllocate(remaining);
        break;
      default:
        LOG(ERROR) << "Invalid matching methods";
        return {};
        break;
    }
    // auto matches = BipartiteAllocate(remaining);
    // LOG(INFO)<<"BipartiteAllocate: matches.size()="<<matches.size();
    if (matches.empty()) break;
    std::vector<size_t> indices_to_remove;

    // 执行实际分配
    size_t allocate_size = 0;
    for (const auto& [idx, region] : matches) {
      // LOG(INFO)<<"Match: "<<remaining[idx].tg_index.size<<" -
      // "<<region->toString();
      allocate_size += remaining[idx];
      auto allocated = AllocateFreeRegion(region, remaining[idx],
                                          "mock_fp_" + std::to_string(id++));
      // 更新allocated_regions
      if (allocated) {
        indices_to_remove.push_back(idx);
      } else {
        LOG(ERROR) << "AllocateFreeRegion failed";
        return {};
      }
    }

    // 倒序删除，避免索引错乱
    std::sort(indices_to_remove.rbegin(), indices_to_remove.rend());
    for (size_t idx : indices_to_remove) {
      remaining.erase(remaining.begin() + idx);
    }
  }

  return remaining;
    }

    void MockGreedyMergeAllocate(std::vector<size_t> requests, bool merge=1) {
        size_t id=0;
        for(size_t request_size : requests) {
            auto region = GreedyMerge(request_size, merge);
            if(region) {
                auto reg=AllocateFreeRegion(region, request_size, "mock_fp_merged" + std::to_string(id++));
                if(reg){
                    allocated_regions[reg->fingerprint] = reg;
                }
            } else {
                LOG(ERROR) << "Failed to allocate " << request_size << " bytes";
            }
        }
    }

    struct RegionAllocateGroup{
        std::shared_ptr<GPUMemoryRegion> start;
        std::shared_ptr<GPUMemoryRegion> end;
        std::vector<TGNeedAllocates> tg_to_loads;

        bool skip;
        RegionAllocateGroup() : start(nullptr), end(nullptr), skip(false) {}
    };

    struct CanGroupSplitResult {
      bool success;
      std::vector<TGNeedAllocates> to_left_requests;
      std::vector<TGNeedAllocates> to_right_requests;
    };

    // 检查是否可以将request_sizes分成两部分，使得左右两部分的总大小分别不超过left_total_size和right_total_size
    /*
    贪心实现：
      假设request_size已经按照大小降序排列
      遍历request_size：
        检查是否能够分配给左桶和右桶中大的那个
        如果不能分配，返回false
        如果能分配：
          执行更新桶大小
          记录分配结果
      返回true和分配结果
    */
    CanGroupSplitResult CanGroupSplit(size_t left_total_size,
                                      size_t right_total_size,
                                      std::vector<TGNeedAllocates> requests) {
      CanGroupSplitResult result{true, {}, {}};
      size_t left_remaining = left_total_size;
      size_t right_remaining = right_total_size;

      for (auto req : requests) {
        // std::cout<<"left_remaining: "<<left_remaining<<" right_remaining:"<<right_remaining<<" req: "<<req.tg_index.size<<std::endl;
        bool can_assign_left = (req.tg_index.size <= left_remaining);
        bool can_assign_right = (req.tg_index.size <= right_remaining);

        // 优先分配给剩余空间较大的桶
        if (can_assign_left &&
            (left_remaining >= right_remaining || !can_assign_right)) {
          result.to_left_requests.push_back(req);
          left_remaining -= req.tg_index.size;
        } else if (can_assign_right) {
          result.to_right_requests.push_back(req);
          right_remaining -= req.tg_index.size;
        } else {
          return {false, {}, {}};
        }
      }
      return result;
    }

    std::pair<RegionAllocateGroup, RegionAllocateGroup> BestSplitAllocate(
        RegionAllocateGroup original) {
      if (original.start == nullptr || original.end == nullptr) {
        LOG(ERROR) << "BestSplit: original is empty";
        return {RegionAllocateGroup(), RegionAllocateGroup()};
      }
      if (original.start == original.end) {
        LOG(ERROR) << "BestSplit: original is single node";
        return {RegionAllocateGroup(), RegionAllocateGroup()};
      }

      // 收集所有可能的拆分点（相邻空闲空间对）及拆分收益
      std::vector<std::pair<size_t, size_t>>
          split_points;  // {拆分收益, 拆分位置索引}
      std::vector<std::shared_ptr<GPUMemoryRegion>>
          regions;  // 合并区域内的所有段（空闲/已分配）
      std::vector<size_t>
          prefix_sums;  // 前缀和数组，用于快速计算已分配空间之和
      int prefix_idx = 0;
      size_t all_sum = 0;

      // 从 original 中提取合并区域的段列表（假设 RegionGroup 包含段列表成员）
      auto current = original.start;
      while (current && current != original.end->next) {
        if (current->status == FREE) {
          regions.push_back(current);

          size_t sum = current->size;
          if (prefix_idx > 0) {
            sum += prefix_sums[prefix_idx - 1];
          }
          prefix_sums.push_back(sum);
          prefix_idx++;
          all_sum += current->size;
        }
        current = current->next;
      }
      if (regions.size() < 2) {
        return {RegionAllocateGroup(), RegionAllocateGroup()};
      }

      // 计算相邻空闲空间的拆分收益（已分配空间之和）
      for (size_t i = 0; i < regions.size() - 1; ++i) {
        auto left = regions[i];
        auto right = regions[i + 1];
        // 计算中间已分配空间的大小之和（即拆分收益）
        size_t split_profit = 0;
        auto mid = left->next;
        while (mid != right) {
          if (mid->status == ALLOCATED) {
            split_profit += mid->size;
          }
          mid = mid->next;
        }
        split_points.emplace_back(split_profit, i);  // 在第i个空闲区域之后拆分
      }

      // 按拆分收益从大到小排序
      std::sort(split_points.begin(), split_points.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });

      // 遍历拆分点，检查可行性
      for (const auto& sp : split_points) {
        size_t split_idx = sp.second;
        auto left_free = regions[split_idx];
        auto right_free = regions[split_idx + 1];

        // 计算拆分后的左右空闲空间总大小（合并各自相邻的空闲段）
        size_t left_total = prefix_sums[split_idx];
        size_t right_total = all_sum - left_total;

        // 调用 CanSplit 检查请求是否可拆分
        // LOG(INFO)<<"CanSplit with point: "<<regions[split_idx]->toString();
        CanGroupSplitResult split_result =
            CanGroupSplit(left_total, right_total, original.tg_to_loads);

        if (split_result.success) {
          // LOG(INFO)<<"Split with point: "<<regions[split_idx]->toString()<<"get split profit: "<<sp.first; LOG(INFO)<<" left requests: ";
          // for(auto req : split_result.to_left_requests) {
          //     LOG(INFO)<<"    "<<req.tg_index.size;
          // }
          // LOG(INFO)<<" right requests: ";
          // for (auto req : split_result.to_right_requests) {
          //     LOG(INFO)<<"    "<<req.tg_index.size;
          // }

          // 步骤4：构造左右子区域的 RegionGroup
          RegionAllocateGroup left_dispatch, right_dispatch;
          left_dispatch.start = regions[0];
          left_dispatch.end = regions[split_idx];
          left_dispatch.tg_to_loads = split_result.to_left_requests;

          right_dispatch.start = regions[split_idx + 1];
          right_dispatch.end = regions.back();
          right_dispatch.tg_to_loads = split_result.to_right_requests;

          return {left_dispatch, right_dispatch};
        }
      }

      // 无可行拆分点，返回空对
      original.skip = true;
      return {RegionAllocateGroup(), RegionAllocateGroup()};
    }
    std::vector<RegionAllocateGroup> RecursiveSplitAllocate(
        std::vector<TGNeedAllocates> tg_to_loads) {
      if(tg_to_loads.empty()) {
        return {};
      }
      // 将tg_to_loads按照tg size降序排序
      std::sort(tg_to_loads.begin(), tg_to_loads.end(),
                [](const auto& a, const auto& b) {
                  return a.tg_index.size > b.tg_index.size;
                });

      // 步骤1: 初始化第0层
      RegionAllocateGroup initial_group;
      initial_group.tg_to_loads = tg_to_loads;
      auto current = memory_regions;
      while (current && current->status != FREE) {
        current = current->next;  // 找到第一个空闲区域
      }
      initial_group.start = current;
      // current = memory_regions;
      std::shared_ptr<GPUMemoryRegion> last = nullptr;
      while (current) {
        if (current->status == FREE) {
          last = current;
        }
        current = current->next;
      }
      initial_group.end = last;  // 指向最后一个空闲区域
      initial_group.skip = false;

      // 使用双向链表管理当前层的拆分区域
      DoublyLinkedList<RegionAllocateGroup> region_groups;
      region_groups.push_back(initial_group);
      // 遍历所有组，拆分可拆分的组
      while (true) {

        // LOG(INFO)<<"===============Current layer: ================";
        // for (auto g = region_groups.begin(); g!= region_groups.end(); g++) {
        //   LOG(INFO)<<"  Group: ";
        //   for(auto current=g->start; current && current!= g->end->next; current = current->next) {
        //     LOG(INFO)<<"    "<<current->toString();
        //   }
        //   std::string sizes;
        //   for (auto req : g->tg_to_loads) {
        //     sizes += std::to_string(req.tg_index.size) + " ";
        //   }
        //   LOG(INFO)<<"  tg_to_loads: "<<sizes;
        // }
        // LOG(INFO)<<"===============================================";

        bool split_occurred = false;
        // 处理当前层的每个组
        for (auto g = region_groups.begin(); g != region_groups.end(); g++) {
          if (g->skip) continue;             // 跳过已经不能再拆分的组
          if (g->start == g->end) continue;  // 跳过只有一个空闲区域的组
          auto [left, right] = BestSplitAllocate(*g);
          if (left.tg_to_loads.empty() && right.tg_to_loads.empty()) {
            // 如果左右子组都为空，则表明拆分失败
            g->skip = true;
            // 什么都不做，继续下一个区域
            continue;
          }
          auto old_group = g;  // 记录旧组在链表中的位置
          split_occurred = true;

          if (left.tg_to_loads.empty()) {
            // 如果只有左子组为空，表明所有请求都分配到了右子组，此时左子组可以全部拆分：
            // 找到左子组中的全部空闲区域，创建新的组，加入链表
            auto current = left.start;
            while (current && current != left.end->next) {
              if (current->status == FREE) {
                auto new_dispatch = RegionAllocateGroup();
                new_dispatch.start = current;
                new_dispatch.end = current;
                new_dispatch.tg_to_loads = {};  // dispatched_sizes为空
                new_dispatch.skip = true;
                g = region_groups.insert_after(
                    g, new_dispatch);  // 在当前组之后插入新的组
              }
              current = current->next;
            }
            g=region_groups.insert_after(g, right); // 将右子组插入到当前组之后
          } else if (right.tg_to_loads.empty()) {
            // 同理
            g=region_groups.insert_after(g, left);  // 插入左子组
            auto current = right.start;
            while (current && current != right.end->next) {
              if (current->status == FREE) {
                auto new_dispatch = RegionAllocateGroup();
                new_dispatch.start = current;
                new_dispatch.end = current;
                new_dispatch.tg_to_loads = {};
                new_dispatch.skip = true;
                g = region_groups.insert_after(g, new_dispatch);
              }
              current = current->next;
            }
          } else {
            // 如果左右子组都不为空，则表明拆分成两个新的组
            g = region_groups.insert_after(g, left);
            g = region_groups.insert_after(g, right);
          }

          // 删除旧组
          region_groups.erase(old_group);
        }
        // 如果没有发生拆分，退出循环
        if (!split_occurred) {
          break;
        }
      }
      return region_groups.toVector();
    }

    void MockRecursiveSplitAllocate(std::vector<size_t> requests){
        size_t fp_id=0;
        std::vector<TGNeedAllocates> tg_to_loads;
        for(size_t request_size : requests) {
            tg_to_loads.push_back(TGNeedAllocates(fp_id, TensorGroupIndex(0, request_size, std::to_string(fp_id),{})));
            fp_id++;
        }

        auto region_groups = RecursiveSplitAllocate(tg_to_loads);
        size_t move_cost=0;
        for(auto region_group : region_groups) {
            
            for(auto current = region_group.start; current && current!= region_group.end->next; current = current->next) {
                if(current->status == ALLOCATED) {
                    move_cost+=current->size;
                }
            }
        }
        move_data_volume.push_back(move_cost);
    }

};

class VRAMManager{
private:
    std::unordered_map<std::string, std::shared_ptr<RegisteredModel>> registered_models_;
    std::unordered_map<int, std::shared_ptr<GPUTensorPool>> gpu_tensor_pools_;
    std::mutex mutex_;
    std::vector<double> model_load_latencies_;
    std::vector<std::string> model_paths_;
    std::vector<double> merge_latencies_;
    std::vector<double> allocate_latencies_;

public:
    VRAMManager(size_t gpu_tensor_pool_size, const std::vector<int>& gpu_ids, 
                  double gpu_bw, double cpu_bw) {
        for (int gpu_id : gpu_ids) {
            gpu_tensor_pools_[gpu_id] = std::make_shared<GPUTensorPool>(
                gpu_id, gpu_tensor_pool_size, gpu_bw, cpu_bw);
        }
    }

    ~VRAMManager() {
        // 统计平均模型加载和合并延迟
        if (!model_load_latencies_.empty()) {
            double avg_load_latency =
                std::accumulate(model_load_latencies_.begin(),
                                model_load_latencies_.end(), 0.0) /
                model_load_latencies_.size();
            LOG(INFO) << "Average model load latency: " << avg_load_latency
                      << " ms" << " , count: " << model_load_latencies_.size();
        }
        if (!merge_latencies_.empty()) {
            double avg_merge_latency =
                std::accumulate(merge_latencies_.begin(), merge_latencies_.end(),
                                0.0) /
                merge_latencies_.size();
            LOG(INFO) << "Average merge latency: " << avg_merge_latency << " ms"<< " , count: " << merge_latencies_.size();
            
        }
        if (!allocate_latencies_.empty()) {
            double avg_allocate_latency =
                std::accumulate(allocate_latencies_.begin(), allocate_latencies_.end(),
                                0.0) /
                allocate_latencies_.size();
            LOG(INFO) << "Average allocate latency: " << avg_allocate_latency << " ms"<< " , count: " << allocate_latencies_.size();
        }

        std::unordered_map<std::string, std::vector<double>> model_load_latency_map;
        // 输出所有的LoadModel调用的延迟
        LOG(INFO)<<"Loading Model Latency";
        for (size_t i = 0; i < model_load_latencies_.size(); i++) {
            model_load_latency_map[model_paths_[i]].push_back(model_load_latencies_[i]);
        }
        for (const auto& pair : model_load_latency_map) {
            LOG(INFO)<<"  Model: "<<pair.first;
            for(auto latency : pair.second) {
                std::cout<<"    "<<latency<<endl;
            }
        }
    }

    // 新增：模型注册方法（参考V3）
    int64_t RegisterModel(const std::string& model_path, int sensitive = 1) {
      std::unique_lock<std::mutex> lock(mutex_);
      if (registered_models_.find(model_path) != registered_models_.end()) {
        LOG(WARNING) << "Model already registered: " << model_path;
        return registered_models_[model_path]->model_size();
      }
      auto model = std::make_shared<RegisteredModel>(model_path, sensitive);
      // 可选：合并张量组（根据需求调整参数）
      // model->MergeTGs(100LL * 1024 * 1024);    // 100MB合并阈值
      if (model->LoadModelFromDisk(8) != 0) {  // 8线程加载
        LOG(ERROR) << "Load model from disk failed: " << model_path;
        return -1;
      }
      registered_models_[model_path] = model;
      LOG(INFO) << "Model registered: " << model_path
                << ", size: " << model->model_size();
      return model->model_size();
    }

    // 新增：内存使用统计（参考V3）
    void MemoryUsage() {
      // 输出移动的总数据量
      for (auto& pair : gpu_tensor_pools_) {
        auto& pool = pair.second;
        LOG(INFO) << "GPU: " << pair.first << ", total_move_data_volume: "
                  << pool->GetTotalMove();
      }
    }

    int WithoutReuse(const std::vector<TGNeedAllocates>& tg_to_load,
                     const std::shared_ptr<RegisteredModel> model,
                     const std::shared_ptr<GPUTensorPool> pool,
                     std::vector<char*>& allocated_regions, int device_id) {
      cudaSetDevice(device_id);

      // 清理GPU内存
      cudaError_t cuda_err;
      for(auto current=pool->memory_regions; current; current=current->next) {
        if(current->addr){
          cuda_err = cudaFree(current->addr);
          if (cuda_err!= cudaSuccess) {
            LOG(ERROR) << "cudaFree failed: " << cudaGetErrorString(cuda_err);
            return -1;
          }
          current->addr = nullptr;
        }
      }
      cuda_err = cudaFree(pool->gpu_base_addr);
      if (cuda_err!= cudaSuccess) {
        LOG(ERROR) << "cudaFree failed: " << cudaGetErrorString(cuda_err);
        return -1;
      }
      


      cuda_err = cudaMalloc(
          &(pool->gpu_base_addr), model->model_size());
      if (cuda_err != cudaSuccess) {
        LOG(ERROR) << "cudaMalloc failed: " << cudaGetErrorString(cuda_err);
        return -1;
      }
  
      for(auto tg : tg_to_load) {
        allocated_regions[tg.tg_id] = (pool->gpu_base_addr)+tg.tg_index.file_offset;
      }
      return 0;
    }
    int GlobalMerge(const std::vector<TGNeedAllocates>& tg_to_load,
                    const std::shared_ptr<RegisteredModel> model,
                    const std::shared_ptr<GPUTensorPool> pool,
                    std::vector<char*>& allocated_regions) {
      auto start_allocate_time = std::chrono::high_resolution_clock::now();
      if (!tg_to_load.empty()) {
        if (pool->GlobalDeFrag()) {
          LOG(ERROR) << "GlbalDe Frag Failed";
          return -1;
        }
        // 更新allocated_regions
        for (int i = 0; i < tg_to_load.size(); i++) {
          auto tg = tg_to_load[i];
          auto region = pool->AllocateRegionFromList(tg.tg_index.size,
                                                     tg.tg_index.fingerprint);
          if (!region) {
            LOG(ERROR) << "AllocateRegionFromList failed";
            return -1;
          }
          region->model_ref = model;
          allocated_regions[tg_to_load[i].tg_id] = region->addr;
        }
      }
      auto end_allocate_time = std::chrono::high_resolution_clock::now();
      auto allocate_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              end_allocate_time - start_allocate_time);
      merge_latencies_.push_back(allocate_duration.count());

      return 0;
    }

    int PartitionedBinPacking(const std::vector<TGNeedAllocates>& tg_to_load, const std::shared_ptr<RegisteredModel> model, const std::shared_ptr<GPUTensorPool> pool,  std::vector<char*>& allocated_regions) {
        auto start_merge_time=std::chrono::high_resolution_clock::now();
        auto start_allocate_time=std::chrono::high_resolution_clock::now();
        auto region_groups = pool->RecursiveSplitAllocate(tg_to_load);
        auto end_allocate_time=std::chrono::high_resolution_clock::now();
        auto allocate_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_allocate_time - start_allocate_time).count();
        allocate_latencies_.push_back(allocate_duration);
        
        // 按照空闲空间组的最终结果合并空闲空间
        for (auto group : region_groups) {
          // LOG(INFO)<<"Merge Region: "<<group.start->toString()<<" to "<<group.end->toString();
          auto merged_region = pool->MergeRegions(group.start, group.end);
          if (!merged_region) {
            LOG(ERROR) << "MergeRegions failed";
            return -1;
          }
          // LOG(INFO)<<"Merged Region: "<<merged_region->toString();
          // 分配TG
          shared_ptr<GPUMemoryRegion> next_free = merged_region;
          shared_ptr<GPUMemoryRegion> allocated = nullptr;
          // LOG(INFO)<<"TG to load: ";
          for (auto tg : group.tg_to_loads) {
            // LOG(INFO)<<"    "<<tg.tg_index.toString();
            auto ret = pool->AllocateNewRegion(next_free, tg.tg_index.size,
                                               tg.tg_index.fingerprint);
            allocated = ret.first;
            next_free = ret.second;
            if (!allocated) {
              LOG(ERROR) << "AllocateNewRegion failed";
              return -1;
            }
            allocated->model_ref = model;
            allocated_regions[tg.tg_id] = allocated->addr;
          }
        }
        // pool->MemoryRegionView();
        auto end_merge_time=std::chrono::high_resolution_clock::now();
        auto merge_duration=std::chrono::duration_cast<std::chrono::milliseconds>(end_merge_time-start_merge_time);
        merge_latencies_.push_back(merge_duration.count());
        return 0;
    }

    int WeightedBipartiteMatch_GreedyMerge(const std::vector<TGNeedAllocates>& tg_to_load, const std::shared_ptr<RegisteredModel> model, const std::shared_ptr<GPUTensorPool> pool,  std::vector<char*>& allocated_regions) {
        auto start_merge_time=std::chrono::high_resolution_clock::now();
        // LOG(INFO)<<"Total allocate size: "<<total_need<<", current free size="<<pool->GetFreeSize();

        // LOG(INFO)<<"After GreedyDrop, memory usage: ";
        // pool->MemoryRegionView();

        // 步骤4: 循环调用BipartiteAllocate分配
        std::vector<TGNeedAllocates> remaining = tg_to_load;
        auto start_match_time=std::chrono::high_resolution_clock::now();
        while (!remaining.empty()) {
            // 输出remaining的内容
            // LOG(INFO)<<"Remaining TG to allocate: "<<remaining.size();
            // for (auto tg : remaining) {
            //     LOG(INFO)<<"TG: "<<tg.tg_id<<" - "<<tg.tg_index.toString();
            // }

            std::vector<size_t> sizes;
            for (const auto& tg : remaining) {
                sizes.push_back(tg.tg_index.size);
            }
            // auto matches = pool->BipartiteAllocate(sizes);
            
            auto matches = pool->WeightedBipartiteAllocate(sizes);
           
            // LOG(INFO)<<"BipartiteAllocate: matches.size()="<<matches.size();
            if (matches.empty()) break;
            std::vector<size_t> indices_to_remove;

            // 执行实际分配
            size_t allocate_size=0;
            for (const auto& [idx, region] : matches) {
                // LOG(INFO)<<"Match: "<<remaining[idx].tg_index.size<<" - "<<region->toString();
                allocate_size+=remaining[idx].tg_index.size;
                auto allocated = pool->AllocateFreeRegion(region, remaining[idx].tg_index.size, remaining[idx].tg_index.fingerprint);
                // 更新allocated_regions
                if (allocated) {
                    allocated->model_ref = model;
                    allocated_regions[remaining[idx].tg_id] = allocated->addr;
                    // LOG(INFO)<<"Allocate TG_ID: "<<remaining[idx].tg_id<<" to region: "<<allocated->toString();
                    indices_to_remove.push_back(idx);
                    
                }else{
                    LOG(ERROR)<<"AllocateFreeRegion failed";
                    return -1;
                }
            }
            // LOG(INFO)<<"BM-Allocate, allocate_size="<<allocate_size<<", current free size="<<pool->GetFreeSize();

            // 倒序删除，避免索引错乱
            std::sort(indices_to_remove.rbegin(), indices_to_remove.rend());
            for (size_t idx : indices_to_remove) {
              remaining.erase(remaining.begin() + idx);
            }
        }
        auto end_match_time=std::chrono::high_resolution_clock::now();
        auto match_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_match_time - start_match_time).count();
        allocate_latencies_.push_back(match_duration);
        // 步骤5: 剩余TG按大小排序后调用GreedyMerge
        LOG(INFO)<<"Remaining TGs need to merge: "<<remaining.size();
        if (!remaining.empty()) {
            // 将remaining按照size从大到小排序
            std::sort(remaining.begin(), remaining.end(),
                      [](const TGNeedAllocates& a, const TGNeedAllocates& b) {
                        return a.tg_index.size > b.tg_index.size;
                      });

            for (auto tg : remaining) {
                auto merged = pool->GreedyMerge(tg.tg_index.size);
                if(!merged){
                    LOG(ERROR) << "GreedyMerge failed for TG size=" << tg.tg_index.size << ", fingerprint=" << tg.tg_index.fingerprint;
                    return -1;
                }
                auto reg=pool->AllocateFreeRegion(merged, tg.tg_index.size, tg.tg_index.fingerprint);
                if(!reg){
                    LOG(ERROR) << "AllocateFreeRegion failed for TG size=" << tg.tg_index.size << ", fingerprint=" << tg.tg_index.fingerprint;
                    return -1;
                }
                reg->model_ref = model;
                allocated_regions[tg.tg_id] = reg->addr;
            
            }
        }
        auto end_merge_time=std::chrono::high_resolution_clock::now();
        auto merge_duration=std::chrono::duration_cast<std::chrono::milliseconds>(end_merge_time-start_merge_time);
        merge_latencies_.push_back(merge_duration.count());
        return 0;
    }

    int GreedyMerge(const std::vector<TGNeedAllocates>& tg_to_load, const std::shared_ptr<RegisteredModel> model, const std::shared_ptr<GPUTensorPool> pool,  std::vector<char*>& allocated_regions) {
        auto start_merge_time=std::chrono::high_resolution_clock::now();
        std::vector<TGNeedAllocates> remaining = tg_to_load;
        // 步骤5: 剩余TG按大小排序后调用GreedyMerge
        LOG(INFO)<<"Remaining TGs need to merge: "<<remaining.size();
        if (!remaining.empty()) {
            // 将remaining按照size从大到小排序
            std::sort(remaining.begin(), remaining.end(),
                      [](const TGNeedAllocates& a, const TGNeedAllocates& b) {
                        return a.tg_index.size > b.tg_index.size;
                      });

            for (auto tg : remaining) {
                auto merged = pool->GreedyMerge(tg.tg_index.size);
                if(!merged){
                    LOG(ERROR) << "GreedyMerge failed for TG size=" << tg.tg_index.size << ", fingerprint=" << tg.tg_index.fingerprint;
                    return -1;
                }
                auto reg=pool->AllocateFreeRegion(merged, tg.tg_index.size, tg.tg_index.fingerprint);
                if(!reg){
                    LOG(ERROR) << "AllocateFreeRegion failed for TG size=" << tg.tg_index.size << ", fingerprint=" << tg.tg_index.fingerprint;
                    return -1;
                }
                reg->model_ref = model;
                allocated_regions[tg.tg_id] = reg->addr;
            
            }
        }
        auto end_merge_time=std::chrono::high_resolution_clock::now();
        auto merge_duration=std::chrono::duration_cast<std::chrono::milliseconds>(end_merge_time-start_merge_time);
        merge_latencies_.push_back(merge_duration.count());
        return 0;
    }

    std::string LoadModel(const std::string& model_path, int device_id, int strategy=4) {
        auto start_time=std::chrono::high_resolution_clock::now();
        // 步骤1: 检查模型和设备
        auto model_it = registered_models_.find(model_path);
        auto pool_it = gpu_tensor_pools_.find(device_id);
        if (model_it == registered_models_.end() || pool_it == gpu_tensor_pools_.end()) {
            LOG(ERROR) << "Model or device not found";
            return "ERROR";
        }

        // 步骤2: 收集待装载的TG
        auto& model = model_it->second;
        auto& pool = pool_it->second;
        const auto& tg_index = model->GetTensorGroupIndexes();
        std::vector<TGNeedAllocates> tg_to_load;
        std::vector<char*> allocated_regions(tg_index.size(), nullptr);
        for (int i=0; i<tg_index.size(); i++) {
            auto allocated = pool->GetTensor(tg_index[i].fingerprint);
            if(allocated){
                allocated_regions[i]=allocated->addr;
            }else{
                tg_to_load.push_back({i, tg_index[i]});
            }
        }
        LOG(INFO)<<"LoadModel: model_path="<<model_path<<" to_allocates.size()="<<tg_to_load.size()<<" / "<<tg_index.size();

        // 步骤3: 调用GreedyDrop确保空间足够
        size_t total_need = 0;
        for (auto tg : tg_to_load) {
            total_need += tg.tg_index.size;
        }
        if(pool->GreedyDrop(total_need, model_path)){
            LOG(ERROR)<<"GreedyDrop failed";
            return "ERROR";
        }

        // 分配空闲空间给TG，更新allocated_regions
        switch (strategy) {
          case 0:
            if(WithoutReuse(tg_to_load, model, pool, allocated_regions, device_id)!=0){
              LOG(ERROR)<<"WithoutReuse failed";
              return "ERROR";
            }
            break;
          case 1:
            if(GlobalMerge(tg_to_load, model, pool, allocated_regions)!=0){
              LOG(ERROR)<<"GlobalMerge failed";
              return "ERROR";
            }
            break;
          case 2:
            if(GreedyMerge(tg_to_load, model, pool, allocated_regions)!=0){
              LOG(ERROR)<<"PartitionedBinPacking failed";
              return "ERROR";
            }
            break;
          case 3:
            if(WeightedBipartiteMatch_GreedyMerge(tg_to_load, model, pool, allocated_regions)!=0){
              LOG(ERROR)<<"WeightedBipartiteMatch_GreedyMerge failed";
              return "ERROR";
            }
            break;
          case 4:
            if(PartitionedBinPacking(tg_to_load, model, pool, allocated_regions)!=0){
              LOG(ERROR)<<"GreedyMerge failed";
              return "ERROR";
            }
            break;
          default:
            LOG(ERROR)<<"Invalid strategy";
            return "ERROR";
        }

        pool->UseModel(model_path);
        if (!tg_to_load.empty()) {
          std::vector<int> tg_to_load_ids;
          for (auto tg : tg_to_load) {
            tg_to_load_ids.push_back(tg.tg_id);
          }

          // 检查allocated_regions是否正确
          for(auto region : allocated_regions) {
            if(region==nullptr){
              LOG(ERROR)<<"allocated_regions is nullptr";
              return "ERROR";
            }
          }

          if (model->LoadModelFromMem(allocated_regions, tg_to_load_ids,
                                      device_id) != 0) {
            LOG(ERROR) << "Failed to load model to GPU";
            return "ERROR";
          }
        }
        auto end_time=std::chrono::high_resolution_clock::now();
        auto duration=std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time);
        model_load_latencies_.push_back(duration.count());
        model_paths_.push_back(model_path);
        return "SUCCESS";
    }

};