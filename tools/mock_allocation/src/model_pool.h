#pragma once

#include <cuda_runtime.h>

#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "LRUList.h"
#include "logger.h"
#include "registered_model.h"

#define MAX_IN_GPU_TENSOR_GROUP 100000
#define MAX_IN_CPU_MODEL 100
#define PINNED_GPU_TENSOR_POOL_RATIO 55
class GPUMemoryRegion {
 public:
  bool is_allocated;
  char* addr;
  size_t size;
  std::string fingerprint;
  std::shared_ptr<GPUMemoryRegion> prev;
  std::shared_ptr<GPUMemoryRegion> next;

  GPUMemoryRegion()
      : is_allocated(false), size(0), prev(nullptr), next(nullptr) {}

  GPUMemoryRegion(void* addr_, size_t size)
      : is_allocated(false), size(size), prev(nullptr), next(nullptr) {
    this->addr = static_cast<char*>(addr_);
  }

  void SetAddr(void* addr) { this->addr = static_cast<char*>(addr); }

  bool isAdjacent(std::shared_ptr<GPUMemoryRegion> other) {
    return (static_cast<char*>(this->addr) + this->size == other->addr) ||
           (static_cast<char*>(other->addr) + other->size == this->addr);
  }

  void merge(std::shared_ptr<GPUMemoryRegion> other) {
    if (this->isAdjacent(other)) {
      this->size += other->size;
      this->next = other->next;
      if (other->next) {
        other->next->prev = std::make_shared<GPUMemoryRegion>(*this);
      }
    }
  }
};

struct CompareMemoryRegion {
  bool operator()(
      const std::pair<size_t, std::shared_ptr<GPUMemoryRegion>>& lhs,
      const std::pair<size_t, std::shared_ptr<GPUMemoryRegion>>& rhs) const {
    if (lhs.first == rhs.first) {
      return lhs.second < rhs.second;  // 比较 std::shared_ptr
    }
    return lhs.first < rhs.first;  // 比较大小
  }
};

class GPUTensorPool_V3 {
 private:
  size_t total_size;
  std::string uuid;
  int device_id;
  cudaStream_t stream_;

  std::vector<std::shared_ptr<GPUMemoryRegion>> memory_regions;
  std::unordered_map<std::string, std::shared_ptr<GPUMemoryRegion>>
      allocated_regions;  // Fingerprint -> Region
  std::set<std::pair<size_t, std::shared_ptr<GPUMemoryRegion>>,
           CompareMemoryRegion>
      free_regions;  // Use set for fast allocation (ordered by size)

  LRUCache<std::string, std::shared_ptr<RegisteredModel>>
      lru_models;  // Use LRUCache for managing LRU list

  std::shared_ptr<GPUMemoryRegion>
      memory_region_view;  // Pointer to the start of the memory region

 public:
  GPUTensorPool_V3(int device_id_, size_t total_size)
      : device_id(device_id_),
        total_size(total_size),
        lru_models(MAX_IN_CPU_MODEL) {  // Cache size is an example
    cudaSetDevice(device_id_);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id_);
    // Get GPU UUID
    char uuidStr[80];
    snprintf(
        uuidStr, sizeof(uuidStr),
        "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
        (unsigned char)props.uuid.bytes[0], (unsigned char)props.uuid.bytes[1],
        (unsigned char)props.uuid.bytes[2], (unsigned char)props.uuid.bytes[3],
        (unsigned char)props.uuid.bytes[4], (unsigned char)props.uuid.bytes[5],
        (unsigned char)props.uuid.bytes[6], (unsigned char)props.uuid.bytes[7],
        (unsigned char)props.uuid.bytes[8], (unsigned char)props.uuid.bytes[9],
        (unsigned char)props.uuid.bytes[10],
        (unsigned char)props.uuid.bytes[11],
        (unsigned char)props.uuid.bytes[12],
        (unsigned char)props.uuid.bytes[13],
        (unsigned char)props.uuid.bytes[14],
        (unsigned char)props.uuid.bytes[15]);
    uuid = std::string(uuidStr);
    cudaError_t err = cudaStreamCreate(&stream_);
    if (err != cudaSuccess) {
      LOG(ERROR) << "cudaStreamCreate error: " << cudaGetErrorString(err);
    }

    void* gpu_memory;
    err = cudaMalloc(&gpu_memory, total_size);
    if (err != cudaSuccess) {
      LOG(ERROR) << "cudaMalloc error: " << cudaGetErrorString(err);
      exit(1);
    }
    LOG(INFO) << "GPU device: " << device_id
              << " allocated: " << total_size / 1024.0 / 1024.0 / 1024.0
              << " GB";

    auto initial_region =
        std::make_shared<GPUMemoryRegion>(gpu_memory, total_size);
    memory_regions.push_back(initial_region);
    free_regions.insert({total_size, initial_region});
    memory_region_view = initial_region;
  }

  ~GPUTensorPool_V3() {
    for (auto region : memory_regions) {
      cudaSetDevice(device_id);
      cudaFree(region->addr);
    }

    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
    LOG(INFO) << "Destroy GPUTensorPool for device " << device_id;
  }

  int GetDeviceId() { return device_id; }
  size_t GetMemorySize() { return total_size; }

  char* GetBaseAddr() { return memory_region_view->addr; }

  // Get a memory region by its fingerprint
  std::shared_ptr<GPUMemoryRegion> GetTensor(const std::string& fingerprint) {
    auto it = allocated_regions.find(fingerprint);
    if (it != allocated_regions.end()) {
      return it->second;
    } else {
      return nullptr;
    }
  }

  int UseModel(std::shared_ptr<RegisteredModel> model) {
    lru_models.put(model->model_path(), model);
    return 0;
  }

  // Find the best fit memory region, or evict LRU regions if necessary
  std::shared_ptr<GPUMemoryRegion> BestFitAllocation(
      size_t requested_size, const std::string& fingerprint) {
    auto ret = GetTensor(fingerprint);
    if (ret) {
      return ret;
    }

    // Search for the best fit in the free regions (using set)
    auto it = free_regions.lower_bound({requested_size, nullptr});
    if (it != free_regions.end()) {
      auto region = it->second;

      if (region->size > requested_size) {
        // Split the region
        auto remaining_region = std::make_shared<GPUMemoryRegion>(
            static_cast<char*>(region->addr) + requested_size,
            region->size - requested_size);
        remaining_region->is_allocated = false;

        // Update the original region's size
        region->size = requested_size;
        region->is_allocated = true;
        region->fingerprint = fingerprint;

        // Update the linked list pointers
        remaining_region->prev = region;
        remaining_region->next = region->next;
        if (region->next) {
          region->next->prev =
              remaining_region;  // Update next region's prev pointer
        }
        region->next =
            remaining_region;  // Link the current region to the new region

        // Update free regions and insert remaining region
        free_regions.erase(it);
        auto ret =
            free_regions.insert({remaining_region->size, remaining_region});
        if (!ret.second) {
          LOG(ERROR) << "Remaining Free Region Insert Failed";
        }

        allocated_regions[region->fingerprint] = region;

        return region;
      } else {
        free_regions.erase(it);
        region->is_allocated = true;
        region->fingerprint = fingerprint;
        allocated_regions[region->fingerprint] = region;

        return region;
      }
    }

    bool find_victim = false;
    while (lru_models.getLRUKeySize() > 0) {
      auto model = lru_models.getLRUValue();
      if (model) {
        auto tg_index = model->GetTensorGroupIndexes();
        for (int i = tg_index.size() - 1; i >= 0; i--) {
          auto region_it = allocated_regions.find(tg_index[i].fingerprint);
          if (region_it != allocated_regions.end()) {
            // LOG(INFO) << "Evicting region: " << region_it->second->fingerprint;
            FreeRegion(region_it->second);
            find_victim = true;
            break;
          }
        }
      }
      if (find_victim) {
        break;
      }
      // this model is not used anymore, remove it from lru
      lru_models.remove(model->model_path());
      LOG(INFO) << "Evicting model: " << model->model_path();
    }
    if (find_victim) {
      return BestFitAllocation(requested_size, fingerprint);
    }
    return nullptr;
  }

  void FreeRegion(std::shared_ptr<GPUMemoryRegion> region) {
    auto it = region;
    auto prev = it->prev;
    auto next = it->next;
    // 1. Update Allocated Region map
    allocated_regions.erase(it->fingerprint);

    // 2. Update Free Region Set
    if (next && !next->is_allocated) {
      // next region is merged, delete it from free region set
      auto target_pair = std::make_pair(next->size, next);
      auto it = free_regions.find(target_pair);
      if (it == free_regions.end()) {
        LOG(ERROR) << "Free Merge Next Find Failed";
      } else {
        free_regions.erase(it);
      }
    }
    if (prev && !prev->is_allocated) {
      // prev region is merged, delete it from free region set
      auto target_pair = std::make_pair(prev->size, prev);
      auto it = free_regions.find(target_pair);
      if (it == free_regions.end()) {
        LOG(ERROR) << "Free Merge Prev Find Failed";
      } else {
        free_regions.erase(it);
      }
    }

    // 3. create new free region
    // 3.1 Update Region List
    if (next && !next->is_allocated) {
      it->merge(next);
    }
    if (prev && !prev->is_allocated) {
      prev->merge(it);
      it = prev;
    }
    it->is_allocated = false;
    // 3.2 insert into free region set
    auto ret = free_regions.insert({it->size, it});
    if (!ret.second) {
      LOG(ERROR) << "Free Region Insert Failed";
    }
  }

  void MemoryRegionView() {
    auto mr = memory_region_view;
    while (mr) {
      std::cout << "[" << mr->fingerprint << "]: (" << mr->is_allocated << ", "
                << mr->size << ", " << reinterpret_cast<void*>(mr->addr)
                << " ) ->";

      mr = mr->next;
    }
    std::cout << std::endl;
  }

  // used space / total space
  void MemoryUsage() {
    size_t used = 0;
    auto mr = memory_region_view;
    while (mr) {
      if (mr->is_allocated) {
        used += mr->size;
      }
      mr = mr->next;
    }
    std::cout << "Fragmentation: " << (double)(total_size - used) / total_size
              << std::endl;
  }
};

class ModelPool {
 public:
  ModelPool(size_t total_size, int num_threads);
  ~ModelPool();
  int64_t RegisterModel(const std::string& model_path);
  size_t GetModelSize() const { return cpu_model_pool_size_; }
  std::string LoadModelFromAsync(const std::string& model_path);

 private:
  // CPU Model Pool
  std::mutex mutex_;
  size_t cpu_model_pool_size_;
  size_t cpu_model_pool_allocated_;
  int num_threads_;
  std::queue<std::future<int>> async_tasks_;

  std::unordered_map<std::string, std::shared_ptr<RegisteredModel>>
      registered_models_;
  std::shared_ptr<LRUCache<std::string, std::shared_ptr<RegisteredModel>>>
      in_cpu_models;

  // GPU Tensor Pool
  std::vector<std::shared_ptr<GPUTensorPool_V3>> gpu_tensor_pools_;
};
