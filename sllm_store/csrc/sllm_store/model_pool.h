#pragma once

#include <glog/logging.h>

#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "LRUList.h"
#include "binary_utils.h"
#include "registered_model.h"

#define MAX_IN_GPU_TENSOR_GROUP 100000
#define MAX_IN_CPU_MODEL 100
#define PINNED_GPU_TENSOR_POOL_RATIO 65

class GPUMemoryRegion : public std::enable_shared_from_this<GPUMemoryRegion> {
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
        other->next->prev = shared_from_this();
      }
    }
  }

  std::string toString() {
    return "[GPUMemoryRegion: addr=" +
           std::to_string(reinterpret_cast<size_t>(addr)) +
           ", size=" + std::to_string(size) + ", fingerprint=" + fingerprint +
           ", is_allocated=" + std::to_string(is_allocated) + "]";
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

class GPUTensorPool {
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
  GPUTensorPool(int device_id_, size_t total_size)
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

  ~GPUTensorPool() {
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
      size_t requested_size, const std::string& fingerprint,
      std::string model_path) {
    if (fingerprint == "kvcache") {
      // LOG(INFO) << "allocate kvcache";
    } else {
      auto ret = GetTensor(fingerprint);
      if (ret) {
        return ret;
      }
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
      if (model->model_path() == model_path) {
        LOG(ERROR) << "Found the request as victim, No enough memory for any "
                      "allocation";
        return nullptr;
      }
      if (model) {
        auto tg_index = model->GetTensorGroupIndexes();
        for (int i = tg_index.size() - 1; i >= 0; i--) {
          auto region_it = allocated_regions.find(tg_index[i].fingerprint);
          if (region_it != allocated_regions.end()) {
            // LOG(INFO) << "Evicting region: " <<
            // region_it->second->fingerprint;
            // LOG(INFO) << "Evicting TG " << i << " of model "
            //           << model->model_path();
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
      return BestFitAllocation(requested_size, fingerprint, model_path);
    }
    return nullptr;
  }

  void FreeRegion(std::shared_ptr<GPUMemoryRegion> region) {
    auto prev = region->prev;
    auto next = region->next;
    // 1. Update Allocated Region map
    allocated_regions.erase(region->fingerprint);

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
        // output elements in free_regions
        LOG(INFO) << "Free Region Set:";
        for (auto it = free_regions.begin(); it != free_regions.end(); ++it) {
          std::string ptr =
              std::to_string(reinterpret_cast<uintptr_t>(it->second.get()));
          LOG(INFO) << "Free Region: " << it->first << " " << ptr << " "
                    << it->second->toString();
        }
        std::string target_pair_str =
            std::to_string(reinterpret_cast<uintptr_t>(prev.get()));
        LOG(INFO) << "Target Pair: " << target_pair.first << " "
                  << target_pair_str << " " << prev->toString();
      } else {
        free_regions.erase(it);
      }
    }

    // 3. create new free region
    // 3.1 Update Region List
    if (next && !next->is_allocated) {
      region->merge(next);
    }
    if (prev && !prev->is_allocated) {
      prev->merge(region);
      region = prev;
    }
    region->is_allocated = false;
    // 3.2 insert into free region set
    auto ret = free_regions.insert({region->size, region});
    if (!ret.second) {
      LOG(ERROR) << "Free Region Insert Failed";
    }
  }

  void CheckConsistency() {
    size_t allocated_regions_num = allocated_regions.size();
    size_t free_regions_num = free_regions.size();

    size_t actual_allocated_regions = 0;
    size_t actual_free_regions = 0;
    auto it = memory_region_view;
    while (it) {
      if (it->is_allocated) {
        actual_allocated_regions++;
      } else {
        actual_free_regions++;
      }
      it = it->next;
    }
    if (allocated_regions_num != actual_allocated_regions) {
      LOG(ERROR) << "allocated_regions!=actual_allocated_regions"
                 << allocated_regions_num << " " << actual_allocated_regions;
    }
    if (free_regions_num != actual_free_regions) {
      LOG(ERROR) << "free_regions!=actual_free_regions" << free_regions_num
                 << " " << actual_free_regions;
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

  int GetAvilableBlocks(std::unordered_set<std::string>& fp_set,
                        size_t block_size) {
    int count = 0;
    auto mr = memory_region_view;
    while (mr) {
      if (!mr->is_allocated && fp_set.find(mr->fingerprint) == fp_set.end()) {
        count += mr->size / block_size;
      }
      mr = mr->next;
    }
    return count;
  }
};

class ModelPool {
 public:
  ModelPool(size_t cpu_memoery_size, int num_threads, size_t gpu_pool_size = 0)
      : cpu_model_pool_size_(cpu_memoery_size), num_threads_(num_threads) {
    LOG(INFO) << "Create ModelPool with "
              << cpu_memoery_size / 1024.0 / 1024.0 / 1024.0 << " GB";
    cpu_model_pool_allocated_ = 0;
    in_cpu_models = std::make_shared<
        LRUCache<std::string, std::shared_ptr<RegisteredModel>>>(
        MAX_IN_CPU_MODEL);

    // get the number of GPUs in the system
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    // int num_gpus = 1;
    gpu_tensor_pools_.resize(num_gpus);
    for (int i = 0; i < num_gpus; i++) {
      // get the total memory size of the GPU
      cudaSetDevice(i);
      if (gpu_pool_size == 0) {
        size_t total_memory;
        cudaMemGetInfo(NULL, &total_memory);

        gpu_tensor_pools_[i] = std::make_shared<GPUTensorPool>(
            i, total_memory * PINNED_GPU_TENSOR_POOL_RATIO / 100);
      } else {
        gpu_tensor_pools_[i] =
            std::make_shared<GPUTensorPool>(i, gpu_pool_size);
      }
    }
  }
  ~ModelPool() { LOG(INFO) << "Destroy ModelPool"; }
  int64_t RegisterModel(const std::string& model_path) {
    std::unique_lock<std::mutex> lock_info(mutex_);
    auto rmodel = registered_models_.find(model_path);
    if (rmodel != registered_models_.end()) {
      LOG(WARNING) << "Model " << model_path << " already registered";
      return rmodel->second->model_size();
    }

    auto model = std::make_shared<RegisteredModel>(model_path);
    registered_models_[model_path] = model;
    return model->model_size();
  }
  size_t GetModelSize() const { return cpu_model_pool_size_; }
  std::string LoadModelAsync(const std::string& model_path, int device_id) {
    std::unique_lock<std::mutex> lock_info(mutex_);

    if (registered_models_.find(model_path) == registered_models_.end()) {
      LOG(ERROR) << "Model " << model_path << " not registered";
      return "ERROR";
    }

    LOG(INFO) << "load registered model " << model_path;

    //   TODO: first check tensor groups in GPU Tensor Pool, so only need to
    //   load the missing tensor groups from disk. To do this, the CPU Model
    //   Pool should also be managed in Tensor Group level.
    //   1. if model not in CPU model pool, load it from disk asyncronously
    if (!in_cpu_models->contains(model_path)) {
      LOG(INFO) << "model not in cpu model pool";
      async_tasks_.emplace(std::async(std::launch::async, [this, model_path]() {
        // check if cpu model pool free memory is enough
        LOG(INFO) << "LoadModelFromDiskAsync: " << model_path
                  << " cpu_model_pool_allocated_ "
                  << cpu_model_pool_allocated_ / 1024.0 / 1024.0 / 1024.0
                  << " cpu_model_pool_size_ "
                  << cpu_model_pool_size_ / 1024.0 / 1024.0 / 1024.0
                  << " model_size "
                  << registered_models_[model_path]->model_size() / 1024.0 /
                         1024.0 / 1024.0;

        if (cpu_model_pool_size_ <
            registered_models_[model_path]->model_size()) {
          LOG(ERROR) << "cpu_model_pool_size_ < "
                        "registed_models_[model_path]->model_size()";
          return -1;
        }
        if (cpu_model_pool_size_ - cpu_model_pool_allocated_ <
            registered_models_[model_path]->model_size()) {
          // iterate through the LRU cache and evict the least recently used
          // models
          while (cpu_model_pool_size_ - cpu_model_pool_allocated_ <
                 registered_models_[model_path]->model_size()) {
            if (in_cpu_models->getLRUKeySize() > 0) {
              auto lru_model = in_cpu_models->getandRemoveLRUKey();
              //   TODO: might check whether the model is still in use
              int freed = registered_models_[lru_model]->UnloadModel();
              if (freed <= 0) {
                LOG(WARNING) << "unloaded model " << lru_model << " failed";
                return -1;
              } else {
                cpu_model_pool_allocated_ -= freed;
                LOG(INFO) << "unloaded model " << lru_model << " success";
              }
            } else {
              LOG(ERROR) << "no more models in cpu_model_pool";
              return -1;
            }
          }
        }

        if (registered_models_[model_path]->LoadModelFromDisk(num_threads_) !=
            0) {
          LOG(ERROR) << "LoadModelFromDiskAsync: " << model_path << " failed";
          return -1;
        }
        in_cpu_models->put(model_path, registered_models_[model_path]);
        cpu_model_pool_allocated_ +=
            registered_models_[model_path]->model_size();
        // LOG(INFO) << "LoadModelFromDiskAsync: " << model_path
        //           << " cpu_model_pool_allocated_ " <<
        //           cpu_model_pool_allocated_
        //           << " cpu_model_pool_size_ " << cpu_model_pool_size_;
        return 0;
      }));
    }
    // 2. check tensor groups in GPU Tensor Pool
    // TODO: if multiple GPU Tensor Pool, need to check all of them and pick the
    // best one currently, only check the first GPU Tensor Pool
    if (gpu_tensor_pools_.size() < 1) {
      LOG(ERROR) << "No GPU Available";
      return "ERROR";
    }
    auto gpu_tensor_pool = gpu_tensor_pools_[device_id];
    gpu_tensor_pool->UseModel(registered_models_[model_path]);

    const std::vector<TensorGroupIndex>& tg_index =
        registered_models_[model_path]->GetTensorGroupIndexes();
    std::vector<char*> allocated_region(tg_index.size(), nullptr);
    std::vector<int> region_to_load;
    int tensor_groups_need_to_load = tg_index.size();

    for (int i = 0; i < tg_index.size(); i++) {
      auto mem_region = gpu_tensor_pool->GetTensor(tg_index[i].fingerprint);
      if (mem_region && mem_region->is_allocated) {
        // already allocated
        allocated_region[i] = mem_region->addr;
        tensor_groups_need_to_load--;
      } else {
        // allocate new memory region
        mem_region = gpu_tensor_pool->BestFitAllocation(
            tg_index[i].size, tg_index[i].fingerprint, model_path);
        if (!mem_region) {
          LOG(ERROR) << "No enough memory in GPU Tensor Pool";
          return "ERROR";
        }
        // gpu_tensor_pool->CheckConsistency();
        allocated_region[i] = mem_region->addr;
        region_to_load.push_back(i);
      }
    }

    LOG(INFO) << "Tensor Groups need to load: " << tensor_groups_need_to_load
              << "/" << tg_index.size();

    // 3. copy data from CPU to GPU async
    async_tasks_.emplace(
        std::async(std::launch::async, [this, model_path, region_to_load,
                                        allocated_region, gpu_tensor_pool]() {
          return registered_models_[model_path]->LoadModelFromMem(
              allocated_region, region_to_load, gpu_tensor_pool->GetDeviceId());
        }));

    // 4. create response
    // each 8bytes represents: cudaIPCMemHandle, device_id, offset for each
    // tensor
    std::string ret;
    // cudaIpcMemHandle_t handle;
    // cudaSetDevice(gpu_tensor_pools_[0]->GetDeviceId());
    // cudaIpcGetMemHandle(&handle, gpu_tensor_pool->GetBaseAddr());
    // // ret=std::string(reinterpret_cast<const char*>(&handle),
    // // sizeof(cudaIpcMemHandle_t));
    // std::string handle_str = std::string(reinterpret_cast<const
    // char*>(&handle),
    //                                      sizeof(cudaIpcMemHandle_t));
    // ret = toHex(std::vector<uint8_t>(handle_str.begin(), handle_str.end()));
    // response.push_back(gpu_tensor_pools_[0]->GetDeviceId());

    std::vector<size_t> response;
    for (int i = 0; i < allocated_region.size(); i++) {
      size_t tensor_group_base_offset =
          allocated_region[i] -
          static_cast<char*>(gpu_tensor_pool->GetBaseAddr());
      for (auto it = tg_index[i].tensor_indexes.begin();
           it != tg_index[i].tensor_indexes.end(); it++) {
        response.push_back(tensor_group_base_offset + it->offset);
      }
    }
    // ret+=std::string(reinterpret_cast<const char*>(response.data()),
    // response.size()*sizeof(size_t));
    std::string response_str(reinterpret_cast<const char*>(response.data()),
                             response.size() * sizeof(size_t));
    ret +=
        toHex(std::vector<uint8_t>(response_str.begin(), response_str.end()));
    // LOG(INFO)<<"ModelPool::LoadModelFromMem: load model from memory, ret:
    // "<<ret;

    // make sure all async tasks are done
    while (!async_tasks_.empty()) {
      std::future<int>& task = async_tasks_.front();
      task.wait();
      async_tasks_.pop();
    }

    return ret;
  }

  std::string getPoolHandle(int device_id) {
    if (device_id < 0 || device_id >= gpu_tensor_pools_.size()) {
      LOG(ERROR) << "ModelPool::getPoolHandle: invalid device_id: "
                 << device_id;
      return "";
    }
    std::string ret;
    cudaIpcMemHandle_t handle;
    cudaSetDevice(gpu_tensor_pools_[device_id]->GetDeviceId());
    cudaIpcGetMemHandle(&handle, gpu_tensor_pools_[device_id]->GetBaseAddr());
    std::string handle_str = std::string(reinterpret_cast<const char*>(&handle),
                                         sizeof(cudaIpcMemHandle_t));
    ret = toHex(std::vector<uint8_t>(handle_str.begin(), handle_str.end()));
    return ret;
  }

  int GetAvailableBlocks(int device_id, size_t block_size,
                         std::string model_path) {
    if (registered_models_.find(model_path) == registered_models_.end()) {
      LOG(ERROR) << "Model " << model_path << " not registered";
      return 0;
    }
    if (device_id < 0 || device_id >= gpu_tensor_pools_.size()) {
      LOG(ERROR) << "ModelPool::GetAvailuableBlocks: invalid device_id: "
                 << device_id;
      return 0;
    }

    auto fp_set = registered_models_[model_path]->GetFingerPrints();
    // LOG(INFO)<<"fp_set size "<<fp_set.size();
    return gpu_tensor_pools_[device_id]->GetAvilableBlocks(fp_set, block_size);
  }

  std::vector<size_t> AllocateBlocks(int device_id, size_t block_size,
                                     std::string model_path, int block_number) {
    auto gpu = gpu_tensor_pools_[device_id];
    std::vector<size_t> allocated_block_offsets;
    for (int i = 0; i < block_number; i++) {
      auto mem_region =
          gpu->BestFitAllocation(block_size, "kvcache", model_path);
      if (!mem_region) {
        LOG(ERROR) << "No enough memory in GPU Tensor Pool";
        return {};
      }
      allocated_block_offsets.push_back(mem_region->addr - gpu->GetBaseAddr());
    }
    return allocated_block_offsets;
  }

  void MemoryUsage() {
    for (auto pool : gpu_tensor_pools_) {
      pool->MemoryUsage();
    }
  }

 private:
  // CPU Model Pool
  std::mutex mutex_;
  size_t cpu_model_pool_size_;
  size_t cpu_model_pool_allocated_;
  int num_threads_;
  std::queue<std::future<int>> async_tasks_;
  size_t gpu_tensor_pool_size;

  std::unordered_map<std::string, std::shared_ptr<RegisteredModel>>
      registered_models_;
  std::shared_ptr<LRUCache<std::string, std::shared_ptr<RegisteredModel>>>
      in_cpu_models;

  // GPU Tensor Pool
  std::vector<std::shared_ptr<GPUTensorPool>> gpu_tensor_pools_;
};
