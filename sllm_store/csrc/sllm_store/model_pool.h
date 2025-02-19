#pragma once

#include <mutex>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <memory>
#include <queue>
#include <future>

#include "registered_model.h"
#include "LRUList.h"

#define MAX_IN_GPU_TENSOR_GROUP 100000
#define MAX_IN_CPU_MODEL 100
#define PINNED_GPU_TENSOR_POOL_RATIO 60


class GPUMemoryRegion {
    public:
        bool is_allocated;
        char* addr;
        size_t size;
        std::string fingerprint;

        GPUMemoryRegion() : is_allocated(false), size(0) {}
    
        GPUMemoryRegion(void* addr_, size_t size) : is_allocated(false), size(size) {
            this->addr = static_cast<char*>(addr_);
        }

        void SetAddr(void* addr) {
            this->addr= static_cast<char*>(addr);
        }

    };

class GPUTensorPool {
public:
    GPUTensorPool(int gpu_device_id, size_t total_size);
    ~GPUTensorPool();

    std::shared_ptr<LRUCache<std::string, std::shared_ptr<GPUMemoryRegion>>> GetInGPUTensor() { return in_gpu_tensor_groups; }

    std::shared_ptr<std::list<GPUMemoryRegion>> GetMemoryViewList() { return memory_view; }

    int GetDeviceId() { return device_id; }
    size_t GetMemorySize() { return gpu_memory_size; }
    void* GetMemory() { return gpu_memory; }
    

    
private:
    std::string uuid;
    cudaStream_t stream_;
    int device_id;
    size_t gpu_memory_size;
    void* gpu_memory;
    std::shared_ptr<std::list<GPUMemoryRegion>> memory_view;
    std::shared_ptr<LRUCache<std::string, std::shared_ptr<GPUMemoryRegion>>> in_gpu_tensor_groups;
};

class ModelPool {
public:
    ModelPool(size_t total_size, int num_threads);
    ~ModelPool();
    int64_t RegisterModel(const std::string& model_path);
    size_t GetModelSize() const { return cpu_model_pool_size_; }
    std::string LoadModelFromDiskAsync(const std::string& model_path);

private:
    // CPU Model Pool
    std::mutex mutex_;
    size_t cpu_model_pool_size_;
    size_t cpu_model_pool_allocated_;
    int num_threads_;
    std::queue<std::future<int>> async_tasks_;

    std::unordered_map<std::string, std::shared_ptr<RegisteredModel>> registered_models_;
    std::shared_ptr<LRUCache<std::string, std::shared_ptr<RegisteredModel>>> in_cpu_models;

    // GPU Tensor Pool
    std::vector<std::shared_ptr<GPUTensorPool>> gpu_tensor_pools_;
};
