#include <cstdint>
#include "reuse_store_v1.h"
#include <chrono>
#include <iomanip>

ReuseStoreV1::ReuseStoreV1(const std::string& storage_path, size_t memory_pool_size,
    int num_thread, int load_strategy=4)
: storage_path_(storage_path) {
    // the filed memory_pool_size is not used in V1, since we assume the CPU Memory is larger enough, so this filed is used to specify the reversed space for each GPU
    // gpu_bandwidth is 400GB, cpu_bandwidht is set to be 20GB
    vram_manager_= std::make_shared<VRAMManager>(memory_pool_size, 400.0*1024*1024, 20.0*1024*1024, load_strategy);
}

ReuseStoreV1::~ReuseStoreV1() {
    vram_manager_.reset();
}


int64_t ReuseStoreV1::RegisterModelInfo(const std::string& model_path) {
    if(model_path[0] == '/'){
      return vram_manager_->RegisterModel(model_path);
    }
    return vram_manager_->RegisterModel(storage_path_ + "/" + model_path);
}

std::string ReuseStoreV1::LoadModelFromDiskAsync(const std::string& model_path, int device_id) {

    // 打印当前的时间
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    double timestamp = std::chrono::duration_cast<std::chrono::microseconds>(duration).count() / 1e6;
    std::cout << std::fixed << std::setprecision(6) << timestamp << std::endl;

    vram_manager_->CachedModels();
    if(model_path[0] == '/'){
      return vram_manager_->LoadModel(model_path, device_id);
    }
  
    return vram_manager_->LoadModel(storage_path_ + "/" + model_path, device_id);
}

int64_t ReuseStoreV1::ToLoadSize(std::string model_path){
    if(model_path[0] == '/'){
        return vram_manager_->GetToLoadSize(model_path);
      }
    
      return vram_manager_->GetToLoadSize(storage_path_ + "/" + model_path);
}

std::vector<int64_t> ReuseStoreV1::ToLoadSizes(std::vector<std::string> model_paths){
    std::vector<int64_t> load_sizes;
    for(auto model_path: model_paths){
        auto s=ToLoadSize(model_path);
        load_sizes.push_back(s);
    }
    return load_sizes;
}

std::string ReuseStoreV1::GetPoolHandle(int pool_id) {
    return vram_manager_->getPoolHandle(pool_id);
}

int ReuseStoreV1::GetAvailableBlocksonGPU(std::string use_model, size_t block_size, int pool_id) {
    return vram_manager_->GetAvailableBlocks(use_model, block_size, pool_id);
}

std::vector<size_t> ReuseStoreV1::AllocateBlocksonGPU(int device_id, size_t block_size, std::string model_path, int block_number) {
    return vram_manager_->AllocateBlocks(device_id, block_size, model_path, block_number);
}
