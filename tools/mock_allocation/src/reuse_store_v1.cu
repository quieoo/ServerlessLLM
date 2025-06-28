#include "reuse_store.h"

ReuseStore::ReuseStore(const std::string& storage_path, size_t memory_pool_size,
    int num_thread)
: storage_path_(storage_path) {
    // the filed memory_pool_size is not used in V1, since we assume the CPU Memory is larger enough, so this filed is used to specify the reversed space for each GPU
    // gpu_bandwidth is 400GB, cpu_bandwidht is set to be 20GB
    vram_manager_= std::make_shared<VRAMManager>(memory_pool_size, 400.0*1024*1024, 20.0*1024*1024);
}

ReuseStore::~ReuseStore() {
    vram_manager_.reset();
}


int64_t ReuseStore::RegisterModelInfo(const std::string& model_path) {
    if(model_path[0] == '/'){
      return vram_manager_->RegisterModel(model_path);
    }
    return vram_manager_->RegisterModel(storage_path_ + "/" + model_path);
}

std::string ReuseStore::LoadModelFromDiskAsync(const std::string& model_path, int device_id) {
    if(model_path[0] == '/'){
      return vram_manager_->LoadModel(model_path, device_id);
    }
  
    return vram_manager_->LoadModel(storage_path_ + "/" + model_path, device_id);
}

std::string ReuseStore::GetPoolHandle(int pool_id) {
    return vram_manager_->getPoolHandle(pool_id);
}

int ReuseStore::GetAvailableBlocksonGPU(std::string use_model, size_t block_size, int pool_id) {
    return vram_manager_->GetAvailableBlocks(use_model, block_size, pool_id);
}

std::vector<size_t> ReuseStore::AllocateBlocksonGPU(int device_id, size_t block_size, std::string model_path, int block_number) {
    return vram_manager_->AllocateBlocks(device_id, block_size, model_path, block_number);
}
