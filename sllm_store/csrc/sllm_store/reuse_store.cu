#include "reuse_store.h"

ReuseStore::ReuseStore(const std::string& storage_path, size_t memory_pool_size,
                       int num_thread)
    : storage_path_(storage_path) {
  model_pool_ = std::make_shared<ModelPool>(memory_pool_size, num_thread);
}

ReuseStore::~ReuseStore() {}

int64_t ReuseStore::RegisterModelInfo(const std::string& model_path) {
  if(model_path[0] == '/'){
    return model_pool_->RegisterModel(model_path);
  }
  return model_pool_->RegisterModel(storage_path_ + "/" + model_path);
}

std::string ReuseStore::LoadModelFromDiskAsync(const std::string& model_path) {
  if(model_path[0] == '/'){
    return model_pool_->LoadModelAsync(model_path);
  }

  return model_pool_->LoadModelAsync(storage_path_ + "/" + model_path);
}

std::string ReuseStore::GetPoolHandle(int pool_id) {
  return model_pool_->getPoolHandle(pool_id);
}

int ReuseStore::GetAvailableBlocksonGPU(std::string use_model,
                                        size_t block_size, int pool_id) {
  return model_pool_->GetAvailableBlocks(pool_id, block_size, use_model);
}

std::vector<size_t> ReuseStore::AllocateBlocksonGPU(int device_id, size_t block_size,
                                        std::string model_path,
                                        int block_number) {
  return model_pool_->AllocateBlocks(device_id, block_size, model_path,
                                     block_number);
}