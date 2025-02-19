#include "reuse_store.h"

ReuseStore::ReuseStore(const std::string& storage_path, size_t memory_pool_size,
                       int num_thread)
    : storage_path_(storage_path) {
  model_pool_ = std::make_shared<ModelPool>(memory_pool_size, num_thread);
}

ReuseStore::~ReuseStore() { LOG(INFO) << "Destroy ReuseStore"; }

int64_t ReuseStore::RegisterModelInfo(const std::string& model_path) {
  return model_pool_->RegisterModel(storage_path_ + "/" + model_path);
}


std::string ReuseStore::LoadModelFromDiskAsync(const std::string& model_path) {
  return model_pool_->LoadModelFromDiskAsync(storage_path_ + "/" + model_path);
}