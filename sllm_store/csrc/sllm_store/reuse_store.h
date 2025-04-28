#pragma once

#include <string>

#include "model_pool.h"

class ReuseStore {
 public:
  ReuseStore(const std::string& storage_path, size_t memory_pool_size,
             int num_thread);
  ~ReuseStore();

  int64_t RegisterModelInfo(const std::string& model_path);
  std::string LoadModelFromDiskAsync(const std::string& model_path);
  size_t GetMemPoolSize() const { return model_pool_->GetModelSize(); }
  size_t GetChunkSize() const { return 0; }
  std::string GetPoolHandle(int pool_id);
  int GetAvailableBlocksonGPU(std::string use_model, size_t block_size,
                              int pool_id);
  std::vector<size_t> AllocateBlocksonGPU(int device_id, size_t block_size,
                                     std::string model_path, int block_number);

 private:
  std::string storage_path_;
  std::shared_ptr<ModelPool> model_pool_;
};