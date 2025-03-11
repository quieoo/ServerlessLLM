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

 private:
  std::string storage_path_;
  std::shared_ptr<ModelPool> model_pool_;
};