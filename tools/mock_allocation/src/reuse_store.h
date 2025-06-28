#pragma once

#include <memory>
#include <string>
#include <vector>
#include "vram_manager.h"

class ReuseStore {
 public:
  ReuseStore(const std::string& storage_path, size_t memory_pool_size,
             int num_thread);
  ~ReuseStore();

  int64_t RegisterModelInfo(const std::string& model_path);
  std::string LoadModelFromDiskAsync(const std::string& model_path, int device_id=0);
  size_t GetMemPoolSize();
  size_t GetChunkSize() const { return 0; }
  std::string GetPoolHandle(int pool_id);
  int GetAvailableBlocksonGPU(std::string use_model, size_t block_size,
                              int pool_id);
  std::vector<size_t> AllocateBlocksonGPU(int device_id, size_t block_size,
                                     std::string model_path, int block_number);

 private:
  std::string storage_path_;
  std::shared_ptr<VRAMManager> vram_manager_;
};