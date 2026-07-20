#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// The simulator uses this narrow interface so that the legacy contiguous pool
// and the CUDA VMM pool can be selected without changing request replay code.
class IVRAMManager {
 public:
  virtual ~IVRAMManager() = default;
  virtual int64_t RegisterModel(const std::string& path, double sensitivity,
                                bool mock_copy, int reuse_granularity) = 0;
  virtual void WarmupModelAccess(
      const std::unordered_map<std::string, size_t>& counts) = 0;
  virtual std::string LoadModel(const std::string& path, int device_id,
                                int free_strategy, int allocate_strategy,
                                bool verbose,
                                bool disable_parameter_reuse) = 0;
  virtual bool EstimateModelLoad(const std::string& path, int device_id,
                                 size_t& cached_bytes, size_t& to_load_bytes,
                                 size_t& total_bytes,
                                 bool disable_parameter_reuse) = 0;
  virtual int64_t GetGPUToLoad(const std::string& path, int policy) = 0;
  virtual int GetAvailableBlocks(const std::string& model, size_t block_size,
                                 int device_id) = 0;
  virtual std::vector<size_t> AllocateBlocks(int device_id, size_t block_size,
                                              const std::string& model,
                                              int block_number) = 0;
  virtual void MemoryUsage() = 0;
  virtual double get_memory_utilization() = 0;
};
