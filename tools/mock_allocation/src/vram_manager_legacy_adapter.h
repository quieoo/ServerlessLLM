#pragma once

#include "vram_manager.h"
#include "vram_manager_interface.h"

class LegacyVRAMManagerAdapter final : public IVRAMManager {
 public:
  LegacyVRAMManagerAdapter(size_t pool_size, const std::vector<int>& gpu_ids,
                           double gpu_bw, double cpu_bw, bool mock_copy)
      : manager_(pool_size, gpu_ids, gpu_bw, cpu_bw, mock_copy) {}
  int64_t RegisterModel(const std::string& p, double s, bool m, int r) override {
    return manager_.RegisterModel(p, s, m, r);
  }
  void WarmupModelAccess(const std::unordered_map<std::string, size_t>& c) override {
    manager_.WarmupModelAccess(c);
  }
  std::string LoadModel(const std::string& p, int d, int f, int a, bool v,
                        bool no_reuse) override {
    return manager_.LoadModel(p, d, f, a, v, no_reuse);
  }
  bool EstimateModelLoad(const std::string& p, int d, size_t& cached,
                         size_t& to_load, size_t& total, bool no_reuse) override {
    return manager_.EstimateModelLoad(p, d, cached, to_load, total, no_reuse);
  }
  int64_t GetGPUToLoad(const std::string& p, int policy) override {
    return manager_.GetGPUToLoad(p, policy);
  }
  int GetAvailableBlocks(const std::string& m, size_t b, int d) override {
    return manager_.GetAvailableBlocks(m, b, d);
  }
  std::vector<size_t> AllocateBlocks(int d, size_t b, const std::string& m,
                                     int n) override {
    return manager_.AllocateBlocks(d, b, m, n);
  }
  void MemoryUsage() override { manager_.MemoryUsage(); }
  double get_memory_utilization() override { return manager_.get_memory_utilization(); }
 private:
  VRAMManager manager_;
};
