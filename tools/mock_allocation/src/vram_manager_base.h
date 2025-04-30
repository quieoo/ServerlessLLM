#pragma once

#include <string>
#include <unordered_map>


class VRAMManagerBase {
 public:
  virtual ~VRAMManagerBase() = default;
  virtual int64_t RegisterModel(const std::string& model_path, int sensitivity) = 0;
  virtual std::string LoadModel(const std::string& model_path, int device_id) = 0;
  virtual void MemoryUsage() = 0;
};