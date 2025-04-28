#pragma once

#include <string>
#include <unordered_map>


class VRAMManagerBase {
 public:
  virtual ~VRAMManagerBase() = default;
  virtual int64_t RegisterModel(const std::string& model_path) = 0;
  virtual size_t GetModelSize() const = 0;
  virtual std::string LoadModel(const std::string& model_path) = 0;
  virtual void MemoryUsage() = 0;
};