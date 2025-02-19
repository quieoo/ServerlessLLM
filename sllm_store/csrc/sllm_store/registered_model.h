#pragma once

#include <fcntl.h>
#include <glog/logging.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "concurrent_array.h"

class TensorIndex {
 public:
  TensorIndex() = default;

  TensorIndex(std::string name, size_t offset, size_t size)
      : name(name),
        offset(offset),
        size(size){}

  std::string name;
  size_t offset;
  size_t size;
};

class TensorGroupIndex {
 public:
  TensorGroupIndex() = default;

  TensorGroupIndex(size_t file_offset, size_t size, std::string fingerprint,
                   const std::vector<TensorIndex>& tensor_indexes)
      : file_offset(file_offset),
        size(size),
        fingerprint(fingerprint),
        tensor_indexes(tensor_indexes) {}

  size_t file_offset;
  size_t size;
  std::string fingerprint;
  std::vector<TensorIndex> tensor_indexes;
};

inline void parse_line(std::string line, std::vector<int64_t>& dims) {
  std::regex re("\\d+");
  std::sregex_iterator it(line.begin(), line.end(), re);
  std::sregex_iterator end;

  while (it != end) {
    dims.push_back(std::stoi((*it)[0]));
    ++it;
  }
}

inline void ParseTensorGroupIndex(
    const std::string& file_path,
    std::vector<TensorGroupIndex>& tensor_group_indexes_) {
  std::ifstream fin(file_path);
  if (!fin.is_open()) {
    std::cerr << "Failed to open tensor group index file: " << file_path
              << std::endl;
    return;
  }

  std::string line;
  std::string current_fingerprint;
  TensorGroupIndex current_tensor_group;
  bool reading_tensor_group = false;

  while (std::getline(fin, line)) {
    // Skip empty lines or comments
    if (line.empty() || line.find("#") == 0) continue;

    // Read the tensor group details
    if (line.find("Group Offset:") != std::string::npos) {
      if (reading_tensor_group) {
        // Store the current tensor group once it's finished
        tensor_group_indexes_.push_back(current_tensor_group);
      }

      // Start reading a new tensor group
      reading_tensor_group = true;
      current_tensor_group = TensorGroupIndex();
      std::stringstream ss(line);
      std::string tmp;
      ss >> tmp >> tmp;  // "Group" "Offset:"
      ss >> current_tensor_group.file_offset;
    }

    if (line.find("Group Size:") != std::string::npos) {
      std::stringstream ss(line);
      std::string tmp;
      ss >> tmp >> tmp;  // "Group" "Size:"
      ss >> current_tensor_group.size;
    }

    if (line.find("Fingerprint:") != std::string::npos) {
      std::stringstream ss(line);
      std::string tmp;
      ss >> tmp;  // "Fingerprint:"
      ss >> current_tensor_group.fingerprint;
      current_fingerprint =
          current_tensor_group.fingerprint;  // Update the fingerprint
    }

    if (line.find("Tensor Name:") != std::string::npos) {
      // Read tensor index details
      TensorIndex tensor_index;
      std::stringstream ss(line);
      std::string tensor_name;
      std::string tmp;
      ss >> tmp >> tmp >> tensor_index.name;

      // Read offset
      std::getline(fin, line);
      ss.clear();
      ss.str(line);
      ss >> tmp >> tensor_index.offset;

      // Read size
      std::getline(fin, line);
      ss.clear();
      ss.str(line);
      ss >> tmp >> tensor_index.size;

      // Add tensor index to tensor group
      current_tensor_group.tensor_indexes.push_back(tensor_index);
    }
  }

  // Add the last tensor group to the map
  if (reading_tensor_group) {
    tensor_group_indexes_.push_back(current_tensor_group);
  }

  fin.close();
}
template <typename T>
std::string Join(const std::vector<T>& vec, const std::string& delimiter) {
  std::ostringstream oss;
  for (size_t i = 0; i < vec.size(); ++i) {
    oss << vec[i];
    if (i != vec.size() - 1) {
      oss << delimiter;
    }
  }
  return oss.str();
}
inline void OutputTensorGroupIndex(
    std::vector<TensorGroupIndex>& tensor_group_indexes_) {
  if (tensor_group_indexes_.empty()) {
    LOG(INFO) << "No tensor group indexes to output.";
    return;
  }

  for (const auto& tg : tensor_group_indexes_) {
    LOG(INFO) << "Tensor Group - Fingerprint: " << tg.fingerprint;
    LOG(INFO) << "  File Offset: " << tg.file_offset << ", Size: " << tg.size;
    LOG(INFO) << "  Tensor Group Contains " << tg.tensor_indexes.size()
              << " Tensors:";

    for (const TensorIndex& tensor : tg.tensor_indexes) {
      LOG(INFO) << "    Tensor Name: " << tensor.name;
      LOG(INFO) << "      Offset: " << tensor.offset
                << ", Size: " << tensor.size;
    }
    LOG(INFO) << "";  // Blank line between tensor groups
  }
}

class RegisteredModel {
 public:
  RegisteredModel(const std::string& model_path) : model_path_(model_path) {
    // load tensor group index file
    std::string tensor_group_index_path =
        model_path_ + "/tensor_group_index.txt";
    ParseTensorGroupIndex(tensor_group_index_path, tensor_group_indexes_);
    // OutputTensorGroupIndex(tensor_group_indexes_);
    LOG(INFO) << "get tensor_group_indexes_ size: "
              << tensor_group_indexes_.size();

    model_size_ = 0;
    partition_sizes_.clear();
    partition_paths_.clear();
    for (int partition_id = 0;; ++partition_id) {
      auto tensor_path =
          model_path_ + ("/tensor.data_" + std::to_string(partition_id));
      if (access(tensor_path.c_str(), F_OK) == -1) {
        LOG(INFO) << "Tensor file " << tensor_path << " does not exist";
        break;
      }
      struct stat st;
      if (stat(tensor_path.c_str(), &st) != 0) {
        LOG(ERROR) << "Failed to get file size of " << tensor_path;
        return;
      }
      model_size_ += st.st_size;
      partition_sizes_.push_back(st.st_size);
      partition_paths_.push_back(tensor_path);
    }
    if (model_size_ == 0) {
      LOG(ERROR) << "Model " << model_path_ << " does not exist";
      return;
    }
    for (int i = 0; i < partition_paths_.size(); i++) {
      LOG(INFO) << "partition " << i << ": " << partition_paths_[i]
                << ", size: " << partition_sizes_[i];
    }

    tensor_group_host_ptr = std::make_shared<ConcurrentArray<void*>>(
        tensor_group_indexes_.size(), nullptr);
  }
  ~RegisteredModel() = default;

  std::string model_path() const { return model_path_; }
  size_t model_size() const { return model_size_; }
  const std::vector<size_t>& partition_sizes() const {
    return partition_sizes_;
  }
  const std::vector<std::filesystem::path>& partition_paths() const {
    return partition_paths_;
  }

  int UnloadModel() {
    int free_ptrs = 0;
    for (int i = 0; i < tensor_group_host_ptr->size(); i++) {
      if (tensor_group_host_ptr->get(i) != nullptr) {
        free(tensor_group_host_ptr->get(i));
        tensor_group_host_ptr->set(i, nullptr);
        free_ptrs++;
      }
    }
    if (free_ptrs < tensor_group_indexes_.size()) {
      LOG(WARNING) << "actual free ptrs: " << free_ptrs
                   << " expected: " << tensor_group_indexes_.size();
      return -1;
    }
    return model_size_;
  }

  int LoadModelFromDisk(int num_threads) {
    std::vector<int> file_descriptors;
    // Attempt to read from 0 until the file is not found
    for (int partition_id = 0; partition_id < partition_sizes_.size();
         ++partition_id) {
      auto tensor_path = partition_paths_[partition_id];
      if (access(tensor_path.c_str(), F_OK) == -1) {
        LOG(ERROR) << "File " << tensor_path << " does not exist";
        return -1;
      }

      // Open file
      int fd = open(tensor_path.c_str(), O_DIRECT | O_RDONLY);
      // TODO: align the host buffer to support O_DIRECT
      // int fd = open(tensor_path.c_str(), O_RDONLY);
      // Note: use aligned_alloc sove this problem

      if (fd < 0) {
        std::string err = "open() failed for file: " + tensor_path.string() +
                          ", error: " + strerror(errno);
        LOG(ERROR) << err;
        return -1;
      }

      file_descriptors.push_back(fd);
    }

    if (file_descriptors.size() <= 0 ||
        file_descriptors.size() != partition_sizes_.size()) {
      LOG(ERROR) << "Failed to open file descriptors";
      return -1;
    }

    LOG(INFO) << "Loading model: " << model_path_ << " with " << num_threads
              << " threads";

    std::vector<std::future<int>> futures;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
      futures.emplace_back(std::async(
          std::launch::async,
          [this, &file_descriptors, thread_idx, num_threads]() {
            // each thread loads a tensor group at a time
            // pick tensor group in a cross-threaded manner
            for (size_t tensor_group_idx = thread_idx;
                 tensor_group_idx < tensor_group_indexes_.size();
                 tensor_group_idx += num_threads) {
              auto& tg = tensor_group_indexes_[tensor_group_idx];

              // LOG(INFO) << "Thread: " << thread_idx << " loading tensor
              // group: " << tensor_group_idx  << " with offset: " <<
              // tg.file_offset;

              size_t in_partition_offset = tg.file_offset;
              size_t partition_id = 0;
              while (partition_id < partition_sizes_.size() &&
                     in_partition_offset >= partition_sizes_[partition_id]) {
                in_partition_offset -= partition_sizes_[partition_id];
                partition_id++;
              }

              if (partition_id >= partition_sizes_.size()) {
                LOG(ERROR) << "Failed to find partition for tensor group: "
                           << tensor_group_idx;
                return -1;
              }
              void* host_ptr = tensor_group_host_ptr->get(tensor_group_idx);
              if (host_ptr == nullptr) {
                // host_ptr = malloc(tg.size);
                host_ptr = aligned_alloc(4096, tg.size);
                if (host_ptr == nullptr) {
                  LOG(ERROR) << "Failed to allocate memory for tensor group: "
                             << tensor_group_idx << " with size: " << tg.size;
                  return -1;
                }
              }

              int fd = file_descriptors[partition_id];
              ssize_t ret = pread(fd, host_ptr, tg.size, in_partition_offset);
              if (ret < 0) {
                LOG(ERROR) << "Failed to read tensor group: "
                           << tensor_group_idx
                           << " from file: " << partition_paths_[partition_id];
                // check parameters
                LOG(ERROR) << "tg.size: " << tg.size << " ret: " << ret
                           << " in_partition_offset: " << in_partition_offset
                           << " partition_id: " << partition_id
                           << " partition_sizes_.size(): "
                           << partition_sizes_.size();

                // check host ptr writable
                if (tg.size > 0) {
                  char* test_ptr = (char*)host_ptr;
                  test_ptr[0] = 1;
                }
                LOG(ERROR) << "host_ptr: " << host_ptr << " writable";

                return -1;
              } else if (ret != tg.size) {
                // one case is that the tensor group is splited into multiple
                // partitions
                // TODO: align the tensor group with the partition size
                if (ret < tg.size &&
                    partition_id + 1 < file_descriptors.size()) {
                  // read the next partition
                  partition_id++;
                  in_partition_offset = 0;
                  size_t remaining_size = tg.size - ret;
                  fd = file_descriptors[partition_id];
                  ret = pread(fd, (char*)host_ptr + ret, remaining_size,
                              in_partition_offset);
                  if (ret != remaining_size) {
                    LOG(ERROR)
                        << "Failed to read tensor group: " << tensor_group_idx
                        << " from file: " << partition_paths_[partition_id];
                    return -1;
                  }
                } else {
                  LOG(ERROR)
                      << "Failed to read tensor group: " << tensor_group_idx
                      << " from file: " << partition_paths_[partition_id];
                  return -1;
                }
              }

              tensor_group_host_ptr->set(tensor_group_idx, host_ptr);
              // LOG(INFO) << "Loaded tensor group: " << tensor_group_idx << " from file: " << partition_paths_[partition_id];
            }

            return 0;
          }));
    }

    bool error = false;
    for (auto& future : futures) {
      int ret = future.get();
      if (ret != 0) {
        LOG(ERROR) << "Error reading from disk, ret " << ret;
        error = true;
      }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - start_time)
                        .count();
    LOG(INFO) << "*** RegisteredModel.LocalModelFromDisk takes " << duration
              << " ms";

    // close file
    for (int fd : file_descriptors) {
      close(fd);
    }

    if (error) {
      LOG(ERROR) << "Error reading from disk";

      // recycle tensor_group_host_ptr
      for (int i = 0; i < tensor_group_host_ptr->size(); i++) {
        if (tensor_group_host_ptr->get(i) != nullptr) {
          free(tensor_group_host_ptr->get(i));
          tensor_group_host_ptr->set(i, nullptr);
        }
      }
      return -1;
    }

    return 0;
  }

  int LodaModelFromMem(std::vector<char*>& allocated_regions, int device_id) {
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < tensor_group_indexes_.size(); i++) {
      // wait from disk loading finish
      size_t pooling = 0;
      while (tensor_group_host_ptr->get(i) == nullptr) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        pooling++;
        if (pooling > 1000) {
          LOG(ERROR) << "Loading from disk timeout";
          return -1;
        }
      }
      cudaError_t err = cudaSetDevice(device_id);
      if (err != cudaSuccess) {
        LOG(ERROR) << "Error setting device " << cudaGetErrorString(err);
        return 1;
      }
      void* gpu_ptr = static_cast<void*>(allocated_regions[i]);
      err = cudaMemcpy(gpu_ptr, tensor_group_host_ptr->get(i),
                       tensor_group_indexes_[i].size, cudaMemcpyHostToDevice);
      // LOG(INFO) << "Copying tensor group " << i << " to GPU " << device_id << " from host";
      if (err != cudaSuccess) {
        LOG(ERROR) << "Error copying to GPU " << cudaGetErrorString(err);
        return -1;
      }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - start_time)
                        .count();
    LOG(INFO) << "*** RegisteredModel.LodaModelFromMem takes " << duration
              << " ms";
    return 0;
  }

  const std::vector<TensorGroupIndex>& GetTensorGroupIndexes() const {
    return tensor_group_indexes_;
  }

  std::shared_ptr<ConcurrentArray<void*>> GetTensorGroupHostPtr() {
    return tensor_group_host_ptr;
  }

 private:
  std::string model_path_;
  size_t model_size_;
  std::vector<size_t> partition_sizes_;
  std::vector<std::filesystem::path> partition_paths_;
  // std::unordered_map<std::string, TensorGroupIndex> tensor_group_indexes_;
  std::vector<TensorGroupIndex> tensor_group_indexes_;

  std::shared_ptr<ConcurrentArray<void*>> tensor_group_host_ptr;
};

class DispatchedModel {
 public:
  DispatchedModel(const std::string& uuid, int device_id,
                  std::shared_ptr<RegisteredModel> model_source)
      : uuid(uuid),
        device_id(device_id),
        model_source(std::move(model_source)) {}

 private:
  std::string uuid;
  int device_id;
  std::shared_ptr<RegisteredModel> model_source;
  ConcurrentArray<void*> tensor_group_gpu_ptr;
};