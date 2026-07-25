#pragma once

#include <cuda_runtime.h>
#include <fcntl.h>
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

#include "binary_utils.h"
#include "concurrent_array.h"
#include "logger.h"

// #define CUDAMALLOCHOST
#define ALLCACHE

// inline void* allocateAlignedPinnedMemory(size_t size, size_t alignment) {
//   // 1. posix_memalign allocates aligned memory
//   void* aligned_mem = NULL;
//   int ret = posix_memalign(&aligned_mem, alignment, size);
//   if (ret != 0 || aligned_mem == NULL) {
//     perror("posix_memalign");
//     return nullptr;
//   }

//   // 2. register the allocated memory to the CUDA device
//   cudaError_t cuda_status =
//       cudaHostRegister(aligned_mem, size, cudaHostRegisterDefault);
//   if (cuda_status != cudaSuccess) {
//     fprintf(stderr, "cudaHostRegister failed: %s\n",
//             cudaGetErrorString(cuda_status));
//     free(aligned_mem);
//     return nullptr;
//   }

//   return aligned_mem;
// }

// inline void freeAlignedPinnedMemory(void* ptr) {
//   // 1. unregister the memory from the CUDA device
//   cudaError_t cuda_status = cudaHostUnregister(ptr);
//   if (cuda_status != cudaSuccess) {
//     fprintf(stderr, "cudaHostUnregister failed: %s\n",
//             cudaGetErrorString(cuda_status));
//   }
//   // 2. free the memory
//   free(ptr);
// }

class TensorIndex {
 public:
  TensorIndex() = default;

  TensorIndex(std::string name, size_t offset, size_t size)
      : name(name), offset(offset), size(size) {}

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

  std::string toString(){
    std::stringstream ss;
    ss<<"file_offset: "<<file_offset<<", size: "<<size<<", fingerprint: "<<fingerprint<<", tensor_indexes: ";
    return ss.str();
  }
  
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
// template <typename T>
// std::string Join(const std::vector<T>& vec, const std::string& delimiter) {
//   std::ostringstream oss;
//   for (size_t i = 0; i < vec.size(); ++i) {
//     oss << vec[i];
//     if (i != vec.size() - 1) {
//       oss << delimiter;
//     }
//   }
//   return oss.str();
// }

class RegisteredModel {
 private:
  std::string model_path_;
  size_t model_size_;
  std::vector<size_t> partition_sizes_;
  std::vector<std::filesystem::path> partition_paths_;
  // std::unordered_map<std::string, TensorGroupIndex> tensor_group_indexes_;
  std::vector<TensorGroupIndex> tensor_group_indexes_;

  std::shared_ptr<ConcurrentArray<void*>> tensor_group_host_ptr;
  double load_sensitive_ = 1;
  double load_penalty_ = 1;

 public:
  RegisteredModel() {}
  void SetMeta(const std::string& model_path, double load_sensitive,
               double load_penalty) {
    model_path_ = model_path;
    load_sensitive_ = load_sensitive;
    load_penalty_ = load_penalty;
  }
  RegisteredModel(const std::string& model_path, double load_sensitive = 1.0, int reuse_granularity = 1)
      : model_path_(model_path), load_sensitive_(load_sensitive) {
    // load tensor group index file
    std::string tensor_group_index_path =
        model_path_ + "/tensor_group_index.txt";
    ParseTensorGroupIndex(tensor_group_index_path, tensor_group_indexes_);

    // 如果重用粒度为0，则将所有TG合并成一个
    if(reuse_granularity == 0){
      TensorGroupIndex merged_tg;
      merged_tg.file_offset = tensor_group_indexes_[0].file_offset;
      merged_tg.size = 0;
      merged_tg.fingerprint = "merged-"+model_path_;
      for(auto& tg: tensor_group_indexes_){
        merged_tg.size += tg.size;
        merged_tg.fingerprint += "-"+tg.fingerprint;
        merged_tg.tensor_indexes.insert(merged_tg.tensor_indexes.end(), tg.tensor_indexes.begin(), tg.tensor_indexes.end());
      }
      tensor_group_indexes_.clear();
      tensor_group_indexes_.push_back(merged_tg);
    }

    // OutputTensorGroupIndex(tensor_group_indexes_);
    // LOG(INFO) << "get tensor_group_indexes_ size: "
    //           << tensor_group_indexes_.size();

    // for(int i=0;i<tensor_group_indexes_.size();i++){
    //   while(tensor_group_indexes_[i].size >= 525348864LL){
    //     tensor_group_indexes_[i].size=tensor_group_indexes_[i].size/2;
    //     std::cout<<"reduce TG "<<tensor_group_indexes_[i].fingerprint<<" to
    //     "<<tensor_group_indexes_[i].size<<std::endl;
    //   }
    // }

    model_size_ = 0;
    partition_sizes_.clear();
    partition_paths_.clear();
    for (int partition_id = 0;; ++partition_id) {
      auto tensor_path =
          model_path_ + ("/tensor.data_" + std::to_string(partition_id));
      if (access(tensor_path.c_str(), F_OK) == -1) {
        break;
      }
      struct stat st;
      if (stat(tensor_path.c_str(), &st) != 0) {
        std::cout << "Failed to get file size of " << tensor_path << std::endl;
        return;
      }
      model_size_ += st.st_size;
      partition_sizes_.push_back(st.st_size);
      partition_paths_.push_back(tensor_path);
    }
    if (model_size_ == 0) {
      std::cout << "Model " << model_path_ << " does not exist" << std::endl;
      return;
    }

    tensor_group_host_ptr = std::make_shared<ConcurrentArray<void*>>(
        tensor_group_indexes_.size(), nullptr);
  }

  // 根据模型存储位置确定装载惩罚系数，如果所有TG都在SSD上则惩罚系数为4，如果所有TG都在内存中则惩罚系数为1
  // 通过检查tensor_group_host_ptr的值来判断TG是否已经加载到内存中
  // int GetLoadPenalty() {
  //   bool all_in_memory = true;
  //   for (int i = 0; i < tensor_group_indexes_.size(); i++) {
  //     auto& tg = tensor_group_indexes_[i];
  //     if (tensor_group_host_ptr->get(i) == nullptr) {
  //       all_in_memory = false;
  //       break;
  //     }
  //   }
  //   if (all_in_memory) {
  //     return 1;
  //   } else {
  //     return 4;
  //   }
  // }

  void SetLoadPenalty(double penalty) { load_sensitive_ = penalty; }
  void SetLoadSensitive(double sensitive) { load_sensitive_ = sensitive; }

  double GetLoadSensitive() { return load_sensitive_; }
  double GetLoadPenalty() { return load_penalty_; }

  ~RegisteredModel() {
    LOG(INFO) << "Clean Registered Model:" << model_path_;
    UnloadModel();
  }

  std::string model_path() const { return model_path_; }
  size_t model_size() const { return model_size_; }
  const std::vector<size_t>& partition_sizes() const {
    return partition_sizes_;
  }
  const std::vector<std::filesystem::path>& partition_paths() const {
    return partition_paths_;
  }

  int UnloadModel() {
    if (!tensor_group_host_ptr) {
      return 0;
    }
    int free_ptrs = 0;
    for (int i = 0; i < tensor_group_host_ptr->size(); i++) {
      if (tensor_group_host_ptr->get(i) != nullptr) {
        // free(tensor_group_host_ptr->get(i));
        // cudaFreeHost(tensor_group_host_ptr->get(i));

        freeAlignedPinnedMemory(tensor_group_host_ptr->get(i));
        tensor_group_host_ptr->set(i, nullptr);
        free_ptrs++;
      }
    }
    if (free_ptrs < tensor_group_indexes_.size()) {
      std::cout << "actual free ptrs: " << free_ptrs
                << " expected: " << tensor_group_indexes_.size() << std::endl;
      return -1;
    }
    return model_size_;
  }

  int LoadModelFromDisk(int num_threads) {
    std::vector<int> file_descriptors;
    const bool safetensors_adapter =
        access((model_path_ + "/safetensors_adapter.json").c_str(), F_OK) == 0;
    // Attempt to read from 0 until the file is not found
    for (int partition_id = 0; partition_id < partition_sizes_.size();
         ++partition_id) {
      auto tensor_path = partition_paths_[partition_id];
      if (access(tensor_path.c_str(), F_OK) == -1) {
        LOG(INFO) << "Tensor file " << tensor_path
                  << " does not exist, stop searching";
        // std::cout << "File " << tensor_path << " does not exist" <<
        // std::endl;
        return -1;
      }

      // Open file
      // Safetensors payloads start immediately after a variable-sized JSON
      // header, so their tensor offsets are not guaranteed to satisfy
      // O_DIRECT alignment. Packed tensor.data_* checkpoints retain the
      // original direct-I/O path.
      int fd = open(tensor_path.c_str(),
                    (safetensors_adapter ? O_RDONLY
                                         : (O_DIRECT | O_RDONLY)));
      // TODO: align the host buffer to support O_DIRECT
      // int fd = open(tensor_path.c_str(), O_RDONLY);
      // Note: use aligned_alloc sove this problem

      if (fd < 0) {
        std::string err = "open() failed for file: " + tensor_path.string() +
                          ", error: " + strerror(errno);
        std::cout << err << std::endl;
        return -1;
      }

      file_descriptors.push_back(fd);
    }

    if (file_descriptors.size() <= 0 ||
        file_descriptors.size() != partition_sizes_.size()) {
      std::cout << "Failed to open file descriptors" << std::endl;
      return -1;
    }

    // std::cout << "Loading model: " << model_path_ << " with " << num_threads
    //           << " threads" << std::endl;

    std::vector<std::future<int>> futures;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
      futures.emplace_back(std::async(std::launch::async, [this,
                                                           &file_descriptors,
                                                           thread_idx,
                                                           num_threads]() {
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
            std::cout << "Failed to find partition for tensor group: "
                      << tensor_group_idx << std::endl;
            return -1;
          }
#ifdef CUDAMALLOCHOST
          void* pinned_host_ptr;
#endif
          void* host_ptr = tensor_group_host_ptr->get(tensor_group_idx);

          if (host_ptr == nullptr) {
            // Way-1: Malloc
            // host_ptr = malloc(tg.size);
            // Way-2: Aligned Malloc
            // host_ptr = aligned_alloc(4096, tg.size);
            // if (host_ptr == nullptr) {
            //   std::cout << "Failed to allocate memory for tensor group: "
            //             << tensor_group_idx << " with size: " << tg.size
            //             << std::endl;
            //   return -1;
            // }
            // Aligned Malloc provides a better performance when loading data

            // from disk Way-3: Pinned Malloc
            // host_ptr = aligned_alloc(4096, tg.size);
            // if (host_ptr == nullptr) {
            //   std::cout << "Failed to allocate memory for tensor group: "
            //             << tensor_group_idx << " with size: " << tg.size
            //             << std::endl;
            //   return -1;
            // }

            // Way-4
            host_ptr = allocateAlignedPinnedMemory(tg.size, 4096);
            if (host_ptr == nullptr) {
              std::cout << "Failed to allocate memory for tensor group: "
                        << tensor_group_idx << " with size: " << tg.size
                        << std::endl;
              return -1;
            }

            // cudaMallocHost, however, providers a better performance when
            // loading data from memory to GPU
#ifdef CUDAMALLOCHOST
            cudaError_t err = cudaMallocHost(&pinned_host_ptr, tg.size);
            if (err != cudaSuccess) {
              std::cerr << "CUDA error in cudaMallocHost: "
                        << cudaGetErrorString(err) << std::endl;
            }
#endif
          }

          int fd = file_descriptors[partition_id];
          ssize_t ret = pread(fd, host_ptr, tg.size, in_partition_offset);
          if (ret < 0) {
            std::cout << "Failed to read tensor group: " << tensor_group_idx
                      << " from file: " << partition_paths_[partition_id]
                      << std::endl;
            // check parameters
            std::cout << "tg.size: " << tg.size << " ret: " << ret
                      << " in_partition_offset: " << in_partition_offset
                      << " partition_id: " << partition_id
                      << " partition_sizes_.size(): " << partition_sizes_.size()
                      << std::endl;

            // check host ptr writable
            // if (tg.size > 0) {
            //   char* test_ptr = (char*)host_ptr;
            //   test_ptr[0] = 1;
            // }
            // std::cout << "host_ptr: " << host_ptr << " writable" <<
            // std::endl;

            return -1;
          } else if (ret != tg.size) {
            // one case is that the tensor group is splited into multiple
            // partitions
            // TODO: align the tensor group with the partition size
            if (ret < tg.size && partition_id + 1 < file_descriptors.size()) {
              // read the next partition
              partition_id++;
              in_partition_offset = 0;
              size_t remaining_size = tg.size - ret;
              fd = file_descriptors[partition_id];
              ret = pread(fd, (char*)host_ptr + ret, remaining_size,
                          in_partition_offset);
              if (ret != remaining_size) {
                std::cout << "Failed to read tensor group: " << tensor_group_idx
                          << " from file: " << partition_paths_[partition_id]
                          << std::endl;
                return -1;
              }
            } else {
              std::cout << "Failed to read tensor group: " << tensor_group_idx
                        << " from file: " << partition_paths_[partition_id]
                        << std::endl;
              return -1;
            }
          }
#ifdef CUDAMALLOCHOST
          memcpy(pinned_host_ptr, host_ptr, tg.size);
          free(host_ptr);
          host_ptr = pinned_host_ptr;
#endif
          tensor_group_host_ptr->set(tensor_group_idx, host_ptr);
          // LOG(INFO) << "Loaded tensor group: " << tensor_group_idx << " from
          // file: " << partition_paths_[partition_id];
        }

        return 0;
      }));
    }

    bool error = false;
    for (auto& future : futures) {
      int ret = future.get();
      if (ret != 0) {
        std::cout << "Error reading from disk, ret " << ret << std::endl;
        error = true;
      }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - start_time)
                        .count();
    // std::cout << "*** RegisteredModel.LocalModelFromDisk takes " << duration
    //           << " ms" << std::endl;

    // close file
    for (int fd : file_descriptors) {
      close(fd);
    }

    if (error) {
      std::cout << "Error reading from disk" << std::endl;

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

  int LoadModelFromMem(const std::vector<char*>& allocated_regions,
                       const std::vector<int>& tg_to_load, int device_id) {
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t copy_size=0;
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
      LOG(ERROR) << "Error setting device " << cudaGetErrorString(err);
      return 1;
    }
    for (auto tg_id : tg_to_load) {
      size_t pooling = 0;
      while (tensor_group_host_ptr->get(tg_id) == nullptr) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        pooling++;
        if (pooling > 10000) {
          std::cout << "Loading from disk timeout" << std::endl;
          return -1;
        }
      }
      // LOG(INFO)<<"Polled: "<<pooling<<" times";
      void* gpu_ptr = static_cast<void*>(allocated_regions[tg_id]);
      err =
          cudaMemcpy(gpu_ptr, tensor_group_host_ptr->get(tg_id),
                     tensor_group_indexes_[tg_id].size, cudaMemcpyHostToDevice);
      if (err != cudaSuccess) {
        LOG(ERROR) << "Error copying to device " << cudaGetErrorString(err);
        return 1;
      }
      copy_size+=tensor_group_indexes_[tg_id].size;
      // memcpy(gpu_ptr, tensor_group_host_ptr->get(tg_id),
      //                  tensor_group_indexes_[tg_id].size);
    }

    LOG(DetailMetrics)<<"CopySize: "<<copy_size;
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - start_time)
                        .count();

                        // std::cout << "*** RegisteredModel.LodaModelFromMem takes " << duration
    //           << " ms" << std::endl;
    return 0;
  }

  const std::vector<TensorGroupIndex>& GetTensorGroupIndexes() const {
    return tensor_group_indexes_;
  }

  std::shared_ptr<ConcurrentArray<void*>> GetTensorGroupHostPtr() {
    return tensor_group_host_ptr;
  }

  // Expand checkpoint tensor groups into one file/copy unit per tensor.
  // The VMM backend may still pack these units into a shared stable model VA
  // arena; this split controls metadata and H2D copy boundaries, not page
  // ownership.
  void SplitTensorGroupsToTensors() {
    std::vector<TensorGroupIndex> tensor_units;
    for (const auto& group : tensor_group_indexes_) {
      for (const auto& tensor : group.tensor_indexes) {
        TensorIndex standalone = tensor;
        standalone.offset = 0;
        tensor_units.emplace_back(
            group.file_offset + tensor.offset, tensor.size,
            group.fingerprint + ":tensor:" + tensor.name + ":" +
                std::to_string(tensor.offset) + ":" +
                std::to_string(tensor.size),
            std::vector<TensorIndex>{standalone});
      }
    }
    tensor_group_indexes_ = std::move(tensor_units);
    tensor_group_host_ptr = std::make_shared<ConcurrentArray<void*>>(
        tensor_group_indexes_.size(), nullptr);
  }

  // ... existing code ...
  // ... existing code ...
  // ... existing code ...
  void MergeTGs(size_t min_size) {
    bool merged;
    do {
      merged = false;
      size_t i = 0;
      while (i < tensor_group_indexes_.size()) {
        TensorGroupIndex& current = tensor_group_indexes_[i];
        if (current.size >= min_size) {
          i++;
          continue;
        }

        size_t merge_end = i + 1;
        if (merge_end >= tensor_group_indexes_.size()) {
          // 如果没有下一个 TensorGroup，跳出循环
          break;
        }

        size_t total_size = current.size;
        std::string combined_fingerprint = current.fingerprint;
        std::vector<TensorIndex> combined_tensor_indexes =
            current.tensor_indexes;

        // 查找可以合并的所有 TensorGroup
        while (merge_end < tensor_group_indexes_.size() &&
               total_size < min_size) {
          TensorGroupIndex& next = tensor_group_indexes_[merge_end];
          combined_fingerprint += "+" + next.fingerprint;

          // 调整下一个 TensorGroup 中所有 tensor 的偏移量
          for (auto& tensor : next.tensor_indexes) {
            TensorIndex adjusted_tensor = tensor;
            adjusted_tensor.offset += total_size;
            combined_tensor_indexes.push_back(adjusted_tensor);
          }

          total_size += next.size;
          merge_end++;
        }

        // 更新当前 TensorGroup 的信息
        current.size = total_size;
        current.fingerprint = combined_fingerprint;
        current.tensor_indexes = combined_tensor_indexes;

        // 移除已合并的 TensorGroup
        tensor_group_indexes_.erase(tensor_group_indexes_.begin() + i + 1,
                                    tensor_group_indexes_.begin() + merge_end);

        merged = true;
      }
    } while (merged);

    // 检查最后一个 tensor_group 的大小，如果小于 min_size，则尝试合并到前一个
    if (tensor_group_indexes_.size() > 1) {
      TensorGroupIndex& last = tensor_group_indexes_.back();
      if (last.size < min_size) {
        TensorGroupIndex& second_last =
            tensor_group_indexes_[tensor_group_indexes_.size() - 2];
        const size_t last_base_offset = second_last.size;
        second_last.size += last.size;
        second_last.fingerprint += "+" + last.fingerprint;
        for (auto& tensor : last.tensor_indexes) {
          TensorIndex adjusted_tensor = tensor;
          adjusted_tensor.offset += last_base_offset;
          second_last.tensor_indexes.push_back(adjusted_tensor);
        }
        tensor_group_indexes_.pop_back();
      }
    }

    // 重建 host 指针数组
    tensor_group_host_ptr = std::make_shared<ConcurrentArray<void*>>(
        tensor_group_indexes_.size(), nullptr);
  }

  void MergeTGsRatio(int final_cnt){
    size_t average_tg_size=model_size_/final_cnt;
    MergeTGs(average_tg_size);
  }
};
