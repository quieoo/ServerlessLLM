#pragma once
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "gpu_tensor_pool_v4.h"  // 包含 GPUMemoryRegion_V2 和 GPUTensorPool_V4 接口，假设兼容 GPUTensorPool_V2
#include "registered_model.h"

#define RELOADMODEL 1
#define TESTMODELCOUNT 30
#define REGENERATE 1

inline std::vector<char> GeneratePatternData(size_t size, int model_idx,
                                             int tg_idx) {
  std::vector<char> data(size);
  char base_value = static_cast<char>((model_idx * 10 + tg_idx) % 256);
  for (size_t i = 0; i < size; i++) {
    data[i] = static_cast<char>(base_value + (i % 13));  // 使用周期性模式
  }
  return data;
}

inline bool ValidateTensorData(void* gpu_ptr, size_t size, int model_idx,
                               int tg_idx) {
  std::vector<char> host_data(size);
  cudaError_t err =
      cudaMemcpy(host_data.data(), gpu_ptr, size, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    std::cerr << "Failed to copy data from GPU for validation: "
              << cudaGetErrorString(err) << std::endl;
    return false;
  }

  auto expected_data = GeneratePatternData(size, model_idx, tg_idx);
  return std::equal(host_data.begin(), host_data.end(), expected_data.begin());
}

// 辅助函数：创建测试模型目录及文件
inline bool CreateComplexTestModelFiles(const std::string& model_dir,
                                        const std::vector<size_t>& tg_sizes) {
  // 如果文件存在则直接返回
  if (std::filesystem::exists(model_dir)) {
    std::cout << "Model directory already exists: " << model_dir << std::endl;
    return true;
  }
  try {
    std::filesystem::create_directories(model_dir);
    std::ofstream index_file(model_dir + "/tensor_group_index.txt");
    if (!index_file) return false;

    size_t file_offset = 0;
    int model_idx =
        std::stoi(model_dir.substr(model_dir.find_last_of('_') + 1));

    for (size_t i = 0; i < tg_sizes.size(); i++) {
      // 写入索引信息
      index_file << "Group Offset: " << file_offset << "\n";
      index_file << "Group Size: " << tg_sizes[i] << "\n";
      index_file << "Fingerprint: " << model_dir << "-tg-" << i << "\n";
      index_file << "Tensor Name: dummy" << i << "\n";
      index_file << "Offset: " << file_offset << "\n";
      index_file << "Size: " << tg_sizes[i] << "\n";

      file_offset += tg_sizes[i];
    }
    index_file.close();

    // 为每个TG写入独特的数据模式
    std::ofstream data_file(model_dir + "/tensor.data_0", std::ios::binary);
    if (!data_file) return false;

    for (size_t i = 0; i < tg_sizes.size(); i++) {
      auto pattern_data = GeneratePatternData(tg_sizes[i], model_idx, i);
      data_file.write(pattern_data.data(), pattern_data.size());
    }
    data_file.close();
  } catch (std::exception& e) {
    std::cerr << "Exception in CreateComplexTestModelFiles: " << e.what()
              << std::endl;
    return false;
  }
  return true;
}

inline std::vector<std::string> GenerateModelAccessSequence(
    const std::vector<std::string>& model_dirs, int total_requests) {
  std::vector<std::string> sequence;
  std::random_device rd;
  std::mt19937 gen(rd());

  // 使用正态分布生成每个模型的访问次数
  std::normal_distribution<> normal(total_requests / model_dirs.size(),
                                    total_requests / (2 * model_dirs.size()));
  std::vector<int> target_counts(model_dirs.size());

  // 首先生成每个模型的目标访问次数
  int total_count = 0;
  for (size_t i = 0; i < model_dirs.size(); i++) {
    target_counts[i] = std::max(1, static_cast<int>(std::round(normal(gen))));
    total_count += target_counts[i];
  }

  // 按照目标次数生成序列
  for (size_t i = 0; i < model_dirs.size(); i++) {
    for (int j = 0; j < target_counts[i]; j++) {
      sequence.push_back(model_dirs[i]);
    }
  }

  // 打乱访问顺序
  std::shuffle(sequence.begin(), sequence.end(), gen);

  // 如果生成的序列长度与要求不符，进行调整
  if (sequence.size() > total_requests) {
    sequence.resize(total_requests);
  } else
    while (sequence.size() < total_requests) {
      sequence.push_back(model_dirs[gen() % model_dirs.size()]);
    }

  // 统计并输出每个模型的实际访问次数
  std::vector<int> actual_counts(model_dirs.size(), 0);
  for (const auto& model : sequence) {
    for (size_t i = 0; i < model_dirs.size(); i++) {
      if (model == model_dirs[i]) {
        actual_counts[i]++;
        break;
      }
    }
  }

  // 输出访问统计信息
  for (size_t i = 0; i < model_dirs.size(); i++) {
    std::cout << "Model " << model_dirs[i]
              << " target count: " << target_counts[i]
              << " actual count: " << actual_counts[i] << std::endl;
  }

  // 返回生成的访问序列
  // std::cout << "Generated access sequence of size: " << sequence.size() <<
  // std::endl; std::cout << "Access sequence: "; for (const auto& model :
  // sequence) {
  //     std::cout << model << " ";
  // }
  // std::cout << std::endl;

  return sequence;
}

inline std::vector<std::string> GetAccessSequence(
    const std::vector<std::string>& model_dirs, int total_requests) {
  std::vector<std::string> sequence;
  // 检查序列文件是否存在
  auto filename = "sequence";
  if (std::filesystem::exists(filename) && REGENERATE == 0) {
    std::string line;
    std::ifstream file(filename);
    while (std::getline(file, line)) {
      sequence.push_back(line);
    }
    return sequence;
  } else {
    sequence = GenerateModelAccessSequence(model_dirs, total_requests);
    std::ofstream file(filename);
    if (!file) {
      std::cerr << "Failed to open file for writing: " << filename << std::endl;
      return {};
    }
    for (const auto& model : sequence) {
      file << model << "\n";
    }
    file.close();
    std::cout << "Access sequence saved to " << filename << std::endl;
    return sequence;
  }
}

inline int evaluate_complex_gpu_tensor_pool() {
  // 设置测试参数：例如 64MB GPU Pool
  // size_t gpu_pool_size = 64LL * 1024;

  // // 每个模型包含3~5个Tensor Group，大小各异（总和应小于
  // // gpu_pool_size，便于测试替换与合并）
  // std::vector<std::vector<size_t>> model_tg_sizes = {
  //     {1024 * 10, 1024 * 20, 1024 * 15},  // 模型1：10KB, 20KB, 15KB
  //     {1024 * 8, 1024 * 16, 1024 * 12,
  //      1024 * 6},                       // 模型2：8KB, 16KB, 12KB, 6KB
  //     {1024 * 18, 1024 * 10, 1024 * 7}  // 模型3：18KB, 10KB, 7KB
  // };

  int device_id = 4;
  size_t gpu_pool_size = 20LL * 1024 * 1024 * 1024;
  std::vector<std::vector<size_t>> model_tg_sizes;
  std::vector<size_t> llama8b;
  llama8b.push_back(1050673152LL);
  for (int i = 0; i < 32; i++) {
    llama8b.push_back(436224000LL);
  }
  llama8b.push_back(1050673152LL);
  std::vector<size_t> opt6b;
  opt6b.push_back(16793600LL + 412090368LL);
  for (int i = 0; i < 32; i++) {
    opt6b.push_back(402759680LL);
  }
  std::vector<size_t> opt2b;
  opt2b.push_back(10496000LL + 257556480LL);
  for (int i = 0; i < 32; i++) {
    opt2b.push_back(157352960LL);
  }
  std::vector<size_t> opt1b;
  opt1b.push_back(8396800LL + 206045184LL + 8192LL);
  for (int i = 0; i < 23; i++) {
    opt1b.push_back(100716544LL);
  }
  model_tg_sizes.push_back(llama8b);
  model_tg_sizes.push_back(opt6b);
  model_tg_sizes.push_back(opt2b);
  model_tg_sizes.push_back(opt1b);

  // 确保任意一个模型的总大小不会超过 GPU 池大小
  size_t total_size = 0;
  for (const auto& tg_sizes : model_tg_sizes) {
    size_t model_size = 0;
    for (const auto& size : tg_sizes) {
      model_size += size;
    }
    printf("model size: %zu\n", model_size);
    if (model_size > gpu_pool_size) {
      std::cerr << "Model size exceeds GPU pool size." << std::endl;
      return EXIT_FAILURE;
    }
  }

  // 创建多个临时模型目录
  std::vector<std::string> model_dirs;
  for (size_t i = 0; i < model_tg_sizes.size(); i++) {
    std::string dir = "./complex_model_" + std::to_string(i);
    if (!CreateComplexTestModelFiles(dir, model_tg_sizes[i])) {
      std::cerr << "Failed to create complex test model files for " << dir
                << std::endl;
      return EXIT_FAILURE;
    }
    model_dirs.push_back(dir);
  }

  // 构造 GPU Tensor 池（用 GPUTensorPool_V4 模拟 GPUTensorPool_V2 接口）
  GPUTensorPool_V4 gpu_pool(device_id, gpu_pool_size);

  // 对每个模型创建 RegisteredModel 并加载CPU数据
  std::vector<std::shared_ptr<RegisteredModel>> reg_models;
  for (const auto& dir : model_dirs) {
    auto reg_model = std::make_shared<RegisteredModel>(dir);
    int ret = reg_model->LoadModelFromDisk(8);
    if (ret != 0) {
      std::cerr << "LoadModelFromDisk failed for model " << dir
                << ", ret: " << ret << std::endl;
      return EXIT_FAILURE;
    }
    reg_models.push_back(reg_model);
    // 绑定模型到GPU池（装载成本设为1.0）
    gpu_pool.BoundModel(dir, reg_model, 1.0);
  }

  std::chrono::nanoseconds total_latency(0);
  int load_count = 0;
  std::unordered_map<std::string, std::chrono::nanoseconds> latencies_map;
  std::unordered_map<std::string, int> latencies_count;

  auto sequence = GetAccessSequence(model_dirs, TESTMODELCOUNT);
  for (const auto& model : sequence) {
    // for (int i = 0; i < 10; i++) {
    // for (const auto& model : model_dirs) {
    auto start = std::chrono::high_resolution_clock::now();
    int ret = gpu_pool.LoadModel(model);
    // int ret=gpu_pool.LoadModelSync(model);
    auto end = std::chrono::high_resolution_clock::now();
    if (ret != 0) {
      std::cerr << "LoadModel failed for model " << model << ", ret: " << ret
                << std::endl;
      return EXIT_FAILURE;
    }
    // unload model
    if (RELOADMODEL) {
      ret = gpu_pool.UnloadModel(model);
      if (ret != 0) {
        std::cerr << "UnloadModel failed for model " << model << std::endl;
        return EXIT_FAILURE;
      }
    }
    // printf("load model %s, current fragmentation: %f\n", model.c_str(),
    //        gpu_pool.GetFragmentation());

    // // 添加数据验证
    // for (const auto& reg_model : reg_models) {
    //   if (reg_model->model_path() == model) {
    //     auto tg_indexes = reg_model->GetTensorGroupIndexes();
    //     int model_idx = std::stoi(model.substr(model.find_last_of('_') +
    //     1));

    //     for (size_t i = 0; i < tg_indexes.size(); i++) {
    //       const auto& tg_index = tg_indexes[i];
    //       auto tensor = gpu_pool.GetTensor(tg_index.fingerprint);
    //       if (!tensor) {
    //         std::cerr << "Failed to find Tensor " << tg_index.fingerprint
    //                   << " in GPU pool for model " << model << std::endl;
    //         return EXIT_FAILURE;
    //       }

    //       // 验证数据正确性
    //       if (!ValidateTensorData(tensor->addr, tensor->size, model_idx,
    //       i))
    //       {
    //         std::cerr << "Data validation failed for " <<
    //         tg_index.fingerprint
    //                   << std::endl;
    //         return EXIT_FAILURE;
    //       }
    //     }
    //   }
    // }

    auto latency =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    total_latency += latency;
    latencies_map[model] += latency;
    latencies_count[model]++;
    load_count++;
  }
  // }
  if (load_count > 0) {
    double avg_latency_ms = total_latency.count() / 1e6 / load_count;
    std::cout << "Average LoadModel latency: " << avg_latency_ms << " ms over "
              << load_count << " runs." << std::endl;
  }

  for (const auto& pair : latencies_map) {
    const std::string& dir = pair.first;
    auto latency = pair.second;
    int count = latencies_count[dir];
    double avg_latency_ms = latency.count() / 1e6 / count;
    std::cout << "Average LoadModel latency for " << dir << ": "
              << avg_latency_ms << " ms over " << count << " runs."
              << std::endl;
  }

  // 输出GPU池当前内存使用情况和碎片化信息
  // gpu_pool.CollectTimeMetrics();
  // gpu_pool.CollectTimeMetrics2();

  return EXIT_SUCCESS;
}