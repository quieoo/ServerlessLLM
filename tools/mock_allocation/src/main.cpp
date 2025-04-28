#include <chrono>
#include <random>

// #include "model_pool.h"



// #include "test_AllocateASAP.h"
// #include "test_gpu_tensor_pool_v2_1.h"
#include "vram_manager_v1.h"

#define MODELREGENERATE 1
// 添加新的辅助函数用于生成和保存请求序列
inline std::vector<int> GenerateModelRequests(
    const std::vector<std::string>& model_dirs_list, bool random,
    int distribution, size_t scale) {
  std::vector<int> requests;
  if (random) {
    std::random_device rd;
    std::mt19937 gen(rd());

    size_t total_requests = model_dirs_list.size() * scale;
    if (distribution == 0) {  // 均匀分布
      std::uniform_int_distribution<> dis(0, model_dirs_list.size() - 1);
      for (size_t i = 0; i < total_requests; i++) {
        requests.push_back(dis(gen));
      }
    } else if (distribution == 1) {  // 正态分布
      double mean = model_dirs_list.size() / 2;
      double stddev = model_dirs_list.size() / 4;
      std::normal_distribution<> dis(mean, stddev);
      for (size_t i = 0; i < total_requests; i++) {
        int j = std::clamp(static_cast<int>(dis(gen)), 0,
                           static_cast<int>(model_dirs_list.size()) - 1);
        requests.push_back(j);
      }
    }
  } else {  // 顺序访问
    for (size_t i = 0; i < scale; i++) {
      for (size_t j = 0; j < model_dirs_list.size(); j++) {
        requests.push_back(j);
      }
    }
  }

  return requests;
}

int evaluate_load_latency(int argc, char* argv[]) {
  size_t memory_pool_size = 50LL * 1024 * 1024 * 1024;  // 12GB
  int num_thread = 4;
  std::string model_dirs;
  std::vector<std::string> model_dirs_list = {
      "/mnt/n0/models/vllm/opt6.7b_tmp/rank_0",
      "/mnt/n0/models/vllm/opt2.7_tmp/rank_0",
      // "/mnt/n0/models/vllm/falcon_7b_tmp/rank_0",
      "/mnt/n0/models/vllm/llama3_chinese_tmp/rank_0",
      "/mnt/n0/models/vllm/opt1.3b_tmp/rank_0",
  };
  int model_pool_type = 2;
  size_t scale = 10;
  size_t gpu_pool_size = 0;
  bool random = false;
  int distribution = 0;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "-m" || arg == "--memory-size") {
      memory_pool_size = std::stoull(argv[i + 1]);
      i++;
    } else if (arg == "-t" || arg == "--thread-num") {
      num_thread = std::stoi(argv[i + 1]);
      i++;
    } else if (arg == "-d" || arg == "--model-dirs") {
      model_dirs = argv[i + 1];
      // assume the model dir is a comma separated list
      std::stringstream ss(model_dirs);
      std::string item;
      while (std::getline(ss, item, ',')) {
        model_dirs_list.push_back(item);
      }
      i++;
    } else if (arg == "-p" || arg == "--model-pool") {
      if (strcmp(argv[i + 1], "native") == 0) {
        std::cout << "Using NativeModelPool" << std::endl;
        model_pool_type = 1;
      } else if (strcmp(argv[i + 1], "reuse") == 0) {
        std::cout << "Using ReuseModelPool" << std::endl;
        model_pool_type = 2;
      }
      i++;
    } else if (arg == "-s" || arg == "--scale") {
      scale = std::stoi(argv[i + 1]);
      i++;
    } else if (arg == "-g" || arg == "--gpu-pool") {
      if (model_pool_type != 2) {
        std::cout << "GPU pool only support reuse model pool" << std::endl;
        return 0;
      }
      float size_gbs = std::stof(argv[i + 1]);
      gpu_pool_size = static_cast<size_t>(size_gbs * 1024 * 1024 * 1024);
      // gpu_pool_size = std::stoull(argv[i + 1]);
      i++;
    } else if (arg == "-r" || arg == "--random") {
      std::cout << "Using random model allocation" << std::endl;
      random = true;
      if (strcmp(argv[i + 1], "even") == 0) {
        std::cout << "Using even distribution" << std::endl;
        distribution = 0;
      } else if (strcmp(argv[i + 1], "guas") == 0) {
        std::cout << "Using guas distribution" << std::endl;
        distribution = 1;
      } else {
        std::cout << "Unknown distribution: " << argv[i + 1] << "\n";
        return 1;
      }
      i++;
    } else {
      std::cout << "Unknown argument: " << arg << "\n";
      return 1;
    }
  }

  if (model_dirs_list.empty()) {
    std::cout << "No model dirs specified\n";
  }

  // 检查或生成请求序列
  std::string sequence_file = "model_requests.seq";
  std::vector<int> model_requests;

  if (std::filesystem::exists(sequence_file)) {
    // 从文件加载现有序列
    std::ifstream file(sequence_file, std::ios::binary);
    if (file) {
      int request;
      while (file.read(reinterpret_cast<char*>(&request), sizeof(int))) {
        model_requests.push_back(request);
      }
      std::cout << "Loaded " << model_requests.size()
                << " requests from existing file" << std::endl;
    }
  }

  // 如果文件不存在或加载失败，重新生成序列
  if (model_requests.empty() || MODELREGENERATE) {
    model_requests =
        GenerateModelRequests(model_dirs_list, random, distribution, scale);
    // 保存到文件
    std::ofstream file(sequence_file, std::ios::binary);
    if (file) {
      for (int request : model_requests) {
        file.write(reinterpret_cast<const char*>(&request), sizeof(int));
      }
      std::cout << "Generated and saved " << model_requests.size()
                << " requests to file" << std::endl;
    }
  }

  // 输出请求序列
  std::cout << "Generated request sequence: ";
  for (const auto& request : model_requests) {
    std::cout << request << " ";
  }
  std::cout << std::endl;

  std::shared_ptr<VRAMManagerBase> model_pool_;
  if (model_pool_type == 1) {
    model_pool_ =
        std::make_shared<VRAMManager_V0>(memory_pool_size, num_thread);
  } else if (model_pool_type == 2) {
    model_pool_ = std::make_shared<VRAMManager_V1>(memory_pool_size, num_thread,
                                              gpu_pool_size);
    
  } else {
    std::cout << "Invalid model pool type" << std::endl;
    return 0;
  }

  for (auto& model_dir : model_dirs_list) {
    auto size = model_pool_->RegisterModel(model_dir);
    // model_sizes.push_back(size);
  }

  // warm the cpu model cache
  for (auto& model_path : model_dirs_list) {
    model_pool_->LoadModel(model_path);
  }

  std::vector<std::chrono::nanoseconds> latencies(model_dirs_list.size());
  std::vector<int> model_indices(model_dirs_list.size(), 0);

  for (int request_idx : model_requests) {
    auto start = std::chrono::high_resolution_clock::now();
    auto ret = model_pool_->LoadModel(model_dirs_list[request_idx]);
    if (model_pool_type == 2) {
      model_pool_->MemoryUsage();
    }
    if (ret == "ERROR") {
      std::cout << "Load model failed: " << model_dirs_list[request_idx]
                << std::endl;
      break;
    }
    auto end = std::chrono::high_resolution_clock::now();
    latencies[request_idx] += end - start;
    model_indices[request_idx]++;
  }

  // 输出统计信息
  auto total_cnt = 0;
  for (size_t i = 0; i < model_dirs_list.size(); i++) {
    std::cout << "Model: " << model_dirs_list[i];
    if (random) {
      std::cout << " Count: " << model_indices[i];
    }
    double avg_latency = latencies[i].count() / 1000000.0 /
                         (model_indices[i] ? model_indices[i] : 1);
    std::cout << " Latency (ms): " << avg_latency << std::endl;
    total_cnt += model_indices[i];
  }

  std::cout << "Load model counts: " << total_cnt << std::endl;
  return 0;
}

int main(int argc, char* argv[]) {
  evaluate_load_latency(argc, argv);

  // test_region_swap();
  // test_allocate_blocks();
  // TestSimpleAllocation();
  // TestFullAllocation();
  // evaluate_complex_gpu_tensor_pool();
  return 0;
}