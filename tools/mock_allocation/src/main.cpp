#include <chrono>
#include <random>

#include "model_pool.h"
void TestGPUTensorPool() {
  GPUTensorPool_V3 pool(0, 1024 * 1024 * 1024);

  std::shared_ptr<GPUMemoryRegion> region1 =
      pool.BestFitAllocation(256 * 1024 * 1024, "Tensor_1");
  if (region1) {
    std::cout << "Allocated Tensor_1 at " << static_cast<void*>(region1->addr)
              << "\n";
  }

  std::shared_ptr<GPUMemoryRegion> region2 =
      pool.BestFitAllocation(512 * 1024 * 1024, "Tensor_2");
  if (region2) {
    std::cout << "Allocated Tensor_2 at " << static_cast<void*>(region2->addr)
              << "\n";
  }

  std::shared_ptr<GPUMemoryRegion> retrieved_region =
      pool.GetTensor("Tensor_1");
  if (retrieved_region) {
    std::cout << "Retrieved Tensor_1 from cache at "
              << static_cast<void*>(retrieved_region->addr) << "\n";
  } else {
    std::cout << "Tensor_1 not found\n";
  }

  retrieved_region = pool.GetTensor("Tensor_3");
  if (retrieved_region) {
    std::cout << "Retrieved Tensor_3 from cache at "
              << static_cast<void*>(retrieved_region->addr) << "\n";
  } else {
    std::cout << "Tensor_3 not found\n";
  }

  std::shared_ptr<GPUMemoryRegion> region3 =
      pool.BestFitAllocation(128 * 1024 * 1024, "Tensor_3");
  if (region3) {
    std::cout << "Allocated Tensor_3 at " << static_cast<void*>(region3->addr)
              << "\n";
  }

  retrieved_region = pool.GetTensor("Tensor_2");
  if (retrieved_region) {
    std::cout << "Retrieved Tensor_2 from cache at "
              << static_cast<void*>(retrieved_region->addr) << "\n";
  } else {
    std::cout << "Tensor_2 not found\n";
  }
  pool.MemoryRegionView();
  std::shared_ptr<GPUMemoryRegion> region4 =
      pool.BestFitAllocation(700 * 1024 * 1024, "Tensor_4");
  if (region4) {
    std::cout << "Allocated Tensor_4 at " << static_cast<void*>(region4->addr)
              << "\n";
  }
  pool.MemoryRegionView();
  pool.MemoryUsage();
}

void TestGpuTensorPoolV2() {
  GPUTensorPool_V3 pool(0, 30LL * 1024 * 1024 * 1024);
  std::string storage_path = "/mnt/n0/models";
  std::vector<std::string> model_paths = {"vllm/opt6.7b_tmp/rank_0",
                                          "vllm/opt1.3b_tmp/rank_0"};

  std::vector<size_t> tensor_sizes;
  std::vector<std::string> tensor_names;
  for (auto& model_path : model_paths) {
    std::string index_file =
        storage_path + "/" + model_path + "/tensor_group_index.txt";
    std::vector<TensorGroupIndex> tensor_group_index;
    ParseTensorGroupIndex(index_file, tensor_group_index);
    for (auto& tensor_group : tensor_group_index) {
      tensor_sizes.push_back(tensor_group.size);
      tensor_names.push_back(tensor_group.fingerprint);
    }
  }

  size_t Scales = 1000;
  std::chrono::nanoseconds total_best_fit_time(0);
  std::chrono::nanoseconds total_get_tensor_time(0);

  for (int i = 0; i < Scales; i++) {
    for (int j = 0; j < tensor_sizes.size(); j++) {
      auto start_best_fit = std::chrono::high_resolution_clock::now();
      pool.BestFitAllocation(tensor_sizes[j], tensor_names[j]);
      auto end_best_fit = std::chrono::high_resolution_clock::now();
      total_best_fit_time += end_best_fit - start_best_fit;

      auto start_get_tensor = std::chrono::high_resolution_clock::now();
      auto region = pool.GetTensor(tensor_names[j]);
      auto end_get_tensor = std::chrono::high_resolution_clock::now();
      total_get_tensor_time += end_get_tensor - start_get_tensor;
      if (!region) {
        std::cout << "Tensor_2 not found\n";
        return;
      } else if (!region->is_allocated ||
                 (region->fingerprint != tensor_names[j])) {
        std::cout << "Retrieved Tensor_2 error " << "\n";
        return;
      }
    }
  }

  std::chrono::nanoseconds average_best_fit_time =
      total_best_fit_time / tensor_sizes.size() / Scales;
  std::chrono::nanoseconds average_get_tensor_time =
      total_get_tensor_time / tensor_sizes.size() / Scales;

  std::cout << "Average BestFitAllocation time: "
            << average_best_fit_time.count() << " ns\n";
  std::cout << "Average GetTensor time: " << average_get_tensor_time.count()
            << " ns\n";
  // pool->MemoryUsage();
  // pool.MemoryRegionView();
}

int main(int argc, char* argv[]) {
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
      gpu_pool_size = std::stoull(argv[i + 1]);
      i++;
    } else if (arg == "-r" || arg == "--random") {
      std::cout << "Using random model allocation" << std::endl;
      random = true;
      if (strcmp(argv[i + 1], "even") == 0) {
        std::cout<<"Using even distribution"<<std::endl;
        distribution = 0;
      } else if (strcmp(argv[i + 1], "guas") == 0) {
        std::cout<<"Using guas distribution"<<std::endl;
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

  std::shared_ptr<ModelPoolBase> model_pool_;
  if (model_pool_type == 1) {
    model_pool_ =
        std::make_shared<NativeModelPool>(memory_pool_size, num_thread);
  } else if (model_pool_type == 2) {
    model_pool_ = std::make_shared<ModelPool>(memory_pool_size, num_thread,
                                              gpu_pool_size);
  } else {
    std::cout << "Invalid model pool type" << std::endl;
    return 0;
  }

  for (auto& model_dir : model_dirs_list) {
    auto size = model_pool_->RegisterModel(model_dir);
    model_sizes.push_back(size);
  }

  if (kv_num_blocks > 0) {
    // check if global memory is enough to run the model
    for (int i = 0; i < model_dirs_list.size(); i++) {
      auto total_size = model_sizes[i] + kv_num_blocks * kv_block_sizes[i];
      if (total_size > gpu_pool_size) {
        std::cout << "GPU Tensor Pool has not enough memory for model "
                  << model_dirs_list[i] << " with kv num blocks "
                  << kv_num_blocks << std::endl;
        return 0;
      }
    }
  }

  // warm the cpu model cache
  for (auto& model_path : model_dirs_list) {
    model_pool_->LoadModelAsync(model_path);
  }

  if (random) {
    std::vector<std::chrono::nanoseconds> latencies(model_dirs_list.size());
    std::vector<int> model_indices(model_dirs_list.size());
    if (distribution == 0) {
      std::random_device rd;
      std::mt19937 gen(rd());  
      std::uniform_int_distribution<> dis(
          0, model_dirs_list.size() - 1);  // 均匀分布

      for (int i = 0; i < model_dirs_list.size() * scale; i++) {
        int j = dis(gen);
        auto start = std::chrono::high_resolution_clock::now();
        model_pool_->LoadModelAsync(model_dirs_list[j]);
        auto end = std::chrono::high_resolution_clock::now();
        latencies[j] += end - start;
        model_indices[j]++;
      }
    } else if (distribution == 1) {
      std::random_device rd;
      std::mt19937 gen(rd());
      double mean = model_dirs_list.size() / 2;      // 均值靠近中间
      double stddev = model_dirs_list.size() / 4;    // 标准差控制分布范围
      std::normal_distribution<> dis(mean, stddev);  // 正态分布
      for (int i = 0; i < model_dirs_list.size() * scale; i++) {
        // 生成的浮动值可能不是整数，需要将其转换为有效的索引
        int j = static_cast<int>(dis(gen));

        // 确保 j 在合理范围内
        j = std::max(0,
                     std::min(j, static_cast<int>(model_dirs_list.size()) - 1));

        // 记录加载模型的延迟
        auto start = std::chrono::high_resolution_clock::now();
        model_pool_->LoadModelAsync(model_dirs_list[j]);
        auto end = std::chrono::high_resolution_clock::now();

        // 更新延迟
        latencies[j] += end - start;
        model_indices[j]++;
      }
    }

    for (int i = 0; i < model_dirs_list.size(); i++) {
      std::cout << "Model: " << model_dirs_list[i];
      std::cout << "Count: " << model_indices[i] << " Average Latency (ms): "
                << latencies[i].count() / 1000000.0 / model_indices[i]
                << std::endl;
    }
  } else {
    std::vector<std::chrono::nanoseconds> latencies(model_dirs_list.size());
    for (int i = 0; i < scale; i++) {
      for (int j = 0; j < model_dirs_list.size(); j++) {
        auto start = std::chrono::high_resolution_clock::now();
        model_pool_->LoadModelAsync(model_dirs_list[j]);
        auto end = std::chrono::high_resolution_clock::now();
        latencies[j] += end - start;
      }
    }

    for (int i = 0; i < model_dirs_list.size(); i++) {
      std::cout << "Model: " << model_dirs_list[i];
      std::cout << " Latency (ms): " << latencies[i].count() / 1000000.0 / scale
                << std::endl;
    }
  }

  if (model_pool_type == 2) {
    model_pool_->MemoryUsage();
  }
}