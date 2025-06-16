#include <chrono>
#include <memory>
#include <random>

// #include "model_pool.h"
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
// #include "test_AllocateASAP.h"
// #include "test_gpu_tensor_pool_v2_1.h"
#include "vram_manager_v1.h"
#include "vram_manager_v2.h"
#include "vram_manager_v3.h"
#include "vram_manager_v4.h"

#include "vram_manager.h"

// #define MODELREGENERATE 1
// #define DEVICEID 0

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
      // double mean = model_dirs_list.size() / 2;
      // double stddev = model_dirs_list.size() / 4;
      double mean = (model_dirs_list.size() - 1) / 2.0;
      double stddev = model_dirs_list.size() / 4.0;
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

inline std::vector<int> AffinityModelRequestGenerate(
  const std::vector<std::string>& model_dirs_list, 
  const std::vector<int>& model_affinity,
  size_t scale) {
std::vector<int> requests;

// 验证权重列表长度匹配
if (model_affinity.size() != model_dirs_list.size()) {
  throw std::invalid_argument("Model affinity list size mismatch");
}

std::random_device rd;
std::mt19937 gen(rd());

// 创建基于权重的离散分布
std::discrete_distribution<> dis(model_affinity.begin(), model_affinity.end());

size_t total_requests = model_dirs_list.size() * scale;
for (size_t i = 0; i < total_requests; i++) {
  requests.push_back(dis(gen));
}

return requests;
}


// int evaluate_load_latency(int argc, char* argv[]) {
//   size_t memory_pool_size = 50LL * 1024 * 1024 * 1024;
//   int num_thread = 4;
//   std::string model_dirs;
//   int model_pool_type = 2;
//   size_t scale = 10;
//   size_t gpu_pool_size = 0;
//   bool random = false;
//   int distribution = 0;
//   bool model_request_regenerate = false;
//   int device_id = 0;
//   bool affinity = false;
//   std::string config_path="configs/model_config.json";

//   for (int i = 1; i < argc; i++) {
//     std::string arg = argv[i];
//     if (arg == "-m" || arg == "--memory-size") {
//       memory_pool_size = std::stoull(argv[i + 1]);
//       i++;
//     } else if (arg == "-t" || arg == "--thread-num") {
//       num_thread = std::stoi(argv[i + 1]);
//       i++;
//     } else if (arg == "-p" || arg == "--model-pool") {
//       model_pool_type = std::stoi(argv[i + 1]);
//       i++;
//     } else if (arg == "-s" || arg == "--scale") {
//       scale = std::stoi(argv[i + 1]);
//       i++;
//     } else if (arg == "-g" || arg == "--gpu-pool") {
//       if (model_pool_type != 2) {
//         std::cout << "GPU pool only support reuse model pool" << std::endl;
//         return 0;
//       }
//       float size_gbs = std::stof(argv[i + 1]);
//       gpu_pool_size = static_cast<size_t>(size_gbs * 1024 * 1024 * 1024);
//       // gpu_pool_size = std::stoull(argv[i + 1]);
//       i++;
//     } else if (arg == "-r" || arg == "--random") {
//       random = true;
//       if (strcmp(argv[i + 1], "even") == 0) {
//         distribution = 0;
//       } else if (strcmp(argv[i + 1], "guas") == 0) {
//         distribution = 1;
//       } else {
//         std::cout << "Unknown distribution: " << argv[i + 1] << "\n";
//         return 1;
//       }
//       i++;
//     } else if (arg == "--regenerate") {
//       model_request_regenerate = true;
//     } else if (arg == "--gpu") {
//       device_id = std::stoi(argv[i + 1]);
//       i++;
//     } else if (arg == "--affinity") {
//       affinity = true;
//     } else if (arg =="-c" || arg == "--config") {
//       config_path = argv[i + 1];
//       i++;
//     } else {
//       std::cout << "Unknown argument: " << arg << "\n";
//       return 1;
//     }
//   }

//   // 加载配置文件
//   std::ifstream config_file(config_path);
//   if (!config_file.is_open()) {
//       std::cerr << "无法打开配置文件！" << std::endl;
//       exit(1);
//   }
//   json config = json::parse(config_file);

//   // 从配置文件读取数据
//   std::vector<std::string> model_dirs_list = config["model_dirs"].get<std::vector<std::string>>();
//   std::vector<int> model_affinity = config["model_affinity"].get<std::vector<int>>();

//   if (model_dirs_list.empty()) {
//     std::cout << "No model dirs specified\n";
//   }

//   // 检查或生成请求序列
//   std::string sequence_file = "model_requests.seq";
//   std::vector<int> model_requests;

//   if (std::filesystem::exists(sequence_file)) {
//     // 从文件加载现有序列
//     std::ifstream file(sequence_file, std::ios::binary);
//     if (file) {
//       int request;
//       while(file >> request){
//         model_requests.push_back(request);
//       }
//       // std::cout << "Loaded " << model_requests.size()
//       //           << " requests from existing file" << std::endl;
//     }
//   }

//   // 如果文件不存在或加载失败，重新生成序列
//   if (model_requests.empty() || model_request_regenerate) {
//     if(affinity){
//       model_requests = AffinityModelRequestGenerate(model_dirs_list, model_affinity, scale);
//     }else{
//       model_requests = GenerateModelRequests(model_dirs_list, random, distribution, scale);
//     }
//     // 保存到文件
//     std::ofstream file(sequence_file, std::ios::binary);
//     if (file) {
//       for (int request : model_requests) {
//         file<<request<<"\n";
//       }
//       std::cout << "Generated and saved " << model_requests.size()
//                 << " requests to file" << std::endl;
//     }
//   }

//   // 输出请求序列
//   std::cout << "Generated request sequence: ";
//   for (const auto& request : model_requests) {
//     std::cout << request << " ";
//   }
//   std::cout << std::endl;

//   std::shared_ptr<VRAMManagerBase> model_pool_;
//   std::vector<int> gpu_ids = {device_id};
//   if (model_pool_type == 0) {
//     model_pool_ = std::make_shared<VRAMManager_V0>(memory_pool_size, num_thread, gpu_ids);
//   } else if (model_pool_type == 1) {
//     model_pool_ = std::make_shared<VRAMManager_V1>(memory_pool_size, num_thread,
//                                                    gpu_pool_size, gpu_ids);
//   } else if (model_pool_type == 2) {
//     model_pool_ = std::make_shared<VRAMManager_V2>(gpu_pool_size, gpu_ids, 400.0*1024*1024);
//   } else if (model_pool_type==3){
//     model_pool_ = std::make_shared<VRAMManager_V3>(gpu_pool_size, gpu_ids, 400.0*1024*1024, 20.0*1024*1024);
//   }else if(model_pool_type==4){
//     model_pool_ = std::make_shared<VRAMManager_V4>(gpu_pool_size, gpu_ids, 400.0*1024*1024, 20.0*1024*1024);
//   }else {
//     std::cout << "Invalid model pool type" << std::endl;
//     return 0;
//   }

//   for (auto& model_dir : model_dirs_list) {
//     auto size = model_pool_->RegisterModel(model_dir, 1);
//     // model_sizes.push_back(size);
//   }


//   std::vector<std::chrono::nanoseconds> latencies(model_dirs_list.size());
//   std::vector<int> model_indices(model_dirs_list.size(), 0);

//   for (int request_idx : model_requests) {
//     auto start = std::chrono::high_resolution_clock::now();
//     auto ret = model_pool_->LoadModel(model_dirs_list[request_idx], device_id);
//     // if (model_pool_type == 2 ) {
//     //   model_pool_->MemoryUsage();
//     // }
//     if (ret == "ERROR") {
//       std::cout << "Load model failed: " << model_dirs_list[request_idx]
//                 << std::endl;
//       break;
//     }
//     auto end = std::chrono::high_resolution_clock::now();
//     latencies[request_idx] += end - start;
//     model_indices[request_idx]++;
//   }

//   // 输出统计信息
//   auto total_cnt = 0;
//   for (size_t i = 0; i < model_dirs_list.size(); i++) {
//     std::cout << "Model: " << model_dirs_list[i];
//     if (random) {
//       std::cout << " Count: " << model_indices[i];
//     }
//     double avg_latency = latencies[i].count() / 1000000.0 /
//                          (model_indices[i] ? model_indices[i] : 1);
//     std::cout << " Latency (ms): " << avg_latency << std::endl;
//     total_cnt += model_indices[i];
//   }
//   if(model_pool_type==2 || 4){
//     model_pool_->MemoryUsage();
//   }

//   std::cout << "Load model counts: " << total_cnt << std::endl;
//   return 0;
// }

int evaluate_vram_manager(int argc, char* argv[]) {
  size_t memory_pool_size = 50LL * 1024 * 1024 * 1024;
  int num_thread = 4;
  std::string model_dirs;
  int model_pool_type = 2;
  size_t scale = 10;
  size_t gpu_pool_size = 0;
  bool random = false;
  int distribution = 0;
  bool model_request_regenerate = false;
  int device_id = 0;
  bool affinity = false;
  std::string config_path="configs/model_config.json";

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "-m" || arg == "--memory-size") {
      memory_pool_size = std::stoull(argv[i + 1]);
      i++;
    } else if (arg == "-t" || arg == "--thread-num") {
      num_thread = std::stoi(argv[i + 1]);
      i++;
    } else if (arg == "-p" || arg == "--model-pool") {
      model_pool_type = std::stoi(argv[i + 1]);
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
      random = true;
      if (strcmp(argv[i + 1], "even") == 0) {
        distribution = 0;
      } else if (strcmp(argv[i + 1], "guas") == 0) {
        distribution = 1;
      } else {
        std::cout << "Unknown distribution: " << argv[i + 1] << "\n";
        return 1;
      }
      i++;
    } else if (arg == "--regenerate") {
      model_request_regenerate = true;
    } else if (arg == "--gpu") {
      device_id = std::stoi(argv[i + 1]);
      i++;
    } else if (arg == "--affinity") {
      affinity = true;
    } else if (arg =="-c" || arg == "--config") {
      config_path = argv[i + 1];
      i++;
    } else {
      std::cout << "Unknown argument: " << arg << "\n";
      return 1;
    }
  }

  // 加载配置文件
  std::ifstream config_file(config_path);
  if (!config_file.is_open()) {
      std::cerr << "无法打开配置文件！" << std::endl;
      exit(1);
  }
  json config = json::parse(config_file);

  // 从配置文件读取数据
  std::vector<std::string> model_dirs_list = config["model_dirs"].get<std::vector<std::string>>();
  std::vector<int> model_affinity = config["model_affinity"].get<std::vector<int>>();

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
      while(file >> request){
        model_requests.push_back(request);
      }
      // std::cout << "Loaded " << model_requests.size()
      //           << " requests from existing file" << std::endl;
    }
  }

  // 如果文件不存在或加载失败，重新生成序列
  if (model_requests.empty() || model_request_regenerate) {
    if(affinity){
      model_requests = AffinityModelRequestGenerate(model_dirs_list, model_affinity, scale);
    }else{
      model_requests = GenerateModelRequests(model_dirs_list, random, distribution, scale);
    }
    // 保存到文件
    std::ofstream file(sequence_file, std::ios::binary);
    if (file) {
      for (int request : model_requests) {
        file<<request<<"\n";
      }
      std::cout << "Generated and saved " << model_requests.size()
                << " requests to file" << std::endl;
    }
  }

  size_t num_request=scale * model_dirs_list.size();
  // 输出请求序列
  std::cout << "Generated request sequence: ";
  for (const auto& request : model_requests) {
    std::cout << request << " ";
  }
  std::cout << std::endl;
  std::vector<int> gpu_ids = {device_id};
  std::shared_ptr<VRAMManager> model_pool_=std::make_shared<VRAMManager>(gpu_pool_size, gpu_ids, 400.0*1024*1024, 20.0*1024*1024);

  for (auto& model_dir : model_dirs_list) {
    auto size = model_pool_->RegisterModel(model_dir, 1);
    // model_sizes.push_back(size);
  }


  std::vector<std::chrono::nanoseconds> latencies(model_dirs_list.size());
  std::vector<int> model_indices(model_dirs_list.size(), 0);
  for(int i=0;i<num_request;i++){
    int request_idx=model_requests[i];
    auto start = std::chrono::high_resolution_clock::now();
    auto ret = model_pool_->LoadModel(model_dirs_list[request_idx], device_id, model_pool_type);
    // if (model_pool_type == 2 ) {
    //   model_pool_->MemoryUsage();
    // }
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
  if(model_pool_type==2 || 4){
    model_pool_->MemoryUsage();
  }

  std::cout << "Load model counts: " << total_cnt << std::endl;
  return 0;
}

int main(int argc, char* argv[]) {
  // evaluate_load_latency(argc, argv);
  evaluate_vram_manager(argc, argv);
  // TestCostAwareDropWithModels();
  return 0;
}