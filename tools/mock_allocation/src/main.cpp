#include <chrono>
#include <memory>
#include <random>

// #include "model_pool.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include <unordered_map>

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


int evaluate_vram_manager(int argc, char* argv[]) {
  size_t memory_pool_size = 50LL * 1024 * 1024 * 1024;
  int num_thread = 4;
  std::string model_dirs;
  int allocate_strategy = 4;
  int free_strategy=1;
  size_t scale = 10;
  size_t gpu_pool_size = 0;
  bool random = false;
  int distribution = 0;
  bool model_request_regenerate = false;
  int device_id = 0;
  bool affinity = false;
  std::string config_path="configs/model_config.json";
  float kv_cache_ratio=0.0;
  size_t block_size = 8 * 1024 * 1024;
  int tokens_in_block=16;
  std::string kv_block_file_path="";
  int kv_batch_size=1;

  std::string req_file_path="";

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "-m" || arg == "--memory-size") {
      memory_pool_size = std::stoull(argv[i + 1]);
      i++;
    } else if (arg == "-t" || arg == "--thread-num") {
      num_thread = std::stoi(argv[i + 1]);
      i++;
    } else if (arg == "-p" || arg == "--model-pool") {
      allocate_strategy = std::stoi(argv[i + 1]);
      i++;
    } else if (arg == "-f" || arg == "--free-strategy") {
      free_strategy = std::stoi(argv[i + 1]);
      i++;
    } else if (arg == "-s" || arg == "--scale") {
      scale = std::stoi(argv[i + 1]);
      i++;
    } else if (arg == "-g" || arg == "--gpu-pool") {
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
    } else if (arg=="--kv_cache_ratio"){
      kv_cache_ratio=std::stof(argv[i + 1]);
      i++;
    } else if(arg=="--kv_block_file_path"){
      kv_block_file_path=argv[i+1];
      i++;
    } else if(arg=="--kv_batch_size"){
      kv_batch_size=std::stoi(argv[i+1]);
      i++;
    }else if (arg=="--block"){
      block_size=std::stoi(argv[i + 1]) * 1024 * 1024;
      i++;
    }else if (arg=="--req_file_path"){
      req_file_path=argv[i + 1];
      i++;
    }
    else {
      std::cout << "Unknown argument: " << arg << "\n";
      return 1;
    }
  }

  std::vector<std::string> model_dirs_list;
  size_t num_request;
  std::vector<int> model_requests;  // 请求模型的序号

  // 加载配置文件
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
      std::cerr << "无法打开配置文件！" << std::endl;
      exit(1);
    }
    json config = json::parse(config_file);

    // 从配置文件读取数据
    model_dirs_list =
        config["model_dirs"].get<std::vector<std::string>>();
    std::vector<int> model_affinity =
        config["model_affinity"].get<std::vector<int>>();

    if (model_dirs_list.empty()) {
      std::cout << "No model dirs specified\n";
    }

    // 读取kv blocks
    std::vector<size_t> num_blocks;
    if(kv_block_file_path != ""){
      std::ifstream kv_block_file(kv_block_file_path);
      if(kv_block_file.is_open()){
        size_t nt;
        while(kv_block_file >> nt){
          // 向上取整
          num_blocks.push_back((nt + tokens_in_block - 1) / tokens_in_block);
        }
      }
    }
    std::cout<<"KV Block Allocate: "<<num_blocks.size()<<std::endl;

    

  if (req_file_path == "") {
    // 检查或生成请求序列
    std::string sequence_file = "model_requests.seq";
    if (std::filesystem::exists(sequence_file)) {
      // 从文件加载现有序列
      std::ifstream file(sequence_file, std::ios::binary);
      if (file) {
        int request;
        while (file >> request) {
          model_requests.push_back(request);
        }
        // std::cout << "Loaded " << model_requests.size()
        //           << " requests from existing file" << std::endl;
      }
    }

    // 如果文件不存在或加载失败，重新生成序列
    if (model_requests.empty() || model_request_regenerate) {
      if (affinity) {
        model_requests = AffinityModelRequestGenerate(model_dirs_list,
                                                      model_affinity, scale);
      } else {
        model_requests =
            GenerateModelRequests(model_dirs_list, random, distribution, scale);
      }
      // 保存到文件
      std::ofstream file(sequence_file, std::ios::binary);
      if (file) {
        for (int request : model_requests) {
          file << request << "\n";
        }
        std::cout << "Generated and saved " << model_requests.size()
                  << " requests to file" << std::endl;
      }
    }

    num_request = scale * model_dirs_list.size();
    // 输出请求序列
    std::cout << "Generated request sequence: ";
    for (const auto& request : model_requests) {
      std::cout << request << " ";
    }
    std::cout << std::endl;
  }else{
    // 从指定的文件读取模型请求
    std::ifstream file(req_file_path);
    if (file.is_open()) {
      int request;
      while (file >> request) {
        model_requests.push_back(request);
      }
      std::cout << "Loaded " << model_requests.size()
                << " requests from existing file" << std::endl;
    }else{
      std::cerr << "无法打开请求文件！" << std::endl;
      exit(1);
    }

    num_request=model_requests.size();
  }

  std::vector<int> gpu_ids = {device_id};
  std::shared_ptr<VRAMManager> model_pool_=std::make_shared<VRAMManager>(gpu_pool_size, gpu_ids, 400.0*1024*1024*1024, 20.0*1024*1024*1024);

  for (auto& model_dir : model_dirs_list) {
    auto size = model_pool_->RegisterModel(model_dir, 1);
  }

  std::vector<std::chrono::nanoseconds> latencies(model_dirs_list.size());
  std::vector<int> model_indices(model_dirs_list.size(), 0);
  size_t batch_id=0;

  for(int i=0;i<num_request;i++){
    int request_idx=model_requests[i];
    std::string req=model_dirs_list[model_requests[i]];

    auto start = std::chrono::high_resolution_clock::now();
    auto ret = model_pool_->LoadModel(req, device_id, free_strategy, allocate_strategy);
    if (ret == "ERROR") {
      std::cout << "Load model failed: " << req
                << std::endl;
      break;
    }
    auto end = std::chrono::high_resolution_clock::now();
    latencies[request_idx] += end - start;
    model_indices[request_idx]++;

    if (kv_cache_ratio > 0.0) {
      // 指定KV Cache占剩余空间的比例
      size_t avai_blk_cnt = model_pool_->GetAvailableBlocks(
          req, block_size, device_id);
      avai_blk_cnt *= kv_cache_ratio;
      auto alret = model_pool_->AllocateBlocks(device_id, block_size,
                                               req,
                                               avai_blk_cnt / 2);
      if (alret.size() != avai_blk_cnt / 2) {
        std::cout << "Allocate Blocks failed" << std::endl;
        return 1;
      }

      for (int i = 0; i < avai_blk_cnt - avai_blk_cnt / 2; i++) {
        auto ret = model_pool_->AllocateBlocks(device_id, block_size,
                                               req, 1);
        if (ret.empty()) {
          std::cout << "Allocate Blocks failed" << std::endl;
          return 1;
        }
      }
    } else if (num_blocks.size() > 0) {
      size_t total_block_cnt = num_blocks.size();
      size_t avai_blk_cnt =
          model_pool_->GetAvailableBlocks(req, block_size, device_id);
      size_t to_allocate_blk_cnt = 0;
      // 数据集特定的token数量和batch size
      for (int b = 0; b < kv_batch_size; b++) {
        to_allocate_blk_cnt +=
            num_blocks[(batch_id * kv_batch_size + b) % total_block_cnt];
      }
      // if (to_allocate_blk_cnt > avai_blk_cnt)
      //   to_allocate_blk_cnt = avai_blk_cnt;
      std::cout << "to_allocate_blk_cnt: " << to_allocate_blk_cnt << " / "
                << avai_blk_cnt << std::endl;
      // 分两次申请，第一次是Prefill，第二次是Decode
      auto alret = model_pool_->AllocateBlocks(device_id, block_size, req,
                                               to_allocate_blk_cnt);
      // if (alret.empty()) {
      //   // std::cout << "Allocate Blocks failed" << std::endl;
      //   // return 1;
      // }
      // for (int i = 0; i < to_allocate_blk_cnt - to_allocate_blk_cnt / 2; i++) {
      //   auto ret = model_pool_->AllocateBlocks(device_id, block_size, req, 1);
      //   if (ret.empty()) {
      //     // std::cout << "Allocate Blocks failed" << std::endl;
      //     // return 1;
      //   }
      //   // 生成16个token，每个需要26ms
      //   // std::this_thread::sleep_for(std::chrono::milliseconds(16*26));
      // }
      batch_id++;
    }
  }

  // 输出统计信息
  auto total_cnt = 0;
  double total_latency=0.0;
  for (size_t i = 0; i < model_dirs_list.size(); i++) {
    std::cout << "Model: " << model_dirs_list[i];
    std::cout << " Count: " << model_indices[i];
    // 统计平均延迟
    double avg_latency = latencies[i].count() / 1000000.0 /
                         (model_indices[i] ? model_indices[i] : 1);
    std::cout << " Latency (ms): " << avg_latency << std::endl;

    total_latency+=latencies[i].count() ;
    total_cnt += model_indices[i];
  }

  std::cout<<"Average Latency: "<<total_latency/1000000.0 /total_cnt<<std::endl;
  
  if(allocate_strategy==2 || 4){
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