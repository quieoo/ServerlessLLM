#include <chrono>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

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
#include "vram_manager_interface.h"
#include "vram_manager_legacy_adapter.h"
#include "vram_manager_vmm.h"

// #define MODELREGENERATE 1
// #define DEVICEID 0

namespace {

struct TraceRequest {
  double arrival_ts = 0.0;
  int model_id = 0;
  int model_index = 0;
  size_t input_tokens = 0;
  size_t output_tokens = 0;
  size_t kv_blocks = 0;
};

double ParseJsonDouble(const json& value, double default_value) {
  if (value.is_null()) {
    return default_value;
  }
  if (value.is_number()) {
    return value.get<double>();
  }
  if (value.is_string()) {
    return std::stod(value.get<std::string>());
  }
  return default_value;
}

void LoadModelConfig(const json& config, std::vector<std::string>& model_dirs_list,
                     std::vector<int>& model_affinity,
                     std::vector<double>& model_sensitivity,
                     std::unordered_map<int, int>& model_id_to_index) {
  model_dirs_list.clear();
  model_affinity.clear();
  model_sensitivity.clear();
  model_id_to_index.clear();

  if (config.contains("model_dirs")) {
    model_dirs_list = config["model_dirs"].get<std::vector<std::string>>();
    if (config.contains("model_affinity")) {
      model_affinity = config["model_affinity"].get<std::vector<int>>();
    } else {
      model_affinity.assign(model_dirs_list.size(), 1);
    }
    model_sensitivity.assign(model_dirs_list.size(), 1.0);
    if (config.contains("model_sensitivity")) {
      const auto& sensitivities = config["model_sensitivity"];
      for (size_t i = 0; i < model_dirs_list.size() && i < sensitivities.size();
           i++) {
        model_sensitivity[i] = ParseJsonDouble(sensitivities[i], 1.0);
      }
    } else if (config.contains("sensitivity")) {
      const auto& sensitivities = config["sensitivity"];
      for (size_t i = 0; i < model_dirs_list.size() && i < sensitivities.size();
           i++) {
        model_sensitivity[i] = ParseJsonDouble(sensitivities[i], 1.0);
      }
    }
    for (size_t i = 0; i < model_dirs_list.size(); i++) {
      model_id_to_index[static_cast<int>(i)] = static_cast<int>(i);
    }
    return;
  }

  if (config.contains("model_lists")) {
    struct ModelConfigEntry {
      int id;
      std::string path;
      double sensitivity;
    };
    std::vector<ModelConfigEntry> models;
    for (const auto& model : config["model_lists"]) {
      int id = model.value("id", static_cast<int>(models.size()));
      std::string path = model.at("path").get<std::string>();
      double sensitivity = model.contains("sensitivity")
                               ? ParseJsonDouble(model["sensitivity"], 1.0)
                               : 1.0;
      models.push_back({id, path, sensitivity});
    }
    std::sort(models.begin(), models.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.id < rhs.id; });
    for (const auto& model : models) {
      int id = model.id;
      model_id_to_index[id] = static_cast<int>(model_dirs_list.size());
      model_dirs_list.push_back(model.path);
      model_affinity.push_back(1);
      model_sensitivity.push_back(model.sensitivity);
    }
    return;
  }

  throw std::invalid_argument("Config must contain model_dirs or model_lists");
}

int ResolveModelRequest(int model_id,
                        const std::unordered_map<int, int>& model_id_to_index,
                        size_t model_count) {
  auto it = model_id_to_index.find(model_id);
  if (it != model_id_to_index.end()) {
    return it->second;
  }
  if (model_id >= 0 && static_cast<size_t>(model_id) < model_count) {
    return model_id;
  }
  throw std::out_of_range("Request model id out of range: " +
                          std::to_string(model_id));
}

void LoadRequestFile(const std::string& req_file_path,
                     const std::unordered_map<int, int>& model_id_to_index,
                     size_t model_count, int tokens_in_block,
                     std::vector<int>& model_requests,
                     std::vector<size_t>& request_num_blocks,
                     std::vector<TraceRequest>& trace_requests) {
  std::ifstream file(req_file_path);
  if (!file.is_open()) {
    throw std::runtime_error("无法打开请求文件: " + req_file_path);
  }

  std::string line;
  size_t line_no = 0;
  while (std::getline(file, line)) {
    line_no++;
    if (line.empty() || line[0] == '#') {
      continue;
    }

    std::istringstream iss(line);
    std::vector<std::string> fields;
    std::string field;
    while (iss >> field) {
      fields.push_back(field);
    }
    if (fields.empty()) {
      continue;
    }

    double arrival_ts = static_cast<double>(model_requests.size());
    int model_id = 0;
    size_t input_len = 0;
    size_t output_len = 0;
    size_t kv_blocks = 0;
    if (fields.size() == 1) {
      model_id = std::stoi(fields[0]);
    } else {
      // ServeGen trace format: timestamp model_id input_len output_len.
      arrival_ts = std::stod(fields[0]);
      model_id = std::stoi(fields[1]);
      if (fields.size() >= 4) {
        input_len = std::stoull(fields[2]);
        output_len = std::stoull(fields[3]);
        kv_blocks =
            (input_len + output_len + tokens_in_block - 1) / tokens_in_block;
        request_num_blocks.push_back(kv_blocks);
      }
    }

    try {
      int model_index =
          ResolveModelRequest(model_id, model_id_to_index, model_count);
      model_requests.push_back(model_index);
      trace_requests.push_back({arrival_ts, model_id, model_index, input_len,
                                output_len, kv_blocks});
    } catch (const std::exception& e) {
      throw std::runtime_error("Invalid request at line " +
                               std::to_string(line_no) + ": " + e.what());
    }
  }
}

void TruncateRequests(size_t max_requests, std::vector<int>& model_requests,
                      std::vector<size_t>& request_num_blocks,
                      std::vector<TraceRequest>& trace_requests) {
  if (max_requests == 0 || model_requests.size() <= max_requests) {
    return;
  }

  model_requests.resize(max_requests);
  if (!request_num_blocks.empty() && request_num_blocks.size() > max_requests) {
    request_num_blocks.resize(max_requests);
  }
  if (!trace_requests.empty() && trace_requests.size() > max_requests) {
    trace_requests.resize(max_requests);
  }
  std::cout << "Truncated requests to " << max_requests << std::endl;
}

std::unordered_map<std::string, size_t> ConsumeWarmupRequests(
    size_t warmup_step, const std::vector<std::string>& model_dirs_list,
    std::vector<int>& model_requests,
    std::vector<size_t>& request_num_blocks,
    std::vector<TraceRequest>& trace_requests) {
  std::unordered_map<std::string, size_t> model_access_counts;
  if (warmup_step == 0 || model_requests.empty()) {
    return model_access_counts;
  }

  size_t actual_warmup_step = std::min(warmup_step, model_requests.size());
  for (size_t i = 0; i < actual_warmup_step; i++) {
    int request_idx = model_requests[i];
    if (request_idx < 0 ||
        static_cast<size_t>(request_idx) >= model_dirs_list.size()) {
      throw std::runtime_error("Warmup request index out of range at request " +
                               std::to_string(i) + ": " +
                               std::to_string(request_idx));
    }
    model_access_counts[model_dirs_list[request_idx]]++;
  }

  model_requests.erase(model_requests.begin(),
                       model_requests.begin() + actual_warmup_step);
  if (!request_num_blocks.empty()) {
    request_num_blocks.erase(
        request_num_blocks.begin(),
        request_num_blocks.begin() +
            std::min(actual_warmup_step, request_num_blocks.size()));
  }
  if (!trace_requests.empty()) {
    trace_requests.erase(
        trace_requests.begin(),
        trace_requests.begin() +
            std::min(actual_warmup_step, trace_requests.size()));
  }

  std::cout << "Warmup consumed " << actual_warmup_step
            << " requests, remaining requests: " << model_requests.size()
            << std::endl;
  return model_access_counts;
}

int WriteLoadLatencyCDF(const std::string& load_cdf_path,
                        std::vector<double> load_latencies_ms) {
  if (load_cdf_path.empty()) {
    return 0;
  }

  std::filesystem::path output_path(load_cdf_path);
  if (output_path.has_parent_path()) {
    std::filesystem::create_directories(output_path.parent_path());
  }

  std::ofstream file(load_cdf_path);
  if (!file.is_open()) {
    std::cerr << "Failed to open load CDF output file: " << load_cdf_path
              << std::endl;
    return 1;
  }

  std::sort(load_latencies_ms.begin(), load_latencies_ms.end());
  file << "latency_ms cdf\n";
  for (size_t i = 0; i < load_latencies_ms.size(); i++) {
    double cdf = static_cast<double>(i + 1) /
                 static_cast<double>(load_latencies_ms.size());
    file << std::fixed << std::setprecision(6) << load_latencies_ms[i] << " "
         << cdf << "\n";
  }

  std::cout << "Wrote LoadModel latency CDF to " << load_cdf_path
            << " with " << load_latencies_ms.size() << " samples"
            << std::endl;
  return 0;
}

double Percentile(std::vector<double> values, double pct) {
  if (values.empty()) {
    return 0.0;
  }

  std::sort(values.begin(), values.end());
  double rank =
      (pct / 100.0) * static_cast<double>(values.size() - 1);
  size_t lower = static_cast<size_t>(std::floor(rank));
  size_t upper = static_cast<size_t>(std::ceil(rank));
  if (lower == upper) {
    return values[lower];
  }
  double fraction = rank - static_cast<double>(lower);
  return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

}  // namespace

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
  int device_id = -1;
  bool affinity = false;
  std::string config_path="configs/model_config.json";
  float kv_cache_ratio=0.0;
  size_t block_size = 8 * 1024 * 1024;
  int tokens_in_block=16;
  std::string kv_block_file_path="";
  int kv_batch_size=0;
  
  bool mock_copy=false;
  int gpu_num=0;
  int schedule_policy=1;
  int reuse_granularity=1;  // 0-model, 1-tensor
  bool tensor_only=false;
  bool verbose=false;
  size_t max_requests=0;
  size_t warmup_step=0;
  bool disable_parameter_reuse=false;
  std::string load_cdf_path;
  unsigned int random_seed=1;
  std::string memory_backend="legacy";
  size_t vmm_page_size_bytes=0;

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
    } else if (arg == "--gpu_num") {
      gpu_num = std::stoi(argv[i + 1]);
      i++;
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
    }else if (arg=="--max_requests"){
      max_requests=std::stoull(argv[i + 1]);
      i++;
    }else if (arg=="--warmup_step"){
      warmup_step=std::stoull(argv[i + 1]);
      i++;
    }else if (arg=="--load_cdf_path"){
      load_cdf_path=argv[i + 1];
      i++;
    }else if(arg=="--mock_copy"){
      mock_copy=true;
    }else if(arg=="--schedule_policy"){
      schedule_policy=std::stoi(argv[i+1]);
      i++;
    }else if(arg=="--reuse_granularity"){
      reuse_granularity=std::stoi(argv[i+1]);
      i++;
    }else if(arg=="--tensor-only" || arg=="--tensor_only"){
      tensor_only=true;
    }else if(arg=="--verbose"){
      verbose=true;
    }else if(arg=="--disable_parameter_reuse" || arg=="--no_parameter_reuse"){
      disable_parameter_reuse=true;
    }else if(arg=="--random_seed"){
      random_seed=static_cast<unsigned int>(std::stoul(argv[i+1]));
      i++;
    }else if(arg=="--memory_backend"){
      memory_backend=argv[i+1];
      i++;
    }else if(arg=="--vmm_page_size_mb"){
      const size_t page_size_mb=std::stoull(argv[i+1]);
      vmm_page_size_bytes=page_size_mb * 1024ULL * 1024ULL;
      i++;
    }
    else {
      std::cout << "Unknown argument: " << arg << "\n";
      return 1;
    }
  }
  std::srand(random_seed);

  // std::cout<<"================ ALL CONFIG =================="<<std::endl;
  // std::cout<<"schedule_policy: "<<schedule_policy<<std::endl;
  // std::cout<<"memory_pool_size: "<<memory_pool_size<<std::endl;
  // std::cout<<"num_thread: "<<num_thread<<std::endl;
  // std::cout<<"allocate_strategy: "<<allocate_strategy<<std::endl;
  // std::cout<<"free_strategy: "<<free_strategy<<std::endl;
  // std::cout<<"scale: "<<scale<<std::endl;
  // std::cout<<"gpu_pool_size: "<<gpu_pool_size<<std::endl;
  // std::cout<<"random: "<<random<<std::endl;
  // std::cout<<"distribution: "<<distribution<<std::endl;
  // std::cout<<"model_request_regenerate: "<<model_request_regenerate<<std::endl;
  // std::cout<<"device_id: "<<device_id<<std::endl;
  // std::cout<<"affinity: "<<affinity<<std::endl;
  // std::cout<<"config_path: "<<config_path<<std::endl;
  // std::cout<<"kv_cache_ratio: "<<kv_cache_ratio<<std::endl;
  // std::cout<<"kv_block_file_path: "<<kv_block_file_path<<std::endl;
  // std::cout<<"kv_batch_size: "<<kv_batch_size<<std::endl;
  // std::cout<<"block_size: "<<block_size<<std::endl;
  // std::cout<<"tokens_in_block: "<<tokens_in_block<<std::endl;
  // std::cout<<"mock_copy: "<<mock_copy<<std::endl;
  // std::cout<<"gpu_num: "<<gpu_num<<std::endl;
  // std::cout<<"req_file_path: "<<req_file_path<<std::endl;
  // std::cout<<"reuse_granularity: "<<reuse_granularity<<std::endl;
  // std::cout<<"schedule_policy: "<<schedule_policy<<std::endl;

  // std::cout<<"===== END CONFIG ====="<<std::endl;  


  std::vector<std::string> model_dirs_list;
  size_t num_request = 0;
  std::vector<int> model_requests;  // 请求模型的序号
  std::vector<size_t> request_num_blocks;
  std::vector<TraceRequest> trace_requests;
  std::vector<int> model_affinity;
  std::vector<double> model_sensitivity;
  std::unordered_map<int, int> model_id_to_index;

  // 加载配置文件
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
      std::cerr << "无法打开配置文件！" << std::endl;
      exit(1);
    }
    json config = json::parse(config_file);

    // 从配置文件读取数据
    try {
      LoadModelConfig(config, model_dirs_list, model_affinity,
                      model_sensitivity, model_id_to_index);
    } catch (const std::exception& e) {
      std::cerr << "配置文件格式错误: " << e.what() << std::endl;
      return 1;
    }

    if (model_dirs_list.empty()) {
      std::cout << "No model dirs specified\n";
      return 1;
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

    num_request = model_requests.size();
    trace_requests.clear();
    trace_requests.reserve(model_requests.size());
    for (size_t i = 0; i < model_requests.size(); i++) {
      trace_requests.push_back(
          {static_cast<double>(i), model_requests[i], model_requests[i], 0, 0,
           0});
    }
    // 输出请求序列
    std::cout << "Generated request sequence: ";
    for (const auto& request : model_requests) {
      std::cout << request << " ";
    }
    std::cout << std::endl;
  }else{
    // 从指定的文件读取模型请求
    try {
      LoadRequestFile(req_file_path, model_id_to_index, model_dirs_list.size(),
                      tokens_in_block, model_requests, request_num_blocks,
                      trace_requests);
    } catch (const std::exception& e) {
      std::cerr << e.what() << std::endl;
      return 1;
    }
    std::cout << "Loaded " << model_requests.size()
              << " requests from existing file" << std::endl;
    if (!request_num_blocks.empty()) {
      std::cout << "Loaded " << request_num_blocks.size()
                << " KV block entries from trace token lengths" << std::endl;
    }
  }

  std::unordered_map<std::string, size_t> warmup_model_access;
  try {
    warmup_model_access = ConsumeWarmupRequests(
        warmup_step, model_dirs_list, model_requests, request_num_blocks,
        trace_requests);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  TruncateRequests(max_requests, model_requests, request_num_blocks,
                   trace_requests);
  num_request=model_requests.size();

  std::vector<int> gpu_ids;
  if(gpu_num>0){
    gpu_ids.resize(gpu_num);
    for(int i=0;i<gpu_num;i++){
      gpu_ids[i]=i;
    }
  }else if(device_id>=0){
    gpu_ids={device_id};
  }else{
    std::cerr << "No GPU specified, please use --gpu or --gpu_num" << std::endl;
    exit(1);
  }

  if (memory_backend != "legacy" && memory_backend != "vmm") {
    std::cerr << "Unknown memory backend: " << memory_backend
              << " (expected legacy or vmm)" << std::endl;
    return 1;
  }
  if (memory_backend == "vmm" && mock_copy) {
    std::cerr << "--memory_backend vmm does not support --mock_copy" << std::endl;
    return 1;
  }
  if (memory_backend == "vmm" && allocate_strategy != 4) {
    std::cout << "VMM backend ignores -p/--model-pool; physical fragmentation "
                 "strategies are disabled" << std::endl;
  }
  if (tensor_only && memory_backend != "vmm") {
    std::cerr << "--tensor-only requires --memory_backend vmm" << std::endl;
    return 1;
  }

  std::shared_ptr<IVRAMManager> model_pool_;
  try {
    if (memory_backend == "vmm") {
      model_pool_ = std::make_shared<VmmVRAMManager>(
          gpu_pool_size, gpu_ids, 400.0 * 1024 * 1024 * 1024,
          20.0 * 1024 * 1024 * 1024, mock_copy, vmm_page_size_bytes);
    } else {
      model_pool_ = std::make_shared<LegacyVRAMManagerAdapter>(
          gpu_pool_size, gpu_ids, 400.0 * 1024 * 1024 * 1024,
          20.0 * 1024 * 1024 * 1024, mock_copy);
    }
  } catch (const std::exception& e) {
    std::cerr << "Failed to create " << memory_backend
              << " memory backend: " << e.what() << std::endl;
    return 1;
  }


  for (size_t i = 0; i < model_dirs_list.size(); i++) {
    double sensitivity =
        i < model_sensitivity.size() ? model_sensitivity[i] : 1.0;
    int64_t size = 0;
    if (memory_backend == "vmm") {
      auto vmm_pool = std::dynamic_pointer_cast<VmmVRAMManager>(model_pool_);
      size = vmm_pool->RegisterModelWithMergeTarget(
          model_dirs_list[i], sensitivity, mock_copy, tensor_only ? -2 : -1);
    } else {
      size = model_pool_->RegisterModel(model_dirs_list[i], sensitivity,
                                        mock_copy, reuse_granularity);
    }
    // std::cout << "Registered model sensitivity: " << model_dirs_list[i]
    //           << " sensitivity=" << sensitivity << std::endl;
  }

  model_pool_->WarmupModelAccess(warmup_model_access);

  std::vector<std::chrono::nanoseconds> latencies(model_dirs_list.size());
  std::vector<double> load_latencies_ms;
  std::vector<int> model_indices(model_dirs_list.size(), 0);
  size_t batch_id=0;

  for(size_t i=0;i<num_request;i++){
    int request_idx=model_requests[i];
    if (request_idx < 0 || static_cast<size_t>(request_idx) >= model_dirs_list.size()) {
      std::cerr << "Request index out of range at request " << i << ": "
                << request_idx << std::endl;
      return 1;
    }
    std::string req=model_dirs_list[model_requests[i]];

    auto start = std::chrono::high_resolution_clock::now();
    if(gpu_num>0){
      device_id=model_pool_->GetGPUToLoad(req, schedule_policy);
    }
    // std::cout<<"Load "<<req<<" to GPU "<<device_id<<std::endl;
    if(verbose){
      std::cout<<"Memory Utilization: "<<model_pool_->get_memory_utilization()<<std::endl;
    }
    auto load_start = std::chrono::high_resolution_clock::now();
    auto ret = model_pool_->LoadModel(req, device_id, free_strategy,
                                      allocate_strategy, verbose,
                                      disable_parameter_reuse);
    auto load_end = std::chrono::high_resolution_clock::now();
    double measured_load_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(load_end -
                                                              load_start)
            .count() /
        1000.0;
    load_latencies_ms.push_back(measured_load_ms);
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
    } else if (!request_num_blocks.empty() && kv_batch_size > 0) {
      size_t total_block_cnt = request_num_blocks.size();
      size_t to_allocate_blk_cnt = 0;
      for (int b = 0; b < kv_batch_size; b++) {
        to_allocate_blk_cnt +=
            request_num_blocks[(batch_id * kv_batch_size + b) % total_block_cnt];
      }
      if (to_allocate_blk_cnt > 0) {
        model_pool_->AllocateBlocks(device_id, block_size, req, to_allocate_blk_cnt);
      }
      batch_id++;
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
      // std::cout << "to_allocate_blk_cnt: " << to_allocate_blk_cnt << " / "
      //           << avai_blk_cnt << std::endl;
      // 分两次申请，第一次是Prefill，第二次是Decode
      if(to_allocate_blk_cnt>0){
        auto alret = model_pool_->AllocateBlocks(device_id, block_size, req,
                                               to_allocate_blk_cnt);
      }
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
  constexpr int kModelColWidth = 70;
  constexpr int kCountColWidth = 12;
  constexpr int kLatencyColWidth = 18;

  std::cout << "\n================ Model Load Summary ================\n";
  std::cout << std::left << std::setw(kModelColWidth) << "Model"
            << std::right << std::setw(kCountColWidth) << "Count"
            << std::setw(kLatencyColWidth) << "Avg Latency(ms)" << "\n";
  std::cout << std::string(kModelColWidth + kCountColWidth + kLatencyColWidth,
                           '-')
            << "\n";

  for (size_t i = 0; i < model_dirs_list.size(); i++) {
    // 统计平均延迟
    double avg_latency = latencies[i].count() / 1000000.0 /
                         (model_indices[i] ? model_indices[i] : 1);
    std::cout << std::left << std::setw(kModelColWidth) << model_dirs_list[i]
              << std::right << std::setw(kCountColWidth) << model_indices[i]
              << std::setw(kLatencyColWidth) << std::fixed
              << std::setprecision(3) << avg_latency << "\n";

    total_latency+=latencies[i].count() ;
    total_cnt += model_indices[i];
  }
  std::cout << std::string(kModelColWidth + kCountColWidth + kLatencyColWidth,
                           '-')
            << "\n";

  std::cout << std::left << std::setw(kModelColWidth) << "Overall"
            << std::right << std::setw(kCountColWidth) << total_cnt
            << std::setw(kLatencyColWidth) << std::fixed
            << std::setprecision(3)
            << (total_cnt ? total_latency / 1000000.0 / total_cnt : 0.0)
            << "\n";

  std::cout << "p50_load_ms: " << std::fixed << std::setprecision(3)
            << Percentile(load_latencies_ms, 50.0) << "\n";
  std::cout << "p95_load_ms: " << std::fixed << std::setprecision(3)
            << Percentile(load_latencies_ms, 95.0) << "\n";
  std::cout << "p99_load_ms: " << std::fixed << std::setprecision(3)
            << Percentile(load_latencies_ms, 99.0) << "\n";
  
  model_pool_->MemoryUsage();

  if (WriteLoadLatencyCDF(load_cdf_path, load_latencies_ms) != 0) {
    return 1;
  }

  std::cout << "====================================================\n";
  return 0;
}

int main(int argc, char* argv[]) {
  evaluate_vram_manager(argc, argv);
  // TestCostAwareDropWithModels();
  return 0;
}
