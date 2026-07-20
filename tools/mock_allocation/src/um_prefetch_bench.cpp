#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "registered_model.h"

using json = nlohmann::json;

namespace {

struct TraceRequest {
  double arrival_ts = 0.0;
  int model_id = 0;
  int model_index = 0;
};

struct BenchResult {
  size_t request_idx = 0;
  int model_id = 0;
  int model_index = 0;
  std::string model_path;
  std::string mode;
  std::string scenario;
  int gpu_id = 0;
  size_t model_bytes = 0;
  size_t bytes_moved_est = 0;
  size_t cached_bytes_est = 0;
  double load_ms = 0.0;
  double prepare_managed_ms = 0.0;
  double prefetch_ms = 0.0;
  int is_warmup = 0;
  int is_full_model_transfer = 0;
};

struct BenchOptions {
  std::string config_path;
  std::string req_file_path;
  std::string mode = "explicit-copy";
  std::string scenario = "warm-repeat";
  std::string load_cdf_path;
  std::string request_csv_path;
  std::string summary_csv_path;
  int gpu_id = 0;
  int gpu_num = 1;
  std::vector<int> gpu_ids;
  size_t max_requests = 0;
  size_t warmup_step = 0;
  unsigned int random_seed = 1;
  bool verbose = false;
  size_t progress_interval = 50;
  size_t prepare_threads = 4;
  double max_gpu_memory_gb = 0.0;
  std::string gpu_assignment_mode = "fixed_model_owner";
};

struct PrepareManagedStats {
  double alloc_ms = 0.0;
  double copy_ms = 0.0;
  double advise_ms = 0.0;
  double total_ms = 0.0;
  bool prepared = false;
};

struct ManagedMemorySupport {
  int device_id = 0;
  int managed_memory = 0;
  int concurrent_managed_access = 0;
  int pageable_memory_access = 0;
  int pageable_memory_access_uses_host_page_tables = 0;
};

struct BenchModel {
  int model_id = 0;
  int model_index = 0;
  std::string model_path;
  std::shared_ptr<RegisteredModel> registered_model;
  void* managed_ptr = nullptr;
  void* explicit_device_ptr = nullptr;
  int managed_resident_gpu = -1;
  int explicit_resident_gpu = -1;
  int last_target_gpu = -1;
  size_t model_size = 0;
  double best_full_prefetch_ms = 0.0;

  ~BenchModel() {
    if (managed_ptr != nullptr) {
      cudaFree(managed_ptr);
      managed_ptr = nullptr;
    }
    if (explicit_device_ptr != nullptr) {
      if (explicit_resident_gpu >= 0) {
        cudaSetDevice(explicit_resident_gpu);
      }
      cudaFree(explicit_device_ptr);
      explicit_device_ptr = nullptr;
    }
  }
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

double Percentile(std::vector<double> values, double pct) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  double index = (pct / 100.0) * static_cast<double>(values.size() - 1);
  size_t left = static_cast<size_t>(std::floor(index));
  size_t right = static_cast<size_t>(std::ceil(index));
  if (left == right) {
    return values[left];
  }
  double frac = index - static_cast<double>(left);
  return values[left] * (1.0 - frac) + values[right] * frac;
}

size_t BytesFromGiB(double gib) {
  if (gib <= 0.0) {
    return 0;
  }
  return static_cast<size_t>(gib * 1024.0 * 1024.0 * 1024.0);
}

void LoadModelConfig(const json& config, std::vector<std::string>& model_dirs_list,
                     std::unordered_map<int, int>& model_id_to_index) {
  model_dirs_list.clear();
  model_id_to_index.clear();

  if (config.contains("model_dirs")) {
    model_dirs_list = config["model_dirs"].get<std::vector<std::string>>();
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
      model_id_to_index[model.id] = static_cast<int>(model_dirs_list.size());
      model_dirs_list.push_back(model.path);
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
                     size_t model_count,
                     std::vector<TraceRequest>& trace_requests) {
  std::ifstream file(req_file_path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open request file: " + req_file_path);
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

    double arrival_ts = static_cast<double>(trace_requests.size());
    int model_id = 0;
    if (fields.size() == 1) {
      model_id = std::stoi(fields[0]);
    } else {
      arrival_ts = std::stod(fields[0]);
      model_id = std::stoi(fields[1]);
    }

    try {
      int model_index =
          ResolveModelRequest(model_id, model_id_to_index, model_count);
      trace_requests.push_back({arrival_ts, model_id, model_index});
    } catch (const std::exception& e) {
      throw std::runtime_error("Invalid request at line " +
                               std::to_string(line_no) + ": " + e.what());
    }
  }
}

void TruncateRequests(size_t max_requests, std::vector<TraceRequest>& trace_requests) {
  if (max_requests == 0 || trace_requests.size() <= max_requests) {
    return;
  }
  trace_requests.resize(max_requests);
}

std::vector<int> BuildGpuIds(int gpu_id, int gpu_num) {
  std::vector<int> gpu_ids;
  if (gpu_num <= 0) {
    gpu_ids.push_back(gpu_id);
    return gpu_ids;
  }
  for (int i = 0; i < gpu_num; ++i) {
    gpu_ids.push_back(gpu_id + i);
  }
  return gpu_ids;
}

ManagedMemorySupport QueryManagedMemorySupport(int device_id) {
  ManagedMemorySupport support;
  support.device_id = device_id;
  cudaDeviceGetAttribute(&support.managed_memory, cudaDevAttrManagedMemory,
                         device_id);
  cudaDeviceGetAttribute(&support.concurrent_managed_access,
                         cudaDevAttrConcurrentManagedAccess, device_id);
  cudaDeviceGetAttribute(&support.pageable_memory_access,
                         cudaDevAttrPageableMemoryAccess, device_id);
  cudaDeviceGetAttribute(&support.pageable_memory_access_uses_host_page_tables,
                         cudaDevAttrPageableMemoryAccessUsesHostPageTables,
                         device_id);
  return support;
}

void ValidateManagedMemorySupport(const BenchOptions& options) {
  for (int gpu_id : options.gpu_ids) {
    ManagedMemorySupport support = QueryManagedMemorySupport(gpu_id);
    std::cout << "[um-support] device=" << support.device_id
              << " managed_memory=" << support.managed_memory
              << " concurrent_managed_access="
              << support.concurrent_managed_access
              << " pageable_memory_access=" << support.pageable_memory_access
              << " pageable_memory_access_uses_host_page_tables="
              << support.pageable_memory_access_uses_host_page_tables
              << std::endl;
    if (support.managed_memory != 1 ||
        support.concurrent_managed_access != 1) {
      throw std::runtime_error(
          "UM+prefetch baseline requires cudaDevAttrConcurrentManagedAccess=1, "
          "managed_memory=1");
    }
  }
}

void ParseArgs(int argc, char* argv[], BenchOptions& options) {
  options.config_path = "configs/model_config.json";
  options.req_file_path = "evaluation/traces/servegen_tangram.trace";
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--config") {
      options.config_path = argv[++i];
    } else if (arg == "--req_file_path") {
      options.req_file_path = argv[++i];
    } else if (arg == "--mode") {
      options.mode = argv[++i];
    } else if (arg == "--scenario") {
      options.scenario = argv[++i];
    } else if (arg == "--gpu") {
      options.gpu_id = std::stoi(argv[++i]);
    } else if (arg == "--gpu_num") {
      options.gpu_num = std::stoi(argv[++i]);
    } else if (arg == "--max_requests") {
      options.max_requests = std::stoull(argv[++i]);
    } else if (arg == "--warmup_step") {
      options.warmup_step = std::stoull(argv[++i]);
    } else if (arg == "--load_cdf_path") {
      options.load_cdf_path = argv[++i];
    } else if (arg == "--request_csv") {
      options.request_csv_path = argv[++i];
    } else if (arg == "--summary_csv") {
      options.summary_csv_path = argv[++i];
    } else if (arg == "--random_seed") {
      options.random_seed = static_cast<unsigned int>(std::stoul(argv[++i]));
    } else if (arg == "--verbose") {
      options.verbose = std::stoi(argv[++i]) != 0;
    } else if (arg == "--progress_interval") {
      options.progress_interval = std::stoull(argv[++i]);
    } else if (arg == "--prepare_threads") {
      options.prepare_threads = std::max<size_t>(1, std::stoull(argv[++i]));
    } else if (arg == "--max_gpu_memory_gb") {
      options.max_gpu_memory_gb = std::stod(argv[++i]);
    } else if (arg == "--gpu_assignment_mode") {
      options.gpu_assignment_mode = argv[++i];
    } else {
      throw std::invalid_argument("Unknown argument: " + arg);
    }
  }

  if (options.mode != "explicit-copy" && options.mode != "um-prefetch") {
    throw std::invalid_argument("--mode must be explicit-copy or um-prefetch");
  }
  if (options.scenario != "flush-each-time" &&
      options.scenario != "warm-repeat") {
    throw std::invalid_argument(
        "--scenario must be flush-each-time or warm-repeat");
  }
  if (options.gpu_assignment_mode != "fixed_model_owner" &&
      options.gpu_assignment_mode != "random_request" &&
      options.gpu_assignment_mode != "round_robin" &&
      options.gpu_assignment_mode != "sticky") {
    throw std::invalid_argument(
        "--gpu_assignment_mode must be fixed_model_owner, random_request, round_robin, or sticky");
  }
  options.gpu_ids = BuildGpuIds(options.gpu_id, options.gpu_num);
}

int WriteLoadLatencyCDF(const std::string& load_cdf_path,
                        std::vector<double> load_latencies_ms) {
  if (load_cdf_path.empty()) {
    return 0;
  }
  std::ofstream file(load_cdf_path);
  if (!file.is_open()) {
    std::cerr << "Failed to open load CDF file: " << load_cdf_path << std::endl;
    return 1;
  }
  std::sort(load_latencies_ms.begin(), load_latencies_ms.end());
  file << "latency_ms,cdf\n";
  for (size_t i = 0; i < load_latencies_ms.size(); ++i) {
    double cdf =
        static_cast<double>(i + 1) / static_cast<double>(load_latencies_ms.size());
    file << std::fixed << std::setprecision(6) << load_latencies_ms[i] << ","
         << cdf << "\n";
  }
  return 0;
}

void ParallelCopyToManaged(BenchModel& model, size_t prepare_threads) {
  const auto& tg_indexes = model.registered_model->GetTensorGroupIndexes();
  auto host_ptrs = model.registered_model->GetTensorGroupHostPtr();
  char* managed_base = static_cast<char*>(model.managed_ptr);
  size_t num_threads =
      std::max<size_t>(1, std::min(prepare_threads, tg_indexes.size()));
  std::vector<std::future<void>> futures;
  futures.reserve(num_threads);

  for (size_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
    futures.emplace_back(std::async(std::launch::async, [&, thread_idx]() {
      for (size_t tg_idx = thread_idx; tg_idx < tg_indexes.size();
           tg_idx += num_threads) {
        void* host_ptr = host_ptrs->get(static_cast<int>(tg_idx));
        if (host_ptr == nullptr) {
          throw std::runtime_error("Host tensor group missing while preparing UM "
                                   "buffer for " +
                                   model.model_path);
        }
        std::memcpy(managed_base + tg_indexes[tg_idx].file_offset, host_ptr,
                    tg_indexes[tg_idx].size);
      }
    }));
  }
  for (auto& future : futures) {
    future.get();
  }
}

PrepareManagedStats PrepareManagedBuffer(BenchModel& model,
                                         const std::vector<int>& gpu_ids,
                                         size_t prepare_threads) {
  PrepareManagedStats stats;
  if (model.managed_ptr != nullptr) {
    return stats;
  }

  auto total_start = std::chrono::high_resolution_clock::now();

  int load_ret = model.registered_model->LoadModelFromDisk(
      static_cast<int>(std::max<size_t>(1, prepare_threads)));
  if (load_ret != 0) {
    throw std::runtime_error("LoadModelFromDisk failed for " + model.model_path);
  }

  auto alloc_start = std::chrono::high_resolution_clock::now();
  cudaError_t err = cudaMallocManaged(&model.managed_ptr, model.model_size);
  if (err != cudaSuccess) {
    throw std::runtime_error("cudaMallocManaged failed for " + model.model_path +
                             ": " + cudaGetErrorString(err));
  }
  auto alloc_end = std::chrono::high_resolution_clock::now();

  auto copy_start = std::chrono::high_resolution_clock::now();
  ParallelCopyToManaged(model, prepare_threads);
  auto copy_end = std::chrono::high_resolution_clock::now();

  auto advise_start = std::chrono::high_resolution_clock::now();
  err = cudaMemAdvise(model.managed_ptr, model.model_size,
                      cudaMemAdviseSetReadMostly, cudaCpuDeviceId);
  if (err != cudaSuccess) {
    throw std::runtime_error("cudaMemAdvise(SetReadMostly) failed for " +
                             model.model_path + ": " +
                             cudaGetErrorString(err));
  }
  err = cudaMemAdvise(model.managed_ptr, model.model_size,
                      cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
  if (err != cudaSuccess) {
    throw std::runtime_error("cudaMemAdvise(SetPreferredLocation, CPU) failed for " +
                             model.model_path + ": " +
                             cudaGetErrorString(err));
  }
  for (int gpu_id : gpu_ids) {
    err = cudaMemAdvise(model.managed_ptr, model.model_size,
                        cudaMemAdviseSetAccessedBy, gpu_id);
    if (err != cudaSuccess) {
      throw std::runtime_error("cudaMemAdvise(SetAccessedBy) failed for " +
                               model.model_path + ": " +
                               cudaGetErrorString(err));
    }
  }
  auto advise_end = std::chrono::high_resolution_clock::now();

  stats.alloc_ms =
      std::chrono::duration_cast<std::chrono::microseconds>(alloc_end - alloc_start)
          .count() /
      1000.0;
  stats.copy_ms =
      std::chrono::duration_cast<std::chrono::microseconds>(copy_end - copy_start)
          .count() /
      1000.0;
  stats.advise_ms =
      std::chrono::duration_cast<std::chrono::microseconds>(advise_end - advise_start)
          .count() /
      1000.0;
  stats.total_ms =
      std::chrono::duration_cast<std::chrono::microseconds>(advise_end - total_start)
          .count() /
      1000.0;
  stats.prepared = true;
  return stats;
}

void FlushExplicitDevice(BenchModel& model) {
  if (model.explicit_device_ptr == nullptr) {
    model.explicit_resident_gpu = -1;
    return;
  }
  if (model.explicit_resident_gpu >= 0) {
    cudaSetDevice(model.explicit_resident_gpu);
  }
  cudaFree(model.explicit_device_ptr);
  model.explicit_device_ptr = nullptr;
  model.explicit_resident_gpu = -1;
}

void FlushManagedToCPU(BenchModel& model) {
  if (model.managed_ptr == nullptr) {
    model.managed_resident_gpu = -1;
    return;
  }
  cudaError_t err =
      cudaMemPrefetchAsync(model.managed_ptr, model.model_size, cudaCpuDeviceId, nullptr);
  if (err != cudaSuccess) {
    throw std::runtime_error("cudaMemPrefetchAsync to CPU failed for " +
                             model.model_path + ": " +
                             cudaGetErrorString(err));
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    throw std::runtime_error("cudaDeviceSynchronize failed after CPU prefetch: " +
                             std::string(cudaGetErrorString(err)));
  }
  model.managed_resident_gpu = -1;
}

size_t ExplicitResidentBytesOnGpu(const std::vector<BenchModel>& models, int target_gpu) {
  size_t total = 0;
  for (const auto& model : models) {
    if (model.explicit_resident_gpu == target_gpu &&
        model.explicit_device_ptr != nullptr) {
      total += model.model_size;
    }
  }
  return total;
}

size_t ManagedResidentBytesOnGpu(const std::vector<BenchModel>& models, int target_gpu) {
  size_t total = 0;
  for (const auto& model : models) {
    if (model.managed_resident_gpu == target_gpu && model.managed_ptr != nullptr) {
      total += model.model_size;
    }
  }
  return total;
}

std::tuple<size_t, size_t> EstimateManagedIoBytes(BenchModel& model,
                                                  double prefetch_ms,
                                                  bool full_transfer_hint) {
  if (full_transfer_hint) {
    if (prefetch_ms > 0.0 &&
        (model.best_full_prefetch_ms <= 0.0 ||
         prefetch_ms < model.best_full_prefetch_ms)) {
      model.best_full_prefetch_ms = prefetch_ms;
    }
    return {model.model_size, 0};
  }

  if (model.best_full_prefetch_ms <= 0.0) {
    return {0, model.model_size};
  }

  double moved_ratio = prefetch_ms / model.best_full_prefetch_ms;
  moved_ratio = std::clamp(moved_ratio, 0.0, 1.0);
  size_t moved_bytes = static_cast<size_t>(
      moved_ratio * static_cast<double>(model.model_size));
  size_t cached_bytes = model.model_size - moved_bytes;
  return {moved_bytes, cached_bytes};
}

void EnsureExplicitCapacity(size_t required_bytes, std::vector<BenchModel>& models,
                            int target_gpu, size_t max_gpu_memory_bytes,
                            bool verbose) {
  if (max_gpu_memory_bytes == 0) {
    return;
  }
  if (required_bytes > max_gpu_memory_bytes) {
    std::ostringstream oss;
    oss << "Requested model exceeds configured GPU memory limit: required_bytes="
        << required_bytes
        << ", max_gpu_memory_bytes=" << max_gpu_memory_bytes;
    throw std::runtime_error(oss.str());
  }

  size_t resident_bytes = ExplicitResidentBytesOnGpu(models, target_gpu);
  if (resident_bytes + required_bytes <= max_gpu_memory_bytes) {
    return;
  }

  if (verbose) {
    std::cout << "[capacity] explicit gpu=" << target_gpu
              << " resident_bytes=" << resident_bytes
              << " required_bytes=" << required_bytes
              << " max_gpu_memory_bytes=" << max_gpu_memory_bytes
              << " -> evicting explicit-copy residents on target gpu"
              << std::endl;
  }

  for (auto& model : models) {
    if (model.explicit_resident_gpu == target_gpu) {
      FlushExplicitDevice(model);
    }
  }

  size_t free_bytes = max_gpu_memory_bytes -
                      ExplicitResidentBytesOnGpu(models, target_gpu);
  if (free_bytes < required_bytes) {
    std::ostringstream oss;
    oss << "Insufficient GPU memory even after evicting all explicit-copy "
           "residents: free_bytes="
        << free_bytes << ", required_bytes=" << required_bytes;
    throw std::runtime_error(oss.str());
  }
}

void EnsureManagedCapacity(size_t required_bytes, std::vector<BenchModel>& models,
                           int target_gpu, size_t max_gpu_memory_bytes,
                           bool verbose) {
  if (max_gpu_memory_bytes == 0) {
    return;
  }
  if (required_bytes > max_gpu_memory_bytes) {
    std::ostringstream oss;
    oss << "Requested model exceeds configured GPU memory limit: required_bytes="
        << required_bytes
        << ", max_gpu_memory_bytes=" << max_gpu_memory_bytes;
    throw std::runtime_error(oss.str());
  }

  size_t resident_bytes = ManagedResidentBytesOnGpu(models, target_gpu);
  if (resident_bytes + required_bytes <= max_gpu_memory_bytes) {
    return;
  }

  if (verbose) {
    std::cout << "[capacity] managed gpu=" << target_gpu
              << " resident_bytes=" << resident_bytes
              << " required_bytes=" << required_bytes
              << " max_gpu_memory_bytes=" << max_gpu_memory_bytes
              << " -> prefetching UM residents on target gpu back to CPU"
              << std::endl;
  }

  for (auto& model : models) {
    if (model.managed_resident_gpu == target_gpu) {
      FlushManagedToCPU(model);
    }
  }
}

int ChooseTargetGpu(const BenchOptions& options, const BenchModel& model,
                    size_t request_idx) {
  if (options.gpu_ids.empty()) {
    return options.gpu_id;
  }
  if (options.gpu_ids.size() == 1) {
    return options.gpu_ids.front();
  }
  if (options.gpu_assignment_mode == "fixed_model_owner") {
    size_t idx = static_cast<size_t>(model.model_index) % options.gpu_ids.size();
    return options.gpu_ids[idx];
  }
  if (options.gpu_assignment_mode == "sticky") {
    if (model.last_target_gpu >= 0) {
      for (int gpu_id : options.gpu_ids) {
        if (gpu_id == model.last_target_gpu) {
          return gpu_id;
        }
      }
    }
    size_t idx = request_idx % options.gpu_ids.size();
    return options.gpu_ids[idx];
  }
  if (options.gpu_assignment_mode == "round_robin") {
    size_t idx = request_idx % options.gpu_ids.size();
    return options.gpu_ids[idx];
  }
  size_t idx = static_cast<size_t>(std::rand()) % options.gpu_ids.size();
  return options.gpu_ids[idx];
}

BenchResult RunExplicitCopy(BenchModel& model, std::vector<BenchModel>& models,
                            const BenchOptions& options, size_t request_idx,
                            bool is_warmup) {
  int target_gpu = ChooseTargetGpu(options, model, request_idx);
  if (options.scenario == "flush-each-time" &&
      model.explicit_resident_gpu == target_gpu &&
      model.explicit_device_ptr != nullptr) {
    FlushExplicitDevice(model);
  }

  bool full_transfer_hint = model.explicit_resident_gpu != target_gpu;
  size_t bytes_moved = full_transfer_hint ? model.model_size : 0;
  size_t cached_bytes = full_transfer_hint ? 0 : model.model_size;
  auto start = std::chrono::high_resolution_clock::now();

  if (full_transfer_hint) {
    if (options.scenario == "warm-repeat" || options.max_gpu_memory_gb > 0.0) {
      EnsureExplicitCapacity(model.model_size, models, target_gpu,
                             BytesFromGiB(options.max_gpu_memory_gb),
                             options.verbose);
    }
    if (model.explicit_device_ptr != nullptr) {
      FlushExplicitDevice(model);
    }
    cudaSetDevice(target_gpu);
    cudaError_t err = cudaMalloc(&model.explicit_device_ptr, model.model_size);
    if (err != cudaSuccess) {
      throw std::runtime_error("cudaMalloc failed for " + model.model_path + ": " +
                               cudaGetErrorString(err));
    }

    char* device_base = static_cast<char*>(model.explicit_device_ptr);
    const auto& tg_indexes = model.registered_model->GetTensorGroupIndexes();
    auto host_ptrs = model.registered_model->GetTensorGroupHostPtr();
    int disk_ret = model.registered_model->LoadModelFromDisk(
        static_cast<int>(std::max<size_t>(1, options.prepare_threads)));
    if (disk_ret != 0) {
      throw std::runtime_error("LoadModelFromDisk failed for " + model.model_path);
    }
    for (size_t i = 0; i < tg_indexes.size(); i++) {
      void* host_ptr = host_ptrs->get(static_cast<int>(i));
      if (host_ptr == nullptr) {
        throw std::runtime_error("Host tensor group missing for " +
                                 model.model_path);
      }
      err = cudaMemcpy(device_base + tg_indexes[i].file_offset, host_ptr,
                       tg_indexes[i].size, cudaMemcpyHostToDevice);
      if (err != cudaSuccess) {
        throw std::runtime_error("cudaMemcpy failed for " + model.model_path +
                                 ": " + cudaGetErrorString(err));
      }
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      throw std::runtime_error("cudaDeviceSynchronize failed after explicit copy: " +
                               std::string(cudaGetErrorString(err)));
    }
    model.explicit_resident_gpu = target_gpu;
  }

  auto end = std::chrono::high_resolution_clock::now();
  double load_ms =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count() /
      1000.0;

  BenchResult result;
  result.request_idx = request_idx;
  result.model_id = model.model_id;
  result.model_index = model.model_index;
  result.model_path = model.model_path;
  result.mode = options.mode;
  result.scenario = options.scenario;
  result.gpu_id = target_gpu;
  result.model_bytes = model.model_size;
  result.bytes_moved_est = bytes_moved;
  result.cached_bytes_est = cached_bytes;
  result.load_ms = load_ms;
  result.is_warmup = is_warmup ? 1 : 0;
  result.is_full_model_transfer = bytes_moved == model.model_size ? 1 : 0;
  model.last_target_gpu = target_gpu;
  return result;
}

BenchResult RunUnifiedMemory(BenchModel& model, std::vector<BenchModel>& models,
                             const BenchOptions& options, size_t request_idx,
                             bool is_warmup) {
  int target_gpu = ChooseTargetGpu(options, model, request_idx);
  auto total_start = std::chrono::high_resolution_clock::now();
  PrepareManagedStats prepare_stats =
      PrepareManagedBuffer(model, options.gpu_ids, options.prepare_threads);

  if (options.scenario == "flush-each-time") {
    if (model.managed_resident_gpu == target_gpu) {
      FlushManagedToCPU(model);
    }
  } else if (options.max_gpu_memory_gb > 0.0) {
    EnsureManagedCapacity(model.model_size, models, target_gpu,
                          BytesFromGiB(options.max_gpu_memory_gb),
                          options.verbose);
  }

  bool full_transfer_hint = model.managed_resident_gpu != target_gpu;
  auto prefetch_start = std::chrono::high_resolution_clock::now();
  cudaSetDevice(target_gpu);
  cudaError_t err = cudaMemPrefetchAsync(model.managed_ptr, model.model_size,
                                         target_gpu, nullptr);
  if (err != cudaSuccess) {
    throw std::runtime_error("cudaMemPrefetchAsync to GPU failed for " +
                             model.model_path + ": " +
                             cudaGetErrorString(err));
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    throw std::runtime_error("cudaDeviceSynchronize failed after GPU prefetch: " +
                             std::string(cudaGetErrorString(err)));
  }
  auto prefetch_end = std::chrono::high_resolution_clock::now();
  double prefetch_ms =
      std::chrono::duration_cast<std::chrono::microseconds>(prefetch_end -
                                                            prefetch_start)
          .count() /
      1000.0;

  model.managed_resident_gpu = target_gpu;
  auto [bytes_moved, cached_bytes] =
      EstimateManagedIoBytes(model, prefetch_ms, full_transfer_hint);

  BenchResult result;
  result.request_idx = request_idx;
  result.model_id = model.model_id;
  result.model_index = model.model_index;
  result.model_path = model.model_path;
  result.mode = options.mode;
  result.scenario = options.scenario;
  result.gpu_id = target_gpu;
  result.model_bytes = model.model_size;
  result.bytes_moved_est = bytes_moved;
  result.cached_bytes_est = cached_bytes;
  result.prepare_managed_ms = prepare_stats.total_ms;
  result.prefetch_ms = prefetch_ms;
  result.load_ms =
      std::chrono::duration_cast<std::chrono::microseconds>(prefetch_end -
                                                            total_start)
          .count() /
      1000.0;
  result.is_warmup = is_warmup ? 1 : 0;
  result.is_full_model_transfer = bytes_moved == model.model_size ? 1 : 0;
  model.last_target_gpu = target_gpu;
  return result;
}

void WriteRequestCsv(const std::string& path, const std::vector<BenchResult>& results) {
  if (path.empty()) {
    return;
  }
  std::ofstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open request csv: " + path);
  }
  file << "request_idx,model_id,model_index,model_path,mode,scenario,gpu_id,"
          "model_bytes,bytes_moved_est,cached_bytes_est,load_ms,prepare_managed_ms,prefetch_ms,"
          "is_warmup,is_full_model_transfer\n";
  for (const auto& result : results) {
    file << result.request_idx << "," << result.model_id << ","
         << result.model_index << "," << result.model_path << ","
         << result.mode << "," << result.scenario << "," << result.gpu_id
         << "," << result.model_bytes << "," << result.bytes_moved_est << ","
         << result.cached_bytes_est << "," << std::fixed << std::setprecision(6)
         << result.load_ms << "," << result.prepare_managed_ms << ","
         << result.prefetch_ms << "," << result.is_warmup << ","
         << result.is_full_model_transfer << "\n";
  }
}

void WriteSummaryCsv(const std::string& path, const std::vector<BenchResult>& results,
                     const BenchOptions& options) {
  if (path.empty()) {
    return;
  }
  std::ofstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open summary csv: " + path);
  }

  std::vector<double> load_latencies;
  std::vector<double> prefetch_latencies;
  double avg_load_ms = 0.0;
  double avg_prepare_ms = 0.0;
  double avg_prefetch_ms = 0.0;
  size_t total_bytes = 0;
  size_t total_cached_bytes = 0;
  size_t total_model_bytes = 0;
  double max_load_ms = 0.0;
  for (const auto& result : results) {
    load_latencies.push_back(result.load_ms);
    prefetch_latencies.push_back(result.prefetch_ms);
    avg_load_ms += result.load_ms;
    avg_prepare_ms += result.prepare_managed_ms;
    avg_prefetch_ms += result.prefetch_ms;
    total_bytes += result.bytes_moved_est;
    total_cached_bytes += result.cached_bytes_est;
    total_model_bytes += result.model_bytes;
    max_load_ms = std::max(max_load_ms, result.load_ms);
  }
  if (!results.empty()) {
    avg_load_ms /= static_cast<double>(results.size());
    avg_prepare_ms /= static_cast<double>(results.size());
    avg_prefetch_ms /= static_cast<double>(results.size());
  }

  file << "mode,scenario,num_requests,avg_load_ms,avg_prepare_managed_ms,"
          "avg_prefetch_ms,p50_load_ms,p95_load_ms,p99_load_ms,"
          "p50_prefetch_ms,p95_prefetch_ms,p99_prefetch_ms,max_load_ms,"
          "avg_bytes_moved,total_bytes_moved,reduced_io\n";
  file << options.mode << "," << options.scenario << "," << results.size() << ","
       << std::fixed << std::setprecision(6) << avg_load_ms << ","
       << avg_prepare_ms << "," << avg_prefetch_ms << ","
       << Percentile(load_latencies, 50.0) << ","
       << Percentile(load_latencies, 95.0) << ","
       << Percentile(load_latencies, 99.0) << ","
       << Percentile(prefetch_latencies, 50.0) << ","
       << Percentile(prefetch_latencies, 95.0) << ","
       << Percentile(prefetch_latencies, 99.0) << ","
       << max_load_ms << ","
       << (results.empty() ? 0.0
                           : static_cast<double>(total_bytes) /
                                 static_cast<double>(results.size()))
       << "," << total_bytes << ","
       << (total_model_bytes == 0
               ? 0.0
               : static_cast<double>(total_cached_bytes) /
                     static_cast<double>(total_model_bytes))
       << "\n";
}

std::vector<BenchModel> BuildBenchModels(
    const std::vector<std::string>& model_dirs_list) {
  std::vector<BenchModel> models;
  models.reserve(model_dirs_list.size());
  for (size_t i = 0; i < model_dirs_list.size(); ++i) {
    BenchModel model;
    model.model_id = static_cast<int>(i);
    model.model_index = static_cast<int>(i);
    model.model_path = model_dirs_list[i];
    // Temporary workaround: keep original tensor-group granularity here so
    // multi-partition models do not collapse into one oversized disk read.
    model.registered_model =
        std::make_shared<RegisteredModel>(model.model_path, 1.0, 1);
    model.model_size = model.registered_model->model_size();
    models.push_back(std::move(model));
  }
  return models;
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    BenchOptions options;
    ParseArgs(argc, argv, options);
    std::srand(options.random_seed);

    if (options.mode == "um-prefetch") {
      ValidateManagedMemorySupport(options);
    }

    std::ifstream config_stream(options.config_path);
    if (!config_stream.is_open()) {
      throw std::runtime_error("Failed to open config file: " +
                               options.config_path);
    }
    json config;
    config_stream >> config;

    std::vector<std::string> model_dirs_list;
    std::unordered_map<int, int> model_id_to_index;
    LoadModelConfig(config, model_dirs_list, model_id_to_index);

    std::vector<TraceRequest> trace_requests;
    LoadRequestFile(options.req_file_path, model_id_to_index,
                    model_dirs_list.size(), trace_requests);
    TruncateRequests(options.max_requests, trace_requests);

    std::vector<BenchModel> models = BuildBenchModels(model_dirs_list);
    std::vector<BenchResult> results;
    results.reserve(trace_requests.size());
    std::unordered_map<std::string, size_t> model_cached_bytes;
    std::unordered_map<std::string, size_t> model_total_bytes;

    size_t warmup_requests = std::min(options.warmup_step, trace_requests.size());
    for (size_t i = 0; i < trace_requests.size(); ++i) {
      const auto& request = trace_requests[i];
      bool is_warmup = i < warmup_requests;
      BenchModel& model = models.at(static_cast<size_t>(request.model_index));

      if (options.progress_interval > 0 &&
          (i == 0 || (i + 1) % options.progress_interval == 0 ||
           i + 1 == trace_requests.size())) {
        std::cout << "[progress] request " << (i + 1) << "/"
                  << trace_requests.size() << " mode=" << options.mode
                  << " scenario=" << options.scenario
                  << " model=" << model.model_path
                  << " warmup=" << (is_warmup ? 1 : 0) << std::endl;
      }

      BenchResult result;
      try {
        if (options.mode == "explicit-copy") {
          result = RunExplicitCopy(model, models, options, i, is_warmup);
        } else {
          result = RunUnifiedMemory(model, models, options, i, is_warmup);
        }
      } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "Benchmark failed at request_idx=" << i
            << ", model_id=" << request.model_id
            << ", model_index=" << request.model_index
            << ", model_path=" << model.model_path << ", mode=" << options.mode
            << ", scenario=" << options.scenario << ": " << e.what();
        throw std::runtime_error(oss.str());
      }

      if (!is_warmup) {
        results.push_back(result);
        model_cached_bytes[result.model_path] += result.cached_bytes_est;
        model_total_bytes[result.model_path] += result.model_bytes;
      }

      if (options.verbose) {
        std::cout << "[request] idx=" << i
                  << " mode=" << result.mode
                  << " scenario=" << result.scenario
                  << " gpu=" << result.gpu_id
                  << " model=" << result.model_path
                  << " bytes_moved_est=" << result.bytes_moved_est
                  << " cached_bytes_est=" << result.cached_bytes_est
                  << " prepare_managed_ms=" << result.prepare_managed_ms
                  << " prefetch_ms=" << result.prefetch_ms
                  << " load_ms=" << result.load_ms << std::endl;
      }
    }

    WriteRequestCsv(options.request_csv_path, results);
    WriteSummaryCsv(options.summary_csv_path, results, options);

    std::vector<double> latency_samples;
    std::vector<double> prefetch_samples;
    double avg_ms = 0.0;
    double avg_prepare_ms = 0.0;
    double avg_prefetch_ms = 0.0;
    size_t total_bytes = 0;
    size_t total_cached_bytes = 0;
    size_t total_model_bytes = 0;
    for (const auto& result : results) {
      latency_samples.push_back(result.load_ms);
      prefetch_samples.push_back(result.prefetch_ms);
      avg_ms += result.load_ms;
      avg_prepare_ms += result.prepare_managed_ms;
      avg_prefetch_ms += result.prefetch_ms;
      total_bytes += result.bytes_moved_est;
      total_cached_bytes += result.cached_bytes_est;
      total_model_bytes += result.model_bytes;
    }
    if (!results.empty()) {
      avg_ms /= static_cast<double>(results.size());
      avg_prepare_ms /= static_cast<double>(results.size());
      avg_prefetch_ms /= static_cast<double>(results.size());
    }

    std::cout << "================ UM Prefetch Benchmark Summary ================\n";
    std::cout << "mode: " << options.mode << "\n";
    std::cout << "scenario: " << options.scenario << "\n";
    std::cout << "requests: " << results.size() << "\n";
    std::cout << "avg_load_ms: " << std::fixed << std::setprecision(6) << avg_ms
              << "\n";
    std::cout << "avg_prepare_managed_ms: " << avg_prepare_ms << "\n";
    std::cout << "avg_prefetch_ms: " << avg_prefetch_ms << "\n";
    std::cout << "p50_load_ms: " << Percentile(latency_samples, 50.0) << "\n";
    std::cout << "p95_load_ms: " << Percentile(latency_samples, 95.0) << "\n";
    std::cout << "p99_load_ms: " << Percentile(latency_samples, 99.0) << "\n";
    std::cout << "p50_prefetch_ms: "
              << Percentile(prefetch_samples, 50.0) << "\n";
    std::cout << "p95_prefetch_ms: "
              << Percentile(prefetch_samples, 95.0) << "\n";
    std::cout << "p99_prefetch_ms: "
              << Percentile(prefetch_samples, 99.0) << "\n";
    std::cout << "avg_bytes_moved: "
              << (results.empty() ? 0.0
                                  : static_cast<double>(total_bytes) /
                                        static_cast<double>(results.size()))
              << "\n";
    std::cout << "total_bytes_moved: " << total_bytes << "\n";
    std::cout << "==================================\n";
    std::cout << "Reduced IO: " << total_cached_bytes << " / " << total_model_bytes
              << " = "
              << (total_model_bytes == 0
                      ? 0.0
                      : static_cast<double>(total_cached_bytes) /
                            static_cast<double>(total_model_bytes))
              << "\n";
    for (const auto& pair : model_total_bytes) {
      size_t cached_bytes = model_cached_bytes[pair.first];
      std::cout << "Model: " << pair.first << " Reduced IO: " << cached_bytes
                << " / " << pair.second << " = "
                << (pair.second == 0
                        ? 0.0
                        : static_cast<double>(cached_bytes) /
                              static_cast<double>(pair.second))
                << "\n";
    }

    if (WriteLoadLatencyCDF(options.load_cdf_path, latency_samples) != 0) {
      return 1;
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
