#include "vram_manager.h"

#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct TangramVRAMHandle {
  std::unique_ptr<VRAMManager> manager;
  std::unordered_map<int, std::string> model_paths;
  bool mock_copy{true};
  int free_strategy{1};
  int allocate_strategy{4};
  bool disable_parameter_reuse{false};
  double load_bandwidth_gbps{20.0};
  double load_overhead_ms{0.0};
  std::string last_error;
};

double EstimateLoadMs(uint64_t to_load_bytes, double bandwidth_gbps,
                      double overhead_ms) {
  if (to_load_bytes == 0 || bandwidth_gbps <= 0.0) {
    return 0.0;
  }
  return (static_cast<double>(to_load_bytes) /
          (bandwidth_gbps * 1000.0 * 1000.0 * 1000.0)) *
             1000.0 +
         overhead_ms;
}

void SetError(TangramVRAMHandle* handle, const std::string& error) {
  if (handle) {
    handle->last_error = error;
  }
}

}  // namespace

extern "C" {

struct TangramVRAMEstimate {
  uint64_t cached_bytes;
  uint64_t to_load_bytes;
  uint64_t total_model_bytes;
  double load_ms;
  int full_model_hit;
};

struct TangramVRAMLoadResult {
  uint64_t cached_bytes;
  uint64_t to_load_bytes;
  uint64_t total_model_bytes;
  double estimated_load_ms;
  double wall_load_ms;
  int full_model_hit;
};

TangramVRAMHandle* tangram_vram_create(
    int num_gpus, uint64_t gpu_pool_size_bytes, double gpu_bandwidth_bytes,
    double cpu_bandwidth_bytes, int mock_copy, double load_bandwidth_gbps,
    double load_overhead_ms, int free_strategy, int allocate_strategy,
    int disable_parameter_reuse) {
  auto handle = std::make_unique<TangramVRAMHandle>();
  try {
    std::vector<int> gpu_ids;
    gpu_ids.reserve(num_gpus);
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      gpu_ids.push_back(gpu_id);
    }
    handle->mock_copy = mock_copy != 0;
    handle->free_strategy = free_strategy;
    handle->allocate_strategy = allocate_strategy;
    handle->disable_parameter_reuse = disable_parameter_reuse != 0;
    handle->load_bandwidth_gbps = load_bandwidth_gbps;
    handle->load_overhead_ms = load_overhead_ms;
    handle->manager = std::make_unique<VRAMManager>(
        static_cast<size_t>(gpu_pool_size_bytes), gpu_ids, gpu_bandwidth_bytes,
        cpu_bandwidth_bytes, handle->mock_copy);
  } catch (const std::exception& e) {
    handle->last_error = e.what();
  } catch (...) {
    handle->last_error = "unknown error during tangram_vram_create";
  }
  return handle.release();
}

void tangram_vram_destroy(TangramVRAMHandle* handle) { delete handle; }

const char* tangram_vram_last_error(TangramVRAMHandle* handle) {
  if (!handle) {
    return "null TangramVRAMHandle";
  }
  return handle->last_error.c_str();
}

int tangram_vram_register_model(TangramVRAMHandle* handle, int model_id,
                                const char* model_path, double sensitive,
                                int reuse_granularity) {
  if (!handle || !handle->manager || !model_path) {
    SetError(handle, "invalid arguments to tangram_vram_register_model");
    return -1;
  }
  try {
    const std::string path(model_path);
    int64_t model_size = handle->manager->RegisterModel(
        path, sensitive, handle->mock_copy, reuse_granularity);
    if (model_size < 0) {
      SetError(handle, "VRAMManager::RegisterModel failed for " + path);
      return -1;
    }
    handle->model_paths[model_id] = path;
    return 0;
  } catch (const std::exception& e) {
    SetError(handle, e.what());
    return -1;
  } catch (...) {
    SetError(handle, "unknown error during tangram_vram_register_model");
    return -1;
  }
}

int tangram_vram_estimate(TangramVRAMHandle* handle, int model_id, int gpu_id,
                          TangramVRAMEstimate* out) {
  if (!handle || !handle->manager || !out) {
    SetError(handle, "invalid arguments to tangram_vram_estimate");
    return -1;
  }
  auto path_it = handle->model_paths.find(model_id);
  if (path_it == handle->model_paths.end()) {
    SetError(handle, "unknown model id in tangram_vram_estimate");
    return -1;
  }

  size_t cached = 0;
  size_t to_load = 0;
  size_t total = 0;
  if (!handle->manager->EstimateModelLoad(
          path_it->second, gpu_id, cached, to_load, total,
          handle->disable_parameter_reuse)) {
    SetError(handle, "VRAMManager::EstimateModelLoad failed");
    return -1;
  }

  out->cached_bytes = static_cast<uint64_t>(cached);
  out->to_load_bytes = static_cast<uint64_t>(to_load);
  out->total_model_bytes = static_cast<uint64_t>(total);
  out->load_ms =
      EstimateLoadMs(out->to_load_bytes, handle->load_bandwidth_gbps,
                     handle->load_overhead_ms);
  out->full_model_hit = out->to_load_bytes == 0 ? 1 : 0;
  return 0;
}

int tangram_vram_load_model(TangramVRAMHandle* handle, int model_id, int gpu_id,
                            TangramVRAMLoadResult* out) {
  if (!handle || !handle->manager || !out) {
    SetError(handle, "invalid arguments to tangram_vram_load_model");
    return -1;
  }

  TangramVRAMEstimate estimate{};
  if (tangram_vram_estimate(handle, model_id, gpu_id, &estimate) != 0) {
    return -1;
  }

  auto path_it = handle->model_paths.find(model_id);
  const auto start = std::chrono::high_resolution_clock::now();
  std::string response = handle->manager->LoadModel(
      path_it->second, gpu_id, handle->free_strategy, handle->allocate_strategy,
      false, handle->disable_parameter_reuse);
  const auto end = std::chrono::high_resolution_clock::now();
  if (response == "ERROR") {
    SetError(handle, "VRAMManager::LoadModel failed");
    return -1;
  }

  out->cached_bytes = estimate.cached_bytes;
  out->to_load_bytes = estimate.to_load_bytes;
  out->total_model_bytes = estimate.total_model_bytes;
  out->estimated_load_ms = estimate.load_ms;
  out->wall_load_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  out->full_model_hit = estimate.full_model_hit;
  return 0;
}

}  // extern "C"
