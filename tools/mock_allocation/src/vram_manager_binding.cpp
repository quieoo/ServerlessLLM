#include "vram_manager_interface.h"
#include "vram_manager_legacy_adapter.h"
#include "vram_manager_vmm.h"

#include <chrono>
#include <algorithm>
#include <cstdint>
#include <exception>
#include <memory>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct TangramVRAMHandle {
  std::unique_ptr<IVRAMManager> manager;
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

const char* tangram_vram_vmm_policy() {
  return "stable_model_va_layerweave_prefix_cache_v2";
}

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

struct TangramTensorBinding {
  uint64_t group_base;
  uint64_t offset;
  uint64_t size;
};

struct TangramModelLayout {
  uint64_t page_count;
  uint64_t contiguous_runs;
  uint64_t adjacent_breaks;
  uint64_t extent_id_span;
};

struct TangramLayerWeaveLayout {
  uint64_t model_base;
  uint64_t page_size;
  uint64_t page_count;
};

struct TangramLayerWeaveLoadResult {
  uint64_t requested_pages;
  uint64_t cached_pages;
  uint64_t mapped_pages;
  uint64_t cached_bytes;
  uint64_t to_load_bytes;
};

TangramVRAMHandle* tangram_vram_create_ex2(
    int num_gpus, uint64_t gpu_pool_size_bytes, double gpu_bandwidth_bytes,
    double cpu_bandwidth_bytes, int mock_copy, double load_bandwidth_gbps,
    double load_overhead_ms, int free_strategy, int allocate_strategy,
    int disable_parameter_reuse, const char* memory_backend,
    uint64_t vmm_page_size_bytes) {
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
    const std::string backend = memory_backend ? memory_backend : "legacy";
    if (backend == "legacy") {
      handle->manager = std::make_unique<LegacyVRAMManagerAdapter>(
          static_cast<size_t>(gpu_pool_size_bytes), gpu_ids, gpu_bandwidth_bytes,
          cpu_bandwidth_bytes, handle->mock_copy);
    } else if (backend == "vmm") {
      handle->manager = std::make_unique<VmmVRAMManager>(
          static_cast<size_t>(gpu_pool_size_bytes), gpu_ids, gpu_bandwidth_bytes,
          cpu_bandwidth_bytes, handle->mock_copy,
          static_cast<size_t>(vmm_page_size_bytes));
    } else {
      throw std::invalid_argument("unknown memory backend: " + backend);
    }
  } catch (const std::exception& e) {
    handle->last_error = e.what();
  } catch (...) {
    handle->last_error = "unknown error during tangram_vram_create";
  }
  return handle.release();
}

// Preserve the original extended ABI. A zero page size selects the CUDA
// device's native minimum VMM allocation granularity.
TangramVRAMHandle* tangram_vram_create_ex(
    int num_gpus, uint64_t gpu_pool_size_bytes, double gpu_bandwidth_bytes,
    double cpu_bandwidth_bytes, int mock_copy, double load_bandwidth_gbps,
    double load_overhead_ms, int free_strategy, int allocate_strategy,
    int disable_parameter_reuse, const char* memory_backend) {
  return tangram_vram_create_ex2(
      num_gpus, gpu_pool_size_bytes, gpu_bandwidth_bytes, cpu_bandwidth_bytes,
      mock_copy, load_bandwidth_gbps, load_overhead_ms, free_strategy,
      allocate_strategy, disable_parameter_reuse, memory_backend, 0);
}

// Preserve the original ABI and default it to the contiguous legacy pool.
TangramVRAMHandle* tangram_vram_create(
    int num_gpus, uint64_t gpu_pool_size_bytes, double gpu_bandwidth_bytes,
    double cpu_bandwidth_bytes, int mock_copy, double load_bandwidth_gbps,
    double load_overhead_ms, int free_strategy, int allocate_strategy,
    int disable_parameter_reuse) {
  return tangram_vram_create_ex(
      num_gpus, gpu_pool_size_bytes, gpu_bandwidth_bytes, cpu_bandwidth_bytes,
      mock_copy, load_bandwidth_gbps, load_overhead_ms, free_strategy,
      allocate_strategy, disable_parameter_reuse, "legacy");
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

int tangram_vram_register_model_ex(TangramVRAMHandle* handle, int model_id,
                                   const char* model_path, double sensitive,
                                   int merge_target_count) {
  if (!handle || !handle->manager || !model_path ||
      merge_target_count == 0 || merge_target_count < -2) {
    SetError(handle, "invalid arguments to tangram_vram_register_model_ex");
    return -1;
  }
  try {
    auto* manager = dynamic_cast<VmmVRAMManager*>(handle->manager.get());
    if (!manager) {
      SetError(handle, "tensor-group merge registration requires VMM backend");
      return -1;
    }
    const std::string path(model_path);
    int64_t model_size = manager->RegisterModelWithMergeTarget(
        path, sensitive, handle->mock_copy, merge_target_count);
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
    SetError(handle, "unknown error during tangram_vram_register_model_ex");
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

int tangram_vram_tensor_count(TangramVRAMHandle* handle, int model_id,
                              int gpu_id) {
  if (!handle || !handle->manager) return -1;
  auto path_it = handle->model_paths.find(model_id);
  auto* manager = dynamic_cast<VmmVRAMManager*>(handle->manager.get());
  if (path_it == handle->model_paths.end() || !manager) {
    SetError(handle, "tensor bindings require the VMM backend");
    return -1;
  }
  std::vector<VmmVRAMManager::TensorBinding> bindings;
  if (!manager->GetTensorBindings(path_it->second, gpu_id, &bindings)) {
    SetError(handle, "cannot get VMM tensor bindings");
    return -1;
  }
  return static_cast<int>(bindings.size());
}

int tangram_vram_get_tensor(TangramVRAMHandle* handle, int model_id,
                            int gpu_id, int index, char* name,
                            uint64_t name_capacity, TangramTensorBinding* out) {
  if (!handle || !handle->manager || !name || name_capacity == 0 || !out) {
    SetError(handle, "invalid arguments to tangram_vram_get_tensor");
    return -1;
  }
  auto path_it = handle->model_paths.find(model_id);
  auto* manager = dynamic_cast<VmmVRAMManager*>(handle->manager.get());
  if (path_it == handle->model_paths.end() || !manager) {
    SetError(handle, "tensor bindings require the VMM backend");
    return -1;
  }
  std::vector<VmmVRAMManager::TensorBinding> bindings;
  if (!manager->GetTensorBindings(path_it->second, gpu_id, &bindings) ||
      index < 0 || static_cast<size_t>(index) >= bindings.size()) {
    SetError(handle, "invalid VMM tensor index");
    return -1;
  }
  const auto& binding = bindings[static_cast<size_t>(index)];
  if (binding.name.size() + 1 > name_capacity) {
    SetError(handle, "tensor name buffer is too small");
    return -1;
  }
  std::memcpy(name, binding.name.c_str(), binding.name.size() + 1);
  out->group_base = static_cast<uint64_t>(binding.group_base);
  out->offset = static_cast<uint64_t>(binding.offset);
  out->size = static_cast<uint64_t>(binding.size);
  return 0;
}

int tangram_vram_layerweave_layout(TangramVRAMHandle* handle, int model_id,
                                   int gpu_id,
                                   TangramLayerWeaveLayout* out) {
  if (!handle || !handle->manager || !out) {
    SetError(handle, "invalid arguments to tangram_vram_layerweave_layout");
    return -1;
  }
  auto path_it = handle->model_paths.find(model_id);
  auto* manager = dynamic_cast<VmmVRAMManager*>(handle->manager.get());
  if (path_it == handle->model_paths.end() || !manager) {
    SetError(handle, "LayerWeave layout requires a registered VMM model");
    return -1;
  }
  size_t page_size = 0;
  size_t page_count = 0;
  if (!manager->GetLayerWeaveLayout(
          path_it->second, gpu_id, &out->model_base, &page_size,
          &page_count)) {
    SetError(handle, "cannot query LayerWeave model layout");
    return -1;
  }
  out->page_size = static_cast<uint64_t>(page_size);
  out->page_count = static_cast<uint64_t>(page_count);
  return 0;
}

int tangram_vram_layerweave_begin_binding(
    TangramVRAMHandle* handle, int model_id, int gpu_id) {
  if (!handle || !handle->manager) {
    SetError(handle,
             "invalid arguments to tangram_vram_layerweave_begin_binding");
    return -1;
  }
  auto path_it = handle->model_paths.find(model_id);
  auto* manager = dynamic_cast<VmmVRAMManager*>(handle->manager.get());
  if (path_it == handle->model_paths.end() || !manager ||
      !manager->BeginLayerWeaveBinding(path_it->second, gpu_id)) {
    SetError(handle, "cannot begin LayerWeave tensor binding");
    return -1;
  }
  return 0;
}

int tangram_vram_layerweave_end_binding(
    TangramVRAMHandle* handle, int model_id, int gpu_id) {
  if (!handle || !handle->manager) {
    SetError(handle,
             "invalid arguments to tangram_vram_layerweave_end_binding");
    return -1;
  }
  auto path_it = handle->model_paths.find(model_id);
  auto* manager = dynamic_cast<VmmVRAMManager*>(handle->manager.get());
  if (path_it == handle->model_paths.end() || !manager ||
      !manager->EndLayerWeaveBinding(path_it->second, gpu_id)) {
    SetError(handle, "cannot end LayerWeave tensor binding");
    return -1;
  }
  return 0;
}

int tangram_vram_layerweave_residency(
    TangramVRAMHandle* handle, int model_id, int gpu_id, uint8_t* residency,
    uint64_t capacity) {
  if (!handle || !handle->manager || (capacity && !residency)) {
    SetError(handle, "invalid arguments to tangram_vram_layerweave_residency");
    return -1;
  }
  auto path_it = handle->model_paths.find(model_id);
  auto* manager = dynamic_cast<VmmVRAMManager*>(handle->manager.get());
  if (path_it == handle->model_paths.end() || !manager) {
    SetError(handle, "LayerWeave residency requires a registered VMM model");
    return -1;
  }
  std::vector<uint8_t> value;
  if (!manager->GetLayerWeaveResidency(path_it->second, gpu_id, &value) ||
      value.size() > capacity) {
    SetError(handle, "LayerWeave residency buffer is too small");
    return -1;
  }
  std::copy(value.begin(), value.end(), residency);
  return static_cast<int>(value.size());
}

int tangram_vram_layerweave_configure_cache(
    TangramVRAMHandle* handle, int model_id, int gpu_id,
    const uint64_t* retained_pages, uint64_t count) {
  if (!handle || !handle->manager || (count && !retained_pages)) {
    SetError(handle,
             "invalid arguments to tangram_vram_layerweave_configure_cache");
    return -1;
  }
  auto path_it = handle->model_paths.find(model_id);
  auto* manager = dynamic_cast<VmmVRAMManager*>(handle->manager.get());
  if (path_it == handle->model_paths.end() || !manager) {
    SetError(handle, "LayerWeave cache configuration requires VMM");
    return -1;
  }
  std::vector<size_t> retained;
  retained.reserve(count);
  for (uint64_t index = 0; index < count; ++index) {
    retained.push_back(static_cast<size_t>(retained_pages[index]));
  }
  if (!manager->ConfigureLayerWeaveCache(
          path_it->second, gpu_id, retained)) {
    SetError(handle, "cannot apply LayerWeave cache configuration");
    return -1;
  }
  return 0;
}

int tangram_vram_layerweave_prepare_pages(
    TangramVRAMHandle* handle, int model_id, int gpu_id,
    const uint64_t* pages, uint64_t count, uint64_t cuda_stream,
    TangramLayerWeaveLoadResult* out) {
  if (!handle || !handle->manager || !out ||
      (count && !pages)) {
    SetError(handle,
             "invalid arguments to tangram_vram_layerweave_prepare_pages");
    return -1;
  }
  auto path_it = handle->model_paths.find(model_id);
  auto* manager = dynamic_cast<VmmVRAMManager*>(handle->manager.get());
  if (path_it == handle->model_paths.end() || !manager) {
    SetError(handle, "LayerWeave prepare requires a registered VMM model");
    return -1;
  }
  std::vector<size_t> requested;
  requested.reserve(count);
  for (uint64_t index = 0; index < count; ++index) {
    requested.push_back(static_cast<size_t>(pages[index]));
  }
  VmmVRAMManager::LayerWeaveLoadResult result;
  cudaStream_t stream =
      reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
  if (!manager->PrepareLayerWeavePages(
          path_it->second, gpu_id, requested, stream, &result)) {
    SetError(handle, "LayerWeave page preparation failed");
    return -1;
  }
  out->requested_pages = result.requested_pages;
  out->cached_pages = result.cached_pages;
  out->mapped_pages = result.mapped_pages;
  out->cached_bytes = result.cached_bytes;
  out->to_load_bytes = result.to_load_bytes;
  return 0;
}

int tangram_vram_model_layout(TangramVRAMHandle* handle, int model_id,
                              int gpu_id, TangramModelLayout* out) {
  if (!handle || !handle->manager || !out) {
    SetError(handle, "invalid arguments to tangram_vram_model_layout");
    return -1;
  }
  auto path_it = handle->model_paths.find(model_id);
  auto* manager = dynamic_cast<VmmVRAMManager*>(handle->manager.get());
  if (path_it == handle->model_paths.end() || !manager) {
    SetError(handle, "model layout requires the VMM backend");
    return -1;
  }
  VmmVRAMManager::ModelLayout layout;
  if (!manager->GetModelLayout(path_it->second, gpu_id, &layout)) {
    SetError(handle, "model is not resident on the requested GPU");
    return -1;
  }
  out->page_count = layout.page_count;
  out->contiguous_runs = layout.contiguous_runs;
  out->adjacent_breaks = layout.adjacent_breaks;
  out->extent_id_span = layout.extent_id_span;
  return 0;
}

uint64_t tangram_vram_kv_base(TangramVRAMHandle* handle, int gpu_id) {
  if (!handle || !handle->manager) return 0;
  auto* manager = dynamic_cast<VmmVRAMManager*>(handle->manager.get());
  if (!manager) {
    SetError(handle, "KV base requires the VMM backend");
    return 0;
  }
  const uint64_t base = manager->GetKVBaseAddress(gpu_id);
  if (!base) SetError(handle, "unknown GPU in tangram_vram_kv_base");
  return base;
}

int tangram_vram_kv_available_blocks(TangramVRAMHandle* handle, int model_id,
                                     int gpu_id, uint64_t block_size) {
  if (!handle || !handle->manager || block_size == 0) return -1;
  auto path_it = handle->model_paths.find(model_id);
  auto* manager = dynamic_cast<VmmVRAMManager*>(handle->manager.get());
  if (path_it == handle->model_paths.end() || !manager) {
    SetError(handle, "KV availability requires a registered VMM model");
    return -1;
  }
  const int count = manager->GetAvailableKVBlocks(
      path_it->second, static_cast<size_t>(block_size), gpu_id);
  if (count < 0) SetError(handle, "cannot query available VMM KV blocks");
  return count;
}

int tangram_vram_kv_allocate(TangramVRAMHandle* handle, int model_id,
                             int gpu_id, uint64_t block_size,
                             const uint64_t* logical_ids, uint64_t count,
                             uint64_t* offsets) {
  if (!handle || !handle->manager || block_size == 0 ||
      (count && (!logical_ids || !offsets))) {
    SetError(handle, "invalid arguments to tangram_vram_kv_allocate");
    return -1;
  }
  auto* manager = dynamic_cast<VmmVRAMManager*>(handle->manager.get());
  auto path_it = handle->model_paths.find(model_id);
  if (!manager || (model_id >= 0 && path_it == handle->model_paths.end())) {
    SetError(handle, "KV allocation requires a registered VMM model");
    return -1;
  }
  std::vector<uint64_t> ids;
  if (count) ids.assign(logical_ids, logical_ids + count);
  std::vector<uint64_t> result;
  const bool allocated = model_id < 0
      ? manager->AllocateKVBlocksUnprotected(
            static_cast<size_t>(block_size), gpu_id, ids, &result)
      : manager->AllocateKVBlocks(
            path_it->second, static_cast<size_t>(block_size), gpu_id, ids,
            &result);
  if (!allocated ||
      result.size() != count) {
    SetError(handle, "VMM KV allocation failed");
    return -1;
  }
  std::copy(result.begin(), result.end(), offsets);
  return 0;
}

int tangram_vram_kv_release(TangramVRAMHandle* handle, int gpu_id,
                            const uint64_t* logical_ids, uint64_t count) {
  if (!handle || !handle->manager || (count && !logical_ids)) {
    SetError(handle, "invalid arguments to tangram_vram_kv_release");
    return -1;
  }
  auto* manager = dynamic_cast<VmmVRAMManager*>(handle->manager.get());
  if (!manager) {
    SetError(handle, "KV release requires the VMM backend");
    return -1;
  }
  std::vector<uint64_t> ids;
  if (count) ids.assign(logical_ids, logical_ids + count);
  if (!manager->ReleaseKVBlocks(gpu_id, ids)) {
    SetError(handle, "VMM KV release failed");
    return -1;
  }
  return 0;
}

int tangram_vram_kv_clear(TangramVRAMHandle* handle, int gpu_id) {
  if (!handle || !handle->manager) return -1;
  auto* manager = dynamic_cast<VmmVRAMManager*>(handle->manager.get());
  if (!manager || !manager->ClearKV(gpu_id)) {
    SetError(handle, "VMM KV clear failed");
    return -1;
  }
  return 0;
}

}  // extern "C"
