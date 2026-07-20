#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "binary_utils.h"
#include "logger.h"
#include "registered_model.h"
#include "vram_manager_interface.h"

// One VMM allocation is a contiguous virtual range backed by individually
// created physical pages.  Weight and KV allocations deliberately share the
// same page budget in VmmGpuPagePool.
class VmmGpuPagePool {
 public:
  enum class Kind { kWeight, kKV };
  struct Allocation {
    CUdeviceptr va = 0;
    size_t logical_bytes = 0;
    size_t mapped_bytes = 0;
    struct Mapping {
      size_t physical_extent_id = 0;
      size_t page_count = 0;
      size_t virtual_offset = 0;
    };
    std::vector<Mapping> mappings;
    Kind kind = Kind::kWeight;
    double sensitivity = 1.0;
    uint64_t last_access = 0;
    uint64_t access_count = 0;
  };

  VmmGpuPagePool(int device_id, size_t capacity, size_t requested_page_size)
      : device_id_(device_id), capacity_bytes_(capacity) {
    Check(cuInit(0), "cuInit");
    CUdevice device;
    Check(cuDeviceGet(&device, device_id_), "cuDeviceGet");
    prop_ = {};
    prop_.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop_.location.id = device_id_;
    prop_.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
    Check(cuMemGetAllocationGranularity(&native_granularity_, &prop_,
                                        CU_MEM_ALLOC_GRANULARITY_MINIMUM),
          "cuMemGetAllocationGranularity");
    granularity_ = requested_page_size == 0 ? native_granularity_
                                            : requested_page_size;
    if (granularity_ < native_granularity_ ||
        granularity_ % native_granularity_ != 0 ||
        (granularity_ & (granularity_ - 1)) != 0) {
      throw std::invalid_argument(
          "VMM page size must be a power-of-two multiple of the device native granularity " +
          std::to_string(native_granularity_));
    }
    page_capacity_ = capacity_bytes_ / granularity_;
    if (page_capacity_ == 0) throw std::runtime_error("VMM pool is smaller than one native page");
    PreallocatePhysicalExtents();
  }
  ~VmmGpuPagePool() { Clear(); ReleasePhysicalExtents(); }
  size_t granularity() const { return granularity_; }
  size_t native_granularity() const { return native_granularity_; }
  size_t free_pages() const { return page_capacity_ - used_pages_; }
  double utilization() const { return page_capacity_ ? double(used_pages_) / page_capacity_ : 0.0; }

  bool HasWeight(const std::string& fp) const { return weights_.count(fp) != 0; }
  CUdeviceptr WeightAddress(const std::string& fp) const { return weights_.at(fp).va; }
  void TouchWeight(const std::string& fp) {
    auto& a = weights_.at(fp); a.last_access = ++clock_; ++a.access_count;
  }
  void DropWeight(const std::string& fp) {
    auto it = weights_.find(fp);
    if (it == weights_.end()) return;
    Release(&it->second);
    weights_.erase(it);
  }
  size_t ReclaimablePages(const std::unordered_set<std::string>& protected_fps) const {
    size_t pages = free_pages();
    for (const auto& item : weights_) {
      if (!protected_fps.count(item.first)) pages +=
          item.second.mapped_bytes / granularity_;
    }
    return pages;
  }

  bool MapWeight(const std::string& fp, size_t bytes, double sensitivity,
                 const std::unordered_set<std::string>& protected_fps) {
    if (HasWeight(fp)) { TouchWeight(fp); return true; }
    const size_t pages = PagesFor(bytes);
    if (!EnsurePages(pages, protected_fps)) return false;
    Allocation a;
    a.logical_bytes = bytes; a.mapped_bytes = pages * granularity_;
    a.kind = Kind::kWeight; a.sensitivity = sensitivity;
    if (!Map(&a)) return false;
    a.last_access = ++clock_; a.access_count = 1;
    weights_.emplace(fp, std::move(a));
    return true;
  }

  // KV is intentionally allocated from the same physical-page budget.  It is
  // reclaimed as a group by CleanKV before the next model load in this simulator.
  std::vector<size_t> MapKV(size_t block_size, int block_count,
                            const std::unordered_set<std::string>& protected_fps) {
    if (block_count <= 0) return {};
    const size_t bytes = block_size * static_cast<size_t>(block_count);
    const size_t pages = PagesFor(bytes);
    if (!EnsurePages(pages, protected_fps)) return {};
    Allocation a;
    a.logical_bytes = bytes; a.mapped_bytes = pages * granularity_; a.kind = Kind::kKV;
    if (!Map(&a)) return {};
    const CUdeviceptr base = a.va;
    kv_.push_back(std::move(a));
    std::vector<size_t> result;
    result.reserve(block_count);
    for (int i = 0; i < block_count; ++i) result.push_back(base + size_t(i) * block_size);
    return result;
  }
  void CleanKV() {
    for (auto& a : kv_) {
      reclaimed_kv_bytes_ += a.logical_bytes;
      Release(&a);
    }
    kv_.clear();
  }
  size_t CachedBytes(const std::vector<TensorGroupIndex>& groups) const {
    size_t total = 0; for (const auto& g : groups) if (HasWeight(g.fingerprint)) total += g.size; return total;
  }
  void Clear() {
    CleanKV();
    for (auto& item : weights_) Release(&item.second);
    weights_.clear();
  }
  void ResetOperationStats() {
    evicted_weight_bytes_ = 0;
    reclaimed_kv_bytes_ = 0;
    map_extent_count_ = 0;
  }
  size_t evicted_weight_bytes() const { return evicted_weight_bytes_; }
  size_t reclaimed_kv_bytes() const { return reclaimed_kv_bytes_; }
  size_t map_extent_count() const { return map_extent_count_; }

 private:
  static void Check(CUresult status, const char* where) {
    if (status != CUDA_SUCCESS) {
      const char* text = nullptr; cuGetErrorString(status, &text);
      throw std::runtime_error(std::string(where) + ": " + (text ? text : "CUDA driver error"));
    }
  }
  size_t PagesFor(size_t bytes) const { return (bytes + granularity_ - 1) / granularity_; }
  struct PhysicalExtent {
    CUmemGenericAllocationHandle handle = 0;
    size_t page_count = 0;
    bool in_use = false;
  };
  void PreallocatePhysicalExtents() {
    // The target driver only accepts mappings that cover a complete physical
    // allocation.  Use one allocation per native page so every byte-capacity
    // state is representable and the VMM layer cannot introduce size-class
    // fragmentation of its own.
    physical_extents_.reserve(page_capacity_);
    free_extent_ids_.reserve(page_capacity_);
    try {
      for (size_t page = 0; page < page_capacity_; ++page) {
        CUmemGenericAllocationHandle handle = 0;
        Check(cuMemCreate(&handle, granularity_, &prop_, 0),
              "cuMemCreate(preallocate native page)");
        physical_extents_.push_back({handle, 1, false});
        free_extent_ids_.push_back(page);
      }
    } catch (...) {
      ReleasePhysicalExtents();
      throw;
    }
  }
  void ReleasePhysicalExtents() {
    for (auto& extent : physical_extents_) {
      if (extent.handle) cuMemRelease(extent.handle);
    }
    physical_extents_.clear();
    free_extent_ids_.clear();
  }
  bool AcquirePhysicalExtent(size_t max_pages, size_t* extent_id,
                             size_t* acquired_pages) {
    if (max_pages == 0 || free_extent_ids_.empty()) return false;
    const size_t id = free_extent_ids_.back();
    free_extent_ids_.pop_back();
    physical_extents_[id].in_use = true;
    *extent_id = id;
    *acquired_pages = 1;
    return true;
  }
  bool EnsurePages(size_t pages, const std::unordered_set<std::string>& protected_fps) {
    // KV allocations are normally cleaned before parameter loading.  Keeping
    // this also makes a direct KV allocation reclaim all stale KV first.
    if (free_pages() < pages && !kv_.empty()) CleanKV();
    while (free_pages() < pages) {
      auto victim = weights_.end(); double best = std::numeric_limits<double>::infinity();
      for (auto it = weights_.begin(); it != weights_.end(); ++it) {
        if (protected_fps.count(it->first)) continue;
        const auto& a = it->second;
        const double score = (double(a.access_count) / std::max<uint64_t>(1, clock_)) * a.sensitivity;
        if (score < best) { best = score; victim = it; }
      }
      if (victim == weights_.end()) return false;
      evicted_weight_bytes_ += victim->second.logical_bytes;
      Release(&victim->second); weights_.erase(victim);
    }
    return true;
  }
  bool Map(Allocation* a) {
    try {
      Check(cuMemAddressReserve(&a->va, a->mapped_bytes, granularity_, 0, 0), "cuMemAddressReserve");
      size_t pages_remaining = a->mapped_bytes / granularity_;
      size_t virtual_offset = 0;
      while (pages_remaining > 0) {
        size_t extent_id = 0, page_count = 0;
        if (!AcquirePhysicalExtent(pages_remaining, &extent_id, &page_count)) {
          throw std::runtime_error("preallocated VMM physical-page pool is inconsistent");
        }
        const size_t bytes = page_count * granularity_;
        const auto& extent = physical_extents_[extent_id];
        Check(cuMemMap(a->va + virtual_offset, bytes, 0, extent.handle, 0), "cuMemMap");
        a->mappings.push_back({extent_id, page_count, virtual_offset});
        used_pages_ += page_count;
        ++map_extent_count_;
        virtual_offset += bytes;
        pages_remaining -= page_count;
      }
      CUmemAccessDesc access{};
      access.location.type = CU_MEM_LOCATION_TYPE_DEVICE; access.location.id = device_id_;
      access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      Check(cuMemSetAccess(a->va, a->mapped_bytes, &access, 1), "cuMemSetAccess");
      return true;
    } catch (const std::exception& e) {
      LOG(ERROR) << "VMM map failed on GPU " << device_id_
                 << ", logical_bytes=" << a->logical_bytes
                 << ", mapped_bytes=" << a->mapped_bytes
                 << ": " << e.what();
      Release(a);
      return false;
    }
  }
  void Release(Allocation* a) {
    if (!a || a->va == 0) return;
    for (const auto& mapping : a->mappings) {
      const size_t bytes = mapping.page_count * granularity_;
      cuMemUnmap(a->va + mapping.virtual_offset, bytes);
      auto& extent = physical_extents_[mapping.physical_extent_id];
      extent.in_use = false;
      free_extent_ids_.push_back(mapping.physical_extent_id);
      used_pages_ -= mapping.page_count;
    }
    cuMemAddressFree(a->va, a->mapped_bytes);
    *a = {};
  }
  int device_id_; size_t capacity_bytes_; size_t native_granularity_ = 0;
  size_t granularity_ = 0;
  size_t page_capacity_ = 0, used_pages_ = 0; uint64_t clock_ = 0;
  CUmemAllocationProp prop_{};
  std::vector<PhysicalExtent> physical_extents_;
  std::vector<size_t> free_extent_ids_;
  std::unordered_map<std::string, Allocation> weights_;
  std::vector<Allocation> kv_;
  size_t evicted_weight_bytes_ = 0;
  size_t reclaimed_kv_bytes_ = 0;
  size_t map_extent_count_ = 0;
};

class VmmVRAMManager final : public IVRAMManager {
 public:
  VmmVRAMManager(size_t pool_size, const std::vector<int>& gpu_ids, double, double,
                 bool mock_copy, size_t requested_page_size = 0) {
    if (mock_copy) throw std::invalid_argument("--memory_backend vmm does not support --mock_copy");
    // Match the legacy pool's initialization boundary: CUDA Runtime primary
    // context creation must not be charged to the first model load on a GPU.
    for (int id : gpu_ids) {
      cudaError_t status = cudaSetDevice(id);
      if (status == cudaSuccess) status = cudaFree(nullptr);
      if (status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA Runtime warm-up failed for GPU ") +
                                 std::to_string(id) + ": " + cudaGetErrorString(status));
      }
      pools_.emplace(id, std::make_unique<VmmGpuPagePool>(
                             id, pool_size, requested_page_size));
    }
  }
  int64_t RegisterModel(const std::string& path, double sensitivity, bool mock_copy, int granularity) override {
    if (mock_copy) return -1;
    if (models_.count(path)) return models_.at(path)->model_size();
    auto model = std::make_shared<RegisteredModel>(path, sensitivity, granularity);
    model->MergeTGsRatio(40);
    if (model->LoadModelFromDisk(8) != 0) return -1;
    models_[path] = model; return model->model_size();
  }
  void WarmupModelAccess(const std::unordered_map<std::string, size_t>& counts) override {
    for (const auto& p : counts) warmup_[p.first] += p.second;
  }
  std::string LoadModel(const std::string& path, int device_id, int, int, bool,
                        bool disable_reuse) override {
    auto mit = models_.find(path); auto pit = pools_.find(device_id);
    if (mit == models_.end() || pit == pools_.end()) return "ERROR";
    auto& model = mit->second; auto& pool = *pit->second;
    pool.ResetOperationStats();
    pool.CleanKV();  // Current simulator semantic: a new model load releases old KV.
    const auto& groups = model->GetTensorGroupIndexes();
    std::unordered_set<std::string> protect;
    for (const auto& g : groups) protect.insert(g.fingerprint);
    std::vector<char*> addresses(groups.size(), nullptr); std::vector<int> missing;
    size_t cached_bytes = 0, to_load_bytes = 0;
    for (size_t i = 0; i < groups.size(); ++i) {
      if (disable_reuse) pool.DropWeight(groups[i].fingerprint);
      if (!disable_reuse && pool.HasWeight(groups[i].fingerprint)) {
        pool.TouchWeight(groups[i].fingerprint);
        cached_bytes += groups[i].size;
      } else if (!pool.MapWeight(groups[i].fingerprint, groups[i].size,
                                  model->GetLoadSensitive(), protect)) {
        return "ERROR";
      } else { missing.push_back(static_cast<int>(i)); to_load_bytes += groups[i].size; }
      addresses[i] = reinterpret_cast<char*>(pool.WeightAddress(groups[i].fingerprint));
    }
    if (!missing.empty() && model->LoadModelFromMem(addresses, missing, device_id) != 0) return "ERROR";
    std::cout << "VMM LoadModel: model_path=" << path << " to device " << device_id
              << " cached_bytes=" << cached_bytes << " to_load_bytes=" << to_load_bytes
              << " evicted_weight_bytes=" << pool.evicted_weight_bytes()
              << " reclaimed_kv_bytes=" << pool.reclaimed_kv_bytes()
              << " map_pages=" << pool.map_extent_count() << std::endl;
    return "VMM";
  }
  bool EstimateModelLoad(const std::string& path, int device, size_t& cached,
                         size_t& to_load, size_t& total, bool disable_reuse) override {
    cached = to_load = total = 0;
    auto mit = models_.find(path); auto pit = pools_.find(device);
    if (mit == models_.end() || pit == pools_.end()) return false;
    for (const auto& g : mit->second->GetTensorGroupIndexes()) {
      total += g.size;
      if (!disable_reuse && pit->second->HasWeight(g.fingerprint)) cached += g.size;
      else to_load += g.size;
    }
    return true;
  }
  int64_t GetGPUToLoad(const std::string& path, int policy) override {
    if (pools_.empty() || !models_.count(path)) return -1;
    std::vector<int> ids;
    for (const auto& pool : pools_) ids.push_back(pool.first);
    std::sort(ids.begin(), ids.end());
    if (policy == 0) return ids.front();
    int best = -1; size_t bytes = 0, free_pages = 0;
    for (int id : ids) {
      const auto& pool = *pools_.at(id);
      const size_t cached = pool.CachedBytes(models_.at(path)->GetTensorGroupIndexes());
      const size_t available = pool.free_pages();
      if (best < 0 || cached > bytes || (cached == bytes && available > free_pages)) {
        best = id; bytes = cached; free_pages = available;
      }
    }
    return best;
  }
  int GetAvailableBlocks(const std::string& model, size_t block_size, int device) override {
    auto it = pools_.find(device); if (it == pools_.end() || block_size == 0) return 0;
    std::unordered_set<std::string> protect;
    if (models_.count(model)) {
      for (const auto& g : models_.at(model)->GetTensorGroupIndexes()) protect.insert(g.fingerprint);
    }
    return static_cast<int>((it->second->ReclaimablePages(protect) *
                             it->second->granularity()) / block_size);
  }
  std::vector<size_t> AllocateBlocks(int device, size_t block, const std::string& model, int count) override {
    auto it = pools_.find(device); if (it == pools_.end()) return {};
    std::unordered_set<std::string> protect;
    if (models_.count(model)) for (const auto& g : models_.at(model)->GetTensorGroupIndexes()) protect.insert(g.fingerprint);
    return it->second->MapKV(block, count, protect);
  }
  void MemoryUsage() override { for (const auto& p : pools_) LOG(METRIC) << "VMM GPU " << p.first << ", utilization=" << p.second->utilization() << ", physical_page_bytes=" << p.second->granularity() << ", native_granularity_bytes=" << p.second->native_granularity(); }
  double get_memory_utilization() override { double total = 0; for (const auto& p : pools_) total += p.second->utilization(); return pools_.empty() ? 0 : total / pools_.size(); }
 private:
  std::unordered_map<std::string, std::shared_ptr<RegisteredModel>> models_;
  std::unordered_map<int, std::unique_ptr<VmmGpuPagePool>> pools_;
  std::unordered_map<std::string, size_t> warmup_;
};
