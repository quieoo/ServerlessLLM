#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstring>
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
    bool owns_va = true;
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
  struct StableWeightArena {
    CUdeviceptr va = 0;
    size_t logical_bytes = 0;
    size_t mapped_bytes = 0;
    double sensitivity = 1.0;
    std::vector<int64_t> physical_extent_ids;
    std::vector<uint64_t> last_access;
    std::vector<uint64_t> access_count;
  };
  struct StableLoadEstimate {
    size_t cached_logical_bytes = 0;
    size_t missing_logical_bytes = 0;
    size_t pages_to_evict = 0;
    double eviction_value_bytes = 0.0;
    double placement_cost_bytes = 0.0;
  };
  struct StableBindingAliases {
    CUmemGenericAllocationHandle handle = 0;
    std::vector<size_t> pages;
  };

  VmmGpuPagePool(int device_id, size_t capacity, size_t requested_page_size)
      : device_id_(device_id), capacity_bytes_(capacity) {
    Check(cuInit(0), "cuInit");
    // CUDA Runtime ordinals are remapped by CUDA_VISIBLE_DEVICES, while raw
    // Driver API ordinals are physical. Resolve the runtime-selected device
    // by UUID so cuMemCreate/cuMemSetAccess target the same physical GPU as
    // torch/vLLM when independent single-GPU processes use different cards.
    cudaDeviceProp runtime_prop{};
    cudaError_t runtime_status =
        cudaGetDeviceProperties(&runtime_prop, device_id_);
    if (runtime_status != cudaSuccess) {
      throw std::runtime_error(
          std::string("cudaGetDeviceProperties: ") +
          cudaGetErrorString(runtime_status));
    }
    CUuuid uuid{};
    static_assert(sizeof(uuid.bytes) == sizeof(runtime_prop.uuid.bytes),
                  "CUDA runtime/driver UUID sizes differ");
    std::memcpy(uuid.bytes, runtime_prop.uuid.bytes, sizeof(uuid.bytes));
    int driver_device_count = 0;
    Check(cuDeviceGetCount(&driver_device_count), "cuDeviceGetCount");
    CUdevice device = -1;
    for (int ordinal = 0; ordinal < driver_device_count; ++ordinal) {
      CUdevice candidate;
      CUuuid candidate_uuid{};
      Check(cuDeviceGet(&candidate, ordinal), "cuDeviceGet");
      Check(cuDeviceGetUuid(&candidate_uuid, candidate), "cuDeviceGetUuid");
      if (std::memcmp(
              candidate_uuid.bytes, uuid.bytes, sizeof(uuid.bytes)) == 0) {
        device = candidate;
        break;
      }
    }
    if (device < 0) {
      throw std::runtime_error(
          "cannot map CUDA Runtime device UUID to Driver device ordinal");
    }
    physical_device_id_ = static_cast<int>(device);
    prop_ = {};
    prop_.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop_.location.id = physical_device_id_;
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
    try {
      kv_va_capacity_ = page_capacity_ * granularity_;
      Check(cuMemAddressReserve(&kv_va_base_, kv_va_capacity_, granularity_,
                                0, 0),
            "cuMemAddressReserve(KV arena)");
      kv_va_slots_.assign(page_capacity_, false);
    } catch (...) {
      ReleasePhysicalExtents();
      throw;
    }
  }
  ~VmmGpuPagePool() {
    Clear();
    if (kv_va_base_) cuMemAddressFree(kv_va_base_, kv_va_capacity_);
    ReleasePhysicalExtents();
  }
  size_t granularity() const { return granularity_; }
  size_t native_granularity() const { return native_granularity_; }
  size_t free_pages() const { return page_capacity_ - used_pages_; }
  double utilization() const { return page_capacity_ ? double(used_pages_) / page_capacity_ : 0.0; }
  void SetModelAccessCount(const std::string& model, size_t count) {
    const size_t previous = model_access_counts_[model];
    model_access_counts_[model] = count;
    if (count >= previous) {
      total_model_access_count_ += count - previous;
    } else {
      total_model_access_count_ -= previous - count;
    }
  }

  bool ReserveStableWeight(const std::string& model, size_t bytes,
                           double sensitivity) {
    if (stable_weights_.count(model)) return true;
    StableWeightArena arena;
    arena.logical_bytes = bytes;
    arena.mapped_bytes = PagesFor(bytes) * granularity_;
    arena.sensitivity = sensitivity;
    const size_t pages = arena.mapped_bytes / granularity_;
    try {
      Check(cuMemAddressReserve(&arena.va, arena.mapped_bytes, granularity_,
                                0, 0),
            "cuMemAddressReserve(stable model arena)");
      arena.physical_extent_ids.assign(pages, -1);
      arena.last_access.assign(pages, 0);
      arena.access_count.assign(pages, 0);
      stable_weights_.emplace(model, std::move(arena));
      return true;
    } catch (const std::exception& e) {
      LOG(ERROR) << "Failed to reserve stable VMM model arena on GPU "
                 << device_id_ << " for " << model << ": " << e.what();
      if (arena.va) cuMemAddressFree(arena.va, arena.mapped_bytes);
      return false;
    }
  }
  void DropStableWeight(const std::string& model, bool release_va = false) {
    auto it = stable_weights_.find(model);
    if (it == stable_weights_.end()) return;
    for (size_t page = 0; page < it->second.physical_extent_ids.size(); ++page) {
      ReleaseStablePage(&it->second, page);
    }
    if (release_va) {
      cuMemAddressFree(it->second.va, it->second.mapped_bytes);
      stable_weights_.erase(it);
    }
  }
  CUdeviceptr StableWeightAddress(const std::string& model) const {
    return stable_weights_.at(model).va;
  }
  size_t StableWeightLogicalBytes(const std::string& model) const {
    return stable_weights_.at(model).logical_bytes;
  }
  size_t StableCachedLogicalBytes(const std::string& model) const {
    const auto& arena = stable_weights_.at(model);
    size_t cached = 0;
    for (size_t page = 0; page < arena.physical_extent_ids.size(); ++page) {
      if (arena.physical_extent_ids[page] < 0) continue;
      const size_t begin = page * granularity_;
      cached += std::min(granularity_, arena.logical_bytes - begin);
    }
    return cached;
  }
  bool StableWeightFullyResident(const std::string& model) const {
    const auto& pages = stable_weights_.at(model).physical_extent_ids;
    return std::all_of(pages.begin(), pages.end(),
                       [](int64_t id) { return id >= 0; });
  }
  const StableWeightArena& StableWeight(const std::string& model) const {
    return stable_weights_.at(model);
  }
  bool RetainStablePages(
      const std::string& model,
      const std::unordered_set<size_t>& retained_pages) {
    auto it = stable_weights_.find(model);
    if (it == stable_weights_.end()) return false;
    for (size_t page = 0;
         page < it->second.physical_extent_ids.size(); ++page) {
      if (!retained_pages.count(page)) {
        ReleaseStablePage(&it->second, page);
      }
    }
    return true;
  }
  bool StablePageResident(const std::string& model, size_t page) const {
    const auto& arena = stable_weights_.at(model);
    return page < arena.physical_extent_ids.size() &&
           arena.physical_extent_ids[page] >= 0;
  }
  bool BeginStableBinding(const std::string& model) {
    auto it = stable_weights_.find(model);
    if (it == stable_weights_.end() || binding_aliases_.count(model)) {
      return false;
    }
    auto& arena = it->second;
    StableBindingAliases aliases;
    try {
      Check(cuMemCreate(&aliases.handle, granularity_, &prop_, 0),
            "cuMemCreate(LayerWeave binding alias)");
      CUmemAccessDesc access{};
      access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      access.location.id = physical_device_id_;
      access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      for (size_t page = 0; page < arena.physical_extent_ids.size(); ++page) {
        if (arena.physical_extent_ids[page] >= 0) continue;
        Check(cuMemMap(arena.va + page * granularity_, granularity_, 0,
                       aliases.handle, 0),
              "cuMemMap(LayerWeave binding alias)");
        aliases.pages.push_back(page);
        Check(cuMemSetAccess(arena.va + page * granularity_, granularity_,
                             &access, 1),
              "cuMemSetAccess(LayerWeave binding alias)");
      }
      binding_aliases_.emplace(model, std::move(aliases));
      return true;
    } catch (const std::exception& e) {
      LOG(ERROR) << "LayerWeave binding alias failed for " << model
                 << ": " << e.what();
      for (size_t page : aliases.pages) {
        cuMemUnmap(arena.va + page * granularity_, granularity_);
      }
      if (aliases.handle) cuMemRelease(aliases.handle);
      return false;
    }
  }
  bool EndStableBinding(const std::string& model) {
    auto arena_it = stable_weights_.find(model);
    auto alias_it = binding_aliases_.find(model);
    if (arena_it == stable_weights_.end() ||
        alias_it == binding_aliases_.end()) {
      return false;
    }
    for (size_t page : alias_it->second.pages) {
      cuMemUnmap(
          arena_it->second.va + page * granularity_, granularity_);
    }
    if (alias_it->second.handle) cuMemRelease(alias_it->second.handle);
    binding_aliases_.erase(alias_it);
    return true;
  }
  StableLoadEstimate EstimateStableLoad(const std::string& model) const {
    StableLoadEstimate estimate;
    const auto model_it = stable_weights_.find(model);
    if (model_it == stable_weights_.end()) {
      estimate.placement_cost_bytes =
          std::numeric_limits<double>::infinity();
      return estimate;
    }
    const auto& target = model_it->second;
    size_t missing_pages = 0;
    for (size_t page = 0; page < target.physical_extent_ids.size(); ++page) {
      const size_t useful = StablePageUsefulBytes(target, page);
      if (target.physical_extent_ids[page] >= 0) {
        estimate.cached_logical_bytes += useful;
      } else {
        estimate.missing_logical_bytes += useful;
        ++missing_pages;
      }
    }
    estimate.pages_to_evict =
        missing_pages > free_pages() ? missing_pages - free_pages() : 0;
    std::vector<double> victim_values;
    victim_values.reserve(used_pages_);
    for (const auto& item : stable_weights_) {
      if (item.first == model) continue;
      const auto& arena = item.second;
      for (size_t page = 0; page < arena.physical_extent_ids.size(); ++page) {
        if (arena.physical_extent_ids[page] >= 0) {
          victim_values.push_back(StablePageValue(item.first, arena, page));
        }
      }
    }
    std::sort(victim_values.begin(), victim_values.end());
    if (victim_values.size() < estimate.pages_to_evict) {
      estimate.placement_cost_bytes =
          std::numeric_limits<double>::infinity();
      return estimate;
    }
    for (size_t i = 0; i < estimate.pages_to_evict; ++i) {
      estimate.eviction_value_bytes += victim_values[i];
    }
    estimate.placement_cost_bytes =
        static_cast<double>(estimate.missing_logical_bytes) +
        estimate.eviction_value_bytes;
    return estimate;
  }
  void RollbackStablePages(const std::string& model,
                           const std::vector<size_t>& pages) {
    auto it = stable_weights_.find(model);
    if (it == stable_weights_.end()) return;
    for (size_t page : pages) ReleaseStablePage(&it->second, page);
  }
  bool MapStableWeight(
      const std::string& model,
      const std::unordered_set<std::string>& protected_models,
      std::vector<size_t>* newly_mapped_pages) {
    auto it = stable_weights_.find(model);
    if (it == stable_weights_.end()) return false;
    std::vector<size_t> pages(it->second.physical_extent_ids.size());
    for (size_t page = 0; page < pages.size(); ++page) pages[page] = page;
    return MapStablePages(
        model, pages, protected_models, newly_mapped_pages);
  }
  bool MapStablePages(
      const std::string& model, const std::vector<size_t>& requested_pages,
      const std::unordered_set<std::string>& protected_models,
      std::vector<size_t>* newly_mapped_pages) {
    auto it = stable_weights_.find(model);
    if (it == stable_weights_.end() || !newly_mapped_pages) return false;
    auto& arena = it->second;
    newly_mapped_pages->clear();
    size_t missing = 0;
    for (size_t page : requested_pages) {
      if (page >= arena.physical_extent_ids.size()) return false;
      missing += arena.physical_extent_ids[page] < 0;
    }
    if (!EnsurePages(missing, protected_models)) return false;
    try {
      CUmemAccessDesc access{};
      access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      access.location.id = physical_device_id_;
      access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      for (size_t page : requested_pages) {
        if (arena.physical_extent_ids[page] >= 0) {
          arena.last_access[page] = ++clock_;
          ++arena.access_count[page];
          continue;
        }
        size_t extent_id = 0, acquired_pages = 0;
        if (!AcquirePhysicalExtent(1, &extent_id, &acquired_pages)) {
          throw std::runtime_error("VMM physical-page pool is inconsistent");
        }
        const CUresult map_status =
            cuMemMap(arena.va + page * granularity_, granularity_, 0,
                     physical_extents_[extent_id].handle, 0);
        if (map_status != CUDA_SUCCESS) {
          physical_extents_[extent_id].in_use = false;
          free_extent_ids_.push_back(extent_id);
          Check(map_status, "cuMemMap(stable model page)");
        }
        arena.physical_extent_ids[page] = static_cast<int64_t>(extent_id);
        newly_mapped_pages->push_back(page);
        ++used_pages_;
        ++map_extent_count_;
        Check(cuMemSetAccess(arena.va + page * granularity_, granularity_,
                             &access, 1),
              "cuMemSetAccess(stable model page)");
        arena.last_access[page] = ++clock_;
        arena.access_count[page] = 1;
      }
      return true;
    } catch (const std::exception& e) {
      LOG(ERROR) << "Failed to map stable model pages for " << model
                 << ": " << e.what();
      for (size_t page : *newly_mapped_pages) ReleaseStablePage(&arena, page);
      newly_mapped_pages->clear();
      return false;
    }
  }

  bool HasWeight(const std::string& fp) const { return weights_.count(fp) != 0; }
  CUdeviceptr WeightAddress(const std::string& fp) const { return weights_.at(fp).va; }
  const Allocation& WeightAllocation(const std::string& fp) const {
    return weights_.at(fp);
  }
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

  CUdeviceptr KVBaseAddress() const { return kv_va_base_; }
  size_t AvailableKVBlocks(
      size_t block_size,
      const std::unordered_set<std::string>& protected_fps = {}) const {
    if (block_size == 0) return 0;
    const size_t pages_per_block = PagesFor(block_size);
    return pages_per_block
        ? ReclaimablePages(protected_fps) / pages_per_block
        : 0;
  }

  bool AllocateKVBlocks(
      size_t block_size, const std::vector<uint64_t>& logical_ids,
      const std::unordered_set<std::string>& protected_fps,
      std::vector<uint64_t>* offsets) {
    if (!offsets || block_size == 0) return false;
    offsets->clear();
    offsets->reserve(logical_ids.size());
    const size_t pages_per_block = PagesFor(block_size);
    std::vector<uint64_t> newly_allocated;
    for (uint64_t logical_id : logical_ids) {
      auto existing = kv_blocks_.find(logical_id);
      if (existing != kv_blocks_.end()) {
        offsets->push_back(existing->second.va - kv_va_base_);
        continue;
      }
      if (!EnsurePages(pages_per_block, protected_fps)) {
        RollbackKVBlocks(newly_allocated);
        offsets->clear();
        return false;
      }
      const size_t slot = FindFreeKVSlot(pages_per_block);
      if (slot == kv_va_slots_.size()) {
        RollbackKVBlocks(newly_allocated);
        offsets->clear();
        return false;
      }
      Allocation allocation;
      allocation.va = kv_va_base_ + slot * granularity_;
      allocation.owns_va = false;
      allocation.logical_bytes = block_size;
      allocation.mapped_bytes = pages_per_block * granularity_;
      allocation.kind = Kind::kKV;
      MarkKVSlots(slot, pages_per_block, true);
      if (!MapAtReservedAddress(&allocation)) {
        MarkKVSlots(slot, pages_per_block, false);
        RollbackKVBlocks(newly_allocated);
        offsets->clear();
        return false;
      }
      offsets->push_back(allocation.va - kv_va_base_);
      kv_blocks_.emplace(logical_id, std::move(allocation));
      newly_allocated.push_back(logical_id);
    }
    return true;
  }

  bool ReleaseKVBlocks(const std::vector<uint64_t>& logical_ids) {
    for (uint64_t logical_id : logical_ids) {
      auto it = kv_blocks_.find(logical_id);
      if (it == kv_blocks_.end()) continue;
      const size_t slot = (it->second.va - kv_va_base_) / granularity_;
      const size_t pages = it->second.mapped_bytes / granularity_;
      reclaimed_kv_bytes_ += it->second.logical_bytes;
      Release(&it->second);
      MarkKVSlots(slot, pages, false);
      kv_blocks_.erase(it);
    }
    return true;
  }

  // Compatibility path used by the allocation simulator. Keep its historical
  // batch rounding behavior; real inference uses AllocateKVBlocks.
  std::vector<size_t> MapKV(size_t block_size, int block_count,
                            const std::unordered_set<std::string>& protected_fps) {
    if (block_count <= 0 || block_size == 0) return {};
    const size_t bytes = block_size * static_cast<size_t>(block_count);
    const size_t pages = PagesFor(bytes);
    if (!EnsurePages(pages, protected_fps)) return {};
    Allocation allocation;
    allocation.logical_bytes = bytes;
    allocation.mapped_bytes = pages * granularity_;
    allocation.kind = Kind::kKV;
    if (!Map(&allocation)) return {};
    const CUdeviceptr base = allocation.va;
    legacy_kv_.push_back(std::move(allocation));
    std::vector<size_t> result;
    result.reserve(block_count);
    for (int i = 0; i < block_count; ++i) {
      result.push_back(base + size_t(i) * block_size);
    }
    return result;
  }
  void CleanKV() {
    std::vector<uint64_t> ids;
    ids.reserve(kv_blocks_.size());
    for (const auto& item : kv_blocks_) ids.push_back(item.first);
    ReleaseKVBlocks(ids);
    for (auto& allocation : legacy_kv_) {
      reclaimed_kv_bytes_ += allocation.logical_bytes;
      Release(&allocation);
    }
    legacy_kv_.clear();
  }
  size_t CachedBytes(const std::vector<TensorGroupIndex>& groups) const {
    size_t total = 0; for (const auto& g : groups) if (HasWeight(g.fingerprint)) total += g.size; return total;
  }
  void Clear() {
    std::vector<std::string> binding_models;
    binding_models.reserve(binding_aliases_.size());
    for (const auto& item : binding_aliases_) {
      binding_models.push_back(item.first);
    }
    for (const auto& model : binding_models) EndStableBinding(model);
    CleanKV();
    for (auto& item : weights_) Release(&item.second);
    weights_.clear();
    for (auto& item : stable_weights_) {
      for (size_t page = 0;
           page < item.second.physical_extent_ids.size(); ++page) {
        ReleaseStablePage(&item.second, page);
      }
      if (item.second.va) {
        cuMemAddressFree(item.second.va, item.second.mapped_bytes);
      }
    }
    stable_weights_.clear();
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
  size_t StablePageUsefulBytes(const StableWeightArena& arena,
                               size_t page) const {
    const size_t begin = page * granularity_;
    if (begin >= arena.logical_bytes) return 0;
    return std::min(granularity_, arena.logical_bytes - begin);
  }
  double StablePageValue(const std::string& model,
                         const StableWeightArena& arena,
                         size_t page) const {
    const auto access_it = model_access_counts_.find(model);
    const size_t accesses =
        access_it == model_access_counts_.end() ? 0 : access_it->second;
    const double probability =
        total_model_access_count_ == 0
            ? 1.0
            : static_cast<double>(accesses) / total_model_access_count_;
    return static_cast<double>(StablePageUsefulBytes(arena, page)) *
           probability * arena.sensitivity;
  }
  size_t FindFreeKVSlot(size_t pages) const {
    if (pages == 0 || pages > kv_va_slots_.size()) return kv_va_slots_.size();
    size_t run = 0;
    for (size_t i = 0; i < kv_va_slots_.size(); ++i) {
      run = kv_va_slots_[i] ? 0 : run + 1;
      if (run == pages) return i + 1 - pages;
    }
    return kv_va_slots_.size();
  }
  void MarkKVSlots(size_t start, size_t pages, bool used) {
    for (size_t i = 0; i < pages; ++i) kv_va_slots_[start + i] = used;
  }
  void RollbackKVBlocks(const std::vector<uint64_t>& logical_ids) {
    ReleaseKVBlocks(logical_ids);
  }
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
    while (free_pages() < pages) {
      auto victim = weights_.end(); double best = std::numeric_limits<double>::infinity();
      StableWeightArena* stable_victim = nullptr;
      size_t stable_victim_page = 0;
      uint64_t oldest_access = std::numeric_limits<uint64_t>::max();
      for (auto it = weights_.begin(); it != weights_.end(); ++it) {
        if (protected_fps.count(it->first)) continue;
        const auto& a = it->second;
        const double score = (double(a.access_count) / std::max<uint64_t>(1, clock_)) * a.sensitivity;
        if (score < best) { best = score; victim = it; }
      }
      for (auto& item : stable_weights_) {
        if (protected_fps.count(item.first)) continue;
        auto& arena = item.second;
        for (size_t page = 0; page < arena.physical_extent_ids.size(); ++page) {
          if (arena.physical_extent_ids[page] < 0) continue;
          const double score = StablePageValue(item.first, arena, page);
          if (score < best ||
              (score == best && arena.last_access[page] < oldest_access)) {
            best = score;
            oldest_access = arena.last_access[page];
            victim = weights_.end();
            stable_victim = &arena;
            stable_victim_page = page;
          }
        }
      }
      if (stable_victim) {
        const size_t begin = stable_victim_page * granularity_;
        evicted_weight_bytes_ +=
            std::min(granularity_, stable_victim->logical_bytes - begin);
        ReleaseStablePage(stable_victim, stable_victim_page);
      } else if (victim != weights_.end()) {
        evicted_weight_bytes_ += victim->second.logical_bytes;
        Release(&victim->second);
        weights_.erase(victim);
      } else {
        return false;
      }
    }
    return true;
  }
  void ReleaseStablePage(StableWeightArena* arena, size_t page) {
    if (!arena || page >= arena->physical_extent_ids.size()) return;
    const int64_t id = arena->physical_extent_ids[page];
    if (id < 0) return;
    cuMemUnmap(arena->va + page * granularity_, granularity_);
    physical_extents_[static_cast<size_t>(id)].in_use = false;
    free_extent_ids_.push_back(static_cast<size_t>(id));
    arena->physical_extent_ids[page] = -1;
    --used_pages_;
  }
  bool Map(Allocation* a) {
    try {
      Check(cuMemAddressReserve(&a->va, a->mapped_bytes, granularity_, 0, 0), "cuMemAddressReserve");
      a->owns_va = true;
      return MapAtReservedAddress(a);
    } catch (const std::exception& e) {
      LOG(ERROR) << "VMM address reservation failed on GPU " << device_id_
                 << ", mapped_bytes=" << a->mapped_bytes << ": " << e.what();
      Release(a);
      return false;
    }
  }
  bool MapAtReservedAddress(Allocation* a) {
    try {
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
      access.location.type = CU_MEM_LOCATION_TYPE_DEVICE; access.location.id = physical_device_id_;
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
    const bool owns_va = a->owns_va;
    const CUdeviceptr va = a->va;
    const size_t mapped_bytes = a->mapped_bytes;
    *a = {};
    if (owns_va) cuMemAddressFree(va, mapped_bytes);
  }
  int device_id_;
  int physical_device_id_ = 0;
  size_t capacity_bytes_;
  size_t native_granularity_ = 0;
  size_t granularity_ = 0;
  size_t page_capacity_ = 0, used_pages_ = 0; uint64_t clock_ = 0;
  CUmemAllocationProp prop_{};
  std::vector<PhysicalExtent> physical_extents_;
  std::vector<size_t> free_extent_ids_;
  std::unordered_map<std::string, Allocation> weights_;
  std::unordered_map<std::string, StableWeightArena> stable_weights_;
  std::unordered_map<std::string, StableBindingAliases> binding_aliases_;
  std::unordered_map<std::string, size_t> model_access_counts_;
  size_t total_model_access_count_ = 0;
  CUdeviceptr kv_va_base_ = 0;
  size_t kv_va_capacity_ = 0;
  std::vector<bool> kv_va_slots_;
  std::unordered_map<uint64_t, Allocation> kv_blocks_;
  std::vector<Allocation> legacy_kv_;
  size_t evicted_weight_bytes_ = 0;
  size_t reclaimed_kv_bytes_ = 0;
  size_t map_extent_count_ = 0;
};

class VmmVRAMManager final : public IVRAMManager {
 public:
  struct TensorBinding {
    std::string name;
    CUdeviceptr group_base = 0;
    size_t offset = 0;
    size_t size = 0;
  };
  struct ModelLayout {
    size_t page_count = 0;
    size_t contiguous_runs = 0;
    size_t adjacent_breaks = 0;
    size_t extent_id_span = 0;
  };
  struct LayerWeaveLoadResult {
    size_t requested_pages = 0;
    size_t cached_pages = 0;
    size_t mapped_pages = 0;
    size_t cached_bytes = 0;
    size_t to_load_bytes = 0;
  };
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
    return RegisterModelWithMergeTarget(
        path, sensitivity, mock_copy, granularity == -1 ? -1 : 40);
  }
  int64_t RegisterModelWithMergeTarget(const std::string& path,
                                       double sensitivity, bool mock_copy,
                                       int merge_target_count) {
    if (mock_copy) return -1;
    if (models_.count(path)) return models_.at(path)->model_size();
    auto model = std::make_shared<RegisteredModel>(
        path, sensitivity, merge_target_count);
    // The allocation simulator historically coalesces groups to roughly 40.
    // Real inference needs the exact offsets emitted by save_tensor_group_dict;
    // -1 keeps checkpoint groups; -2 expands them to one unit per tensor.
    if (merge_target_count == -2) {
      model->SplitTensorGroupsToTensors();
    } else if (merge_target_count > 0) {
      model->MergeTGsRatio(merge_target_count);
    }
    if (model->LoadModelFromDisk(8) != 0) return -1;
    std::vector<size_t> offsets;
    offsets.reserve(model->GetTensorGroupIndexes().size());
    constexpr size_t kTensorAlignment = 256;
    size_t packed_bytes = 0;
    for (const auto& group : model->GetTensorGroupIndexes()) {
      packed_bytes =
          (packed_bytes + kTensorAlignment - 1) & ~(kTensorAlignment - 1);
      offsets.push_back(packed_bytes);
      packed_bytes += group.size;
    }
    std::vector<VmmGpuPagePool*> reserved;
    for (auto& item : pools_) {
      if (!item.second->ReserveStableWeight(path, packed_bytes, sensitivity)) {
        for (auto* pool : reserved) pool->DropStableWeight(path, true);
        return -1;
      }
      reserved.push_back(item.second.get());
    }
    model_group_offsets_[path] = std::move(offsets);
    model_packed_bytes_[path] = packed_bytes;
    models_[path] = model;
    return model->model_size();
  }
  void WarmupModelAccess(const std::unordered_map<std::string, size_t>& counts) override {
    for (const auto& p : counts) {
      model_access_counts_[p.first] += p.second;
      for (auto& pool : pools_) {
        pool.second->SetModelAccessCount(
            p.first, model_access_counts_[p.first]);
      }
    }
  }
  std::string LoadModel(const std::string& path, int device_id, int, int, bool,
                        bool disable_reuse) override {
    auto mit = models_.find(path); auto pit = pools_.find(device_id);
    if (mit == models_.end() || pit == pools_.end()) return "ERROR";
    auto& model = mit->second; auto& pool = *pit->second;
    ++model_access_counts_[path];
    for (auto& pool_item : pools_) {
      pool_item.second->SetModelAccessCount(
          path, model_access_counts_[path]);
    }
    pool.ResetOperationStats();
    pool.CleanKV();  // Current simulator semantic: a new model load releases old KV.
    const auto& groups = model->GetTensorGroupIndexes();
    if (disable_reuse) pool.DropStableWeight(path);
    std::unordered_set<std::string> protect{path};
    std::vector<size_t> missing_pages;
    if (!pool.MapStableWeight(path, protect, &missing_pages)) return "ERROR";

    cudaError_t cuda_status = cudaSetDevice(device_id);
    if (cuda_status != cudaSuccess) {
      pool.RollbackStablePages(path, missing_pages);
      return "ERROR";
    }
    const auto host_ptrs = model->GetTensorGroupHostPtr();
    const auto& offsets = model_group_offsets_.at(path);
    const size_t page_size = pool.granularity();
    const CUdeviceptr base = pool.StableWeightAddress(path);
    size_t to_load_bytes = 0;
    for (size_t page : missing_pages) {
      const size_t page_begin = page * page_size;
      const size_t page_end =
          std::min(page_begin + page_size, model_packed_bytes_.at(path));
      for (size_t group_id = 0; group_id < groups.size(); ++group_id) {
        const size_t group_begin = offsets[group_id];
        const size_t group_end = group_begin + groups[group_id].size;
        const size_t copy_begin = std::max(page_begin, group_begin);
        const size_t copy_end = std::min(page_end, group_end);
        if (copy_begin >= copy_end) continue;
        char* host = static_cast<char*>(host_ptrs->get(group_id));
        if (!host) {
          pool.RollbackStablePages(path, missing_pages);
          return "ERROR";
        }
        cuda_status = cudaMemcpy(
            reinterpret_cast<void*>(base + copy_begin),
            host + (copy_begin - group_begin), copy_end - copy_begin,
            cudaMemcpyHostToDevice);
        if (cuda_status != cudaSuccess) {
          LOG(ERROR) << "Stable VMM H2D copy failed: "
                     << cudaGetErrorString(cuda_status);
          pool.RollbackStablePages(path, missing_pages);
          return "ERROR";
        }
        to_load_bytes += copy_end - copy_begin;
      }
    }
    size_t total_weight_bytes = 0;
    for (const auto& group : groups) total_weight_bytes += group.size;
    const size_t cached_bytes = total_weight_bytes - to_load_bytes;
    total_cached_weight_bytes_ += cached_bytes;
    total_weight_bytes_ += total_weight_bytes;
    total_h2d_weight_bytes_ += to_load_bytes;
    model_cached_weight_bytes_[path] += cached_bytes;
    model_weight_bytes_[path] += total_weight_bytes;
    model_h2d_weight_bytes_[path] += to_load_bytes;
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
    const auto& groups = mit->second->GetTensorGroupIndexes();
    const auto& offsets = model_group_offsets_.at(path);
    const size_t page_size = pit->second->granularity();
    for (size_t group_id = 0; group_id < groups.size(); ++group_id) {
      total += groups[group_id].size;
      if (disable_reuse) continue;
      const size_t group_begin = offsets[group_id];
      const size_t group_end = group_begin + groups[group_id].size;
      for (size_t page = group_begin / page_size;
           page <= (group_end - 1) / page_size; ++page) {
        if (!pit->second->StablePageResident(path, page)) continue;
        const size_t begin = std::max(group_begin, page * page_size);
        const size_t end = std::min(group_end, (page + 1) * page_size);
        cached += end - begin;
      }
    }
    to_load = total - cached;
    return true;
  }
  int64_t GetGPUToLoad(const std::string& path, int policy) override {
    if (pools_.empty() || !models_.count(path)) return -1;
    std::vector<int> ids;
    for (const auto& pool : pools_) ids.push_back(pool.first);
    std::sort(ids.begin(), ids.end());
    if (policy == 0) {
      ++gpu_selection_counts_[ids.front()];
      return ids.front();
    }
    int best = -1;
    VmmGpuPagePool::StableLoadEstimate best_estimate;
    for (int id : ids) {
      const auto& pool = *pools_.at(id);
      const auto estimate = pool.EstimateStableLoad(path);
      if (best < 0 ||
          estimate.placement_cost_bytes < best_estimate.placement_cost_bytes ||
          (estimate.placement_cost_bytes ==
               best_estimate.placement_cost_bytes &&
           estimate.missing_logical_bytes <
               best_estimate.missing_logical_bytes) ||
          (estimate.placement_cost_bytes ==
               best_estimate.placement_cost_bytes &&
           estimate.missing_logical_bytes ==
               best_estimate.missing_logical_bytes &&
           id < best)) {
        best = id;
        best_estimate = estimate;
      }
    }
    if (best >= 0) {
      ++gpu_selection_counts_[best];
      gpu_selection_estimated_eviction_value_[best] +=
          best_estimate.eviction_value_bytes;
      gpu_selection_estimated_missing_bytes_[best] +=
          best_estimate.missing_logical_bytes;
    }
    return best;
  }
  int GetAvailableBlocks(const std::string& model, size_t block_size, int device) override {
    auto it = pools_.find(device); if (it == pools_.end() || block_size == 0) return 0;
    std::unordered_set<std::string> protect;
    if (models_.count(model)) protect.insert(model);
    return static_cast<int>((it->second->ReclaimablePages(protect) *
                             it->second->granularity()) / block_size);
  }
  std::vector<size_t> AllocateBlocks(int device, size_t block, const std::string& model, int count) override {
    auto it = pools_.find(device); if (it == pools_.end()) return {};
    std::unordered_set<std::string> protect;
    if (models_.count(model)) protect.insert(model);
    return it->second->MapKV(block, count, protect);
  }
  uint64_t GetKVBaseAddress(int device) const {
    auto it = pools_.find(device);
    return it == pools_.end() ? 0 : it->second->KVBaseAddress();
  }
  int GetAvailableKVBlocks(const std::string& model, size_t block_size,
                           int device) const {
    auto it = pools_.find(device);
    if (it == pools_.end() || !models_.count(model) || block_size == 0) return -1;
    std::unordered_set<std::string> protect;
    protect.insert(model);
    return static_cast<int>(
        it->second->AvailableKVBlocks(block_size, protect));
  }
  bool AllocateKVBlocks(const std::string& model, size_t block_size, int device,
                        const std::vector<uint64_t>& logical_ids,
                        std::vector<uint64_t>* offsets) {
    auto it = pools_.find(device);
    auto model_it = models_.find(model);
    if (it == pools_.end() || model_it == models_.end()) return false;
    std::unordered_set<std::string> protect;
    protect.insert(model);
    return it->second->AllocateKVBlocks(block_size, logical_ids, protect,
                                        offsets);
  }
  bool AllocateKVBlocksUnprotected(
      size_t block_size, int device,
      const std::vector<uint64_t>& logical_ids,
      std::vector<uint64_t>* offsets) {
    auto it = pools_.find(device);
    if (it == pools_.end()) return false;
    return it->second->AllocateKVBlocks(block_size, logical_ids, {}, offsets);
  }
  bool ReleaseKVBlocks(int device,
                       const std::vector<uint64_t>& logical_ids) {
    auto it = pools_.find(device);
    return it != pools_.end() && it->second->ReleaseKVBlocks(logical_ids);
  }
  bool ClearKV(int device) {
    auto it = pools_.find(device);
    if (it == pools_.end()) return false;
    it->second->CleanKV();
    return true;
  }
  void MemoryUsage() override {
    for (const auto& p : pools_) {
      LOG(METRIC) << "VMM GPU " << p.first
                  << ", utilization=" << p.second->utilization()
                  << ", physical_page_bytes=" << p.second->granularity()
                  << ", native_granularity_bytes="
                  << p.second->native_granularity();
    }
    const double reduced_io = total_weight_bytes_ == 0
                                  ? 0.0
                                  : static_cast<double>(total_cached_weight_bytes_) /
                                        static_cast<double>(total_weight_bytes_);
    LOG(METRIC) << "Reduced IO: " << total_cached_weight_bytes_ << " / "
                << total_weight_bytes_ << " = " << reduced_io;
    LOG(METRIC) << "VMM weight H2D bytes: " << total_h2d_weight_bytes_;
    for (const auto& item : model_weight_bytes_) {
      const size_t model_cached = model_cached_weight_bytes_[item.first];
      const double model_reduced_io = item.second == 0
                                          ? 0.0
                                          : static_cast<double>(model_cached) /
                                                static_cast<double>(item.second);
      LOG(METRIC) << "Model: " << item.first << " Reduced IO: "
                  << model_cached << " / " << item.second << " = "
                  << model_reduced_io << ", H2D bytes: "
                  << model_h2d_weight_bytes_[item.first];
    }
    for (const auto& pool : pools_) {
      const int gpu = pool.first;
      LOG(METRIC) << "VMM GPU selection: gpu=" << gpu
                  << ", count=" << gpu_selection_counts_[gpu]
                  << ", estimated_missing_bytes="
                  << gpu_selection_estimated_missing_bytes_[gpu]
                  << ", estimated_eviction_value_bytes="
                  << gpu_selection_estimated_eviction_value_[gpu];
    }
  }
  double get_memory_utilization() override { double total = 0; for (const auto& p : pools_) total += p.second->utilization(); return pools_.empty() ? 0 : total / pools_.size(); }
  bool GetTensorBindings(const std::string& path, int device_id,
                         std::vector<TensorBinding>* out) const {
    if (!out) return false;
    auto mit = models_.find(path);
    auto pit = pools_.find(device_id);
    if (mit == models_.end() || pit == pools_.end()) return false;
    out->clear();
    const CUdeviceptr model_base = pit->second->StableWeightAddress(path);
    const auto& offsets = model_group_offsets_.at(path);
    const auto& groups = mit->second->GetTensorGroupIndexes();
    for (size_t group_id = 0; group_id < groups.size(); ++group_id) {
      const auto& group = groups[group_id];
      const CUdeviceptr base = model_base + offsets[group_id];
      for (const auto& tensor : group.tensor_indexes) {
        out->push_back(
            {tensor.name, base, tensor.offset, tensor.size});
      }
    }
    return true;
  }
  bool GetLayerWeaveLayout(const std::string& path, int device_id,
                           uint64_t* model_base, size_t* page_size,
                           size_t* page_count) const {
    auto mit = models_.find(path);
    auto pit = pools_.find(device_id);
    if (mit == models_.end() || pit == pools_.end() ||
        !model_base || !page_size || !page_count) {
      return false;
    }
    *model_base = pit->second->StableWeightAddress(path);
    *page_size = pit->second->granularity();
    const size_t logical_bytes = model_packed_bytes_.at(path);
    *page_count = (logical_bytes + *page_size - 1) / *page_size;
    return true;
  }
  bool BeginLayerWeaveBinding(const std::string& path, int device_id) {
    auto mit = models_.find(path);
    auto pit = pools_.find(device_id);
    return mit != models_.end() && pit != pools_.end() &&
           pit->second->BeginStableBinding(path);
  }
  bool EndLayerWeaveBinding(const std::string& path, int device_id) {
    auto mit = models_.find(path);
    auto pit = pools_.find(device_id);
    return mit != models_.end() && pit != pools_.end() &&
           pit->second->EndStableBinding(path);
  }
  bool GetLayerWeaveResidency(const std::string& path, int device_id,
                              std::vector<uint8_t>* residency) const {
    auto mit = models_.find(path);
    auto pit = pools_.find(device_id);
    if (mit == models_.end() || pit == pools_.end() || !residency) {
      return false;
    }
    const auto& arena = pit->second->StableWeight(path);
    residency->resize(arena.physical_extent_ids.size());
    for (size_t page = 0; page < arena.physical_extent_ids.size(); ++page) {
      (*residency)[page] = arena.physical_extent_ids[page] >= 0 ? 1 : 0;
    }
    return true;
  }
  bool ConfigureLayerWeaveCache(
      const std::string& path, int device_id,
      const std::vector<size_t>& retained_pages) {
    auto mit = models_.find(path);
    auto pit = pools_.find(device_id);
    if (mit == models_.end() || pit == pools_.end()) return false;
    const auto& arena = pit->second->StableWeight(path);
    std::unordered_set<size_t> retained;
    retained.reserve(retained_pages.size());
    for (size_t page : retained_pages) {
      if (page >= arena.physical_extent_ids.size()) return false;
      retained.insert(page);
    }
    return pit->second->RetainStablePages(path, retained);
  }
  bool PrepareLayerWeavePages(
      const std::string& path, int device_id,
      const std::vector<size_t>& requested_pages, cudaStream_t stream,
      LayerWeaveLoadResult* result) {
    auto mit = models_.find(path);
    auto pit = pools_.find(device_id);
    if (mit == models_.end() || pit == pools_.end() || !result) return false;
    auto& model = mit->second;
    auto& pool = *pit->second;
    *result = {};
    result->requested_pages = requested_pages.size();
    for (size_t page : requested_pages) {
      if (pool.StablePageResident(path, page)) ++result->cached_pages;
    }
    std::unordered_set<std::string> protect{path};
    std::vector<size_t> newly_mapped_pages;
    if (!pool.MapStablePages(
            path, requested_pages, protect, &newly_mapped_pages)) {
      return false;
    }
    result->mapped_pages = newly_mapped_pages.size();
    const auto host_ptrs = model->GetTensorGroupHostPtr();
    const auto& groups = model->GetTensorGroupIndexes();
    const auto& offsets = model_group_offsets_.at(path);
    const size_t page_size = pool.granularity();
    const size_t packed_bytes = model_packed_bytes_.at(path);
    const CUdeviceptr base = pool.StableWeightAddress(path);
    for (size_t page : requested_pages) {
      const size_t page_begin = page * page_size;
      const size_t page_end = std::min(page_begin + page_size, packed_bytes);
      const bool newly_mapped =
          std::find(newly_mapped_pages.begin(), newly_mapped_pages.end(),
                    page) != newly_mapped_pages.end();
      for (size_t group_id = 0; group_id < groups.size(); ++group_id) {
        const size_t group_begin = offsets[group_id];
        const size_t group_end = group_begin + groups[group_id].size;
        const size_t copy_begin = std::max(page_begin, group_begin);
        const size_t copy_end = std::min(page_end, group_end);
        if (copy_begin >= copy_end) continue;
        if (!newly_mapped) {
          result->cached_bytes += copy_end - copy_begin;
          continue;
        }
        char* host = static_cast<char*>(host_ptrs->get(group_id));
        if (!host) {
          pool.RollbackStablePages(path, newly_mapped_pages);
          return false;
        }
        const cudaError_t status = cudaMemcpyAsync(
            reinterpret_cast<void*>(base + copy_begin),
            host + (copy_begin - group_begin), copy_end - copy_begin,
            cudaMemcpyHostToDevice, stream);
        if (status != cudaSuccess) {
          LOG(ERROR) << "LayerWeave async H2D failed: "
                     << cudaGetErrorString(status);
          // Do not unmap pages after an async copy may have entered the stream.
          return false;
        }
        result->to_load_bytes += copy_end - copy_begin;
      }
    }
    return true;
  }
  bool GetModelLayout(const std::string& path, int device_id,
                      ModelLayout* out) const {
    if (!out) return false;
    auto mit = models_.find(path);
    auto pit = pools_.find(device_id);
    if (mit == models_.end() || pit == pools_.end()) return false;
    *out = {};
    if (!pit->second->StableWeightFullyResident(path)) return false;
    bool have_previous = false;
    size_t previous_id = 0;
    size_t min_id = std::numeric_limits<size_t>::max();
    size_t max_id = 0;
    const auto& arena = pit->second->StableWeight(path);
    for (int64_t extent_id : arena.physical_extent_ids) {
      if (extent_id < 0) return false;
      const size_t id = static_cast<size_t>(extent_id);
      if (!have_previous ||
          (id + 1 != previous_id && previous_id + 1 != id)) {
        ++out->contiguous_runs;
        if (have_previous) ++out->adjacent_breaks;
      }
      have_previous = true;
      previous_id = id;
      min_id = std::min(min_id, id);
      max_id = std::max(max_id, id);
      ++out->page_count;
    }
    if (out->page_count) out->extent_id_span = max_id - min_id + 1;
    return true;
  }
 private:
  std::unordered_map<std::string, std::shared_ptr<RegisteredModel>> models_;
  std::unordered_map<std::string, std::vector<size_t>> model_group_offsets_;
  std::unordered_map<std::string, size_t> model_packed_bytes_;
  std::unordered_map<int, std::unique_ptr<VmmGpuPagePool>> pools_;
  std::unordered_map<std::string, size_t> model_access_counts_;
  std::unordered_map<int, size_t> gpu_selection_counts_;
  std::unordered_map<int, double> gpu_selection_estimated_missing_bytes_;
  std::unordered_map<int, double>
      gpu_selection_estimated_eviction_value_;
  size_t total_cached_weight_bytes_ = 0;
  size_t total_weight_bytes_ = 0;
  size_t total_h2d_weight_bytes_ = 0;
  std::unordered_map<std::string, size_t> model_cached_weight_bytes_;
  std::unordered_map<std::string, size_t> model_weight_bytes_;
  std::unordered_map<std::string, size_t> model_h2d_weight_bytes_;
};
