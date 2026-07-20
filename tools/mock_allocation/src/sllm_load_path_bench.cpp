#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <errno.h>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

#include "binary_utils.h"
#include "registered_model.h"

namespace {

using Clock = std::chrono::steady_clock;

struct BenchOptions {
  std::string model_path;
  std::string mode = "sllm";
  int gpu_id = 0;
  int repeats = 3;
  size_t pipeline_depth = 2;
  bool flush_page_cache = false;
};

struct BenchResult {
  int repeat = 0;
  std::string mode;
  size_t model_bytes = 0;
  size_t group_count = 0;
  double total_ms = 0.0;
  double ssd_to_cpu_ms = 0.0;
  double cpu_to_gpu_ms = 0.0;
  double overlap_gap_ms = 0.0;
};

struct ModelFiles {
  size_t model_bytes = 0;
  std::vector<size_t> partition_sizes;
  std::vector<std::filesystem::path> partition_paths;
};

struct SummaryStats {
  double mean = 0.0;
  double min = 0.0;
  double max = 0.0;
  double p50 = 0.0;
  double p95 = 0.0;
  double p99 = 0.0;
};

struct BufferSlot {
  void* host_ptr = nullptr;
  void* device_ptr = nullptr;
  size_t capacity = 0;
  size_t bytes = 0;
  bool ready = false;
  bool consumed = true;
  bool stop = false;
  size_t group_idx = 0;
};

struct GpuBuffer {
  void* ptr = nullptr;
  size_t capacity = 0;

  void EnsureCapacity(size_t bytes) {
    if (capacity >= bytes) {
      return;
    }
    if (ptr != nullptr) {
      cudaFree(ptr);
      ptr = nullptr;
      capacity = 0;
    }
    const cudaError_t error = cudaMalloc(&ptr, bytes);
    if (error != cudaSuccess) {
      throw std::runtime_error(std::string("cudaMalloc: ") +
                               cudaGetErrorString(error));
    }
    capacity = bytes;
  }

  ~GpuBuffer() {
    if (ptr != nullptr) {
      cudaFree(ptr);
      ptr = nullptr;
    }
  }
};

void CheckCuda(cudaError_t error, const std::string& context) {
  if (error != cudaSuccess) {
    throw std::runtime_error(context + ": " + cudaGetErrorString(error));
  }
}

SummaryStats ComputeStats(const std::vector<double>& values) {
  if (values.empty()) {
    return {};
  }
  std::vector<double> sorted = values;
  std::sort(sorted.begin(), sorted.end());
  SummaryStats stats;
  stats.min = sorted.front();
  stats.max = sorted.back();
  stats.mean = std::accumulate(sorted.begin(), sorted.end(), 0.0) /
               static_cast<double>(sorted.size());
  auto percentile = [&](double p) {
    const double idx = p * static_cast<double>(sorted.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(idx));
    const size_t hi = static_cast<size_t>(std::ceil(idx));
    const double frac = idx - static_cast<double>(lo);
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
  };
  stats.p50 = percentile(0.50);
  stats.p95 = percentile(0.95);
  stats.p99 = percentile(0.99);
  return stats;
}

BenchOptions ParseArgs(int argc, char* argv[]) {
  BenchOptions options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const std::string& flag) -> std::string {
      if (i + 1 >= argc) {
        throw std::invalid_argument("Missing value for " + flag);
      }
      return argv[++i];
    };

    if (arg == "--model-path") {
      options.model_path = require_value(arg);
    } else if (arg == "--mode") {
      options.mode = require_value(arg);
    } else if (arg == "--gpu-id") {
      options.gpu_id = std::stoi(require_value(arg));
    } else if (arg == "--repeats") {
      options.repeats = std::stoi(require_value(arg));
    } else if (arg == "--pipeline-depth") {
      options.pipeline_depth = static_cast<size_t>(std::stoul(require_value(arg)));
    } else if (arg == "--flush-page-cache") {
      options.flush_page_cache = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage: sllm_load_path_bench --model-path <rank_dir> [options]\n"
          << "  --mode <sllm|sllm_cpu_only>\n"
          << "  --gpu-id <id>\n"
          << "  --repeats <n>\n"
          << "  --pipeline-depth <n>\n"
          << "  --flush-page-cache\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("Unknown argument: " + arg);
    }
  }

  if (options.model_path.empty()) {
    throw std::invalid_argument("--model-path is required");
  }
  if (options.mode != "sllm" && options.mode != "sllm_cpu_only") {
    throw std::invalid_argument("--mode must be one of: sllm, sllm_cpu_only");
  }
  if (options.repeats <= 0) {
    throw std::invalid_argument("--repeats must be > 0");
  }
  if (options.pipeline_depth == 0) {
    throw std::invalid_argument("--pipeline-depth must be > 0");
  }
  return options;
}

std::vector<int> OpenPartitions(
    const std::vector<std::filesystem::path>& partition_paths) {
  std::vector<int> fds;
  for (const auto& path : partition_paths) {
    int fd = open(path.c_str(), O_DIRECT | O_RDONLY);
    if (fd < 0) {
      throw std::runtime_error("open failed for " + path.string() + ": " +
                               std::strerror(errno));
    }
    fds.push_back(fd);
  }
  return fds;
}

void ClosePartitions(const std::vector<int>& fds) {
  for (int fd : fds) {
    if (fd >= 0) {
      close(fd);
    }
  }
}

std::pair<size_t, size_t> LocatePartitionOffset(
    size_t global_offset, const std::vector<size_t>& partition_sizes) {
  size_t partition_id = 0;
  size_t offset = global_offset;
  while (partition_id < partition_sizes.size() &&
         offset >= partition_sizes[partition_id]) {
    offset -= partition_sizes[partition_id];
    ++partition_id;
  }
  if (partition_id >= partition_sizes.size()) {
    throw std::runtime_error("tensor group offset exceeds partition size");
  }
  return {partition_id, offset};
}

void ReadGroup(const TensorGroupIndex& group, void* host_ptr,
               const std::vector<int>& fds,
               const std::vector<size_t>& partition_sizes) {
  auto [partition_id, in_partition_offset] =
      LocatePartitionOffset(group.file_offset, partition_sizes);
  size_t remaining = group.size;
  char* dst = static_cast<char*>(host_ptr);
  while (remaining > 0) {
    const size_t readable =
        std::min(remaining, partition_sizes[partition_id] - in_partition_offset);
    const ssize_t ret =
        pread(fds[partition_id], dst, readable, in_partition_offset);
    if (ret < 0 || static_cast<size_t>(ret) != readable) {
      throw std::runtime_error("pread failed while reading tensor group at offset " +
                               std::to_string(group.file_offset));
    }
    remaining -= readable;
    dst += readable;
    ++partition_id;
    in_partition_offset = 0;
  }
}

void MaybeFlushPageCache(const std::vector<std::filesystem::path>& partition_paths) {
  if (!partition_paths.empty()) {
    std::cerr << "flush-page-cache requested, but automatic drop_caches is "
                 "not performed by this benchmark. Please clear caches "
                 "manually before running if needed.\n";
  }
}

ModelFiles ScanModelFiles(const std::string& model_path) {
  ModelFiles files;
  for (int partition_id = 0;; ++partition_id) {
    const std::filesystem::path tensor_path =
        std::filesystem::path(model_path) /
        ("tensor.data_" + std::to_string(partition_id));
    if (!std::filesystem::exists(tensor_path)) {
      break;
    }
    const auto size = std::filesystem::file_size(tensor_path);
    files.model_bytes += size;
    files.partition_sizes.push_back(size);
    files.partition_paths.push_back(tensor_path);
  }
  if (files.model_bytes == 0 || files.partition_paths.empty()) {
    throw std::runtime_error("No tensor.data_* files found under " + model_path);
  }
  return files;
}

BenchResult RunCpuOnly(const BenchOptions& options,
                       const std::vector<TensorGroupIndex>& groups,
                       const std::vector<int>& fds,
                       const std::vector<size_t>& partition_sizes,
                       size_t model_bytes) {
  BenchResult result;
  result.mode = "sllm_cpu_only";
  result.model_bytes = model_bytes;
  result.group_count = groups.size();

  std::vector<void*> host_buffers(groups.size(), nullptr);
  GpuBuffer gpu_buffer;
  for (size_t i = 0; i < groups.size(); ++i) {
    host_buffers[i] = allocateAlignedPinnedMemory(groups[i].size, 4096);
    if (host_buffers[i] == nullptr) {
      throw std::runtime_error("Failed to allocate pinned host memory");
    }
  }

  const auto load_start = Clock::now();
  for (size_t i = 0; i < groups.size(); ++i) {
    ReadGroup(groups[i], host_buffers[i], fds, partition_sizes);
  }
  const auto load_end = Clock::now();
  result.ssd_to_cpu_ms =
      std::chrono::duration<double, std::milli>(load_end - load_start).count();

  cudaStream_t stream;
  CheckCuda(cudaSetDevice(options.gpu_id), "cudaSetDevice");
  CheckCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
  const auto copy_start = Clock::now();
  for (size_t i = 0; i < groups.size(); ++i) {
    gpu_buffer.EnsureCapacity(groups[i].size);
    CheckCuda(cudaMemcpyAsync(gpu_buffer.ptr, host_buffers[i], groups[i].size,
                              cudaMemcpyHostToDevice, stream),
              "cudaMemcpyAsync");
    CheckCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  }
  const auto copy_end = Clock::now();
  CheckCuda(cudaStreamDestroy(stream), "cudaStreamDestroy");

  result.cpu_to_gpu_ms =
      std::chrono::duration<double, std::milli>(copy_end - copy_start).count();
  result.total_ms = result.cpu_to_gpu_ms;
  result.overlap_gap_ms = 0.0;

  for (void* ptr : host_buffers) {
    freeAlignedPinnedMemory(ptr);
  }
  return result;
}

BenchResult RunPipelined(const BenchOptions& options,
                         const std::vector<TensorGroupIndex>& groups,
                         const std::vector<int>& fds,
                         const std::vector<size_t>& partition_sizes,
                         size_t model_bytes) {
  BenchResult result;
  result.mode = "sllm";
  result.model_bytes = model_bytes;
  result.group_count = groups.size();

  std::vector<BufferSlot> slots(options.pipeline_depth);
  for (auto& slot : slots) {
    slot.consumed = true;
  }
  GpuBuffer gpu_buffer;

  std::mutex mutex;
  std::condition_variable cv_producer;
  std::condition_variable cv_consumer;
  std::exception_ptr worker_error;
  size_t next_to_produce = 0;

  double producer_ms = 0.0;
  std::thread producer([&]() {
    try {
      while (next_to_produce < groups.size()) {
        BufferSlot* target = nullptr;
        {
          std::unique_lock<std::mutex> lock(mutex);
          cv_producer.wait(lock, [&]() {
            for (auto& slot : slots) {
              if (slot.consumed && !slot.ready) {
                target = &slot;
                return true;
              }
            }
            return false;
          });
          target->consumed = false;
          target->group_idx = next_to_produce;
        }

        const auto start = Clock::now();
        const auto& group = groups[next_to_produce];
        if (target->capacity < group.size) {
          if (target->host_ptr != nullptr) {
            freeAlignedPinnedMemory(target->host_ptr);
          }
          target->host_ptr = allocateAlignedPinnedMemory(group.size, 4096);
          if (target->host_ptr == nullptr) {
            throw std::runtime_error("Failed to allocate pinned host memory");
          }
          target->capacity = group.size;
        }
        ReadGroup(group, target->host_ptr, fds, partition_sizes);
        target->bytes = group.size;
        const auto end = Clock::now();
        producer_ms +=
            std::chrono::duration<double, std::milli>(end - start).count();

        {
          std::lock_guard<std::mutex> lock(mutex);
          target->ready = true;
          ++next_to_produce;
        }
        cv_consumer.notify_one();
      }
    } catch (...) {
      worker_error = std::current_exception();
      cv_consumer.notify_all();
    }

    {
      std::lock_guard<std::mutex> lock(mutex);
      for (auto& slot : slots) {
        slot.stop = true;
      }
    }
    cv_consumer.notify_all();
  });

  CheckCuda(cudaSetDevice(options.gpu_id), "cudaSetDevice");
  cudaStream_t stream;
  CheckCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
  const auto total_start = Clock::now();
  double consumer_ms = 0.0;
  size_t consumed = 0;
  while (consumed < groups.size()) {
    BufferSlot* source = nullptr;
    {
      std::unique_lock<std::mutex> lock(mutex);
      cv_consumer.wait(lock, [&]() {
        if (worker_error != nullptr) {
          return true;
        }
        for (auto& slot : slots) {
          if (slot.ready) {
            source = &slot;
            return true;
          }
        }
        return false;
      });
      if (worker_error != nullptr) {
        break;
      }
      source->ready = false;
    }

    const auto copy_start = Clock::now();
    gpu_buffer.EnsureCapacity(source->bytes);
    CheckCuda(cudaMemcpyAsync(gpu_buffer.ptr, source->host_ptr,
                              source->bytes, cudaMemcpyHostToDevice, stream),
              "cudaMemcpyAsync");
    CheckCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    const auto copy_end = Clock::now();
    consumer_ms +=
        std::chrono::duration<double, std::milli>(copy_end - copy_start).count();

    {
      std::lock_guard<std::mutex> lock(mutex);
      source->consumed = true;
      ++consumed;
    }
    cv_producer.notify_one();
  }
  const auto total_end = Clock::now();
  CheckCuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
  producer.join();

  if (worker_error != nullptr) {
    std::rethrow_exception(worker_error);
  }

  result.ssd_to_cpu_ms = producer_ms;
  result.cpu_to_gpu_ms = consumer_ms;
  result.total_ms =
      std::chrono::duration<double, std::milli>(total_end - total_start).count();
  result.overlap_gap_ms =
      std::max(0.0, result.total_ms -
                        std::max(result.ssd_to_cpu_ms, result.cpu_to_gpu_ms));

  for (auto& slot : slots) {
    if (slot.host_ptr != nullptr) {
      freeAlignedPinnedMemory(slot.host_ptr);
      slot.host_ptr = nullptr;
    }
  }
  return result;
}

void PrintSummary(const BenchOptions& options,
                  const std::vector<BenchResult>& results) {
  std::vector<double> total_ms;
  std::vector<double> ssd_to_cpu_ms;
  std::vector<double> cpu_to_gpu_ms;
  std::vector<double> overlap_gap_ms;
  for (const auto& result : results) {
    total_ms.push_back(result.total_ms);
    ssd_to_cpu_ms.push_back(result.ssd_to_cpu_ms);
    cpu_to_gpu_ms.push_back(result.cpu_to_gpu_ms);
    overlap_gap_ms.push_back(result.overlap_gap_ms);
  }

  const auto total_stats = ComputeStats(total_ms);
  const auto ssd_stats = ComputeStats(ssd_to_cpu_ms);
  const auto h2d_stats = ComputeStats(cpu_to_gpu_ms);
  const auto overlap_stats = ComputeStats(overlap_gap_ms);

  std::cout << "================ SLLM Load Path Benchmark ================\n";
  std::cout << "mode: " << options.mode << "\n";
  std::cout << "model_path: " << options.model_path << "\n";
  std::cout << "repeats: " << results.size() << "\n";
  if (!results.empty()) {
    std::cout << "model_bytes: " << results.front().model_bytes << "\n";
    std::cout << "group_count: " << results.front().group_count << "\n";
  }
  auto print_stats = [](const std::string& name, const SummaryStats& stats) {
    std::cout << std::fixed << std::setprecision(3)
              << name << " ms"
              << " mean=" << stats.mean
              << " min=" << stats.min
              << " max=" << stats.max
              << " p50=" << stats.p50
              << " p95=" << stats.p95
              << " p99=" << stats.p99 << "\n";
  };
  print_stats("total", total_stats);
  print_stats("ssd_to_cpu", ssd_stats);
  print_stats("cpu_to_gpu", h2d_stats);
  print_stats("overlap_gap", overlap_stats);
}

int RunBenchmark(const BenchOptions& options) {
  const ModelFiles files = ScanModelFiles(options.model_path);
  if (options.flush_page_cache) {
    MaybeFlushPageCache(files.partition_paths);
  }
  std::vector<TensorGroupIndex> groups;
  ParseTensorGroupIndex(options.model_path + "/tensor_group_index.txt", groups);
  if (groups.empty()) {
    throw std::runtime_error("No tensor groups found under " + options.model_path);
  }

  std::vector<BenchResult> results;
  for (int repeat = 0; repeat < options.repeats; ++repeat) {
    const auto fds = OpenPartitions(files.partition_paths);
    BenchResult result;
    if (options.mode == "sllm") {
      result = RunPipelined(options, groups, fds, files.partition_sizes,
                            files.model_bytes);
    } else {
      result = RunCpuOnly(options, groups, fds, files.partition_sizes,
                          files.model_bytes);
    }
    ClosePartitions(fds);
    result.repeat = repeat;
    results.push_back(result);
    std::cout << "[" << options.mode << "] repeat " << (repeat + 1) << "/"
              << options.repeats << ": total=" << std::fixed
              << std::setprecision(3) << result.total_ms
              << " ms, ssd_to_cpu=" << result.ssd_to_cpu_ms
              << " ms, cpu_to_gpu=" << result.cpu_to_gpu_ms
              << " ms, overlap_gap=" << result.overlap_gap_ms << " ms\n";
  }

  PrintSummary(options, results);
  return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    const BenchOptions options = ParseArgs(argc, argv);
    return RunBenchmark(options);
  } catch (const std::exception& e) {
    std::cerr << "sllm_load_path_bench failed: " << e.what() << std::endl;
    return 1;
  }
}
