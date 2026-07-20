#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr float kEpsilon = 1e-5f;
constexpr int kThreadsPerBlock = 256;

enum class LayoutType {
  kTensorLevel,
  kPageLevelRandom,
};

enum class KernelType {
  kCopy,
  kSiluMul,
  kRmsNorm,
  kQkDot,
  kAttnProjGemm,
  kFfnUpGemm,
  kDecodeQk,
};

struct BenchOptions {
  int gpu_id = 0;
  LayoutType layout = LayoutType::kTensorLevel;
  KernelType kernel = KernelType::kCopy;
  size_t page_size = 64 * 1024;
  size_t batch_size = 1;
  size_t seq_len = 2048;
  size_t hidden_size = 4096;
  size_t intermediate_size = 11008;
  size_t num_heads = 32;
  size_t head_dim = 128;
  int warmup_iters = 20;
  int measure_iters = 100;
  unsigned int random_seed = 1;
  std::string csv_path;
  std::string run_label;
  bool verbose = false;
};

struct BenchResult {
  std::string label;
  std::string layout;
  std::string kernel;
  size_t page_size = 0;
  size_t batch_size = 0;
  size_t seq_len = 0;
  size_t hidden_size = 0;
  size_t intermediate_size = 0;
  size_t num_heads = 0;
  size_t head_dim = 0;
  int warmup_iters = 0;
  int measure_iters = 0;
  size_t logical_bytes = 0;
  size_t physical_bytes = 0;
  double arithmetic_intensity = 0.0;
  double avg_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
  double bandwidth_gbps = 0.0;
  double tflops = 0.0;
  double checksum = 0.0;
};

struct DeviceTensor {
  LayoutType layout = LayoutType::kTensorLevel;
  size_t numel = 0;
  size_t num_bytes = 0;
  size_t page_size = 0;
  size_t page_count = 0;
  float* contiguous = nullptr;
  std::vector<void*> page_ptrs_host;
  void** page_ptrs_device = nullptr;
};

std::string LayoutToString(LayoutType layout) {
  switch (layout) {
    case LayoutType::kTensorLevel:
      return "tensor-level";
    case LayoutType::kPageLevelRandom:
      return "page-level-random";
  }
  return "unknown";
}

std::string KernelToString(KernelType kernel) {
  switch (kernel) {
    case KernelType::kCopy:
      return "copy";
    case KernelType::kSiluMul:
      return "silu_mul";
    case KernelType::kRmsNorm:
      return "rmsnorm";
    case KernelType::kQkDot:
      return "qk_dot";
    case KernelType::kAttnProjGemm:
      return "attn_proj_gemm";
    case KernelType::kFfnUpGemm:
      return "ffn_up_gemm";
    case KernelType::kDecodeQk:
      return "decode_qk";
  }
  return "unknown";
}

void CheckCuda(cudaError_t err, const std::string& message) {
  if (err != cudaSuccess) {
    throw std::runtime_error(message + ": " + cudaGetErrorString(err));
  }
}

size_t RoundUpDiv(size_t a, size_t b) {
  return (a + b - 1) / b;
}

void EnsureParentDir(const std::string& path) {
  if (path.empty()) {
    return;
  }
  std::filesystem::path output_path(path);
  if (output_path.has_parent_path()) {
    std::filesystem::create_directories(output_path.parent_path());
  }
}

LayoutType ParseLayout(const std::string& value) {
  if (value == "tensor-level") {
    return LayoutType::kTensorLevel;
  }
  if (value == "page-level-random") {
    return LayoutType::kPageLevelRandom;
  }
  throw std::invalid_argument(
      "--layout must be tensor-level or page-level-random");
}

KernelType ParseKernel(const std::string& value) {
  if (value == "copy") return KernelType::kCopy;
  if (value == "silu_mul") return KernelType::kSiluMul;
  if (value == "rmsnorm") return KernelType::kRmsNorm;
  if (value == "qk_dot") return KernelType::kQkDot;
  if (value == "attn_proj_gemm") return KernelType::kAttnProjGemm;
  if (value == "ffn_up_gemm") return KernelType::kFfnUpGemm;
  if (value == "decode_qk") return KernelType::kDecodeQk;
  throw std::invalid_argument(
      "--kernel must be copy, silu_mul, rmsnorm, qk_dot, attn_proj_gemm, "
      "ffn_up_gemm, or decode_qk");
}

BenchOptions ParseArgs(int argc, char* argv[]) {
  BenchOptions options;
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--gpu") {
      options.gpu_id = std::stoi(argv[++i]);
    } else if (arg == "--layout") {
      options.layout = ParseLayout(argv[++i]);
    } else if (arg == "--kernel") {
      options.kernel = ParseKernel(argv[++i]);
    } else if (arg == "--page_size_kb") {
      options.page_size = static_cast<size_t>(std::stoull(argv[++i])) * 1024ULL;
    } else if (arg == "--page_size_mb") {
      options.page_size = static_cast<size_t>(std::stoull(argv[++i])) * 1024ULL *
                          1024ULL;
    } else if (arg == "--batch_size") {
      options.batch_size = std::stoull(argv[++i]);
    } else if (arg == "--seq_len") {
      options.seq_len = std::stoull(argv[++i]);
    } else if (arg == "--hidden_size") {
      options.hidden_size = std::stoull(argv[++i]);
    } else if (arg == "--intermediate_size") {
      options.intermediate_size = std::stoull(argv[++i]);
    } else if (arg == "--num_heads") {
      options.num_heads = std::stoull(argv[++i]);
    } else if (arg == "--head_dim") {
      options.head_dim = std::stoull(argv[++i]);
    } else if (arg == "--warmup_iters") {
      options.warmup_iters = std::stoi(argv[++i]);
    } else if (arg == "--measure_iters") {
      options.measure_iters = std::stoi(argv[++i]);
    } else if (arg == "--random_seed") {
      options.random_seed = static_cast<unsigned int>(std::stoul(argv[++i]));
    } else if (arg == "--csv") {
      options.csv_path = argv[++i];
    } else if (arg == "--label") {
      options.run_label = argv[++i];
    } else if (arg == "--verbose") {
      options.verbose = true;
    } else {
      throw std::invalid_argument("Unknown argument: " + arg);
    }
  }

  if (options.page_size == 0) {
    throw std::invalid_argument("--page_size must be positive");
  }
  if (options.page_size % sizeof(float) != 0) {
    throw std::invalid_argument("--page_size must be a multiple of 4 bytes");
  }
  if (options.measure_iters <= 0) {
    throw std::invalid_argument("--measure_iters must be positive");
  }
  return options;
}

std::vector<float> MakeHostTensor(size_t numel, unsigned int seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
  std::vector<float> host(numel);
  for (size_t i = 0; i < numel; i++) {
    host[i] = dis(gen);
  }
  return host;
}

DeviceTensor CreateDeviceTensor(const std::vector<float>& host_data,
                                LayoutType layout, size_t page_size,
                                unsigned int seed) {
  DeviceTensor tensor;
  tensor.layout = layout;
  tensor.numel = host_data.size();
  tensor.num_bytes = host_data.size() * sizeof(float);
  tensor.page_size = page_size;
  tensor.page_count = RoundUpDiv(tensor.num_bytes, page_size);

  if (layout == LayoutType::kTensorLevel) {
    CheckCuda(cudaMalloc(&tensor.contiguous, tensor.num_bytes),
              "cudaMalloc failed for contiguous tensor");
    CheckCuda(cudaMemcpy(tensor.contiguous, host_data.data(), tensor.num_bytes,
                         cudaMemcpyHostToDevice),
              "cudaMemcpy failed for contiguous tensor");
    return tensor;
  }

  tensor.page_ptrs_host.resize(tensor.page_count, nullptr);
  std::vector<void*> physical_pages(tensor.page_count, nullptr);
  for (size_t i = 0; i < tensor.page_count; i++) {
    CheckCuda(cudaMalloc(&physical_pages[i], page_size),
              "cudaMalloc failed for page allocation");
  }

  std::vector<size_t> permutation(tensor.page_count);
  std::iota(permutation.begin(), permutation.end(), 0);
  std::mt19937 gen(seed);
  std::shuffle(permutation.begin(), permutation.end(), gen);

  const char* host_bytes = reinterpret_cast<const char*>(host_data.data());
  for (size_t logical_page = 0; logical_page < tensor.page_count; logical_page++) {
    size_t physical_page = permutation[logical_page];
    void* page_ptr = physical_pages[physical_page];
    tensor.page_ptrs_host[logical_page] = page_ptr;
    size_t offset = logical_page * page_size;
    size_t bytes_to_copy = std::min(page_size, tensor.num_bytes - offset);
    CheckCuda(cudaMemcpy(page_ptr, host_bytes + offset, bytes_to_copy,
                         cudaMemcpyHostToDevice),
              "cudaMemcpy failed for page payload");
  }

  CheckCuda(cudaMalloc(&tensor.page_ptrs_device,
                       tensor.page_count * sizeof(void*)),
            "cudaMalloc failed for page table");
  CheckCuda(cudaMemcpy(tensor.page_ptrs_device, tensor.page_ptrs_host.data(),
                       tensor.page_count * sizeof(void*), cudaMemcpyHostToDevice),
            "cudaMemcpy failed for page table");
  return tensor;
}

void DestroyDeviceTensor(DeviceTensor& tensor) {
  if (tensor.contiguous != nullptr) {
    cudaFree(tensor.contiguous);
    tensor.contiguous = nullptr;
  }
  if (tensor.page_ptrs_device != nullptr) {
    cudaFree(tensor.page_ptrs_device);
    tensor.page_ptrs_device = nullptr;
  }
  for (void* page_ptr : tensor.page_ptrs_host) {
    if (page_ptr != nullptr) {
      cudaFree(page_ptr);
    }
  }
  tensor.page_ptrs_host.clear();
}

size_t PhysicalBytes(const DeviceTensor& tensor) {
  if (tensor.layout == LayoutType::kTensorLevel) {
    return tensor.num_bytes;
  }
  return tensor.page_count * tensor.page_size +
         tensor.page_count * sizeof(void*);
}

__device__ inline float ReadPagedFloat(void* const* page_table,
                                       size_t page_size_bytes,
                                       size_t element_idx) {
  size_t byte_offset = element_idx * sizeof(float);
  size_t page_idx = byte_offset / page_size_bytes;
  size_t page_offset = byte_offset % page_size_bytes;
  const char* page = static_cast<const char*>(page_table[page_idx]);
  return *reinterpret_cast<const float*>(page + page_offset);
}

__device__ inline void WritePagedFloat(void** page_table,
                                       size_t page_size_bytes,
                                       size_t element_idx, float value) {
  size_t byte_offset = element_idx * sizeof(float);
  size_t page_idx = byte_offset / page_size_bytes;
  size_t page_offset = byte_offset % page_size_bytes;
  char* page = static_cast<char*>(page_table[page_idx]);
  *reinterpret_cast<float*>(page + page_offset) = value;
}

__device__ inline float Silu(float x) {
  return x / (1.0f + expf(-x));
}

__global__ void CopyContiguousKernel(const float* x, float* y, size_t numel) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < numel) {
    y[idx] = x[idx] * 1.0001f + 1.0f;
  }
}

__global__ void CopyPagedKernel(void* const* x_pages, void** y_pages,
                                size_t page_size_bytes, size_t numel) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < numel) {
    float x = ReadPagedFloat(x_pages, page_size_bytes, idx);
    WritePagedFloat(y_pages, page_size_bytes, idx, x * 1.0001f + 1.0f);
  }
}

__global__ void SiluMulContiguousKernel(const float* gate, const float* up,
                                        float* out, size_t numel) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < numel) {
    out[idx] = Silu(gate[idx]) * up[idx];
  }
}

__global__ void SiluMulPagedKernel(void* const* gate_pages, void* const* up_pages,
                                   void** out_pages, size_t page_size_bytes,
                                   size_t numel) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < numel) {
    float gate = ReadPagedFloat(gate_pages, page_size_bytes, idx);
    float up = ReadPagedFloat(up_pages, page_size_bytes, idx);
    WritePagedFloat(out_pages, page_size_bytes, idx, Silu(gate) * up);
  }
}

__global__ void RmsNormContiguousKernel(const float* x, const float* weight,
                                        float* out, size_t rows,
                                        size_t hidden_size) {
  __shared__ float shared_sum[kThreadsPerBlock];
  size_t row = blockIdx.x;
  size_t tid = threadIdx.x;
  if (row >= rows) return;

  float sum = 0.0f;
  size_t row_offset = row * hidden_size;
  for (size_t col = tid; col < hidden_size; col += blockDim.x) {
    float value = x[row_offset + col];
    sum += value * value;
  }
  shared_sum[tid] = sum;
  __syncthreads();

  for (size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      shared_sum[tid] += shared_sum[tid + stride];
    }
    __syncthreads();
  }

  float inv_rms = rsqrtf(shared_sum[0] / static_cast<float>(hidden_size) +
                         kEpsilon);
  for (size_t col = tid; col < hidden_size; col += blockDim.x) {
    float value = x[row_offset + col];
    out[row_offset + col] = value * inv_rms * weight[col];
  }
}

__global__ void RmsNormPagedKernel(void* const* x_pages, void* const* weight_pages,
                                   void** out_pages, size_t page_size_bytes,
                                   size_t rows, size_t hidden_size) {
  __shared__ float shared_sum[kThreadsPerBlock];
  size_t row = blockIdx.x;
  size_t tid = threadIdx.x;
  if (row >= rows) return;

  float sum = 0.0f;
  size_t row_offset = row * hidden_size;
  for (size_t col = tid; col < hidden_size; col += blockDim.x) {
    float value = ReadPagedFloat(x_pages, page_size_bytes, row_offset + col);
    sum += value * value;
  }
  shared_sum[tid] = sum;
  __syncthreads();

  for (size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      shared_sum[tid] += shared_sum[tid + stride];
    }
    __syncthreads();
  }

  float inv_rms = rsqrtf(shared_sum[0] / static_cast<float>(hidden_size) +
                         kEpsilon);
  for (size_t col = tid; col < hidden_size; col += blockDim.x) {
    float value = ReadPagedFloat(x_pages, page_size_bytes, row_offset + col);
    float weight = ReadPagedFloat(weight_pages, page_size_bytes, col);
    WritePagedFloat(out_pages, page_size_bytes, row_offset + col,
                    value * inv_rms * weight);
  }
}

__global__ void QkDotContiguousKernel(const float* q, const float* k, float* out,
                                      size_t rows, size_t head_dim) {
  __shared__ float shared_sum[kThreadsPerBlock];
  size_t row = blockIdx.x;
  size_t tid = threadIdx.x;
  if (row >= rows) return;

  float sum = 0.0f;
  size_t row_offset = row * head_dim;
  for (size_t col = tid; col < head_dim; col += blockDim.x) {
    sum += q[row_offset + col] * k[row_offset + col];
  }
  shared_sum[tid] = sum;
  __syncthreads();

  for (size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      shared_sum[tid] += shared_sum[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    out[row] = shared_sum[0];
  }
}

__global__ void QkDotPagedKernel(void* const* q_pages, void* const* k_pages,
                                 void** out_pages, size_t page_size_bytes,
                                 size_t rows, size_t head_dim) {
  __shared__ float shared_sum[kThreadsPerBlock];
  size_t row = blockIdx.x;
  size_t tid = threadIdx.x;
  if (row >= rows) return;

  float sum = 0.0f;
  size_t row_offset = row * head_dim;
  for (size_t col = tid; col < head_dim; col += blockDim.x) {
    sum += ReadPagedFloat(q_pages, page_size_bytes, row_offset + col) *
           ReadPagedFloat(k_pages, page_size_bytes, row_offset + col);
  }
  shared_sum[tid] = sum;
  __syncthreads();

  for (size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      shared_sum[tid] += shared_sum[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    WritePagedFloat(out_pages, page_size_bytes, row, shared_sum[0]);
  }
}

template <int TileM, int TileN, int TileK>
__global__ void MatmulContiguousKernel(const float* a, const float* b, float* c,
                                       size_t m, size_t n, size_t k) {
  __shared__ float a_tile[TileM][TileK];
  __shared__ float b_tile[TileK][TileN];

  size_t row = static_cast<size_t>(blockIdx.y) * TileM + threadIdx.y;
  size_t col = static_cast<size_t>(blockIdx.x) * TileN + threadIdx.x;
  float acc = 0.0f;

  for (size_t tile_k = 0; tile_k < k; tile_k += TileK) {
    size_t a_col = tile_k + threadIdx.x;
    size_t b_row = tile_k + threadIdx.y;

    a_tile[threadIdx.y][threadIdx.x] =
        (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
    b_tile[threadIdx.y][threadIdx.x] =
        (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;
    __syncthreads();

    for (int inner = 0; inner < TileK; inner++) {
      acc += a_tile[threadIdx.y][inner] * b_tile[inner][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < m && col < n) {
    c[row * n + col] = acc;
  }
}

template <int TileM, int TileN, int TileK>
__global__ void MatmulPagedKernel(void* const* a_pages, void* const* b_pages,
                                  void** c_pages, size_t page_size_bytes,
                                  size_t m, size_t n, size_t k) {
  __shared__ float a_tile[TileM][TileK];
  __shared__ float b_tile[TileK][TileN];

  size_t row = static_cast<size_t>(blockIdx.y) * TileM + threadIdx.y;
  size_t col = static_cast<size_t>(blockIdx.x) * TileN + threadIdx.x;
  float acc = 0.0f;

  for (size_t tile_k = 0; tile_k < k; tile_k += TileK) {
    size_t a_col = tile_k + threadIdx.x;
    size_t b_row = tile_k + threadIdx.y;

    a_tile[threadIdx.y][threadIdx.x] =
        (row < m && a_col < k)
            ? ReadPagedFloat(a_pages, page_size_bytes, row * k + a_col)
            : 0.0f;
    b_tile[threadIdx.y][threadIdx.x] =
        (b_row < k && col < n)
            ? ReadPagedFloat(b_pages, page_size_bytes, b_row * n + col)
            : 0.0f;
    __syncthreads();

    for (int inner = 0; inner < TileK; inner++) {
      acc += a_tile[threadIdx.y][inner] * b_tile[inner][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < m && col < n) {
    WritePagedFloat(c_pages, page_size_bytes, row * n + col, acc);
  }
}

__global__ void DecodeQkContiguousKernel(const float* q, const float* k_cache,
                                         float* out, size_t batch_heads,
                                         size_t seq_len, size_t head_dim) {
  size_t row = blockIdx.x;
  size_t tid = threadIdx.x;
  if (row >= batch_heads) return;

  const float* q_row = q + row * head_dim;
  const float* k_row = k_cache + row * seq_len * head_dim;
  float* out_row = out + row * seq_len;

  for (size_t token = tid; token < seq_len; token += blockDim.x) {
    const float* key = k_row + token * head_dim;
    float acc = 0.0f;
    for (size_t dim = 0; dim < head_dim; dim++) {
      acc += q_row[dim] * key[dim];
    }
    out_row[token] = acc;
  }
}

__global__ void DecodeQkPagedKernel(void* const* q_pages, void* const* k_pages,
                                    void** out_pages, size_t page_size_bytes,
                                    size_t batch_heads, size_t seq_len,
                                    size_t head_dim) {
  size_t row = blockIdx.x;
  size_t tid = threadIdx.x;
  if (row >= batch_heads) return;

  for (size_t token = tid; token < seq_len; token += blockDim.x) {
    float acc = 0.0f;
    for (size_t dim = 0; dim < head_dim; dim++) {
      size_t q_idx = row * head_dim + dim;
      size_t k_idx = row * seq_len * head_dim + token * head_dim + dim;
      acc += ReadPagedFloat(q_pages, page_size_bytes, q_idx) *
             ReadPagedFloat(k_pages, page_size_bytes, k_idx);
    }
    WritePagedFloat(out_pages, page_size_bytes, row * seq_len + token, acc);
  }
}

void LaunchCopy(const BenchOptions& options, const DeviceTensor& x,
                const DeviceTensor& y) {
  size_t blocks = RoundUpDiv(x.numel, static_cast<size_t>(kThreadsPerBlock));
  if (options.layout == LayoutType::kTensorLevel) {
    CopyContiguousKernel<<<blocks, kThreadsPerBlock>>>(x.contiguous, y.contiguous,
                                                       x.numel);
  } else {
    CopyPagedKernel<<<blocks, kThreadsPerBlock>>>(
        x.page_ptrs_device, y.page_ptrs_device, x.page_size, x.numel);
  }
}

void LaunchSiluMul(const BenchOptions& options, const DeviceTensor& gate,
                   const DeviceTensor& up, const DeviceTensor& out) {
  size_t blocks = RoundUpDiv(gate.numel, static_cast<size_t>(kThreadsPerBlock));
  if (options.layout == LayoutType::kTensorLevel) {
    SiluMulContiguousKernel<<<blocks, kThreadsPerBlock>>>(
        gate.contiguous, up.contiguous, out.contiguous, gate.numel);
  } else {
    SiluMulPagedKernel<<<blocks, kThreadsPerBlock>>>(
        gate.page_ptrs_device, up.page_ptrs_device, out.page_ptrs_device,
        gate.page_size, gate.numel);
  }
}

void LaunchRmsNorm(const BenchOptions& options, const DeviceTensor& x,
                   const DeviceTensor& weight, const DeviceTensor& out,
                   size_t rows, size_t hidden_size) {
  if (options.layout == LayoutType::kTensorLevel) {
    RmsNormContiguousKernel<<<rows, kThreadsPerBlock>>>(
        x.contiguous, weight.contiguous, out.contiguous, rows, hidden_size);
  } else {
    RmsNormPagedKernel<<<rows, kThreadsPerBlock>>>(
        x.page_ptrs_device, weight.page_ptrs_device, out.page_ptrs_device,
        x.page_size, rows, hidden_size);
  }
}

void LaunchQkDot(const BenchOptions& options, const DeviceTensor& q,
                 const DeviceTensor& k, const DeviceTensor& out, size_t rows,
                 size_t head_dim) {
  if (options.layout == LayoutType::kTensorLevel) {
    QkDotContiguousKernel<<<rows, kThreadsPerBlock>>>(
        q.contiguous, k.contiguous, out.contiguous, rows, head_dim);
  } else {
    QkDotPagedKernel<<<rows, kThreadsPerBlock>>>(
        q.page_ptrs_device, k.page_ptrs_device, out.page_ptrs_device,
        q.page_size, rows, head_dim);
  }
}

void LaunchMatmul(const BenchOptions& options, const DeviceTensor& a,
                  const DeviceTensor& b, const DeviceTensor& c, size_t m,
                  size_t n, size_t k) {
  constexpr int kTile = 16;
  dim3 block(kTile, kTile);
  dim3 grid(RoundUpDiv(n, static_cast<size_t>(kTile)),
            RoundUpDiv(m, static_cast<size_t>(kTile)));
  if (options.layout == LayoutType::kTensorLevel) {
    MatmulContiguousKernel<kTile, kTile, kTile>
        <<<grid, block>>>(a.contiguous, b.contiguous, c.contiguous, m, n, k);
  } else {
    MatmulPagedKernel<kTile, kTile, kTile><<<grid, block>>>(
        a.page_ptrs_device, b.page_ptrs_device, c.page_ptrs_device, a.page_size,
        m, n, k);
  }
}

void LaunchDecodeQk(const BenchOptions& options, const DeviceTensor& q,
                    const DeviceTensor& k_cache, const DeviceTensor& out,
                    size_t batch_heads, size_t seq_len, size_t head_dim) {
  if (options.layout == LayoutType::kTensorLevel) {
    DecodeQkContiguousKernel<<<batch_heads, kThreadsPerBlock>>>(
        q.contiguous, k_cache.contiguous, out.contiguous, batch_heads, seq_len,
        head_dim);
  } else {
    DecodeQkPagedKernel<<<batch_heads, kThreadsPerBlock>>>(
        q.page_ptrs_device, k_cache.page_ptrs_device, out.page_ptrs_device,
        q.page_size, batch_heads, seq_len, head_dim);
  }
}

double MeasureKernelRuntimeMs(const std::function<void()>& launch,
                              int warmup_iters, int measure_iters) {
  for (int i = 0; i < warmup_iters; i++) {
    launch();
  }
  CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed after warmup");

  cudaEvent_t start;
  cudaEvent_t stop;
  CheckCuda(cudaEventCreate(&start), "cudaEventCreate(start) failed");
  CheckCuda(cudaEventCreate(&stop), "cudaEventCreate(stop) failed");
  CheckCuda(cudaEventRecord(start), "cudaEventRecord(start) failed");
  for (int i = 0; i < measure_iters; i++) {
    launch();
  }
  CheckCuda(cudaEventRecord(stop), "cudaEventRecord(stop) failed");
  CheckCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop) failed");

  float total_ms = 0.0f;
  CheckCuda(cudaEventElapsedTime(&total_ms, start, stop),
            "cudaEventElapsedTime failed");
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return static_cast<double>(total_ms) / static_cast<double>(measure_iters);
}

double FetchChecksum(const DeviceTensor& tensor) {
  if (tensor.numel == 0) return 0.0;
  std::vector<float> host(1, 0.0f);
  if (tensor.layout == LayoutType::kTensorLevel) {
    CheckCuda(cudaMemcpy(host.data(), tensor.contiguous, sizeof(float),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy checksum failed");
    return host[0];
  }

  CheckCuda(cudaMemcpy(host.data(), tensor.page_ptrs_host[0], sizeof(float),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy paged checksum failed");
  return host[0];
}

size_t KernelTrafficBytes(const BenchOptions& options) {
  size_t rows = options.batch_size * options.seq_len;
  switch (options.kernel) {
    case KernelType::kCopy:
      return rows * options.hidden_size * sizeof(float) * 2ULL;
    case KernelType::kSiluMul:
      return rows * options.intermediate_size * sizeof(float) * 3ULL;
    case KernelType::kRmsNorm:
      return rows * options.hidden_size * sizeof(float) * 4ULL;
    case KernelType::kQkDot:
      return rows * options.num_heads *
             (options.head_dim * sizeof(float) * 2ULL + sizeof(float));
    case KernelType::kAttnProjGemm:
      return rows * options.hidden_size * sizeof(float) * 2ULL +
             options.hidden_size * options.hidden_size * sizeof(float);
    case KernelType::kFfnUpGemm:
      return rows * sizeof(float) *
                 (options.hidden_size + options.intermediate_size) +
             options.hidden_size * options.intermediate_size * sizeof(float);
    case KernelType::kDecodeQk:
      return options.batch_size * options.num_heads *
             (options.head_dim * sizeof(float) +
              options.seq_len * options.head_dim * sizeof(float) +
              options.seq_len * sizeof(float));
  }
  return 0;
}

double KernelFlops(const BenchOptions& options) {
  size_t rows = options.batch_size * options.seq_len;
  switch (options.kernel) {
    case KernelType::kCopy:
      return static_cast<double>(rows) * options.hidden_size * 2.0;
    case KernelType::kSiluMul:
      return static_cast<double>(rows) * options.intermediate_size * 8.0;
    case KernelType::kRmsNorm:
      return static_cast<double>(rows) * options.hidden_size * 4.0;
    case KernelType::kQkDot:
      return static_cast<double>(rows) * options.num_heads * options.head_dim *
             2.0;
    case KernelType::kAttnProjGemm:
      return 2.0 * static_cast<double>(rows) * options.hidden_size *
             options.hidden_size;
    case KernelType::kFfnUpGemm:
      return 2.0 * static_cast<double>(rows) * options.hidden_size *
             options.intermediate_size;
    case KernelType::kDecodeQk:
      return 2.0 * static_cast<double>(options.batch_size) * options.num_heads *
             options.seq_len * options.head_dim;
  }
  return 0.0;
}

void FillDerivedMetrics(const BenchOptions& options, BenchResult& result) {
  double traffic_bytes = static_cast<double>(KernelTrafficBytes(options));
  double flops = KernelFlops(options);
  result.arithmetic_intensity =
      traffic_bytes > 0.0 ? flops / traffic_bytes : 0.0;
  result.bandwidth_gbps =
      result.avg_ms > 0.0 ? traffic_bytes / result.avg_ms / 1.0e6 : 0.0;
  result.tflops = result.avg_ms > 0.0 ? flops / result.avg_ms / 1.0e9 : 0.0;
}

BenchResult BuildCommonResult(const BenchOptions& options) {
  BenchResult result;
  result.label = options.run_label;
  result.layout = LayoutToString(options.layout);
  result.kernel = KernelToString(options.kernel);
  result.page_size = options.page_size;
  result.batch_size = options.batch_size;
  result.seq_len = options.seq_len;
  result.hidden_size = options.hidden_size;
  result.intermediate_size = options.intermediate_size;
  result.num_heads = options.num_heads;
  result.head_dim = options.head_dim;
  result.warmup_iters = options.warmup_iters;
  result.measure_iters = options.measure_iters;
  return result;
}

BenchResult RunCopyBench(const BenchOptions& options) {
  size_t numel = options.batch_size * options.seq_len * options.hidden_size;
  auto x_host = MakeHostTensor(numel, options.random_seed + 1);
  auto y_host = std::vector<float>(numel, 0.0f);
  DeviceTensor x = CreateDeviceTensor(x_host, options.layout, options.page_size,
                                      options.random_seed + 101);
  DeviceTensor y = CreateDeviceTensor(y_host, options.layout, options.page_size,
                                      options.random_seed + 201);

  auto launch = [&]() { LaunchCopy(options, x, y); };
  double avg_ms =
      MeasureKernelRuntimeMs(launch, options.warmup_iters, options.measure_iters);
  CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed after copy");

  BenchResult result = BuildCommonResult(options);
  result.logical_bytes = x.num_bytes + y.num_bytes;
  result.physical_bytes = PhysicalBytes(x) + PhysicalBytes(y);
  result.avg_ms = result.min_ms = result.max_ms = avg_ms;
  result.checksum = FetchChecksum(y);
  FillDerivedMetrics(options, result);

  DestroyDeviceTensor(x);
  DestroyDeviceTensor(y);
  return result;
}

BenchResult RunSiluMulBench(const BenchOptions& options) {
  size_t numel = options.batch_size * options.seq_len * options.intermediate_size;
  auto gate_host = MakeHostTensor(numel, options.random_seed + 11);
  auto up_host = MakeHostTensor(numel, options.random_seed + 12);
  auto out_host = std::vector<float>(numel, 0.0f);
  DeviceTensor gate = CreateDeviceTensor(gate_host, options.layout, options.page_size,
                                         options.random_seed + 301);
  DeviceTensor up = CreateDeviceTensor(up_host, options.layout, options.page_size,
                                       options.random_seed + 302);
  DeviceTensor out = CreateDeviceTensor(out_host, options.layout, options.page_size,
                                        options.random_seed + 303);

  auto launch = [&]() { LaunchSiluMul(options, gate, up, out); };
  double avg_ms =
      MeasureKernelRuntimeMs(launch, options.warmup_iters, options.measure_iters);
  CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed after silu_mul");

  BenchResult result = BuildCommonResult(options);
  result.logical_bytes = gate.num_bytes + up.num_bytes + out.num_bytes;
  result.physical_bytes = PhysicalBytes(gate) + PhysicalBytes(up) + PhysicalBytes(out);
  result.avg_ms = result.min_ms = result.max_ms = avg_ms;
  result.checksum = FetchChecksum(out);
  FillDerivedMetrics(options, result);

  DestroyDeviceTensor(gate);
  DestroyDeviceTensor(up);
  DestroyDeviceTensor(out);
  return result;
}

BenchResult RunRmsNormBench(const BenchOptions& options) {
  size_t rows = options.batch_size * options.seq_len;
  size_t x_numel = rows * options.hidden_size;
  auto x_host = MakeHostTensor(x_numel, options.random_seed + 21);
  auto weight_host = MakeHostTensor(options.hidden_size, options.random_seed + 22);
  auto out_host = std::vector<float>(x_numel, 0.0f);
  DeviceTensor x = CreateDeviceTensor(x_host, options.layout, options.page_size,
                                      options.random_seed + 401);
  DeviceTensor weight = CreateDeviceTensor(weight_host, options.layout,
                                           options.page_size,
                                           options.random_seed + 402);
  DeviceTensor out = CreateDeviceTensor(out_host, options.layout, options.page_size,
                                        options.random_seed + 403);

  auto launch = [&]() {
    LaunchRmsNorm(options, x, weight, out, rows, options.hidden_size);
  };
  double avg_ms =
      MeasureKernelRuntimeMs(launch, options.warmup_iters, options.measure_iters);
  CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed after rmsnorm");

  BenchResult result = BuildCommonResult(options);
  result.logical_bytes = x.num_bytes + weight.num_bytes + out.num_bytes;
  result.physical_bytes = PhysicalBytes(x) + PhysicalBytes(weight) + PhysicalBytes(out);
  result.avg_ms = result.min_ms = result.max_ms = avg_ms;
  result.checksum = FetchChecksum(out);
  FillDerivedMetrics(options, result);

  DestroyDeviceTensor(x);
  DestroyDeviceTensor(weight);
  DestroyDeviceTensor(out);
  return result;
}

BenchResult RunQkDotBench(const BenchOptions& options) {
  size_t rows = options.batch_size * options.seq_len * options.num_heads;
  size_t vec_numel = rows * options.head_dim;
  auto q_host = MakeHostTensor(vec_numel, options.random_seed + 31);
  auto k_host = MakeHostTensor(vec_numel, options.random_seed + 32);
  auto out_host = std::vector<float>(rows, 0.0f);
  DeviceTensor q = CreateDeviceTensor(q_host, options.layout, options.page_size,
                                      options.random_seed + 501);
  DeviceTensor k = CreateDeviceTensor(k_host, options.layout, options.page_size,
                                      options.random_seed + 502);
  DeviceTensor out = CreateDeviceTensor(out_host, options.layout, options.page_size,
                                        options.random_seed + 503);

  auto launch = [&]() { LaunchQkDot(options, q, k, out, rows, options.head_dim); };
  double avg_ms =
      MeasureKernelRuntimeMs(launch, options.warmup_iters, options.measure_iters);
  CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed after qk_dot");

  BenchResult result = BuildCommonResult(options);
  result.logical_bytes = q.num_bytes + k.num_bytes + out.num_bytes;
  result.physical_bytes = PhysicalBytes(q) + PhysicalBytes(k) + PhysicalBytes(out);
  result.avg_ms = result.min_ms = result.max_ms = avg_ms;
  result.checksum = FetchChecksum(out);
  FillDerivedMetrics(options, result);

  DestroyDeviceTensor(q);
  DestroyDeviceTensor(k);
  DestroyDeviceTensor(out);
  return result;
}

BenchResult RunAttnProjGemmBench(const BenchOptions& options) {
  size_t m = options.batch_size * options.seq_len;
  size_t k = options.hidden_size;
  size_t n = options.hidden_size;
  auto x_host = MakeHostTensor(m * k, options.random_seed + 41);
  auto w_host = MakeHostTensor(k * n, options.random_seed + 42);
  auto out_host = std::vector<float>(m * n, 0.0f);
  DeviceTensor x = CreateDeviceTensor(x_host, options.layout, options.page_size,
                                      options.random_seed + 601);
  DeviceTensor w = CreateDeviceTensor(w_host, options.layout, options.page_size,
                                      options.random_seed + 602);
  DeviceTensor out = CreateDeviceTensor(out_host, options.layout, options.page_size,
                                        options.random_seed + 603);

  auto launch = [&]() { LaunchMatmul(options, x, w, out, m, n, k); };
  double avg_ms =
      MeasureKernelRuntimeMs(launch, options.warmup_iters, options.measure_iters);
  CheckCuda(cudaDeviceSynchronize(),
            "cudaDeviceSynchronize failed after attn_proj_gemm");

  BenchResult result = BuildCommonResult(options);
  result.logical_bytes = x.num_bytes + w.num_bytes + out.num_bytes;
  result.physical_bytes = PhysicalBytes(x) + PhysicalBytes(w) + PhysicalBytes(out);
  result.avg_ms = result.min_ms = result.max_ms = avg_ms;
  result.checksum = FetchChecksum(out);
  FillDerivedMetrics(options, result);

  DestroyDeviceTensor(x);
  DestroyDeviceTensor(w);
  DestroyDeviceTensor(out);
  return result;
}

BenchResult RunFfnUpGemmBench(const BenchOptions& options) {
  size_t m = options.batch_size * options.seq_len;
  size_t k = options.hidden_size;
  size_t n = options.intermediate_size;
  auto x_host = MakeHostTensor(m * k, options.random_seed + 51);
  auto w_host = MakeHostTensor(k * n, options.random_seed + 52);
  auto out_host = std::vector<float>(m * n, 0.0f);
  DeviceTensor x = CreateDeviceTensor(x_host, options.layout, options.page_size,
                                      options.random_seed + 701);
  DeviceTensor w = CreateDeviceTensor(w_host, options.layout, options.page_size,
                                      options.random_seed + 702);
  DeviceTensor out = CreateDeviceTensor(out_host, options.layout, options.page_size,
                                        options.random_seed + 703);

  auto launch = [&]() { LaunchMatmul(options, x, w, out, m, n, k); };
  double avg_ms =
      MeasureKernelRuntimeMs(launch, options.warmup_iters, options.measure_iters);
  CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed after ffn_up_gemm");

  BenchResult result = BuildCommonResult(options);
  result.logical_bytes = x.num_bytes + w.num_bytes + out.num_bytes;
  result.physical_bytes = PhysicalBytes(x) + PhysicalBytes(w) + PhysicalBytes(out);
  result.avg_ms = result.min_ms = result.max_ms = avg_ms;
  result.checksum = FetchChecksum(out);
  FillDerivedMetrics(options, result);

  DestroyDeviceTensor(x);
  DestroyDeviceTensor(w);
  DestroyDeviceTensor(out);
  return result;
}

BenchResult RunDecodeQkBench(const BenchOptions& options) {
  size_t batch_heads = options.batch_size * options.num_heads;
  size_t q_numel = batch_heads * options.head_dim;
  size_t k_numel = batch_heads * options.seq_len * options.head_dim;
  size_t out_numel = batch_heads * options.seq_len;
  auto q_host = MakeHostTensor(q_numel, options.random_seed + 61);
  auto k_host = MakeHostTensor(k_numel, options.random_seed + 62);
  auto out_host = std::vector<float>(out_numel, 0.0f);
  DeviceTensor q = CreateDeviceTensor(q_host, options.layout, options.page_size,
                                      options.random_seed + 801);
  DeviceTensor k_cache = CreateDeviceTensor(k_host, options.layout, options.page_size,
                                            options.random_seed + 802);
  DeviceTensor out = CreateDeviceTensor(out_host, options.layout, options.page_size,
                                        options.random_seed + 803);

  auto launch = [&]() {
    LaunchDecodeQk(options, q, k_cache, out, batch_heads, options.seq_len,
                   options.head_dim);
  };
  double avg_ms =
      MeasureKernelRuntimeMs(launch, options.warmup_iters, options.measure_iters);
  CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed after decode_qk");

  BenchResult result = BuildCommonResult(options);
  result.logical_bytes = q.num_bytes + k_cache.num_bytes + out.num_bytes;
  result.physical_bytes = PhysicalBytes(q) + PhysicalBytes(k_cache) + PhysicalBytes(out);
  result.avg_ms = result.min_ms = result.max_ms = avg_ms;
  result.checksum = FetchChecksum(out);
  FillDerivedMetrics(options, result);

  DestroyDeviceTensor(q);
  DestroyDeviceTensor(k_cache);
  DestroyDeviceTensor(out);
  return result;
}

BenchResult RunBenchmark(const BenchOptions& options) {
  CheckCuda(cudaSetDevice(options.gpu_id), "cudaSetDevice failed");
  switch (options.kernel) {
    case KernelType::kCopy:
      return RunCopyBench(options);
    case KernelType::kSiluMul:
      return RunSiluMulBench(options);
    case KernelType::kRmsNorm:
      return RunRmsNormBench(options);
    case KernelType::kQkDot:
      return RunQkDotBench(options);
    case KernelType::kAttnProjGemm:
      return RunAttnProjGemmBench(options);
    case KernelType::kFfnUpGemm:
      return RunFfnUpGemmBench(options);
    case KernelType::kDecodeQk:
      return RunDecodeQkBench(options);
  }
  throw std::runtime_error("Unknown kernel type");
}

void AppendCsv(const std::string& path, const BenchResult& result) {
  if (path.empty()) return;
  EnsureParentDir(path);
  bool write_header = !std::filesystem::exists(path) ||
                      std::filesystem::file_size(path) == 0;
  std::ofstream file(path, std::ios::app);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open csv output: " + path);
  }
  if (write_header) {
    file << "label,layout,kernel,page_size_bytes,batch_size,seq_len,hidden_size,"
            "intermediate_size,num_heads,head_dim,warmup_iters,measure_iters,"
            "logical_bytes,physical_bytes,arithmetic_intensity,avg_ms,min_ms,"
            "max_ms,bandwidth_gbps,tflops,checksum\n";
  }
  file << result.label << "," << result.layout << "," << result.kernel << ","
       << result.page_size << "," << result.batch_size << "," << result.seq_len
       << "," << result.hidden_size << "," << result.intermediate_size << ","
       << result.num_heads << "," << result.head_dim << ","
       << result.warmup_iters << "," << result.measure_iters << ","
       << result.logical_bytes << "," << result.physical_bytes << ","
       << std::fixed << std::setprecision(6) << result.arithmetic_intensity
       << "," << result.avg_ms << "," << result.min_ms << ","
       << result.max_ms << "," << result.bandwidth_gbps << ","
       << result.tflops << "," << result.checksum << "\n";
}

void PrintSummary(const BenchResult& result) {
  std::cout << "================ Memory Coalescing Benchmark ================\n";
  std::cout << "label: " << result.label << "\n";
  std::cout << "layout: " << result.layout << "\n";
  std::cout << "kernel: " << result.kernel << "\n";
  std::cout << "page_size_bytes: " << result.page_size << "\n";
  std::cout << "logical_bytes: " << result.logical_bytes << "\n";
  std::cout << "physical_bytes: " << result.physical_bytes << "\n";
  std::cout << "arithmetic_intensity(FLOPs/byte): "
            << result.arithmetic_intensity << "\n";
  std::cout << "avg_ms: " << std::fixed << std::setprecision(6) << result.avg_ms
            << "\n";
  std::cout << "effective_bandwidth_GBps: " << result.bandwidth_gbps << "\n";
  std::cout << "effective_TFLOPS: " << result.tflops << "\n";
  std::cout << "checksum: " << result.checksum << "\n";
  std::cout << "============================================================\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    BenchOptions options = ParseArgs(argc, argv);
    if (options.run_label.empty()) {
      std::ostringstream oss;
      oss << LayoutToString(options.layout) << "-" << KernelToString(options.kernel)
          << "-page" << options.page_size;
      options.run_label = oss.str();
    }

    BenchResult result = RunBenchmark(options);
    PrintSummary(result);
    AppendCsv(options.csv_path, result);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "memory_coalescing_bench failed: " << e.what() << std::endl;
    return 1;
  }
}
