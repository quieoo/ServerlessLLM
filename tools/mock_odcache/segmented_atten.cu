#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "attention_utils.cuh"
#include "cache.h"
#include "cuda_compat.h"
#include "quant_utils.cuh"

// ----------------------Reshape and Cache----------------------
namespace vllm {
template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void reshape_and_cache_segment_kernel(
    const scalar_t* __restrict__ key, const scalar_t* __restrict__ value,
    void* __restrict__ global_memory, const int64_t* __restrict__ slot_mapping,
    const u_int64_t* __restrict__ block_tables, const int layer_id,
    const int key_stride, const int value_stride, const int num_heads,
    const int head_size, const int block_size, const int x,
    const float kv_scale) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    // Padding token that should be ignored.
    return;
  }
  const int64_t block_idx = slot_idx / block_size;  // logical block id
  const int64_t block_offset = slot_idx % block_size;
  u_int64_t phy_block_offset = block_tables[block_idx];
  cache_t* block_key_ptr =
      (cache_t*)((char*)global_memory + phy_block_offset) +
      layer_id * num_heads * (head_size / x) * block_size * x * 2;
  cache_t* block_value_ptr =
      block_key_ptr + num_heads * (head_size / x) * block_size * x;

  const int n = num_heads * head_size;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t src_key_idx = token_idx * key_stride + i;
    const int64_t src_value_idx = token_idx * value_stride + i;
    scalar_t tgt_key = key[src_key_idx];
    scalar_t tgt_value = value[src_value_idx];

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int x_idx = head_offset / x;
    const int x_offset = head_offset % x;

    cache_t* key_cache_ptr =
        block_key_ptr + head_idx * (head_size / x) * block_size * x +
        x_idx * block_size * x + block_offset * x + x_offset;

    cache_t* value_cache_ptr = block_value_ptr +
                               head_idx * head_size * block_size +
                               head_offset * block_size + block_offset;

    if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
      key_cache_ptr[0] = tgt_key;
      value_cache_ptr[0] = tgt_value;
    } else {
      key_cache_ptr[0] =
          fp8::scaled_convert<cache_t, scalar_t, kv_dt>(tgt_key, kv_scale);
      value_cache_ptr[0] =
          fp8::scaled_convert<cache_t, scalar_t, kv_dt>(tgt_value, kv_scale);
    }
  }
}
}  // namespace vllm

#define CALL_RESHAPE_AND_CACHE_SEGMENT(KV_T, CACHE_T, KV_DTYPE)          \
  vllm::reshape_and_cache_segment_kernel<KV_T, CACHE_T, KV_DTYPE>        \
      <<<grid, block, 0, stream>>>(                                      \
          reinterpret_cast<KV_T*>(key.data_ptr()),                       \
          reinterpret_cast<KV_T*>(value.data_ptr()),                     \
          reinterpret_cast<void*>(global_memory),                         \
          slot_mapping.data_ptr<int64_t>(),                              \
          reinterpret_cast<uint64_t*>(block_tables.data_ptr<int64_t>()), \
          layer_id, key_stride, value_stride, num_heads, head_size,      \
          block_size, 16 / sizeof(CACHE_T), kv_scale);

void reshape_and_cache_segment(
    torch::Tensor& key,           // [num_tokens, num_heads, head_size]
    torch::Tensor& value,         // [num_tokens, num_heads, head_size]
    int64_t global_memory,        // global memory pointer
    torch::Tensor& block_tables,  // [num_seqs, max_num_blocks_per_seq]
    int64_t layer_id,             // layer id
    torch::Tensor& slot_mapping,  // [num_tokens]
    int64_t block_size,           // number of slots in a block
    const std::string& kv_cache_dtype, const double kv_scale) {
  int num_tokens = key.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int key_stride = key.stride(0);
  int value_stride = value.stride(0);
  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype,
                             CALL_RESHAPE_AND_CACHE_SEGMENT)
}

// ----------------------Compute Attention----------------------
#ifndef USE_ROCM
#define WARP_SIZE 32
#else
#define WARP_SIZE warpSize
#endif

#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

namespace vllm {

// Utility function for attention softmax.
template <int NUM_WARPS>
inline __device__ float block_sum(float* red_smem, float sum) {
  // Decompose the thread index into warp / lane.
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  // Compute the sum per warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    sum += VLLM_SHFL_XOR_SYNC(sum, mask);
  }

  // Warp leaders store the data to shared memory.
  if (lane == 0) {
    red_smem[warp] = sum;
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The warps compute the final sums.
  if (lane < NUM_WARPS) {
    sum = red_smem[lane];
  }

  // Parallel reduction inside the warp.
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    sum += VLLM_SHFL_XOR_SYNC(sum, mask);
  }

  // Broadcast to other threads.
  return VLLM_SHFL_SYNC(sum, 0);
}

// Grid: (num_heads, num_seqs, max_num_partitions).
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, vllm::Fp8KVCacheDataType KV_DTYPE,
          bool IS_BLOCK_SPARSE,
          int PARTITION_SIZE = 0>  // Zero means no partitioning.
__device__ void segmented_attention_kernel(
    float* __restrict__ exp_sums,  // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                     // max_num_partitions]
    scalar_t* __restrict__ out,  // [num_seqs, num_heads, max_num_partitions,
                                 // head_size]
    const scalar_t* __restrict__ q,    // [num_seqs, num_heads, head_size]
    void* __restrict__ global_memory,  // global memory pointer
    const int num_kv_heads,            // [num_heads]
    const float scale,
    const uint64_t* __restrict__ block_tables,  // [num_seqs,
                                                // max_num_blocks_per_seq]
    const int layer_id,
    const int* __restrict__ seq_lens,  // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float kv_scale, const int tp_rank, const int blocksparse_local_blocks,
    const int blocksparse_vert_stride, const int blocksparse_block_size,
    const int blocksparse_head_sliding_step) {
  const int seq_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  const int max_num_partitions = gridDim.z;
  constexpr bool USE_PARTITIONING = PARTITION_SIZE > 0;
  const int seq_len = seq_lens[seq_idx];
  if (USE_PARTITIONING && partition_idx * PARTITION_SIZE >= seq_len) {
    // No work to do. Terminate the thread block.
    return;
  }

  const int num_seq_blocks = DIVIDE_ROUND_UP(seq_len, BLOCK_SIZE);
  const int num_blocks_per_partition =
      USE_PARTITIONING ? PARTITION_SIZE / BLOCK_SIZE : num_seq_blocks;

  // [start_block_idx, end_block_idx) is the range of blocks to process.
  const int start_block_idx =
      USE_PARTITIONING ? partition_idx * num_blocks_per_partition : 0;
  const int end_block_idx =
      MIN(start_block_idx + num_blocks_per_partition, num_seq_blocks);
  const int num_blocks = end_block_idx - start_block_idx;

  // [start_token_idx, end_token_idx) is the range of tokens to process.
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx =
      MIN(start_token_idx + num_blocks * BLOCK_SIZE, seq_len);
  const int num_tokens = end_token_idx - start_token_idx;

  // Define thread group size and number of groups for parallel processing
  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  constexpr int NUM_THREAD_GROUPS = NUM_THREADS / THREAD_GROUP_SIZE;
  static_assert(NUM_THREADS % THREAD_GROUP_SIZE == 0,
                "NUM_THREADS must be divisible by THREAD_GROUP_SIZE");

  // Determine the number of tokens processed per thread group
  constexpr int NUM_TOKENS_PER_THREAD_GROUP =
      DIVIDE_ROUND_UP(BLOCK_SIZE, WARP_SIZE);

  // Calculate number of warps and indices for threading
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int thread_idx = threadIdx.x;
  const int warp_idx = thread_idx / WARP_SIZE;
  const int lane = thread_idx % WARP_SIZE;

  const int head_idx = blockIdx.x;
  const int num_heads = gridDim.x;
  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;
  const float alibi_slope =
      alibi_slopes == nullptr ? 0.f : alibi_slopes[head_idx];

  // A vector type to store a part of a key or a query.
  // The vector size is configured in such a way that the threads in a thread
  // group fetch or compute 16 bytes at a time. For example, if the size of a
  // thread group is 4 and the data type is half, then the vector size is 16 /
  // (4 * sizeof(half)) == 2.
  constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)), 1);
  using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Quant_vec = typename Vec<cache_t, VEC_SIZE>::Type;

  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

  const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

  // Load the query to registers.
  // Each thread in a thread group has a different part of the query.
  // For example, if the the thread group size is 4, then the first thread in
  // the group has 0, 4, 8, ... th vectors of the query, and the second thread
  // has 1, 5, 9, ... th vectors of the query, and so on. NOTE(woosuk): Because
  // q is split from a qkv tensor, it may not be contiguous.
  const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  __shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];

#pragma unroll
  for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD;
       i += NUM_THREAD_GROUPS) {
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    // 装载Q
    q_vecs[thread_group_offset][i] =
        *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
  }
  __syncthreads();  // TODO(naed90): possible speedup if this is replaced with a
                    // memory wall right before we use q_vecs

  // Memory planning.
  extern __shared__ char shared_mem[];
  float* logits = reinterpret_cast<float*>(shared_mem);
  // Workspace for reduction.
  __shared__ float red_smem[2 * NUM_WARPS];

  // x == THREAD_GROUP_SIZE * VEC_SIZE
  // Each thread group fetches x elements from the key at a time.
  constexpr int x = 16 / sizeof(cache_t);
  float qk_max = -FLT_MAX;

  // 定位到当前seq的block table
  const uint64_t* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

  // blocksparse specific vars
  int bs_block_offset;
  int q_bs_block_id;
  if constexpr (IS_BLOCK_SPARSE) {
    // const int num_blocksparse_blocks = DIVIDE_ROUND_UP(seq_len,
    // blocksparse_block_size);
    q_bs_block_id = (seq_len - 1) / blocksparse_block_size;
    if (blocksparse_head_sliding_step >= 0)
      // sliding on q heads
      bs_block_offset =
          (tp_rank * num_heads + head_idx) * blocksparse_head_sliding_step + 1;
    else
      // sliding on kv heads
      bs_block_offset = (tp_rank * num_kv_heads + kv_head_idx) *
                            (-blocksparse_head_sliding_step) +
                        1;
  }

  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
       block_idx += NUM_WARPS) {
    // NOTE(woosuk): The block number is stored in int32. However, we cast it to
    // int64 because int32 can lead to overflow when this variable is multiplied
    // by large numbers (e.g., kv_block_stride).
    // For blocksparse attention: skip computation on blocks that are not
    // attended
    if constexpr (IS_BLOCK_SPARSE) {
      const int k_bs_block_id = block_idx * BLOCK_SIZE / blocksparse_block_size;
      const bool is_remote =
          ((k_bs_block_id + bs_block_offset) % blocksparse_vert_stride == 0);
      const bool is_local =
          (k_bs_block_id > q_bs_block_id - blocksparse_local_blocks);
      if (!is_remote && !is_local) {
        for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
          const int physical_block_offset =
              (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;
          const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;

          if (thread_group_offset == 0) {
            // NOTE(linxihui): assign very large number to skipped tokens to
            // avoid contribution to the sumexp softmax normalizer. This will
            // not be used at computing sum(softmax*v) as the blocks will be
            // skipped.
            logits[token_idx - start_token_idx] = -FLT_MAX;
          }
        }
        continue;
      }
    }
    // const int64_t physical_block_number =
    //     static_cast<int64_t>(block_table[block_idx]);

    // 通过block table获取当前block在全局内存中的偏移量
    const uint64_t block_global_offset = block_table[block_idx];

    for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
      const int physical_block_offset =
          (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;
      const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
      K_vec k_vecs[NUM_VECS_PER_THREAD];

#pragma unroll
      for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        // const cache_t* k_ptr =
        //     k_cache + physical_block_number * kv_block_stride +
        //     kv_head_idx * kv_head_stride + physical_block_offset * x;
        // 通过全局偏移量访问Key
        const cache_t* k_ptr =
            reinterpret_cast<const cache_t*>(global_memory +
                                             block_global_offset) +
            layer_id * num_kv_heads * (HEAD_SIZE / x) * BLOCK_SIZE * x * 2 +
            kv_head_idx * kv_head_stride + physical_block_offset * x;

        const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
        const int offset1 = (vec_idx * VEC_SIZE) / x;
        const int offset2 = (vec_idx * VEC_SIZE) % x;

        if constexpr (KV_DTYPE == Fp8KVCacheDataType::kAuto) {
          k_vecs[j] = *reinterpret_cast<const K_vec*>(
              k_ptr + offset1 * BLOCK_SIZE * x + offset2);
        } else {
          // Vector conversion from Quant_vec to K_vec.
          Quant_vec k_vec_quant = *reinterpret_cast<const Quant_vec*>(
              k_ptr + offset1 * BLOCK_SIZE * x + offset2);
          k_vecs[j] = fp8::scaled_convert<K_vec, Quant_vec, KV_DTYPE>(
              k_vec_quant, kv_scale);
        }
      }

      // Compute dot product.
      // This includes a reduction across the threads in the same thread group.
      float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(
                             q_vecs[thread_group_offset], k_vecs);
      // Add the ALiBi bias if slopes are given.
      qk += (alibi_slope != 0) ? alibi_slope * (token_idx - seq_len + 1) : 0;
      if (thread_group_offset == 0) {
        // Store the partial reductions to shared memory.
        // NOTE(woosuk): It is required to zero out the masked logits.
        const bool mask = token_idx >= seq_len;
        logits[token_idx - start_token_idx] = mask ? 0.f : qk;
        // Update the max value.
        qk_max = mask ? qk_max : fmaxf(qk_max, qk);
      }
    }
  }

// Perform reduction across the threads in the same warp to get the
// max qk value for each "warp" (not across the thread block yet).
// The 0-th thread of each thread group already has its max qk value.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = qk_max;
  }
  __syncthreads();
  // Get the max qk value for the sequence.
  qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
  }
  // Broadcast the max qk value to all threads.
  qk_max = VLLM_SHFL_SYNC(qk_max, 0);

  // Get the sum of the exp values.
  float exp_sum = 0.f;
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    float val = __expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

  // Compute softmax.
  const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    logits[i] *= inv_sum;
  }
  __syncthreads();

  // If partitioning is enabled, store the max logit and exp_sum.
  if (USE_PARTITIONING && thread_idx == 0) {
    float* max_logits_ptr = max_logits +
                            seq_idx * num_heads * max_num_partitions +
                            head_idx * max_num_partitions + partition_idx;
    *max_logits_ptr = qk_max;
    float* exp_sums_ptr = exp_sums + seq_idx * num_heads * max_num_partitions +
                          head_idx * max_num_partitions + partition_idx;
    *exp_sums_ptr = exp_sum;
  }
  // Each thread will fetch 16 bytes from the value cache at a time.
  constexpr int V_VEC_SIZE = MIN(16 / sizeof(scalar_t), BLOCK_SIZE);
  using V_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using L_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using V_quant_vec = typename Vec<cache_t, V_VEC_SIZE>::Type;
  using Float_L_vec = typename FloatVec<L_vec>::Type;
  constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE;
  constexpr int NUM_ROWS_PER_ITER = WARP_SIZE / NUM_V_VECS_PER_ROW;
  constexpr int NUM_ROWS_PER_THREAD =
      DIVIDE_ROUND_UP(HEAD_SIZE, NUM_ROWS_PER_ITER);
  // NOTE(woosuk): We use FP32 for the accumulator for better accuracy.
  float accs[NUM_ROWS_PER_THREAD];

#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    accs[i] = 0.f;
  }
  scalar_t zero_value;
  zero(zero_value);

  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
       block_idx += NUM_WARPS) {
    if constexpr (IS_BLOCK_SPARSE) {
      int v_bs_block_id = block_idx * BLOCK_SIZE / blocksparse_block_size;
      if (!((v_bs_block_id + bs_block_offset) % blocksparse_vert_stride == 0) &&
          !((v_bs_block_id > q_bs_block_id - blocksparse_local_blocks))) {
        continue;
      }
    }

    // const int64_t physical_block_number =
    //     static_cast<int64_t>(block_table[block_idx]);
    // 获得当前block在全局内存中的偏移量
    uint64_t block_global_offset = block_table[block_idx];
    const int physical_block_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
    L_vec logits_vec;
    from_float(logits_vec, *reinterpret_cast<Float_L_vec*>(logits + token_idx -
                                                           start_token_idx));
    // const cache_t* v_ptr = v_cache + physical_block_number * kv_block_stride
    // +
    //                        kv_head_idx * kv_head_stride;
    // 获得块内数据地址。Value和Key在同一个块内，块内偏移量：num_kv_heads *
    // (HEAD_SIZE / x) * BLOCK_SIZE * x
    const cache_t* v_ptr =
        reinterpret_cast<const cache_t*>(global_memory + block_global_offset) +
        layer_id * num_kv_heads * (HEAD_SIZE / x) * BLOCK_SIZE * x * 2 +
        num_kv_heads * (HEAD_SIZE / x) * BLOCK_SIZE * x +
        kv_head_idx * kv_head_stride;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE) {
        const int offset = row_idx * BLOCK_SIZE + physical_block_offset;
        V_vec v_vec;

        if constexpr (KV_DTYPE == Fp8KVCacheDataType::kAuto) {
          v_vec = *reinterpret_cast<const V_vec*>(v_ptr + offset);
        } else {
          V_quant_vec v_quant_vec =
              *reinterpret_cast<const V_quant_vec*>(v_ptr + offset);
          // Vector conversion from V_quant_vec to V_vec.
          v_vec = fp8::scaled_convert<V_vec, V_quant_vec, KV_DTYPE>(v_quant_vec,
                                                                    kv_scale);
        }
        if (block_idx == num_seq_blocks - 1) {
          // NOTE(woosuk): When v_vec contains the tokens that are out of the
          // context, we should explicitly zero out the values since they may
          // contain NaNs. See
          // https://github.com/vllm-project/vllm/issues/641#issuecomment-1682544472
          scalar_t* v_vec_ptr = reinterpret_cast<scalar_t*>(&v_vec);
#pragma unroll
          for (int j = 0; j < V_VEC_SIZE; j++) {
            v_vec_ptr[j] = token_idx + j < seq_len ? v_vec_ptr[j] : zero_value;
          }
        }
        accs[i] += dot(logits_vec, v_vec);
      }
    }
  }

// Perform reduction within each warp.
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    float acc = accs[i];
#pragma unroll
    for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
      acc += VLLM_SHFL_XOR_SYNC(acc, mask);
    }
    accs[i] = acc;
  }

  // NOTE(woosuk): A barrier is required because the shared memory space for
  // logits is reused for the output.
  __syncthreads();

  // Perform reduction across warps.
  float* out_smem = reinterpret_cast<float*>(shared_mem);
#pragma unroll
  for (int i = NUM_WARPS; i > 1; i /= 2) {
    int mid = i / 2;
    // Upper warps write to shared memory.
    if (warp_idx >= mid && warp_idx < i) {
      float* dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          dst[row_idx] = accs[i];
        }
      }
    }
    __syncthreads();

    // Lower warps update the output.
    if (warp_idx < mid) {
      const float* src = &out_smem[warp_idx * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          accs[i] += src[row_idx];
        }
      }
    }
    __syncthreads();
  }

  // Write the final output.
  if (warp_idx == 0) {
    scalar_t* out_ptr =
        out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
        head_idx * max_num_partitions * HEAD_SIZE + partition_idx * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        from_float(*(out_ptr + row_idx), accs[i]);
      }
    }
  }
}

template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, vllm::Fp8KVCacheDataType KV_DTYPE,
          bool IS_BLOCK_SPARSE>
__global__ void segmented_attention_v1_kernel(
    scalar_t* __restrict__ out,        // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ q,    // [num_seqs, num_heads, head_size]
    void* __restrict__ global_memory,  // global memory pointer
    const int num_kv_heads,            // [num_heads]
    const float scale,
    const uint64_t* __restrict__ block_tables,  // [num_seqs,
                                                // max_num_blocks_per_seq]
    const int layer_id,
    const int* __restrict__ seq_lens,  // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float kv_scale, const int tp_rank, const int blocksparse_local_blocks,
    const int blocksparse_vert_stride, const int blocksparse_block_size,
    const int blocksparse_head_sliding_step) {
  segmented_attention_kernel<scalar_t, cache_t, HEAD_SIZE, BLOCK_SIZE,
                             NUM_THREADS, KV_DTYPE, IS_BLOCK_SPARSE>(
      /* exp_sums */ nullptr, /* max_logits */ nullptr, out, q, global_memory,
      num_kv_heads, scale, block_tables, layer_id, seq_lens,
      max_num_blocks_per_seq, alibi_slopes, q_stride, kv_block_stride,
      kv_head_stride, kv_scale, tp_rank, blocksparse_local_blocks,
      blocksparse_vert_stride, blocksparse_block_size,
      blocksparse_head_sliding_step);
}
}  // namespace vllm

#define LAUNCH_SEGMENTED_ATTENTION_V1(HEAD_SIZE)                               \
  VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(                        \
      ((void*)vllm::segmented_attention_v1_kernel<T, CACHE_T, HEAD_SIZE,       \
                                                  BLOCK_SIZE, NUM_THREADS,     \
                                                  KV_DTYPE, IS_BLOCK_SPARSE>), \
      shared_mem_size);                                                        \
  vllm::segmented_attention_v1_kernel<T, CACHE_T, HEAD_SIZE, BLOCK_SIZE,       \
                                      NUM_THREADS, KV_DTYPE, IS_BLOCK_SPARSE>  \
      <<<grid, block, shared_mem_size, stream>>>(                              \
          out_ptr, query_ptr, global_memory, num_kv_heads, scale,              \
          block_tables_ptr, layer_id, seq_lens_ptr, max_num_blocks_per_seq,    \
          alibi_slopes_ptr, q_stride, kv_block_stride, kv_head_stride,         \
          kv_scale, tp_rank, blocksparse_local_blocks,                         \
          blocksparse_vert_stride, blocksparse_block_size,                     \
          blocksparse_head_sliding_step);

template <typename T, typename CACHE_T, int BLOCK_SIZE,
          vllm::Fp8KVCacheDataType KV_DTYPE, bool IS_BLOCK_SPARSE,
          int NUM_THREADS = 128>
void segmented_attention_v1_launcher(
    torch::Tensor& out, torch::Tensor& query, void* global_memory,
    int num_kv_heads, float scale, torch::Tensor& block_tables, int layer_id,
    torch::Tensor& seq_lens, int max_seq_len,
    const c10::optional<torch::Tensor>& alibi_slopes, float kv_scale,
    const int tp_rank, const int blocksparse_local_blocks,
    const int blocksparse_vert_stride, const int blocksparse_block_size,
    const int blocksparse_head_sliding_step) {
  int num_seqs = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);

  int x = 16 / sizeof(CACHE_T);
  int kv_block_stride = num_kv_heads * head_size / x * BLOCK_SIZE * x;
  int kv_head_stride = head_size / x * BLOCK_SIZE * x;
  // int kv_block_stride = key_cache.stride(0);
  // int kv_head_stride = key_cache.stride(1);

  int thread_group_size = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  assert(head_size % thread_group_size == 0);

  // NOTE: alibi_slopes is optional.
  const float* alibi_slopes_ptr =
      alibi_slopes
          ? reinterpret_cast<const float*>(alibi_slopes.value().data_ptr())
          : nullptr;

  T* out_ptr = reinterpret_cast<T*>(out.data_ptr());
  T* query_ptr = reinterpret_cast<T*>(query.data_ptr());
  // CACHE_T* key_cache_ptr = reinterpret_cast<CACHE_T*>(key_cache.data_ptr());
  // CACHE_T* value_cache_ptr =
  // reinterpret_cast<CACHE_T*>(value_cache.data_ptr());
  uint64_t* block_tables_ptr =
      reinterpret_cast<uint64_t*>(block_tables.data_ptr<int64_t>());

  int* seq_lens_ptr = seq_lens.data_ptr<int>();

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int padded_max_seq_len =
      DIVIDE_ROUND_UP(max_seq_len, BLOCK_SIZE) * BLOCK_SIZE;
  int logits_size = padded_max_seq_len * sizeof(float);
  int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);
  // Python-side check in vllm.worker.worker._check_if_can_support_max_seq_len
  // Keep that in sync with the logic here!
  int shared_mem_size = std::max(logits_size, outputs_size);

  dim3 grid(num_heads, num_seqs, 1);
  dim3 block(NUM_THREADS);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  switch (head_size) {
    // NOTE(woosuk): To reduce the compilation time, we only compile for the
    // head sizes that we use in the model. However, we can easily extend this
    // to support any head size which is a multiple of 16.
    case 64:
      LAUNCH_SEGMENTED_ATTENTION_V1(64);
      break;
    case 80:
      LAUNCH_SEGMENTED_ATTENTION_V1(80);
      break;
    case 96:
      LAUNCH_SEGMENTED_ATTENTION_V1(96);
      break;
    case 112:
      LAUNCH_SEGMENTED_ATTENTION_V1(112);
      break;
    case 128:
      LAUNCH_SEGMENTED_ATTENTION_V1(128);
      break;
    case 192:
      LAUNCH_SEGMENTED_ATTENTION_V1(192);
      break;
    case 256:
      LAUNCH_SEGMENTED_ATTENTION_V1(256);
      break;
    default:
      TORCH_CHECK(false, "Unsupported head size: ", head_size);
      break;
  }
}

#define CALL_SEGMENTED_V1_LAUNCHER(T, CACHE_T, BLOCK_SIZE, KV_DTYPE,          \
                                   IS_BLOCK_SPARSE)                           \
  segmented_attention_v1_launcher<T, CACHE_T, BLOCK_SIZE, KV_DTYPE,           \
                                  IS_BLOCK_SPARSE>(                           \
      out, query, reinterpret_cast<void*>(global_memory), num_kv_heads, scale, block_tables, layer_id, \
      seq_lens, max_seq_len, alibi_slopes, kv_scale, tp_rank,                 \
      blocksparse_local_blocks, blocksparse_vert_stride,                      \
      blocksparse_block_size, blocksparse_head_sliding_step);

#define CALL_SEGMENTED_V1_LAUNCHER_SPARSITY(T, CACHE_T, BLOCK_SIZE,       \
                                            IS_FP8_KV_CACHE)              \
  switch (is_block_sparse) {                                              \
    case true:                                                            \
      CALL_SEGMENTED_V1_LAUNCHER(T, CACHE_T, BLOCK_SIZE, IS_FP8_KV_CACHE, \
                                 true);                                   \
      break;                                                              \
    case false:                                                           \
      CALL_SEGMENTED_V1_LAUNCHER(T, CACHE_T, BLOCK_SIZE, IS_FP8_KV_CACHE, \
                                 false);                                  \
      break;                                                              \
  }
#define CALL_SEGMENTED_V1_LAUNCHER_BLOCK_SIZE(T, CACHE_T, KV_DTYPE)  \
  switch (block_size) {                                              \
    case 8:                                                          \
      CALL_SEGMENTED_V1_LAUNCHER_SPARSITY(T, CACHE_T, 8, KV_DTYPE);  \
      break;                                                         \
    case 16:                                                         \
      CALL_SEGMENTED_V1_LAUNCHER_SPARSITY(T, CACHE_T, 16, KV_DTYPE); \
      break;                                                         \
    case 32:                                                         \
      CALL_SEGMENTED_V1_LAUNCHER_SPARSITY(T, CACHE_T, 32, KV_DTYPE); \
      break;                                                         \
    default:                                                         \
      TORCH_CHECK(false, "Unsupported block size: ", block_size);    \
      break;                                                         \
  }

/*
Current KV Block layout:
layer-0
  |   key cache      |   value cache    |   key cache      |   value cache    |
  |------------------|------------------|------------------|------------------|
  |   key cache      |   value cache    |   key cache      |   value cache    |
  |------------------|------------------|------------------|------------------|
  |   ^^^^^^^^^      |   ^^^^^^^^^^^    |   ^^^^^^^^^      |   ^^^^^^^^^^^    |
layer-1
  |   key cache      |   value cache    |   key cache      |   value cache    |
  |------------------|------------------|------------------|------------------|
  |   key cache      |   value cache    |   key cache      |   value cache    |
  |------------------|------------------|------------------|------------------|
  |   ^^^^^^^^^      |   ^^^^^^^^^^^    |   ^^^^^^^^^      |   ^^^^^^^^^^^    |
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
*/
void segmented_attention_v1(
    torch::Tensor& out,    // [num_seqs, num_heads, head_size]
    torch::Tensor& query,  // [num_seqs, num_heads, head_size]
    int64_t global_memory,
    int64_t num_kv_heads,  // [num_heads]
    double scale,
    torch::Tensor& block_tables,  // [num_seqs, max_num_blocks_per_seq]
    int64_t layer_id,
    torch::Tensor& seq_lens,  // [num_seqs]
    int64_t block_size, int64_t max_seq_len,
    const c10::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, double kv_scale, const int64_t tp_rank,
    const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride, const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step) {
  const bool is_block_sparse = (blocksparse_vert_stride > 1);

  DISPATCH_BY_KV_CACHE_DTYPE(query.dtype(), kv_cache_dtype,
                             CALL_SEGMENTED_V1_LAUNCHER_BLOCK_SIZE)
}
