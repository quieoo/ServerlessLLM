#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include <chrono>
#include <iostream>

#include "cache.h"
#include "quant_utils.cuh"

// global parameters
int num_tokens = 200;
int num_heads = 32;
int num_kv_heads = 32;
int head_size = 128;
int num_layers = 32;
int block_size = 16;
int x = 16 / sizeof(float);
float kv_scale = 1.0f;
float scale = 1.0f;
int num_seqs = 1;
int max_num_blocks_per_seq = 16;
int64_t max_seq_len = 128;
int num_blocks = num_seqs * max_num_blocks_per_seq;
// '(num_tokens + (block_size-1)) / block_size' should be smaller than
// 'num_blocks=num_seqs*max_num_blocks_per_seq'
c10::optional<torch::Tensor> alibi_slopes = c10::nullopt;
int64_t tp_rank = 0;
int64_t blocksparse_local_blocks = 0;
int64_t blocksparse_vert_stride = 0;
int64_t blocksparse_block_size = 64;
int64_t blocksparse_head_sliding_step = 0;

// global tensors
torch::Tensor key;
torch::Tensor value;
torch::Tensor slot_mapping;
torch::Tensor key_cache;
torch::Tensor value_cache;
torch::Tensor query;
torch::Tensor out;
torch::Tensor block_tables;
torch::Tensor block_tables_segmentd;
torch::Tensor seq_lens;
torch::Tensor out_copy;

void prepare_tensors() {
  at::cuda::CUDAGuard device_guard(2);
  query =
      torch::randn({num_seqs, num_heads, head_size}, torch::kFloat32).cuda();
  key = torch::randn({num_tokens, num_kv_heads, head_size}, torch::kFloat32)
            .cuda();
  value = torch::randn({num_tokens, num_kv_heads, head_size}, torch::kFloat32)
              .cuda();
  out = torch::zeros({num_seqs, num_heads, head_size}, torch::kFloat32).cuda();

  slot_mapping = torch::arange(0, num_tokens, torch::kInt64).cuda();
  block_tables = torch::arange(num_seqs * max_num_blocks_per_seq, torch::kInt)
                     .reshape({num_seqs, max_num_blocks_per_seq})
                     .cuda();
  block_tables_segmentd =
      torch::arange(num_seqs * max_num_blocks_per_seq, torch::kInt64)
          .reshape({num_seqs, max_num_blocks_per_seq})
          .cuda();
  int64_t scale = static_cast<int64_t>(num_layers * block_size * num_kv_heads * head_size *
                                       sizeof(float) * 2);
  block_tables_segmentd = block_tables_segmentd * scale;

  // make block_tables_segments more fragmented
  // case-1: leaf-overs
  // int64_t scale = static_cast<int64_t>(block_size * num_kv_heads * head_size *
  //   sizeof(float) * 2.1);
  //
  // case-2: unaligned
  // block_tables_segmentd = block_tables_segmentd + 64;

  // std::cout << "###### key ######" << std::endl;
  // std::cout << key.cpu() << std::endl;
  // std::cout << "###### value ######" << std::endl;
  // std::cout << value.cpu() << std::endl;

  // std::cout << "##### Slot Mapping: " << slot_mapping.cpu() << std::endl;

  key_cache =
      torch::zeros({num_blocks, num_kv_heads, head_size / x, block_size, x},
                   torch::kFloat32)
          .cuda();
  value_cache = torch::zeros({num_blocks, num_kv_heads, head_size, block_size},
                             torch::kFloat32)
                    .cuda();

  // std::cout << "##### Block Tables: " << block_tables.cpu() << std::endl;
  // std::cout << "##### Block Tables Segmentd: " << block_tables_segmentd.cpu()
  //           << std::endl;
  seq_lens = torch::randint(num_tokens, num_tokens + 1, {num_seqs}, torch::kInt)
                 .cuda();
  // std::cout << "##### Seq Lens: " << seq_lens.cpu() << std::endl;
}

void test_paged_attention() {
  at::cuda::CUDAGuard device_guard(2);
  // std::cout << "-----------------test_paged_attention-----------------"
  //           << std::endl;

  auto start = std::chrono::high_resolution_clock::now();
  reshape_and_cache(key, value, key_cache, value_cache, slot_mapping, "auto",
                    kv_scale);

  // std::cout << "##### Key Cache #####" << std::endl;
  // std::cout << key_cache.cpu() << std::endl;
  // std::cout << "##### Value Cache ##### " << std::endl;
  // std::cout << value_cache.cpu() << std::endl;
  paged_attention_v1(out, query, key_cache, value_cache, num_kv_heads, scale,
                     block_tables, seq_lens, block_size, max_seq_len,
                     alibi_slopes, "auto", kv_scale, tp_rank,
                     blocksparse_local_blocks, blocksparse_vert_stride,
                     blocksparse_block_size, blocksparse_head_sliding_step);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::micro> duration = end - start;
  std::cout << "test_paged_attention cost: " << duration.count() << " us"
            << std::endl;

  // std::cout << "##### Query " << std::endl;
  // std::cout << query.cpu() << std::endl;
  // std::cout << "##### Out #####" << std::endl;
  // std::cout << out.cpu() << std::endl;

  out_copy = out.clone();
  out = torch::zeros({num_seqs, num_heads, head_size}, torch::kFloat32).cuda();
}

void test_segmented_attention() {
  at::cuda::CUDAGuard device_guard(2);
  // std::cout <<
  // "-------------------test_segmented_attention-------------------"
  //           << std::endl;
  size_t global_memory_size = num_layers * 2 * num_blocks * block_size * num_kv_heads *
                              head_size * sizeof(float);  // key and value are stored together
  void* global_memory;
  cudaMalloc(&global_memory, global_memory_size);

  int layer_id = 1;

  auto start = std::chrono::high_resolution_clock::now();
  reshape_and_cache_segment(key, value, reinterpret_cast<int64_t>(global_memory), block_tables_segmentd, layer_id,
                            slot_mapping, block_size, "auto", kv_scale);

  // auto key_cache_tensor =
  //     torch::from_blob(global_memory,
  //                      {num_blocks, num_heads, head_size / x, block_size, x},
  //                      torch::kFloat32)
  //         .cuda();
  // // std::cout << "Key Cache: " << std::endl;
  // // std::cout << key_cache_tensor.cpu() << std::endl;
  // auto value_cache_tensor =
  //     torch::from_blob(
  //         (float*)global_memory + num_heads * (head_size / x) * block_size *
  //         x, {num_blocks, num_heads, head_size, block_size}, torch::kFloat32)
  //         .cuda();
  // // std::cout << "Value Cache: " << std::endl;
  // // std::cout << value_cache_tensor.cpu() << std::endl;

  // if(torch::equal(key_cache, key_cache_tensor) && torch::equal(value_cache,
  // value_cache_tensor)){
  //   std::cout << "Key Cache and Value Cache are equal" << std::endl;
  // }else{
  //   std::cout << "FATAL: Key Cache and Value Cache are not equal" <<
  //   std::endl;
  // }

  segmented_attention_v1(out, query, reinterpret_cast<int64_t>(global_memory), num_kv_heads, scale,
                         block_tables_segmentd, layer_id, seq_lens, block_size,
                         max_seq_len, alibi_slopes, "auto", kv_scale, tp_rank,
                         blocksparse_local_blocks, blocksparse_vert_stride,
                         blocksparse_block_size, blocksparse_head_sliding_step);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::micro> duration = end - start;
  std::cout << "test_segmented_attention cost: " << duration.count() << " us"
            << std::endl;
  // std::cout << "##### Query #####" << std::endl;
  // std::cout << query.cpu() << std::endl;
  // std::cout << "##### Out #####" << std::endl;
  // std::cout << out.cpu() << std::endl;
  

  if (torch::equal(out, out_copy)) {
    std::cout << "Passed: out tensor is equal" << std::endl;
  } else {
    std::cout << "FATAL: out tensor is not equal" << std::endl;
  }
  cudaFree(global_memory);
}

void correctness_check() {
  try {
    prepare_tensors();
    test_paged_attention();
    test_segmented_attention();
  } catch (const std::exception& e) {
    std::cerr << "Test failed: " << e.what() << std::endl;
  }
}

void performance_test(int cnt) {
  // test segmentd attention
  size_t global_memory_size = num_layers * num_blocks * block_size * num_kv_heads *
                              head_size * sizeof(float) *
                              2;  // key and value are stored together
  void* global_memory;
  cudaMalloc(&global_memory, global_memory_size);

  auto start2 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < cnt; i++) {
    reshape_and_cache_segment(key, value, reinterpret_cast<int64_t>(global_memory), block_tables_segmentd, 0,
                              slot_mapping, block_size, "auto", kv_scale);
    segmented_attention_v1(
        out, query, reinterpret_cast<int64_t>(global_memory), num_kv_heads, scale, block_tables_segmentd, 0, 
        seq_lens, block_size, max_seq_len, alibi_slopes, "auto", kv_scale,
        tp_rank, blocksparse_local_blocks, blocksparse_vert_stride,
        blocksparse_block_size, blocksparse_head_sliding_step);
  }
  auto end2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::micro> duration2 = end2 - start2;
  std::cout << "test_segmented_attention cost: " << duration2.count() / cnt
            << " us" << std::endl;
  cudaFree(global_memory);
  // test paged attention
  at::cuda::CUDAGuard device_guard(2);
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < cnt; i++) {
    reshape_and_cache(key, value, key_cache, value_cache, slot_mapping, "auto",
                      kv_scale);
    paged_attention_v1(out, query, key_cache, value_cache, num_kv_heads, scale,
                       block_tables, seq_lens, block_size, max_seq_len,
                       alibi_slopes, "auto", kv_scale, tp_rank,
                       blocksparse_local_blocks, blocksparse_vert_stride,
                       blocksparse_block_size, blocksparse_head_sliding_step);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::micro> duration = end - start;
  std::cout << "test_paged_attention cost: " << duration.count() / cnt << " us"
            << std::endl;
}



int main() {
  correctness_check();
  // prepare_tensors();
  // performance_test(1);
}
