#pragma once
#include <torch/all.h>

void reshape_and_cache_segment(
    torch::Tensor& key,           // [num_tokens, num_heads, head_size]
    torch::Tensor& value,         // [num_tokens, num_heads, head_size]
    int64_t global_memory,          // global memory pointer
    torch::Tensor& block_tables,  // [num_seqs, max_num_blocks_per_seq]
    int64_t layer_id,             // layer id
    torch::Tensor& slot_mapping,  // [num_tokens]
    int64_t block_size,               // number of slots in a block
    const std::string& kv_cache_dtype, const double kv_scale);
void segmented_attention_v1(
    torch::Tensor& out,    // [num_seqs, num_heads, head_size]
    torch::Tensor& query,  // [num_seqs, num_heads, head_size]
    int64_t global_memory,
    int64_t num_kv_heads,  // [num_heads]
    double scale,
    torch::Tensor& block_tables,  // [num_seqs, max_num_blocks_per_seq]
    int64_t layer_id,
    torch::Tensor& seq_lens,      // [num_seqs]
    int64_t block_size, int64_t max_seq_len,
    const c10::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, double kv_scale, const int64_t tp_rank,
    const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride, const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step);

void reshape_and_cache(
    torch::Tensor& key,    // [num_tokens, num_heads, head_size]
    torch::Tensor& value,  // [num_tokens, num_heads, head_size]
    torch::Tensor&
        key_cache,  // [num_blocks, num_heads, head_size/x, block_size, x]
    torch::Tensor&
        value_cache,  // [num_blocks, num_heads, head_size, block_size]
    torch::Tensor& slot_mapping,  // [num_tokens]
    const std::string& kv_cache_dtype, const double kv_scale);
void paged_attention_v1(
    torch::Tensor& out,    // [num_seqs, num_heads, head_size]
    torch::Tensor& query,  // [num_seqs, num_heads, head_size]
    torch::Tensor&
        key_cache,  // [num_blocks, num_heads, head_size/x, block_size, x]
    torch::Tensor&
        value_cache,       // [num_blocks, num_heads, head_size, block_size]
    int64_t num_kv_heads,  // [num_heads]
    double scale,
    torch::Tensor& block_tables,  // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor& seq_lens,      // [num_seqs]
    int64_t block_size, int64_t max_seq_len,
    const c10::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, double kv_scale, const int64_t tp_rank,
    const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride, const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step);