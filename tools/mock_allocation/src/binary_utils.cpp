// ----------------------------------------------------------------------------
//  ServerlessLLM
//  Copyright (c) ServerlessLLM Team 2024
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//
//   You may obtain a copy of the License at
//
//                   http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
//  ----------------------------------------------------------------------------

#include "binary_utils.h"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

// Function to print the binary array in hexadecimal format
void PrintBinaryArrayInHex(const unsigned char* data, size_t size) {
  std::cout << "Data in Hex: ";
  for (size_t i = 0; i < size; ++i) {
    std::cout << std::hex << std::setw(2) << std::setfill('0')
              << static_cast<int>(data[i]) << " ";
  }
  std::cout << std::dec
            << std::endl;  // Switch back to decimal for any future output
}
std::string toHex(const std::vector<uint8_t>& data) {
  std::stringstream ss;
  for (uint8_t byte : data) {
    ss << std::setw(2) << std::setfill('0') << std::hex << (int)byte;
  }
  return ss.str();
}

std::vector<uint8_t> fromHex(const std::string& hexStr) {
  std::vector<uint8_t> data;
  for (size_t i = 0; i < hexStr.length(); i += 2) {
    uint8_t byte = (std::stoi(hexStr.substr(i, 2), nullptr, 16));
    data.push_back(byte);
  }
  return data;
}

void* allocateAlignedPinnedMemory(size_t size, size_t alignment) {
  // 1. posix_memalign allocates aligned memory
  void* aligned_mem = NULL;
  int ret = posix_memalign(&aligned_mem, alignment, size);
  if (ret != 0 || aligned_mem == NULL) {
    perror("posix_memalign");
    return nullptr;
  }

  // 2. register the allocated memory to the CUDA device
  cudaError_t cuda_status =
      cudaHostRegister(aligned_mem, size, cudaHostRegisterDefault);
  if (cuda_status != cudaSuccess) {
    fprintf(stderr, "cudaHostRegister failed: %s\n",
            cudaGetErrorString(cuda_status));
    free(aligned_mem);
    return nullptr;
  }

  return aligned_mem;
}

void freeAlignedPinnedMemory(void* ptr) {
  // 1. unregister the memory from the CUDA device
  cudaError_t cuda_status = cudaHostUnregister(ptr);
  if (cuda_status != cudaSuccess) {
    fprintf(stderr, "cudaHostUnregister failed: %s\n",
            cudaGetErrorString(cuda_status));
  }
  // 2. free the memory
  free(ptr);
}


// 实现 cuda_safe_move
// 参数:
//   free_region_base_addr: 目标地址（free region 开始地址）
//   data_addr: 数据源地址
//   data_size: 需要拷贝的数据总字节数
//
// 算法思路：
//   1. 计算两地址之间的 gap，作为一次拷贝的最大安全长度；
//   2. 循环拷贝，每次拷贝 chunk = min(gap, 剩余数据量)；
//   3. 使用 cudaMemcpyDeviceToDevice 进行设备间拷贝。
cudaError_t cuda_safe_move(void* free_region_base_addr, void* data_addr,
                                  size_t data_size) {
  char* dest = static_cast<char*>(free_region_base_addr);
  char* src = static_cast<char*>(data_addr);

  // 处理无数据的情况
  if (data_size == 0) return cudaSuccess;

  // 检查指针是否有效
  if (dest == nullptr || src == nullptr) return cudaErrorInvalidValue;

  if (dest == src) return cudaSuccess;

  // 计算地址是否重叠
  const size_t overlap_cond = (dest > src) ? (dest - src) : (src - dest);
  const bool overlap = (overlap_cond < data_size);

  if (!overlap) {
    // 无重叠，直接拷贝
    cudaError_t err = cudaMemcpy(dest, src, data_size, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) return err;
    // 同步流，确保数据拷贝已经结束
    err = cudaStreamSynchronize(0); // 默认流
    if (err != cudaSuccess) return err;
  } else {
    // 处理重叠情况
    if (dest < src) {
      // 正向分块拷贝
      size_t gap = src - dest;
      size_t offset = 0;
      while (offset < data_size) {
        size_t chunk = std::min(data_size - offset, gap);
        cudaError_t err = cudaMemcpy(dest + offset, src + offset, chunk,
                                     cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) return err;
        // 同步流，确保数据拷贝已经结束
        err = cudaStreamSynchronize(0); // 默认流
        if (err != cudaSuccess) return err;
        offset += chunk;
      }
    } else {
      // 反向分块拷贝
      size_t gap = dest - src;
      size_t remaining = data_size;
      while (remaining > 0) {
        size_t chunk = std::min(remaining, gap);
        remaining -= chunk;
        cudaError_t err = cudaMemcpy(dest + remaining, src + remaining, chunk,
                                     cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) return err;
        // 同步流，确保数据拷贝已经结束
        err = cudaStreamSynchronize(0); // 默认流
        if (err != cudaSuccess) return err;
      }
    }
  }

  return cudaSuccess;
}



#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/maximum_weighted_matching.hpp>

vector<pair<int, int>> MaxWeightBMatchingWithBoost(const vector<vector<BipartEdge>>& left_edges) {
    // 确定左右节点数量
    int left_size = left_edges.size();
    int right_size = 0;
    for (const auto& edges : left_edges) {
        for (const auto& edge : edges) {
            right_size = max(right_size, (int)edge.right_idx + 1);
        }
    }
    int total_nodes = left_size + right_size;  // 左右节点统一编号（左0~left_size-1，右left_size~total_nodes-1）

    // 定义图类型（无向图，带权边）
    typedef boost::adjacency_list<
        boost::vecS, 
        boost::vecS,
        boost::undirectedS,
        boost::no_property,
        boost::property<boost::edge_weight_t, int64_t>
    > Graph;
    typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;

    Graph g(total_nodes);
    // 添加边到图中
    for (int i = 0; i < left_size; ++i) {
        for (const auto& edge : left_edges[i]) {
            int j = edge.right_idx;
            boost::add_edge(i, left_size + j, static_cast<int64_t>(edge.weight), g);
            // std::cout<<"add edge: "<<i<<" "<<left_size + j<<" "<<edge.weight<<std::endl;
        }
    }

    std::vector<Graph::vertex_descriptor> mate(boost::num_vertices(g));
    boost::maximum_weighted_matching(g, &mate[0]);

    std::vector<pair<int, int>> result;

    for (size_t i = 0; i < mate.size(); ++i) {
      if (mate[i] != Graph::null_vertex() && i < mate[i]) {
          result.emplace_back(i, mate[i]-left_size);
          // std::cout << "匹配边: " << i << " - " << mate[i] << "\n";
      }
    } 
    return result;
}

// 检查是否可以将request_sizes分成两部分，使得左右两部分的总大小分别不超过left_total_size和right_total_size
/*
贪心实现：
  假设request_size已经按照大小降序排列
  遍历request_size：
    检查是否能够分配给左桶和右桶中大的那个
    如果不能分配，返回false
    如果能分配：
      执行更新桶大小
      记录分配结果
  返回true和分配结果
*/
CanSplitResult CanSplit(size_t left_total_size, size_t right_total_size, std::vector<size_t> request_sizes){

  // 检查请求是否已排序
  for (size_t i = 1; i < request_sizes.size(); i++) {
    if (request_sizes[i] > request_sizes[i-1]) {
      std::cout << "ERROR: request_sizes is not sorted in descending order"<<std::endl;
      return {false, {}, {}};
    }
  }

  CanSplitResult result{true, {}, {}};
  size_t left_remaining = left_total_size;
  size_t right_remaining = right_total_size;
  
  for (auto req : request_sizes) {
    // std::cout<<"left_remaining: "<<left_remaining<<" right_remaining: "<<right_remaining<<" req: "<<req<<std::endl;
    bool can_assign_left = (req <= left_remaining);
    bool can_assign_right = (req <= right_remaining);
    
    // 优先分配给剩余空间较大的桶
    if (can_assign_left && (left_remaining >= right_remaining || !can_assign_right)) {
      result.to_left_sizes.push_back(req);
      left_remaining -= req;
    } else if (can_assign_right) {
      result.to_right_sizes.push_back(req);
      right_remaining -= req;
    } else {
      return {false, {}, {}};
    }
  }
  return result;
}
