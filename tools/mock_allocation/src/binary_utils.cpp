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

template <typename T>
std::string Join(const std::vector<T>& vec, const std::string& delimiter) {
  std::ostringstream oss;
  for (size_t i = 0; i < vec.size(); ++i) {
    oss << vec[i];
    if (i != vec.size() - 1) {
      oss << delimiter;
    }
  }
  return oss.str();
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
