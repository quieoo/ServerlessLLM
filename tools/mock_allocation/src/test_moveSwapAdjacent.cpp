// test case
#include <cuda_runtime.h>

#include <cassert>
#include <iostream>

#include "model_pool.h"  // 包含 GPUMemoryRegion 与 moveSwapAdjacent 定义
#include "util.h"

void printRegionChain(const std::shared_ptr<GPUMemoryRegion>& head) {
  auto cur = head;
  int index = 1;
  while (cur) {
    std::cout << "Region " << index << ": " << cur->toString() << std::endl;
    cur = cur->next;
    index++;
  }
}
int test_region_swap() {
  cudaSetDevice(0);

  // 分配一块 1024 字节的 GPU 内存，用于构建三个区域
  constexpr size_t totalSize = 1024;
  char* deviceMemory = nullptr;
  cudaError_t err = cudaMalloc(&deviceMemory, totalSize);
  if (err != cudaSuccess) {
    std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
    return -1;
  }

  // 构造三个区域：
  // region1: Free, 大小 256, 起始地址 deviceMemory
  // region2: Allocated, 大小 512, 起始地址
  // deviceMemory+256，且填充测试数据（例如填充 55） region3: Free, 大小 256,
  // 起始地址 deviceMemory+768
  constexpr size_t region1Size = 256;
  constexpr size_t region2Size = 512;
  constexpr size_t region3Size = totalSize - region1Size - region2Size;  // 256

  auto region1 = std::make_shared<GPUMemoryRegion>(deviceMemory, region1Size);
  auto region2 = std::make_shared<GPUMemoryRegion>(deviceMemory + region1Size,
                                                   region2Size);
  auto region3 = std::make_shared<GPUMemoryRegion>(
      deviceMemory + region1Size + region2Size, region3Size);

  // 设置链表关系
  region1->next = region2;
  region2->prev = region1;
  region2->next = region3;
  region3->prev = region2;

  // 设置各自状态
  region1->status = Free;
  region2->status = Allocated;
  region3->status = Free;
  region2->fingerprint = "test_region";

  // 填充 region2 测试数据，全部设置为 55
  err = cudaMemset(region2->addr, 55, region2Size);
  if (err != cudaSuccess) {
    std::cerr << "cudaMemset failed: " << cudaGetErrorString(err) << std::endl;
    cudaFree(deviceMemory);
    return -1;
  }

  std::cout << "Before moveSwapAdjacent, region chain:" << std::endl;
  printRegionChain(region1);

  // 调用 moveSwapAdjacent 对 region1 与 region2 进行交换，
  // 要求 region1 状态必须为 Free，region2 为 Allocated且二者相邻
  region1->moveSwapAdjacent(region2);

  // 由于区域位置已发生改变，但链表关系仍未变更，故 region1->next 仍指向 region2
  std::cout << "\nAfter moveSwapAdjacent, region chain:" << std::endl;
  printRegionChain(region1);

  // 验证数据是否正确拷贝到交换后的区域（region1）
  // 现 region1 已成为 Allocated，大小与原 region2Size 相等
  char* hostData = new char[region1->size];
  err = cudaMemcpy(hostData, region1->addr, region1->size,
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
    delete[] hostData;
    cudaFree(deviceMemory);
    return -1;
  }

  bool valid = true;
  for (size_t i = 0; i < region1->size; i++) {
    if (hostData[i] != 55) {
      valid = false;
      break;
    }
  }

  std::cout << "\nData validation for swapped region: "
            << (valid ? "SUCCESS" : "FAILURE") << std::endl;
  std::cout << "Expected swapped region size: " << region2Size
            << ", Actual: " << region1->size << std::endl;

  delete[] hostData;
  cudaFree(deviceMemory);
  return 0;
}


// Before moveSwapAdjacent, region chain:
// Region 1: [GPUMemoryRegion: addr=139928203689984, size=256, fingerprint=, is_allocated=0]
// Region 2: [GPUMemoryRegion: addr=139928203690240, size=512, fingerprint=test_region, is_allocated=2]
// Region 3: [GPUMemoryRegion: addr=139928203690752, size=256, fingerprint=, is_allocated=0]

// After moveSwapAdjacent, region chain:
// Region 1: [GPUMemoryRegion: addr=139928203689984, size=512, fingerprint=test_region, is_allocated=2]
// Region 2: [GPUMemoryRegion: addr=139928203690496, size=256, fingerprint=, is_allocated=0]
// Region 3: [GPUMemoryRegion: addr=139928203690752, size=256, fingerprint=, is_allocated=0]

// Data validation for swapped region: SUCCESS
// Expected swapped region size: 512, Actual: 512