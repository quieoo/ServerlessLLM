#pragma once

#include <iostream>
#include <cassert>
#include <cmath>
// #include "model_pool.h"
#include "logger.h"
#include "gpu_tensor_pool_v4.h"

// 测试分配一个较小的区域（直接在连续 Free 空间上分裂）
inline void TestSimpleAllocation() {
    std::cout << "Running TestSimpleAllocation..." << std::endl;
    size_t total_size = 1024; // 1KB
    // 创建 GPU Tensor Pool，device_id 为 0
    GPUTensorPool_V4 pool(0, total_size);

    std::cout << "Initial Memory Region View:" << std::endl;
    pool.MemoryRegionView();
    float frag_before = pool.GetFragmentation();
    std::cout << "Initial Fragmentation: " << frag_before << std::endl;

    // 分配 512 字节区域
    auto region = pool.AllocateASAP(512);
    assert(region != nullptr);
    // 分配后 splitAllocated 将所分配区域状态更新为 Allocated
    assert(region->status == Allocated);
    assert(region->size == 512);

    std::cout << "Allocated Region: " << region->toString() << std::endl;
    std::cout << "Memory Region View After Allocation:" << std::endl;
    pool.MemoryRegionView();
    float frag_after = pool.GetFragmentation();
    std::cout << "Fragmentation After Allocation: " << frag_after << std::endl;
    // 剩余空间 512 字节，碎片率应为 512 / 1024 = 0.5
    assert(std::fabs(frag_after - 0.5f) < 0.01f);

    std::cout << "TestSimpleAllocation Passed." << std::endl << std::endl;
}

// 测试分配一个与总空间相等的区域
inline void TestFullAllocation() {
    std::cout << "Running TestFullAllocation..." << std::endl;
    size_t total_size = 1024; // 1KB
    GPUTensorPool_V4 pool(0, total_size);

    // 直接请求整个内存区域
    auto region = pool.AllocateASAP(1024);
    assert(region != nullptr);
    assert(region->status == Allocated);
    assert(region->size == 1024);

    std::cout << "Allocated Region: " << region->toString() << std::endl;
    std::cout << "Memory Region View After Allocation:" << std::endl;
    pool.MemoryRegionView();
    float frag_after = pool.GetFragmentation();
    std::cout << "Fragmentation After Full Allocation: " << frag_after << std::endl;
    // 整个内存被分配，碎片率应为 0
    assert(std::fabs(frag_after - 0.0f) < 0.001f);

    std::cout << "TestFullAllocation Passed." << std::endl << std::endl;
}


inline void TestFragmentationMerge() {
    std::cout << "Running TestFragmentationMerge..." << std::endl;
    size_t total_size = 1024; // 1KB
    // 创建 GPU Tensor Pool，device_id 为 0
    GPUTensorPool_V4 pool(0, total_size);

    std::cout << "Initial Memory Region View:" << std::endl;
    pool.MemoryRegionView();

    // 第一步：分配300字节
    auto region1 = pool.AllocateASAP(300);
    assert(region1 != nullptr);
    assert(region1->status == Allocated);
    assert(region1->size == 300);
    std::cout << "Allocated region1 (300 bytes): " << region1->toString() << std::endl;
    // 此时内存链： [region1 Allocated(300)] -> [region2 Free(724)]

    // 第二步：分配另一个300字节
    auto region2 = pool.AllocateASAP(300);
    assert(region2 != nullptr);
    assert(region2->status == Allocated);
    assert(region2->size == 300);
    std::cout << "Allocated region2 (300 bytes): " << region2->toString() << std::endl;
    // 此时内存链： [region1 Allocated(300)] -> [region2 Allocated(300)] -> [region3 Free(424)]

    // 第三步：释放region1制造碎片，释放后不会与region2合并，因为region2为Allocated
    pool.FreeRegion(region1);
    std::cout << "After freeing region1:" << std::endl;
    pool.MemoryRegionView();
    // 现在内存链为：[region1 Free(300)] -> [region2 Allocated(300)] -> [region3 Free(424)]
    // 单个Free区域都不足以满足500字节的申请

    // 第四步：请求500字节空间，此时只能通过合并region1和region3（中间仅隔了region2，需要移动该区域）
    auto region_merge = pool.AllocateASAP(500);
    assert(region_merge != nullptr);
    assert(region_merge->status == Allocated);
    assert(region_merge->size == 500);
    std::cout << "Allocated merged region (500 bytes): " << region_merge->toString() << std::endl;

    std::cout << "Memory Region View After Merged Allocation:" << std::endl;
    pool.MemoryRegionView();
    float frag_after = pool.GetFragmentation();
    std::cout << "Fragmentation After Merged Allocation: " << frag_after << std::endl;
    // 理论上剩余空间应为1024 - (region2仍然Allocated的300 和本次分配的500)=224
    assert(std::fabs(frag_after - (224.0f / 1024)) < 0.01f);

    std::cout << "TestFragmentationMerge Passed." << std::endl << std::endl;
}
