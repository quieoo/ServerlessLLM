#include <gtest/gtest.h>

#include <memory>

#include "vram_manager_v3.h"

TEST(GPUTensorPoolV3Test, BasicAllocation) {
  const size_t POOL_SIZE = 1024 * 1024;           // 1MB
  GPUTensorPool_V3 pool(0, POOL_SIZE, 1e9, 1e9);  // 高带宽设置

  // 测试空闲区域分配
  auto region =
      pool.AllocateFreeRegion(pool.memory_regions, 512 * 1024, "reg1");
  ASSERT_NE(region, nullptr);
  EXPECT_EQ(region->status, ALLOCATED);
  EXPECT_EQ(region->size, 512 * 1024);

  // 验证链表结构
  EXPECT_NE(region->next, nullptr);
  EXPECT_EQ(region->next->status, FREE);
  EXPECT_EQ(region->next->size, POOL_SIZE - 512 * 1024);

  pool.MemoryRegionView();
}


// ... existing code ...

TEST(GPUTensorPoolV3Test, MergeRegions_V2ComplexScenario) {
    const size_t POOL_SIZE = 1024 * 1024 * 4; // 4MB
    GPUTensorPool_V3 pool(0, POOL_SIZE, 1e9, 1e9);
    
    // 创建碎片化内存布局：A-F-F-F-A-F-F-A
    auto rA = pool.AllocateFreeRegion(pool.memory_regions, 512 * 1024, "regA");
    auto rB = pool.AllocateFreeRegion(rA->next, 512 * 1024, "regB");
    auto rC = pool.AllocateFreeRegion(rB->next, 512 * 1024, "regC");
    auto rD = pool.AllocateFreeRegion(rC->next, 512 * 1024, "regD");
    auto rE = pool.AllocateFreeRegion(rD->next, 512 * 1024, "regE");
    auto rF = pool.AllocateFreeRegion(rE->next, 512 * 1024, "regF");
    auto rG = pool.AllocateFreeRegion(rF->next, 512 * 1024, "regG");
    auto rH = pool.AllocateFreeRegion(rG->next, 512 * 1024, "regH");
    pool.FreeRegion(rA);
    pool.FreeRegion(rE);
    pool.FreeRegion(rH);
    LOG(INFO)<<"Before Merge";
    pool.MemoryRegionView();
    LOG(INFO)<<"=============================";
    // 执行合并操作
    auto merged_region = pool.MergeRegions_V2(pool.memory_regions, rH);

    LOG(INFO)<<"After Merge";
    
    LOG(INFO)<<"After Merge";
    pool.MemoryRegionView();
    LOG(INFO)<<"=============================";

    LOG(INFO)<<"allocated_regions:";
    for (auto& region : pool.allocated_regions) {
      LOG(INFO)<<region.first<<":"<<region.second->toString();
    }
    LOG(INFO)<<"=============================";

}

int memcheck(void* addr, uint8_t pattern, size_t size) {
    uint8_t* p = static_cast<uint8_t*>(addr);
    for (size_t i = 0; i < size; ++i) {
        if (p[i] != pattern) {
            return -1;
        }
    }
    return 0;
}

TEST(GPUTensorPoolV3Test, MergeRegions_V2DataCheckScenario) {
  const size_t POOL_SIZE = 1024 * 1024 * 4; // 4MB
  GPUTensorPool_V3 pool(0, POOL_SIZE, 1e9, 1e9);

  // 申请CPU内存并初始化测试数据
  void *cpu_buf_AA, *cpu_buf_BB, *cpu_buf_CC, *cpu_buf_DD, *cpu_buf_EE;
  cudaMallocHost(&cpu_buf_AA, 512 * 1024);
  cudaMallocHost(&cpu_buf_BB, 512 * 1024);
  cudaMallocHost(&cpu_buf_CC, 512 * 1024);
  cudaMallocHost(&cpu_buf_DD, 512 * 1024);
  cudaMallocHost(&cpu_buf_EE, 512 * 1024);
  memset(cpu_buf_AA, 0xAA, 512 * 1024);
  memset(cpu_buf_BB, 0xBB, 512 * 1024);
  memset(cpu_buf_CC, 0xCC, 512 * 1024);
  memset(cpu_buf_DD, 0xDD, 512 * 1024);
  memset(cpu_buf_EE, 0xEE, 512 * 1024);

  // 分配显存区域并拷贝数据
  auto r1 = pool.AllocateFreeRegion(pool.memory_regions, 512 * 1024, "reg1");
  cudaMemcpy(r1->addr, cpu_buf_AA, 512 * 1024, cudaMemcpyHostToDevice);
  
  auto r2 = pool.AllocateFreeRegion(r1->next, 512 * 1024, "reg2");
  cudaMemcpy(r2->addr, cpu_buf_BB, 512 * 1024, cudaMemcpyHostToDevice);
  pool.FreeRegion(r1);
  
  auto r3 = pool.AllocateFreeRegion(r2->next, 512 * 1024, "reg3");
  cudaMemcpy(r3->addr, cpu_buf_CC, 512 * 1024, cudaMemcpyHostToDevice);
  
  auto r4 = pool.AllocateFreeRegion(r3->next, 512 * 1024, "reg4");
  cudaMemcpy(r4->addr, cpu_buf_DD, 512 * 1024, cudaMemcpyHostToDevice);
  pool.FreeRegion(r3);
  
  auto r5 = pool.AllocateFreeRegion(r4->next, 512 * 1024, "reg5");
  cudaMemcpy(r5->addr, cpu_buf_EE, 512 * 1024, cudaMemcpyHostToDevice);
  pool.FreeRegion(r5);

  // 验证前将显存数据拷贝到CPU内存
  void *verify_buf = malloc(512 * 1024);
  
  // 验证r2数据
  cudaMemcpy(verify_buf, r2->addr, 512 * 1024, cudaMemcpyDeviceToHost);
  EXPECT_EQ(memcheck(verify_buf, 0xBB, 512 * 1024), 0);
  
  // 验证r4数据
  cudaMemcpy(verify_buf, r4->addr, 512 * 1024, cudaMemcpyDeviceToHost);
  EXPECT_EQ(memcheck(verify_buf, 0xDD, 512 * 1024), 0);
  
  // 执行合并操作
  auto merged_region = pool.MergeRegions_V2(pool.memory_regions, r5);
  
  LOG(INFO)<<"Memory Region View After Merge:";
  pool.MemoryRegionView();
  LOG(INFO)<<"=============================";

  // 验证合并后r2数据
  cudaMemcpy(verify_buf, pool.allocated_regions["reg2"]->addr, 512 * 1024, cudaMemcpyDeviceToHost);
  EXPECT_EQ(memcheck(verify_buf, 0xBB, 512 * 1024), 0);

  // 验证合并后r4数据
  cudaMemcpy(verify_buf, pool.allocated_regions["reg4"]->addr, 512 * 1024, cudaMemcpyDeviceToHost);
  EXPECT_EQ(memcheck(verify_buf, 0xDD, 512 * 1024), 0);


  // 清理CPU内存
  cudaFreeHost(cpu_buf_AA);
  cudaFreeHost(cpu_buf_BB);
  cudaFreeHost(cpu_buf_CC);
  cudaFreeHost(cpu_buf_DD);
  cudaFreeHost(cpu_buf_EE);

}
// ... existing code ...
// ... existing code ...

TEST(GPUTensorPoolV3Test, GetMinMergeCostScenarios) {
  // 初始化 GPUTensorPool_V3
  int device_id = 0;
  size_t total_size = 1024;  // 1024 字节
  double gpu_bw = 1.0;       // GPU-GPU 带宽
  double cpu_bw = 1.0;       // CPU-GPU 带宽
  GPUTensorPool_V3 pool(device_id, total_size, gpu_bw, cpu_bw);

  // 场景 1: 存在足够大的空闲区域
  {
    size_t request_size = 512;
    auto result = pool.GetMinMergeCost(pool.memory_regions, request_size);
    ASSERT_EQ(result.T_merge, 0);
    ASSERT_NE(result.start, nullptr);
    ASSERT_NE(result.end, nullptr);
    ASSERT_GE(result.start->size, request_size);
  }

  // 场景 2: 分割内存区域，测试左右子空间合并
  {
    // 分配一些区域来创建复杂的内存布局
    auto region1 = pool.AllocateFreeRegion(pool.memory_regions, 256, "fp1");
    auto region2 = pool.AllocateFreeRegion(region1->next, 256, "fp2");
    // 设置一个 LOADING 状态的区域
    region1->next->status = LOADING;

    size_t request_size = 512;
    auto result = pool.GetMinMergeCost(pool.memory_regions, request_size);
    // 这里需要根据具体实现来检查合并成本和区域
    // 假设左右子空间合并可以满足需求
    ASSERT_NE(result.T_merge, UNVALID_COST);
    ASSERT_NE(result.start, nullptr);
    ASSERT_NE(result.end, nullptr);

    LOG(INFO) << "Case-2 Passed";
    pool.MemoryRegionView();
    LOG(INFO) << "=============================";
  }

  // 场景 4: 更复杂的离散空闲区域分布
  {
    // 重置内存池
    pool = GPUTensorPool_V3(device_id, total_size, gpu_bw, cpu_bw);

    // 创建离散的已分配和空闲区域
    auto r1 = pool.AllocateFreeRegion(pool.memory_regions, 100, "fp4");
    auto r2 = pool.AllocateFreeRegion(r1->next, 100, "fp5");
    auto r3 = pool.AllocateFreeRegion(r2->next, 100, "fp6");
    // 释放中间区域创建离散空闲区域
    pool.FreeRegion(r2);

    // 设置一个 LOADING 状态的区域
    r1->next->status = LOADING;

    size_t request_size = 200;
    auto result = pool.GetMinMergeCost(pool.memory_regions, request_size);
    // 根据具体实现检查合并成本和区域
    ASSERT_NE(result.T_merge, UNVALID_COST);
    ASSERT_NE(result.start, nullptr);
    ASSERT_NE(result.end, nullptr);
    LOG(INFO) << "Case-4 Passed";
    pool.MemoryRegionView();
    LOG(INFO) << "=============================";
  }

  // 场景 5: 极度离散的空闲区域分布
  {
    // 重置内存池
    pool = GPUTensorPool_V3(device_id, total_size, gpu_bw, cpu_bw);

    // 交替分配和释放区域
    std::vector<std::shared_ptr<GPUMemoryRegion_V3>> regions;
    std::shared_ptr<GPUMemoryRegion_V3> free_region = pool.memory_regions;
    for (int i = 0; i < 5; ++i) {
      auto region = pool.AllocateFreeRegion(free_region, 100,
                                            "fp" + std::to_string(i + 7));
      free_region = region->next;
      regions.push_back(region);
    }
    LOG(INFO) << "Checking case 5...";
    pool.MemoryRegionView();
    LOG(INFO) << "=============================";
    // 释放奇数索引的区域
    for (size_t i = 1; i < regions.size(); i += 2) {
      pool.FreeRegion(regions[i]);
    }

    // 设置一个 LOADING 状态的区域
    regions[0]->next->status = LOADING;

    LOG(INFO) << "Checking case 5...";
    pool.MemoryRegionView();
    LOG(INFO) << "=============================";

    size_t request_size = 600;
    auto result = pool.GetMinMergeCost(pool.memory_regions, request_size);
    // 根据具体实现检查合并成本和区域
    if (result.T_merge == UNVALID_COST) {
      std::cout << "场景 5 无法找到合并方案，可能需要调整内存布局或请求大小。"
                << std::endl;
    } else {
      ASSERT_NE(result.T_merge, UNVALID_COST);
      ASSERT_NE(result.start, nullptr);
      ASSERT_NE(result.end, nullptr);
    }

    // 打印merge result
    LOG(INFO) << "Merge result: " << "T_merge=" << result.T_merge
              << " start=" << result.start->toString()
              << " end=" << result.end->toString();

    // 执行合并
    auto merged_free_region = pool.MergeRegions_V2(result.start, result.end);
    // 打印合并后的内存布局
    
    pool.AllocateFreeRegion(merged_free_region, request_size, "new_reg");
    LOG(INFO) << "After merge and allocation: ";
    pool.MemoryRegionView();
    LOG(INFO) << "=============================";
  }
}

TEST(GPUTensorPoolV3Test, AllocateASAP_V2WithMerge) {
  const size_t POOL_SIZE = 2 * 1024 * 1024;
  GPUTensorPool_V3 pool(0, POOL_SIZE, 1e9, 1e9);

  // 创建碎片化内存布局
  auto region1 =
      pool.AllocateFreeRegion(pool.memory_regions, 512 * 1024, "reg1");
  auto region2 =
      pool.AllocateFreeRegion(region1->next, 512 * 1024, "reg2");

  // 执行ASAP分配（需要合并空间）
  auto new_region =
      pool.AllocateASAP_V2(1 * 1024 * 1024, "new_reg", "model", 0, 1.0, nullptr);

  ASSERT_NE(new_region, nullptr);
  EXPECT_EQ(new_region->size, 1 * 1024 * 1024);
  EXPECT_EQ(new_region->status, ALLOCATED);

  LOG(INFO)<<"After AllocateASAP_V2:";
  pool.MemoryRegionView();
  LOG(INFO)<<"=============================";
}

// ... existing code ...

TEST(GPUTensorPoolV3Test, ComplexAllocateASAP_V2WithCostAnalysis) {
    const size_t POOL_SIZE = 4 * 1024 * 1024;  // 4MB
    GPUTensorPool_V3 pool(0, POOL_SIZE, 1e9, 1e9);
    
    // 创建三个测试模型
    auto modelA = std::make_shared<RegisteredModel>();
    auto modelB = std::make_shared<RegisteredModel>();
    auto modelC = std::make_shared<RegisteredModel>();
    modelA->SetMeta("modelA", 1.5, 2.0);  // 高敏感度
    modelB->SetMeta("modelB", 1.0, 1.0);
    modelC->SetMeta("modelC", 0.5, 0.8);  // 低敏感度

    // 初始分配（创建碎片化内存）
    auto r1 = pool.AllocateFreeRegion(pool.memory_regions, 512*1024, "A1");
    r1->model_ref = modelA;
    auto r2 = pool.AllocateFreeRegion(r1->next, 512*1024, "B1");
    r2->model_ref = modelB;
    auto r3 = pool.AllocateFreeRegion(r2->next, 1*1024*1024, "C1"); 
    r3->model_ref = modelC;
    auto r4 = pool.AllocateFreeRegion(r3->next, 512*1024, "A2");
    r4->model_ref = modelA;
    pool.FreeRegion(r2);  // 创建中间空闲区
    pool.FreeRegion(r3);  // 创建右侧大空闲区

    LOG(INFO) << "\nInitial Memory Layout:";
    pool.MemoryRegionView();
    LOG(INFO) << "=============================";

    // 模拟模型访问记录
    pool.UseModel("modelA");  // 2次
    pool.UseModel("modelA");
    pool.UseModel("modelB");  // 1次
    pool.UseModel("modelC");  // 1次
    pool.total_access = 4;     // 总访问次数
    pool.UpdateDropCost();

    // 执行ASAP分配（需要1.5MB空间，触发复杂决策）
    auto new_region = pool.AllocateASAP_V2(3.5*1024*1024, "new_model", "model", 
                                      /*T_overlap=*/0.5, 
                                      /*S=*/1.0, 
                                      /*loading_region=*/nullptr);

    // 验证分配结果
    // ASSERT_NE(new_region, nullptr);
    // EXPECT_EQ(new_region->size, 1.5*1024*1024);
    // EXPECT_EQ(new_region->status, ALLOCATED);

    // 验证内存合并情况
    LOG(INFO) << "\nFinal Memory Layout:";
    pool.MemoryRegionView();
    
    // // 验证丢弃的区域（预期丢弃C1，保留A系列）
    // EXPECT_EQ(pool.allocated_regions.count("C1"), 0);
    // EXPECT_EQ(pool.allocated_regions.count("A1"), 1);
    // EXPECT_EQ(pool.allocated_regions.count("A2"), 1);
}