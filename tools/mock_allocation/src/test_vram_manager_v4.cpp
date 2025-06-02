#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <string>

#include "registered_model.h"
#include "vram_manager_v4.h"

#include "binary_utils.h"

// TEST(GPUTensorPoolV4Test, BasicAllocation) {
//   const size_t POOL_SIZE = 1024 * 1024;           // 1MB
//   GPUTensorPool_V4 pool(0, POOL_SIZE, 1e9, 1e9);  // 高带宽设置

//   // 测试空闲区域分配
//   auto region =
//       pool.AllocateFreeRegion(pool.memory_regions, 512 * 1024, "reg1");
//   ASSERT_NE(region, nullptr);
//   EXPECT_EQ(region->status, ALLOCATED);
//   EXPECT_EQ(region->size, 512 * 1024);

//   // 验证链表结构
//   EXPECT_NE(region->next, nullptr);
//   EXPECT_EQ(region->next->status, FREE);
//   EXPECT_EQ(region->next->size, POOL_SIZE - 512 * 1024);

//   pool.MemoryRegionView();
// }


// // ... existing code ...

// TEST(GPUTensorPoolV4Test, MergeRegions_V2ComplexScenario) {
//     const size_t POOL_SIZE = 1024 * 1024 * 5; // 4MB
//     GPUTensorPool_V4 pool(0, POOL_SIZE, 1e9, 1e9);
    
//     // 创建碎片化内存布局：A-F-F-F-A-F-F-A
//     auto rA = pool.AllocateFreeRegion(pool.memory_regions, 512 * 1024, "regA");
//     auto rB = pool.AllocateFreeRegion(rA->next, 512 * 1024, "regB");
//     auto rC = pool.AllocateFreeRegion(rB->next, 512 * 1024, "regC");
//     auto rD = pool.AllocateFreeRegion(rC->next, 512 * 1024, "regD");
//     auto rE = pool.AllocateFreeRegion(rD->next, 512 * 1024, "regE");
//     auto rF = pool.AllocateFreeRegion(rE->next, 512 * 1024, "regF");
//     auto rG = pool.AllocateFreeRegion(rF->next, 512 * 1024, "regG");
//     auto rH = pool.AllocateFreeRegion(rG->next, 512 * 1024, "regH");
//     pool.FreeRegion(rA);
//     pool.FreeRegion(rE);
//     pool.FreeRegion(rG);
//     pool.MemoryRegionView();
//     pool.FreeRegion(rH, 1);
    
//     LOG(INFO)<<"Before Merge";
//     pool.MemoryRegionView();
//     LOG(INFO)<<"=============================";
//     // 执行合并操作
//     auto merged_region = pool.MergeRegions(pool.memory_regions, rH);

//     LOG(INFO)<<"After Merge";
    
//     LOG(INFO)<<"After Merge";
//     pool.MemoryRegionView();
//     LOG(INFO)<<"=============================";

//     LOG(INFO)<<"allocated_regions:";
//     for (auto& region : pool.allocated_regions) {
//       LOG(INFO)<<region.first<<":"<<region.second->toString();
//     }
//     LOG(INFO)<<"=============================";
// }


// TEST(GPUTensorPoolV4Test, GreedyDrop_InsufficientFreeMemory) {
//   const size_t POOL_SIZE = 1024 * 1024; // 1MB
//   GPUTensorPool_V4 pool(0, POOL_SIZE, 1e9, 1e9);

//   auto modelA = std::make_shared<RegisteredModel>();
//   auto modelB = std::make_shared<RegisteredModel>();
//   auto modelC = std::make_shared<RegisteredModel>();
//   modelA->SetMeta("low_cost_model", 0.5, 2.0);  // 高敏感度
//   modelB->SetMeta("medium_cost_model", 1.0, 1.0);
//   modelC->SetMeta("high_cost_model", 1.5, 0.8);  // 低敏感度

//   // 创建两个低优先级和一个高优先级区域（通过访问次数控制成本）
//   auto low_cost_region = pool.AllocateFreeRegion(pool.memory_regions, 200 * 1024, "low_cost");
//   low_cost_region->model_ref = modelA;
//   auto medium_cost_region = pool.AllocateFreeRegion(low_cost_region->next, 200 * 1024, "medium_cost");
//   medium_cost_region->model_ref = modelB;
//   auto high_cost_region = pool.AllocateFreeRegion(medium_cost_region->next, 200 * 1024, "high_cost");
//   high_cost_region->model_ref = modelC;
  
//   // 设置访问次数（影响成本计算）
//   pool.UseModel("low_cost_model");  // 低访问次数 -> 低丢弃成本
//   pool.UseModel("low_cost_model");
//   pool.UseModel("medium_cost_model");
//   pool.UseModel("medium_cost_model");
//   pool.UseModel("medium_cost_model");
//   pool.UseModel("high_cost_model");  // 高访问次数 -> 高丢弃成本
//   pool.UseModel("high_cost_model");
//   pool.UseModel("high_cost_model");
//   pool.UseModel("high_cost_model");

//   LOG(INFO)<<"Before GreedyDrop";
//   pool.MemoryRegionView();
//   LOG(INFO)<<"=============================";

  
//   pool.GreedyDrop(700 * 1024);

//   LOG(INFO)<<"After GreedyDrop";
//   pool.MemoryRegionView();
//   LOG(INFO)<<"=============================";
// }

// TEST(GPUTensorPoolV4Test, BipartiteAllocate_PerfectMatch) {
//     const size_t POOL_SIZE = 4 * 1024 * 1024; // 4MB
//     GPUTensorPool_V4 pool(0, POOL_SIZE, 1e9, 1e9);


//     auto reg1=pool.AllocateFreeRegion(pool.memory_regions, 1*1024*1024, "reg1");
//     auto reg2=pool.AllocateFreeRegion(reg1->next, 1*1024*1024, "reg2");
//     auto reg3=pool.AllocateFreeRegion(reg2->next, 1*1024*1024, "reg3");
//     auto reg4=pool.AllocateFreeRegion(reg3->next, 1*1024*1024, "reg4");
    
//     pool.FreeRegion(reg1);
//     pool.FreeRegion(reg3);
//     LOG(INFO)<<"Before BipartiteAllocate";
//     pool.MemoryRegionView();
//     LOG(INFO)<<"=============================";

//     std::vector<size_t> requests = {1*1024*1024, 1*1024*1024};
//     std::vector<std::string> fingerprints = {"fingerprint1", "fingerprint2", "fingerprint3"};
//     while(!requests.empty()){
//       auto match=pool.BipartiteAllocate(requests);
//       for(auto& m:match){
//         size_t request_size=requests[m.first];
//         LOG(INFO)<<"Get match:"<<m.first<<":"<<request_size<<":"<<m.second->size;
//         pool.AllocateFreeRegion(m.second, request_size, fingerprints[m.first]);
//         requests.erase(requests.begin()+m.first);
//         fingerprints.erase(fingerprints.begin()+m.first);
//       }
//     }

//     LOG(INFO)<<"After BipartiteAllocate";
//     pool.MemoryRegionView();
//     LOG(INFO)<<"=============================";
//     LOG(INFO)<<"allocated_regions:";
//     for (auto& region : pool.allocated_regions) {
//       LOG(INFO)<<region.first<<":"<<region.second->toString();
//     }
// }

// TEST(GPUTensorPoolV4Test, BipartiteAllocate_ComplexFragmentation) {
//     const size_t POOL_SIZE = 8 * 1024 * 1024; // 8MB 更大内存池
//     GPUTensorPool_V4 pool(0, POOL_SIZE, 1e9, 1e9);

//     // 创建复杂碎片化布局：A(512KB)-F(1MB)-A(256KB)-F(768KB)-A(1.5MB)-F(384KB)-A(1MB)
//     auto regA = pool.AllocateFreeRegion(pool.memory_regions, 512 * 1024, "regA");       // 已分配
//     auto regB = pool.AllocateFreeRegion(regA->next, 1024 * 1024, "regB");             // 空闲（后续释放）
//     auto regC = pool.AllocateFreeRegion(regB->next, 256 * 1024, "regC");              // 已分配
//     auto regD = pool.AllocateFreeRegion(regC->next, 768 * 1024, "regD");              // 空闲（后续释放）
//     auto regE = pool.AllocateFreeRegion(regD->next, 1536 * 1024, "regE");             // 已分配
//     auto regF = pool.AllocateFreeRegion(regE->next, 384 * 1024, "regF");              // 空闲（后续释放）
//     auto regG = pool.AllocateFreeRegion(regF->next, 1024 * 1024, "regG");             // 已分配

//     // 释放部分区域形成空闲块：F(1MB)-F(768KB)-F(384KB)
//     pool.FreeRegion(regB);
//     pool.FreeRegion(regD);
//     pool.FreeRegion(regF);
    
//     LOG(INFO) << "Before BipartiteAllocate - Fragmented State";
//     pool.MemoryRegionView();
//     LOG(INFO) << "=============================";

//     // 构造混合大小的请求（包含部分匹配场景）
//     std::vector<size_t> requests = {
//         256 * 1024,   // 匹配384KB空闲块（剩余128KB）
//         512 * 1024,   // 匹配1MB空闲块（剩余512KB）
//         768 * 1024,   // 完美匹配768KB空闲块
//         1024 * 1024   // 需要合并1MB空闲块的剩余部分（假设前一步分配后剩余512KB，可能需要其他策略）
//     };
//     std::vector<std::string> fingerprints = {"fp1", "fp2", "fp3", "fp4"};

//     // 循环处理请求直到全部匹配
//     while (!requests.empty()) {
//         auto match = pool.BipartiteAllocate(requests);
//         ASSERT_FALSE(match.empty()) << "Should find matches for remaining requests";
        
//         for (auto& m : match) {
//             size_t request_size = requests[m.first];
//             LOG(INFO) << "Matched request " << m.first 
//                       << ": size=" << request_size 
//                       << " with free region size=" << m.second->size;
            
//             // 执行分配（注意：实际分配可能切割空闲块）
//             auto allocated = pool.AllocateFreeRegion(m.second, request_size, fingerprints[m.first]);
//             ASSERT_NE(allocated, nullptr) << "Allocation should succeed";
            
//             // 验证分配后的剩余空间（如果空闲块大于请求）
//             if (m.second->size > request_size) {
//                 EXPECT_EQ(m.second->next->size, m.second->size - request_size) 
//                      << "Should create remaining free region after allocation";
//             }

//             // 移除已处理的请求和指纹
//             requests.erase(requests.begin() + m.first);
//             fingerprints.erase(fingerprints.begin() + m.first);
//         }
//     }

//     LOG(INFO) << "After BipartiteAllocate - Final State";
//     pool.MemoryRegionView();
//     LOG(INFO) << "=============================";
    
// }

// ... existing code ...

// TEST(GPUTensorPoolV4Test, GreedyMerge_ComplexFragmentation) {
//     const size_t POOL_SIZE = 4 * 1024 * 1024; // 8MB 内存池
//     GPUTensorPool_V4 pool(0, POOL_SIZE, 1e9, 1e9);

//     // 构造初始碎片化布局：A(256KB)-F(512KB)-F(512KB)-A(256KB)-F(1MB)-F(1MB)-A(512KB)
//     auto regA = pool.AllocateFreeRegion(pool.memory_regions, 256 * 1024, "regA");  // 已分配
//     auto regB = pool.AllocateFreeRegion(regA->next, 512 * 1024, "regB");          // 空闲（后续释放）
//     auto regC = pool.AllocateFreeRegion(regB->next, 512 * 1024, "regC");          // 空闲（后续释放）
//     auto regD = pool.AllocateFreeRegion(regC->next, 256 * 1024, "regD");          // 已分配
//     auto regE = pool.AllocateFreeRegion(regD->next, 1024 * 1024, "regE");         // 空闲（后续释放）
//     auto regF = pool.AllocateFreeRegion(regE->next, 1024 * 1024, "regF");         // 空闲（后续释放）
//     auto regG = pool.AllocateFreeRegion(regF->next, 512 * 1024, "regG");          // 已分配

//     // 释放空闲块：形成两段连续空闲区（B+C=1MB，E+F=2MB）和中间的已分配块D
//     pool.FreeRegion(regA);
//     pool.FreeRegion(regC);
//     pool.FreeRegion(regG);
    
//     LOG(INFO) << "Before GreedyMerge - Initial Fragmented State";
//     pool.MemoryRegionView();
//     LOG(INFO) << "=============================";

//     // 执行贪心合并（假设GreedyMerge会遍历所有空闲块并合并相邻的）
//     pool.GreedyMerge(1.25*1024*1024);

//     LOG(INFO) << "After GreedyMerge - Merged State";
//     pool.MemoryRegionView();
//     LOG(INFO) << "=============================";
// }

TEST(KMTest, RectangularMatrix) {
  // 3任务 vs 2资源
  std::vector<std::vector<BipartEdge>> adj = {
      {{0, 3}, {1, 1}},
      {{0, 10}, {1, 4}},
      {{0, 1}, {1, 2}},
  };
  auto result = MaxWeightBMatchingWithBoost(adj);
  // 输出result
  for (const auto& [u, v] : result) {
      LOG(INFO) << "(" << u << ", " << v << ")";
  }
}

// TEST(KMTest, ComplexGraph){
//   const int left_nodes=40;
//   const int right_nodes=40;
  
  
//   std::vector<std::vector<BipartEdge>> adj(left_nodes);
//   std::random_device rd;
//   std::mt19937 gen(rd());
//   std::uniform_real_distribution<> dis_weight(1.0, 100.0);
//   std::uniform_real_distribution<> dis_node(0, right_nodes);
  
//   for (size_t i = 0; i < left_nodes; ++i) {
//     // 随机生成10条边, 右节点随机
//     for (size_t j = 0; j < right_nodes; ++j) {
//       adj[i].emplace_back(BipartEdge{static_cast<size_t>(dis_node(gen)), dis_weight(gen)});
//     }
//   }
//   auto result = MaxWeightBMatchingWithBoost(adj);
//   // 输出result
//   for (const auto& [u, v] : result) {
//       LOG(INFO) << "(" << u << ", " << v << ")";
//   }
  
// }

TEST(GPUTensorPoolV4Test, GreedyMerge_ComplexFragmentation2) {
  const size_t POOL_SIZE = 20LL * 1024 * 1024 * 1024;
  const int num_regions = 40;
  const int num_requests = 500;
  const int device_id = 4;
  // 最小大小为100MB
  const size_t min_size = 100LL * 1024 * 1024;
  // 最大大小为1000MB
  const size_t max_size = 1000LL * 1024 * 1024;

  const int test_rounds = 10;
  std::vector<size_t> move_cost_p1;
  std::vector<size_t> move_cost_p2;
  std::vector<size_t> move_cost_p3;
  std::vector<size_t> move_cost_p4;
  std::vector<size_t> optimal_move_cost;

  size_t no_need_merge=0;
  for (int i = 0; i < test_rounds; i++) {
    GPUTensorPool_V4 pool0(device_id, POOL_SIZE, 1e9, 1e9);
    size_t free = pool0.MockGenerateRegions(num_regions, min_size, max_size, 1);
    auto request = pool0.MockGenerateRequests(free, num_requests, min_size, max_size, 1);
    LOG(INFO)<<"request size: "<<request.size();
    optimal_move_cost.push_back(pool0.GetMergeCost());

    LOG(INFO) << "Direct Greedy Merge";
    // 创建pool1, 拷贝pool0的内存布局, 深度拷贝
    std::shared_ptr<GPUTensorPool_V4> pool1(
        new GPUTensorPool_V4(device_id, POOL_SIZE, 1e9, 1e9));
    pool1->CopyMemoryLayout(pool0);
    pool1->MockGreedyMergeAllocate(request, 0);
    move_cost_p1.push_back(pool1->GetTotalMove());
    pool1.reset();
    LOG(INFO) << "===================================";

    LOG(INFO) << "Bipartite Matching then Merge";
    std::shared_ptr<GPUTensorPool_V4> pool2(
        new GPUTensorPool_V4(device_id, POOL_SIZE, 1e9, 1e9));
    pool2->CopyMemoryLayout(pool0);
    auto remaining = pool2->MockBartiteMatching(request, 0);
    // LOG(INFO) << "request size: " << request.size()
    //           << ", remaining size: " << remaining.size();
    // if(remaining.size() == 0){
    //   LOG(INFO) << "#### No remaining requests after bipartite matching";
    // }
    pool2->MockGreedyMergeAllocate(remaining, 0);
    move_cost_p2.push_back(pool2->GetTotalMove());
    pool2.reset();
    LOG(INFO) << "===================================";

    LOG(INFO) << "Weighted Bipartite Matching then Merge";
    std::shared_ptr<GPUTensorPool_V4> pool3(
        new GPUTensorPool_V4(device_id, POOL_SIZE, 1e9, 1e9));
    pool3->CopyMemoryLayout(pool0);
    auto remaining3 = pool3->MockBartiteMatching(request, 1);
    // LOG(INFO) << "request size: " << request.size()
              // << ", remaining size: " << remaining3.size();
    if(remaining3.size() == 0){
      // LOG(INFO) << "#### No remaining requests after weighted bipartite matching";
      no_need_merge++;
    }
    pool3->MockGreedyMergeAllocate(remaining3, 0);
    
    move_cost_p3.push_back(pool3->GetTotalMove());
    pool3.reset();
    LOG(INFO) << "===================================";


    LOG(INFO)<<"Recursive Split";
    std::shared_ptr<GPUTensorPool_V4> pool4(
        new GPUTensorPool_V4(device_id, POOL_SIZE, 1e9, 1e9));
    pool4->CopyMemoryLayout(pool0);
    pool4->MockRecursiveSplitAllocate(request);
    move_cost_p4.push_back(pool4->GetTotalMove());

    LOG(INFO)<<"===================================";

  }

  // 输出平均move cost
  size_t move_cost_p1_total=0;
  size_t move_cost_p2_total=0;
  size_t move_cost_p3_total=0;
  size_t move_cost_p4_total=0;
  size_t optimal_move_cost_total=0;
  for (size_t i = 0; i < move_cost_p1.size(); i++) {
    move_cost_p1_total += move_cost_p1[i];
    move_cost_p2_total += move_cost_p2[i];
    move_cost_p3_total += move_cost_p3[i];
    move_cost_p4_total += move_cost_p4[i];
    optimal_move_cost_total += optimal_move_cost[i];
  }


  LOG(INFO) << "Direct Greedy Merge Average Move Cost: "
            << (double)move_cost_p1_total / (move_cost_p1.size()) / (1024.0*1024.0*1024.0);
  LOG(INFO) << "Bipartite Matching then Merge Average Move Cost: "
            << (double)move_cost_p2_total / (move_cost_p2.size())/ (1024.0*1024.0*1024.0);
  LOG(INFO) << "Weighted Bipartite Matching then Merge Average Move Cost: "
            << (double)move_cost_p3_total / (move_cost_p3.size())/ (1024.0*1024.0*1024.0);
  LOG(INFO) << "Optimal Move Cost: "
            << (double)optimal_move_cost_total / (optimal_move_cost.size())/ (1024.0*1024.0*1024.0);
  LOG(INFO) << "Weighted Improves over Direct Greedy Merge: "
            << (double)move_cost_p3_total/move_cost_p1_total;
  LOG(INFO) << "Weighted Improves over Optimal Move Cost: "
            << (double)move_cost_p3_total/optimal_move_cost_total;
  LOG(INFO)<<"Weighted No Need To Merge Ratio: "<<(double)no_need_merge/move_cost_p3.size();


  LOG(INFO) << "Recursive Split Average Move Cost: "
            << (double)move_cost_p4_total / (move_cost_p4.size())/ (1024.0*1024.0*1024.0);
}
