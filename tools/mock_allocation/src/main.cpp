#include <chrono>

#include "model_pool.h"
void TestGPUTensorPool() {
  GPUTensorPool_V2 pool(0, 1024 * 1024 * 1024);

  std::shared_ptr<GPUMemoryRegion> region1 =
      pool.BestFitAllocation(256 * 1024 * 1024, "Tensor_1");
  if (region1) {
    std::cout << "Allocated Tensor_1 at " << static_cast<void*>(region1->addr)
              << "\n";
  }

  std::shared_ptr<GPUMemoryRegion> region2 =
      pool.BestFitAllocation(512 * 1024 * 1024, "Tensor_2");
  if (region2) {
    std::cout << "Allocated Tensor_2 at " << static_cast<void*>(region2->addr)
              << "\n";
  }

  std::shared_ptr<GPUMemoryRegion> retrieved_region =
      pool.GetTensor("Tensor_1");
  if (retrieved_region) {
    std::cout << "Retrieved Tensor_1 from cache at "
              << static_cast<void*>(retrieved_region->addr) << "\n";
  } else {
    std::cout << "Tensor_1 not found\n";
  }

  retrieved_region = pool.GetTensor("Tensor_3");
  if (retrieved_region) {
    std::cout << "Retrieved Tensor_3 from cache at "
              << static_cast<void*>(retrieved_region->addr) << "\n";
  } else {
    std::cout << "Tensor_3 not found\n";
  }

  std::shared_ptr<GPUMemoryRegion> region3 =
      pool.BestFitAllocation(128 * 1024 * 1024, "Tensor_3");
  if (region3) {
    std::cout << "Allocated Tensor_3 at " << static_cast<void*>(region3->addr)
              << "\n";
  }

  retrieved_region = pool.GetTensor("Tensor_2");
  if (retrieved_region) {
    std::cout << "Retrieved Tensor_2 from cache at "
              << static_cast<void*>(retrieved_region->addr) << "\n";
  } else {
    std::cout << "Tensor_2 not found\n";
  }
  pool.Details();
  std::shared_ptr<GPUMemoryRegion> region4 =
      pool.BestFitAllocation(700 * 1024 * 1024, "Tensor_4");
  if (region4) {
    std::cout << "Allocated Tensor_4 at " << static_cast<void*>(region4->addr)
              << "\n";
  }
  pool.Details();
  pool.MemoryUsage();
}

void TestGpuTensorPoolV2() {
  GPUTensorPool_V2 pool(0, 30LL * 1024 * 1024 * 1024);
  std::string storage_path = "/mnt/n0/models";
  std::vector<std::string> model_paths = {"vllm/opt6.7b_tmp/rank_0",
                                          "vllm/opt1.3b_tmp/rank_0"};

  std::vector<size_t> tensor_sizes;
  std::vector<std::string> tensor_names;
  for (auto& model_path : model_paths) {
    std::string index_file =
        storage_path + "/" + model_path + "/tensor_group_index.txt";
    std::vector<TensorGroupIndex> tensor_group_index;
    ParseTensorGroupIndex(index_file, tensor_group_index);
    for (auto& tensor_group : tensor_group_index) {
      tensor_sizes.push_back(tensor_group.size);
      tensor_names.push_back(tensor_group.fingerprint);
    }
  }

  size_t Scales = 1000;
  std::chrono::nanoseconds total_best_fit_time(0);
  std::chrono::nanoseconds total_get_tensor_time(0);

  for (int i = 0; i < Scales; i++) {
    for (int j = 0; j < tensor_sizes.size(); j++) {
      auto start_best_fit = std::chrono::high_resolution_clock::now();
      pool.BestFitAllocation(tensor_sizes[j], tensor_names[j]);
      auto end_best_fit = std::chrono::high_resolution_clock::now();
      total_best_fit_time += end_best_fit - start_best_fit;

      auto start_get_tensor = std::chrono::high_resolution_clock::now();
      auto region = pool.GetTensor(tensor_names[j]);
      auto end_get_tensor = std::chrono::high_resolution_clock::now();
      total_get_tensor_time += end_get_tensor - start_get_tensor;
      if (!region) {
        std::cout << "Tensor_2 not found\n";
        return;
      } else if (!region->is_allocated ||
                 (region->fingerprint != tensor_names[j])) {
        std::cout << "Retrieved Tensor_2 error " << "\n";
        return;
      }
    }
  }

  std::chrono::nanoseconds average_best_fit_time =
      total_best_fit_time / tensor_sizes.size() / Scales;
  std::chrono::nanoseconds average_get_tensor_time =
      total_get_tensor_time / tensor_sizes.size() / Scales;

  std::cout << "Average BestFitAllocation time: "
            << average_best_fit_time.count() << " ns\n";
  std::cout << "Average GetTensor time: " << average_get_tensor_time.count()
            << " ns\n";
  // pool.MemoryRegionView();
}

int main() {
  //   TestGPUTensorPool();
  //   TestGpuTensorPoolV2();
  //   return 0;
  // micro benchmark passed: Average BestFitAllocation time: 1962 ns, Average
  // GetTensor time: 293 ns

  size_t memory_pool_size = 30LL * 1024 * 1024 * 1024;  // 12GB
  int num_thread = 4;
  std::string storage_path = "/mnt/n0/models";
  // std::vector<std::string> model_paths = {"vllm/opt6.7b_tmp/rank_0"};
  std::vector<std::string> model_paths = {"vllm/opt6.7b_tmp/rank_0",
                                          "vllm/opt1.3b_tmp/rank_0"};

  std::shared_ptr<ModelPool> model_pool_ =
      std::make_shared<ModelPool>(memory_pool_size, num_thread);

  for (auto& model_path : model_paths) {
    model_pool_->RegisterModel(storage_path + "/" + model_path);
  }

  size_t scale = 10;
  for (int i = 0; i < scale; i++) {
    for (auto& model_path : model_paths) {
      model_pool_->LoadModelFromDiskAsync(storage_path + "/" + model_path);
    }
  }
}