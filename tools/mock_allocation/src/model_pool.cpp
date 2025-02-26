
#include "model_pool.h"

#include "binary_utils.h"
#include "logger.h"

ModelPool::ModelPool(size_t cpu_memoery_size, int num_threads)
    : cpu_model_pool_size_(cpu_memoery_size), num_threads_(num_threads) {
  LOG(INFO) << "Create ModelPool with " << cpu_memoery_size/1024.0/1024.0/1024.0 << " GB";
  cpu_model_pool_allocated_ = 0;
  in_cpu_models =
      std::make_shared<LRUCache<std::string, std::shared_ptr<RegisteredModel>>>(
          MAX_IN_CPU_MODEL);

  // get the number of GPUs in the system
  //   int num_gpus;
  //   cudaGetDeviceCount(&num_gpus);
  int num_gpus = 1;
  gpu_tensor_pools_.resize(num_gpus);
  for (int i = 0; i < num_gpus; i++) {
    // get the total memory size of the GPU
    cudaSetDevice(i);    
    size_t total_memory;
    cudaMemGetInfo(NULL, &total_memory);
    
    gpu_tensor_pools_[i] = std::make_shared<GPUTensorPool_V3>(
        i, total_memory * PINNED_GPU_TENSOR_POOL_RATIO / 100);
  }
}

ModelPool::~ModelPool() { LOG(INFO) << "Destroy ModelPool"; }

int64_t ModelPool::RegisterModel(const std::string& model_path) {
  std::unique_lock<std::mutex> lock_info(mutex_);
  auto rmodel = registered_models_.find(model_path);
  if (rmodel != registered_models_.end()) {
    LOG(WARNING) << "Model " << model_path << " already registered";
    return rmodel->second->model_size();
  }

  auto model = std::make_shared<RegisteredModel>(model_path);
  registered_models_[model_path] = model;
  return model->model_size();
}

std::string ModelPool::LoadModelFromAsync(const std::string& model_path) {
  std::unique_lock<std::mutex> lock_info(mutex_);

  if (registered_models_.find(model_path) == registered_models_.end()) {
    LOG(ERROR) << "Model " << model_path << " not registered";
    return "ERROR";
  }

  LOG(INFO) << "load registered model " << model_path;

  //   TODO: first check tensor groups in GPU Tensor Pool, so only need to load
  //   the missing tensor groups from disk. To do this, the CPU Model Pool
  //   should also be managed in Tensor Group level.
  //   1. if model not in CPU model pool, load it from disk asyncronously
  if (!in_cpu_models->contains(model_path)) {
    LOG(INFO) << "model not in cpu model pool";
    async_tasks_.emplace(std::async(std::launch::async, [this, model_path]() {
      // check if cpu model pool free memory is enough
      LOG(INFO) << "LoadModelFromDiskAsync: " << model_path
                << " cpu_model_pool_allocated_ " << cpu_model_pool_allocated_/1024.0/1024.0/1024.0
                << " cpu_model_pool_size_ " << cpu_model_pool_size_/1024.0/1024.0/1024.0
                << " model_size "
                << registered_models_[model_path]->model_size()/1024.0/1024.0/1024.0;

      if (cpu_model_pool_size_ < registered_models_[model_path]->model_size()) {
        LOG(ERROR) << "cpu_model_pool_size_ < "
                      "registed_models_[model_path]->model_size()";
        return -1;
      }
      if (cpu_model_pool_size_ - cpu_model_pool_allocated_ <
          registered_models_[model_path]->model_size()) {
        // iterate through the LRU cache and evict the least recently used
        // models
        while (cpu_model_pool_size_ - cpu_model_pool_allocated_ <
               registered_models_[model_path]->model_size()) {
          if (in_cpu_models->getLRUKeySize() > 0) {
            auto lru_model = in_cpu_models->getandRemoveLRUKey();
            //   TODO: might check whether the model is still in use
            int freed = registered_models_[lru_model]->UnloadModel();
            if (freed <= 0) {
              LOG(WARNING) << "unloaded model " << lru_model << " failed";
              return -1;
            } else {
              cpu_model_pool_allocated_ -= freed;
              LOG(INFO) << "unloaded model " << lru_model << " success";
            }
          } else {
            LOG(ERROR) << "no more models in cpu_model_pool";
            return -1;
          }
        }
      }

      if (registered_models_[model_path]->LoadModelFromDisk(num_threads_) !=
          0) {
        LOG(ERROR) << "LoadModelFromDiskAsync: " << model_path << " failed";
        return -1;
      }
      in_cpu_models->put(model_path, registered_models_[model_path]);
      cpu_model_pool_allocated_ += registered_models_[model_path]->model_size();
      // LOG(INFO) << "LoadModelFromDiskAsync: " << model_path
      //           << " cpu_model_pool_allocated_ " << cpu_model_pool_allocated_
      //           << " cpu_model_pool_size_ " << cpu_model_pool_size_;
      return 0;
    }));
  }
  // 2. check tensor groups in GPU Tensor Pool
  // TODO: if multiple GPU Tensor Pool, need to check all of them and pick the
  // best one currently, only check the first GPU Tensor Pool
  if (gpu_tensor_pools_.size() < 1) {
    LOG(ERROR) << "No GPU Available";
    return "ERROR";
  }
  auto gpu_tensor_pool = gpu_tensor_pools_[0];
  gpu_tensor_pool->UseModel(registered_models_[model_path]);

  const std::vector<TensorGroupIndex>& tg_index =
      registered_models_[model_path]->GetTensorGroupIndexes();
  std::vector<char*> allocated_region(tg_index.size(), nullptr);
  std::vector<int> region_to_load;
  int tensor_groups_need_to_load = tg_index.size();

  for (int i = 0; i < tg_index.size(); i++) {
    auto mem_region = gpu_tensor_pool->GetTensor(tg_index[i].fingerprint);
    if (mem_region && mem_region->is_allocated) {
      // already allocated
      allocated_region[i] = mem_region->addr;
      tensor_groups_need_to_load--;
    } else {
      // allocate new memory region
      mem_region = gpu_tensor_pool->BestFitAllocation(tg_index[i].size,
                                                      tg_index[i].fingerprint);
      if (!mem_region) {
        LOG(ERROR) << "No enough memory in GPU Tensor Pool";
        return "ERROR";
      }
      allocated_region[i] = mem_region->addr;
      region_to_load.push_back(i);
    }
  }

  LOG(INFO) << "Tensor Groups need to load: " << tensor_groups_need_to_load
            << "/" << tg_index.size();

  // 3. copy data from CPU to GPU async
  async_tasks_.emplace(std::async(
      std::launch::async,
      [this, model_path, region_to_load, allocated_region, gpu_tensor_pool]() {
        return registered_models_[model_path]->LoadModelFromMem(
            allocated_region, region_to_load, gpu_tensor_pool->GetDeviceId());
      }));

  // 4. create response
  // each 8bytes represents: cudaIPCMemHandle, device_id, offset for each tensor
  std::string ret;
  //   cudaIpcMemHandle_t handle;
  //   cudaSetDevice(gpu_tensor_pools_[0]->GetDeviceId());
  //   cudaIpcGetMemHandle(&handle, gpu_base_addr);
  //   // ret=std::string(reinterpret_cast<const char*>(&handle),
  //   // sizeof(cudaIpcMemHandle_t));
  //   std::string handle_str = std::string(reinterpret_cast<const
  //   char*>(&handle),
  //                                        sizeof(cudaIpcMemHandle_t));
  //   ret = toHex(std::vector<uint8_t>(handle_str.begin(), handle_str.end()));

  std::vector<size_t> response;
  response.push_back(gpu_tensor_pools_[0]->GetDeviceId());
  for (int i = 0; i < allocated_region.size(); i++) {
    size_t tensor_group_base_offset =
        allocated_region[i] -
        static_cast<char*>(gpu_tensor_pool->GetBaseAddr());
    for (auto it = tg_index[i].tensor_indexes.begin();
         it != tg_index[i].tensor_indexes.end(); it++) {
      response.push_back(tensor_group_base_offset + it->offset);
    }
  }
  // ret+=std::string(reinterpret_cast<const char*>(response.data()),
  // response.size()*sizeof(size_t));
  std::string response_str(reinterpret_cast<const char*>(response.data()),
                           response.size() * sizeof(size_t));
  ret += toHex(std::vector<uint8_t>(response_str.begin(), response_str.end()));
  // LOG(INFO)<<"ModelPool::LoadModelFromMem: load model from memory, ret:
  // "<<ret;

  // make sure all async tasks are done
  while (!async_tasks_.empty()) {
    std::future<int>& task = async_tasks_.front();  
    task.wait();                                   
    async_tasks_.pop();
  }

  return ret;
}