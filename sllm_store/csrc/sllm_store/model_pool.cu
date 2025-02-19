#include <glog/logging.h>

#include "binary_utils.h"
#include "model_pool.h"

GPUTensorPool::GPUTensorPool(int device_id_, size_t max_memory)
    : device_id(device_id_), gpu_memory_size(max_memory) {
  cudaSetDevice(device_id_);
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device_id_);
  // Get GPU UUID
  char uuidStr[80];
  snprintf(
      uuidStr, sizeof(uuidStr),
      "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
      (unsigned char)props.uuid.bytes[0], (unsigned char)props.uuid.bytes[1],
      (unsigned char)props.uuid.bytes[2], (unsigned char)props.uuid.bytes[3],
      (unsigned char)props.uuid.bytes[4], (unsigned char)props.uuid.bytes[5],
      (unsigned char)props.uuid.bytes[6], (unsigned char)props.uuid.bytes[7],
      (unsigned char)props.uuid.bytes[8], (unsigned char)props.uuid.bytes[9],
      (unsigned char)props.uuid.bytes[10], (unsigned char)props.uuid.bytes[11],
      (unsigned char)props.uuid.bytes[12], (unsigned char)props.uuid.bytes[13],
      (unsigned char)props.uuid.bytes[14], (unsigned char)props.uuid.bytes[15]);
  uuid = std::string(uuidStr);
  cudaError_t err = cudaStreamCreate(&stream_);
  if (err != cudaSuccess) {
    LOG(FATAL) << "cudaStreamCreate error: " << cudaGetErrorString(err);
  }

  // allocate memory for the GPU tensor pool
  cudaMalloc(&gpu_memory, gpu_memory_size);

  // create the first GPU Memory Region
  memory_view = std::make_shared<std::list<GPUMemoryRegion>>();
  memory_view->push_back(GPUMemoryRegion(gpu_memory, gpu_memory_size));
  in_gpu_tensor_groups =
      std::make_shared<LRUCache<std::string, std::shared_ptr<GPUMemoryRegion>>>(
          MAX_IN_GPU_TENSOR_GROUP);
  LOG(INFO) << "Crate GPUTensorPool for device " << device_id_ << " with "
            << gpu_memory_size << " bytes";
}

GPUTensorPool::~GPUTensorPool() {
  cudaSetDevice(device_id);
  cudaFree(gpu_memory);
  cudaStreamDestroy(stream_);
  LOG(INFO) << "Destroy GPUTensorPool for device " << device_id;
}

ModelPool::ModelPool(size_t cpu_memoery_size, int num_threads)
    : cpu_model_pool_size_(cpu_memoery_size), num_threads_(num_threads) {
  cpu_model_pool_allocated_ = 0;
  in_cpu_models =
      std::make_shared<LRUCache<std::string, std::shared_ptr<RegisteredModel>>>(
          MAX_IN_CPU_MODEL);

  // get the number of GPUs in the system
  int num_gpus;
  cudaGetDeviceCount(&num_gpus);
  gpu_tensor_pools_.resize(num_gpus);
  for (int i = 0; i < num_gpus; i++) {
    // get the total memory size of the GPU
    size_t total_memory;
    cudaMemGetInfo(NULL, &total_memory);
    gpu_tensor_pools_[i] = std::make_shared<GPUTensorPool>(
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


std::string ModelPool::LoadModelFromDiskAsync(const std::string& model_path) {
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
                << " cpu_model_pool_allocated_ " << cpu_model_pool_allocated_
                << " cpu_model_pool_size_ " << cpu_model_pool_size_
                << " model_size "
                << registered_models_[model_path]->model_size();

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
      LOG(INFO) << "LoadModelFromDiskAsync: " << model_path
                << " cpu_model_pool_allocated_ " << cpu_model_pool_allocated_
                << " cpu_model_pool_size_ " << cpu_model_pool_size_;
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

  LOG(INFO) << "gpu tensor pool size: " << gpu_tensor_pools_.size();
  auto gpu_lru_cache = gpu_tensor_pools_[0]->GetInGPUTensor();
  auto gpu_memory_view = gpu_tensor_pools_[0]->GetMemoryViewList();
  void* gpu_base_addr = gpu_tensor_pools_[0]->GetMemory();

  const std::vector<TensorGroupIndex>& tg_index =
      registered_models_[model_path]->GetTensorGroupIndexes();
  std::vector<char*> allocated_region(tg_index.size(), nullptr);
  int tensor_groups_need_to_load = 0;
  // for (int i = 0; i < tg_index.size(); i++) {
  //   if(gpu_lru_cache->contains(tg_index[i].fingerprint)){
  //     allocated_region[i] =
  //     gpu_lru_cache->get(tg_index[i].fingerprint)->addr;
  //   }
  //   if (allocated_region[i] == nullptr) {
  //     tensor_groups_need_to_load++;
  //   }
  // }

  LOG(INFO) << "Tensor Groups need to load: " << tensor_groups_need_to_load
            << "/" << tg_index.size();

  //   3. allocate the space of tensor groups in GPU Tensor Pool
  //   TODO: better allocation algorithm
  size_t offset = 0;
  for (int i = 0; i < allocated_region.size(); i++) {
    if (allocated_region[i] == nullptr) {
      char* addr = (char*)gpu_base_addr + offset;
      allocated_region[i] = addr;
      offset += tg_index[i].size;

      // iterate the memory view until find the first large enough region
      // for (auto it= gpu_memory_view->begin(); it != gpu_memory_view->end();
      // it++) {
      //   if(it->is_allocated == false && it->size >= need_size) {
      //       // allocate the region
      //       auto alloc_region=std::make_shared<GPUMemoryRegion>(it->addr,
      //       need_size); alloc_region->is_allocated = true;
      //       alloc_region->fingerprint= tg_index[i].fingerprint;
      //       allocated_region[i] = alloc_region;

      //       // erase the memory region
      //       size_t original_region_size= it->size;
      //       it= gpu_memory_view->erase(it);
      //       if(original_region_size > need_size) {
      //           // if need, create and add new region
      //           auto
      //           new_region=std::make_shared<GPUMemoryRegion>(alloc_region->addr+need_size,
      //           original_region_size-need_size); gpu_memory_view->insert(it,
      //           new_region);
      //       }
      //       break;
      //   }
      // }

      //   if can't find a large enough region, remove the lease recently used
      //   tensor group and
      // if(allocated_region[i] == nullptr) {

      // }
    }
  }

  // copy data from CPU to GPU
  registered_models_[model_path]->LodaModelFromMem(
      allocated_region, gpu_tensor_pools_[0]->GetDeviceId());

  // 4. create response
  // each 8bytes represents: cudaIPCMemHandle, device_id, offset for each tensor
  std::string ret;
  cudaIpcMemHandle_t handle;
  cudaSetDevice(gpu_tensor_pools_[0]->GetDeviceId());
  cudaIpcGetMemHandle(&handle, gpu_base_addr);
  // ret=std::string(reinterpret_cast<const char*>(&handle),
  // sizeof(cudaIpcMemHandle_t));
  std::string handle_str = std::string(reinterpret_cast<const char*>(&handle),
                                       sizeof(cudaIpcMemHandle_t));
  ret = toHex(std::vector<uint8_t>(handle_str.begin(), handle_str.end()));

  std::vector<size_t> response;
  response.push_back(gpu_tensor_pools_[0]->GetDeviceId());
  for (int i = 0; i < allocated_region.size(); i++) {
    size_t tensor_group_base_offset =
        allocated_region[i] - static_cast<char*>(gpu_base_addr);
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


  return ret;
}