#include "binary_utils.h"
#include "cuda_runtime.h"
#include "logger.h"
#include "model_pool.h"

NativeModelPool::NativeModelPool(size_t cpu_memoery_size, int num_threads)
    : cpu_model_pool_size_(cpu_memoery_size), num_threads_(num_threads) {
  LOG(INFO) << "Create ModelPool with "
            << cpu_memoery_size / 1024.0 / 1024.0 / 1024.0 << " GB";
  cpu_model_pool_allocated_ = 0;
  in_cpu_models =
      std::make_shared<LRUCache<std::string, std::shared_ptr<RegisteredModel>>>(
          MAX_IN_CPU_MODEL);
  int device_id_ = 0;
  cudaSetDevice(device_id_);
  cudaError_t err = cudaStreamCreate(&stream_);
  if (err != cudaSuccess) {
    LOG(ERROR) << "cudaStreamCreate error: " << cudaGetErrorString(err);
  }
}

NativeModelPool::~NativeModelPool() {
  LOG(INFO) << "Destroy ModelPool";
  cudaStreamSynchronize(stream_);
  cudaStreamDestroy(stream_);
}

int64_t NativeModelPool::RegisterModel(const std::string& model_path) {
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

std::string NativeModelPool::LoadModelAsync(const std::string& model_path) {
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
                << " cpu_model_pool_allocated_ "
                << cpu_model_pool_allocated_ / 1024.0 / 1024.0 / 1024.0
                << " cpu_model_pool_size_ "
                << cpu_model_pool_size_ / 1024.0 / 1024.0 / 1024.0
                << " model_size "
                << registered_models_[model_path]->model_size() / 1024.0 /
                       1024.0 / 1024.0;

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

  const std::vector<TensorGroupIndex>& tg_index =
      registered_models_[model_path]->GetTensorGroupIndexes();
  std::vector<char*> allocated_region(tg_index.size(), nullptr);
  std::vector<int> region_to_load;
  int device_id = 0;
  char* gpu_base_addr;
  cudaSetDevice(device_id);
  cudaError_t cuda_err =
      cudaMalloc(&gpu_base_addr, registered_models_[model_path]->model_size());
  if (cuda_err != cudaSuccess) {
    LOG(ERROR) << "cudaMalloc failed: " << cudaGetErrorString(cuda_err);
    return "cudaMalloc failed";
  }
  for (int i = 0; i < tg_index.size(); i++) {
    allocated_region[i] = gpu_base_addr + tg_index[i].file_offset;
    region_to_load.push_back(i);
  }

  async_tasks_.emplace(std::async(
      std::launch::async,
      [this, model_path, region_to_load, allocated_region, device_id]() {
        return registered_models_[model_path]->LoadModelFromMem(
            allocated_region, region_to_load, device_id);
      }));

  std::string ret;
  std::vector<size_t> response;
  response.push_back(device_id);
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

  // make sure all async tasks are done
  while (!async_tasks_.empty()) {
    std::future<int>& task = async_tasks_.front();
    task.wait();
    async_tasks_.pop();
  }

  cudaError_t err = cudaFree(gpu_base_addr);
  if (err != cudaSuccess) {
    std::cerr << "CUDA error in cudaFree: " << cudaGetErrorString(err)
              << std::endl;
  }

  return ret;
}