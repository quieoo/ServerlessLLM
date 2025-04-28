#pragma once

#include <cuda_runtime.h>


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
inline cudaError_t cuda_safe_move(void* free_region_base_addr, void* data_addr,
                                  size_t data_size) {
  char* dest = static_cast<char*>(free_region_base_addr);
  char* src = static_cast<char*>(data_addr);

  // 处理无数据的情况
  if (data_size == 0) return cudaSuccess;

  // 检查指针是否有效
  if (dest == nullptr || src == nullptr) return cudaErrorInvalidValue;

  // 计算地址是否重叠
  const size_t overlap_cond = (dest > src) ? (dest - src) : (src - dest);
  const bool overlap = (overlap_cond < data_size);

  if (!overlap) {
    // 无重叠，直接拷贝
    return cudaMemcpy(dest, src, data_size, cudaMemcpyDeviceToDevice);
  }

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
    }
  }

  return cudaSuccess;
}

/*
GPUMemoryRegion_V2:
    添加一个读写锁，所有修改和读取操作都需要加锁，
*/
enum RegionStatus { Free = 0, Loading = 1, Allocated = 2 };


class GPUMemoryRegion_V2
    : public std::enable_shared_from_this<GPUMemoryRegion_V2> {
 public:
  // bool is_allocated;
  RegionStatus status;
  char* addr;
  size_t size;
  std::string fingerprint;
  std::shared_ptr<GPUMemoryRegion_V2> prev;
  std::shared_ptr<GPUMemoryRegion_V2> next;
  std::string model_path;

  GPUMemoryRegion_V2()
      : status(RegionStatus::Free), size(0), prev(nullptr), next(nullptr) {}

  GPUMemoryRegion_V2(void* addr_, size_t size)
      : status(RegionStatus::Free), size(size), prev(nullptr), next(nullptr) {
    this->addr = static_cast<char*>(addr_);
  }

  void SetAddr(void* addr) { this->addr = static_cast<char*>(addr); }

  bool isAdjacent(std::shared_ptr<GPUMemoryRegion_V2> other) {
    if (addr == nullptr || other->addr == nullptr) {
      return false;
    }
    return (addr + size == other->addr) || (other->addr + other->size == addr);
  }
  std::shared_ptr<GPUMemoryRegion_V2> clone() const {
    auto new_region = std::make_shared<GPUMemoryRegion_V2>();
    new_region->status = this->status;
    new_region->addr = this->addr;
    new_region->size = this->size;
    new_region->fingerprint = this->fingerprint;
    new_region->model_path = this->model_path;
    return new_region;
  }

  void merge(std::shared_ptr<GPUMemoryRegion_V2> other) {
    if (!isAdjacent(other)) return;
    if (status != RegionStatus::Free || other->status != RegionStatus::Free)
      return;

    // 显式断开other节点的链接
    if (other->prev) other->prev->next = other->next;
    if (other->next) other->next->prev = other->prev;

    if (other->addr + other->size == addr) {  // other在前
      addr = other->addr;
      size += other->size;
      prev = other->prev;  // 更新prev指针
      if (prev) prev->next = shared_from_this();
    } else if (addr + size == other->addr) {  // other在后
      size += other->size;
      next = other->next;  // 更新next指针
      if (next) next->prev = shared_from_this();
    }

    // 显式重置other的指针（安全措施）
    other->prev = nullptr;
    other->next = nullptr;
  }

  std::string toString() {
    return "[GPUMemoryRegion_V2: addr=" +
           std::to_string(reinterpret_cast<size_t>(addr)) +
           ", size=" + std::to_string(size) + ", fingerprint=" + fingerprint +
           ", is_allocated=" + std::to_string(status) +
           ", model_path=" + model_path + "]";
  }

  // 通过移动变换一个空闲区域和已分配区域的位置
  void moveSwapAdjacent(std::shared_ptr<GPUMemoryRegion_V2> other) {
    if (!isAdjacent(other)) {
      LOG(ERROR) << "Regions are not adjacent.";
      return;
    }
    if (status != RegionStatus::Free ||
        other->status != RegionStatus::Allocated) {
      LOG(ERROR) << "Invalid region statuses.";
      return;
    }

    // 检查合并后的总大小是否足够容纳数据
    const size_t total_merged_size = size + other->size;
    if (total_merged_size < other->size) {
      LOG(ERROR) << "Merged region is smaller than allocated region.";
      return;
    }
    // 保存原始信息
    std::string original_fingerprint = other->fingerprint;
    size_t original_size = other->size;

    // 执行数据移动
    cudaError_t err = cuda_safe_move(addr, other->addr, other->size);
    if (err != cudaSuccess) {
      LOG(ERROR) << "cuda_safe_move failed: " << cudaGetErrorString(err);
      return;
    }

    // 更新当前区域状态
    status = RegionStatus::Allocated;
    fingerprint = original_fingerprint;
    size = original_size;
    model_path = other->model_path;  // 更新模型路径

    // 更新other区域状态
    other->status = RegionStatus::Free;
    other->addr = addr + size;
    other->size = total_merged_size - size;
    other->fingerprint.clear();
    other->model_path.clear();
    // // 安全拷贝：假设物理内存连续，目标地址空间足够
    // cudaError_t err = cuda_safe_move(addr, other->addr, other->size);
    // if (err != cudaSuccess) {
    //   LOG(ERROR) << "cuda_safe_move failed: " << cudaGetErrorString(err);
    //   return;
    // }

    // // 更新当前区域为 Allocated
    // status = RegionStatus::Allocated;
    // fingerprint = other->fingerprint;
    // size = other->size;  // 新大小为 other->size

    // // 更新 other 区域为 Free
    // other->status = RegionStatus::Free;
    // other->addr = addr + size;  // 新地址为原空闲区域 + 已分配大小
    // other->size = total_merged_size - size;
    // other->fingerprint.clear();
  }

  // 将当前 Free 区域分裂成两个：左侧区域为 Allocated，大小为 allocated_size，
  // 右侧区域为新的 Free 区域，大小为 (原区域大小 - allocated_size)。
  // 返回新创建的 Allocated 区域
  std::shared_ptr<GPUMemoryRegion_V2> splitAllocated(size_t allocated_size) {
    if (status != RegionStatus::Free) {
      LOG(ERROR) << "splitAllocated 必须在 Free 区域上调用";
      return nullptr;
    }
    if (size < allocated_size) {
      LOG(ERROR) << "splitAllocated 错误：区域大小不足";
      return nullptr;
    }
    // 如果正好相等，则无需分裂，直接将状态更新为 Allocated
    if (size == allocated_size) {
      status = RegionStatus::Allocated;
      return shared_from_this();
    }

    // 计算新 free 区域的起始地址和大小
    char* new_free_addr = addr + allocated_size;
    size_t remaining = size - allocated_size;

    // 更新当前区域：变为 Allocated 区域，其大小为 allocated_size
    size = allocated_size;
    status = RegionStatus::Allocated;

    // 创建新的 Free 区域
    auto new_region =
        std::make_shared<GPUMemoryRegion_V2>(new_free_addr, remaining);
    new_region->status = RegionStatus::Free;
    new_region->fingerprint.clear();

    // 更新链表：将新区域插入当前区域之后
    new_region->next = next;
    if (next) {
      next->prev = new_region;
    }
    next = new_region;
    new_region->prev = shared_from_this();

    return shared_from_this();
  }
};

/*
GPUTensorPool
成员变量：
  - shared_ptr<ResigteredModel> models, 模型被绑定到这个池子
  - total_model_request + {model_path, model_request}字典,用来计算模型的访问概率
  - {model_path, load_cost}字典, 每个模型的装载速度（比如，从CPU或者从SSD装载）
  - allocated_regions， {fingerprint, region}字典, 快速找到已分配的模型的TG
  - memory_region_view，显存池的全局视图
  - shared_ptr<GPUMemoryRegion> to_load_region,
分配好空间，等待从CPU装载TG的空闲区域

成员方法：
-AllocateASAP(size),分配一个空闲区域，返回指向这个区域的指针。如果找不到一个足够大的空闲空间则找到一个最快的方式合并出来一个足够大的空闲区域。注意合并时会跳过状态为正在状态的区域。
-FreeRegion(region), 释放一个区域，更新allocated_regions
-LoadModel(model_path)
    1. 如果不属于当前显存池返回错误
    2. 遍历RegisteredModel对象的tensor_group_indexes_数组，获得所有的TG
    3. 检查TG的fingerprint，获得不在allocated_regions中的待装载的TG
    4.遍历memory_region_view获得空闲区域大小之和，如果小于待装载的TG的大小之和：
        计算待释放空间
        选择一些已装载的TG释放，要求选择TG的总空间大于等于待释放空间，同时选择的TG的总成本最低。成本的计算方式：TG的大小*模型装载速度*模型访问概率。
        调用FreeRegion释放选择的TG
    5. 开始异步装载：
        对于第一个待装载的TG，调用AllocateASAP(size)，更新to_load_region
        遍历所有待装载的TG：
            访问to_load_region,如果还没准备好则等待
            准备好之后：根据to_load_region获得分配的区域，并更新其状态为正在装载
            访问RegisteredModel对象中的tensor_group_host_ptr，判断TG是否已经加载到CPU中，如果没有则等待，否则调用cudaMemcpyAsync将TG从CPU复制到对应的分配区域
            如果还有下一个TG，调用AllocateASAP(size)分配一个新区域并更新to_load_region
                注意此时可能遇到状态为正在装载的区域，必须跳过
            等待cudaMemcpyAsync完成，完成后更新分配区域的状态为已分配
            更新allocated_regions字典

*/
class GPUTensorPool_V4 {
 private:
  size_t total_size;
  std::string uuid;
  int device_id;
  cudaStream_t stream_;
  void* gpu_memory_base;

  // 模型管理
  std::unordered_map<std::string, std::shared_ptr<RegisteredModel>> models;
  std::unordered_map<std::string, int>
      total_model_request;  // key: model_path, value: 请求数
  std::unordered_map<std::string, double> load_cost;

  std::vector<std::shared_ptr<GPUMemoryRegion_V2>> memory_regions;
  std::unordered_map<std::string, std::shared_ptr<GPUMemoryRegion_V2>>
      allocated_regions;  // Fingerprint -> Region
  std::shared_ptr<GPUMemoryRegion_V2>
      memory_region_view;  // Pointer to the start of the memory region
  //   std::mutex memory_region_mutex;  // 成员变量，保护内存区域链表

  // 当前待装载区域（分配好后等待从 CPU 装载 TG）
  std::shared_ptr<GPUMemoryRegion_V2> to_load_region;
  //   std::mutex load_region_mutex;
  //   std::condition_variable load_region_cv;

  //   std::future<bool> async_task;  // 异步任务句柄

  std::vector<double> total_load_time;
  std::vector<double> prepare_free_region_time;
  std::vector<double> copy_time;
  std::vector<double> first_region_time;

 public:
  GPUTensorPool_V4(int device_id_, size_t total_size)
      : device_id(device_id_),
        total_size(total_size) {  // Cache size is an example
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
        (unsigned char)props.uuid.bytes[10],
        (unsigned char)props.uuid.bytes[11],
        (unsigned char)props.uuid.bytes[12],
        (unsigned char)props.uuid.bytes[13],
        (unsigned char)props.uuid.bytes[14],
        (unsigned char)props.uuid.bytes[15]);
    uuid = std::string(uuidStr);
    cudaError_t err = cudaStreamCreate(&stream_);
    if (err != cudaSuccess) {
      LOG(ERROR) << "cudaStreamCreate error: " << cudaGetErrorString(err);
    }

    err = cudaMalloc(&gpu_memory_base, total_size);
    if (err != cudaSuccess) {
      LOG(ERROR) << "cudaMalloc error: " << cudaGetErrorString(err);
      exit(1);
    }
    LOG(INFO) << "GPU device: " << device_id
              << " allocated: " << total_size / 1024.0 / 1024.0 / 1024.0
              << " GB";

    auto initial_region =
        std::make_shared<GPUMemoryRegion_V2>(gpu_memory_base, total_size);
    memory_regions.push_back(initial_region);
    memory_region_view = initial_region;
  }

  ~GPUTensorPool_V4() {
    cudaSetDevice(device_id);
    cudaFree(gpu_memory_base);  // 只释放初始分配的显存

    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
    LOG(INFO) << "Destroy GPUTensorPool for device " << device_id;
  }

  void BoundModel(const std::string& model_path,
                  const std::shared_ptr<RegisteredModel>& model, double cost) {
    // models.push_back(model);
    models[model_path] = model;
    total_model_request[model_path] = 0;
    load_cost[model_path] = cost;
  }
  // Get a memory region by its fingerprint
  std::shared_ptr<GPUMemoryRegion_V2> GetTensor(
      const std::string& fingerprint) {
    auto it = allocated_regions.find(fingerprint);
    if (it != allocated_regions.end()) {
      return it->second;
    } else {
      return nullptr;
    }
  }

  void MergeFreeRegions() {
    auto region = memory_region_view;
    while (region) {
      if (region->status == RegionStatus::Free && region->next &&
          region->next->status == RegionStatus::Free) {
        region->merge(region->next);
      } else {
        region = region->next;
      }
    }
  }

  size_t GetFreeSize() {
    size_t free_size = 0;
    auto region = memory_region_view;
    while (region) {
      if (region->status == RegionStatus::Free) {
        free_size += region->size;
      }
      region = region->next;
    }
    return free_size;
  }

  // - 遍历memory_region_view，找到第一个足够大的且状态为空闲的区域
  // - 如果没有找到，搜索合并成本最低的连续空闲区域：
  //   遍历每一个连续空闲区域的开始s:
  //     找到一个最小的空闲区域e作为连续空闲区域的结尾,e>=s,
  //     从s到e之间的所有空闲区域的大小加起来大于等于要分配的空间
  //     合并成本等于空闲区域s和e之间所有的已分配区域的大小
  //     s和e之间不能包含正在装载的区域，如果遇到一个正在装载的区域，则s直接从正在装载区域右边的空闲区域开始
  //   找到移动成本最低的(s,e)对，利用cudaMemcpy移动其中的已分配区域到(s,e)这个范围内的左侧，合并空闲区域到范围的右侧
  //   将合并后空闲区域分配给TG，如果还有剩余空间则在新的分配空间的右侧创建一个新的空闲区域
  //   返回分配区域的首地址
  std::shared_ptr<GPUMemoryRegion_V2> AllocateASAP(size_t requested_size) {
    // 1. 查找第一个足够大的空闲区域
    std::shared_ptr<GPUMemoryRegion_V2> curr = memory_region_view;
    while (curr) {
      if (curr->status == RegionStatus::Free && curr->size >= requested_size) {
        break;
      }
      curr = curr->next;
    }

    if (!curr) {
      // 2. 寻找最优合并候选 (s, e)
      double best_move_cost = std::numeric_limits<double>::max();
      std::shared_ptr<GPUMemoryRegion_V2> best_start = nullptr,
                                          best_end = nullptr;

      std::shared_ptr<GPUMemoryRegion_V2> s = memory_region_view;
      while (s) {
        if (s->status != RegionStatus::Free) {
          s = s->next;
          continue;
        }

        std::shared_ptr<GPUMemoryRegion_V2> candidate = s;
        size_t free_sum = 0;
        size_t move_cost = 0;
        bool valid = true;
        std::shared_ptr<GPUMemoryRegion_V2> e = nullptr;

        // 寻找当前s对应的e
        while (candidate) {
          if (candidate->status == RegionStatus::Loading) {
            valid = false;
            break;
          }

          if (candidate->status == RegionStatus::Free) {
            free_sum += candidate->size;
          } else {
            move_cost += candidate->size;
          }

          if (free_sum >= requested_size) {
            e = candidate;
            break;
          }

          if (!candidate->next ||
              candidate->next->status == RegionStatus::Loading) {
            valid = false;
            break;
          }

          candidate = candidate->next;
        }

        if (valid && e && move_cost < best_move_cost) {
          best_move_cost = move_cost;
          best_start = s;
          best_end = e;
        }

        // 跳过已处理区域，移动到下一个可能的s
        s = candidate ? candidate->next : nullptr;
        while (s && s->status != RegionStatus::Free) {
          s = s->next;
        }
      }

      if (!best_start) {
        LOG(ERROR) << "AllocateASAP failed: no merge candidate found";
        MemoryRegionView();
        // PrintAllocatedRegions();
        return nullptr;
      }
      //   LOG(INFO)<< "Best merge candidate: (" << best_start->toString()<< ",
      //   " << best_end->toString() << ")";
      // 3. 执行移动交换，彻底移动所有Allocated到左侧
      bool swapped;
      do {
        swapped = false;
        auto merge_ptr = best_start;
        while (merge_ptr && merge_ptr != best_end->next) {
          if (merge_ptr->status == RegionStatus::Free && merge_ptr->next &&
              merge_ptr->next->status == RegionStatus::Allocated) {
            merge_ptr->moveSwapAdjacent(merge_ptr->next);
            swapped = true;
            // 交换后可能需要重新检查当前节点
          } else {
            merge_ptr = merge_ptr->next;
          }
        }
      } while (swapped);

      // 3.1 对于移动后的allocated区域，需要更新allocated_regions字典的内容
      auto merge_ptr = best_start;
      while (merge_ptr && merge_ptr != best_end->next) {
        if (merge_ptr->status == RegionStatus::Allocated) {
          allocated_regions[merge_ptr->fingerprint] = merge_ptr;
        }
        merge_ptr = merge_ptr->next;
      }

      // 4. 合并右侧的Free区域
      auto left_free = best_start;
      while (left_free && left_free->status != RegionStatus::Free) {
        left_free = left_free->next;
      }
      if (left_free) {
        while (left_free->next &&
               left_free->next->status == RegionStatus::Free) {
          left_free->merge(left_free->next);
        }
      }
      curr = left_free;

      if (!curr || curr->size < requested_size) {
        LOG(ERROR) << "Failed to merge enough space";
        return nullptr;
      }
    }

    // 5. 分割并返回分配的区域
    return curr->splitAllocated(requested_size);
  }

  void FreeRegion(std::shared_ptr<GPUMemoryRegion_V2> region) {
    allocated_regions.erase(region->fingerprint);
    region->status = RegionStatus::Free;
  }

  // LoadModel: 装载模型
  // 1. 检查模型是否属于当前池（models 集合中存在该 model_path）
  // 2. 遍历 RegisteredModel 的 tensor_group_indexes_ 获得所有 TG
  // 3. 筛选未在 allocated_regions 中的 TG（待装载）
  // 4. 检查 memory_region_view 空闲总空间；若不足则依成本（TG大小 * 装载速度 *
  // 访问概率）释放部分已分配 TG
  // 5. 对于第一个待加载 TG，调用 AllocateASAP 分配区域更新 to_load_region，
  //    遍历所有待加载 TG：等待 to_load_region 就绪后更新区域状态为 Loading，
  //    异步分配下一个区域，并用 cudaMemcpy 将 CPU 中的数据复制到 GPU 分配区
  int LoadModel(const std::string& model_path) {
    // 1. 检查模型是否存在
    auto target_model = models[model_path];
    if (!target_model) {
      LOG(ERROR) << "Model " << model_path << " not found in pool";
      return -1;
    }

    // 2. 收集需要加载的Tensor Group
    const auto& tg_indexes = target_model->GetTensorGroupIndexes();
    std::vector<size_t> pending;
    for (size_t i = 0; i < tg_indexes.size(); i++) {
      if (allocated_regions.find(tg_indexes[i].fingerprint) ==
          allocated_regions.end()) {
        pending.push_back(i);
      }
    }
    if (pending.empty()) {
      LOG(INFO) << "All tensor groups already loaded for " << model_path;
      return 0;
    }
    LOG(INFO)<<"Pending TG / Total TG: "<<pending.size()<<"/"<<tg_indexes.size();

    // 3. 计算所需空间并释放旧区域（带成本计算）
    size_t required_space = 0;
    for (auto idx : pending) {
      required_space += tg_indexes[idx].size;
    }

    // 改进的空间释放策略（按模型相关成本释放）
    if (GetFreeSize() < required_space) {
      std::vector<std::tuple<std::string, size_t, double>>
          candidates;  // (fp, size, cost)
      // LOG(INFO)<<"-----content of allocated_regions before remove-----";
      // PrintAllocatedRegions();
      // LOG(INFO)<<"-----------------------------------------------------";
      // 步骤1: 收集候选区域
      {
        for (const auto& [fp, region] : allocated_regions) {
          // 直接从区域获取模型路径
          const std::string& model_path = region->model_path;
          if (model_path.empty()) {
            LOG(WARNING) << "Orphan region found: " << fp;
            continue;
          }

          // 获取模型参数（带默认值防御空指针）
          const double model_load_cost =
              load_cost.count(model_path) ? load_cost[model_path] : 1.0;
          const int model_request_count = total_model_request.count(model_path)
                                              ? total_model_request[model_path]
                                              : 1;

          // 成本公式：区域大小 * 模型装载成本 / (1 + 模型请求次数)
          double cost =
              region->size * model_load_cost * (1 + model_request_count);
          candidates.emplace_back(fp, region->size, cost);
        }
      }

      // 步骤2: 按成本升序排序（优先释放低价值区域）
      std::sort(candidates.begin(), candidates.end(),
                [](const auto& a, const auto& b) {
                  return std::get<2>(a) < std::get<2>(b);
                });

      // 步骤3: 执行释放
      size_t released = 0;
      size_t required_release = required_space - GetFreeSize();
      for (const auto& [fp, size, cost] : candidates) {
        if (released >= required_release) break;

        // 获取并释放区域
        std::shared_ptr<GPUMemoryRegion_V2> region = allocated_regions[fp];
        // {
        //   auto it = allocated_regions.find(fp);
        //   if (it == allocated_regions.end()) continue;
        //   region = it->second;
        // }

        if (region) {
          FreeRegion(region);
          released += size;
          // LOG(INFO) << "Released region " << fp << " (size: " << size
          //           << ", cost: " << cost << ")";
        }
      }
      MergeFreeRegions();

      // 最终检查
      if (GetFreeSize() < required_space) {
        LOG(ERROR) << "Insufficient memory after eviction (Need: "
                   << required_space << ", Free: " << GetFreeSize() << ")";
        MemoryRegionView();
        return -1;
      }
    }

    // 4. 异步加载
    for (size_t i = 0; i < pending.size(); ++i) {
      auto start_load = std::chrono::high_resolution_clock::now();
      const auto idx = pending[i];
      const auto& tg_info = tg_indexes[idx];

      // 4.1 分配当前区域
      if (i == 0) {
        auto start_first_region = std::chrono::high_resolution_clock::now();
        to_load_region = AllocateASAP(tg_info.size);
        auto end_first_region = std::chrono::high_resolution_clock::now();
        first_region_time.push_back(std::chrono::duration<double, std::micro>(
                                        end_first_region - start_first_region)
                                        .count());
      }

      if (!to_load_region || to_load_region->size < tg_info.size) {
        LOG(ERROR) << "Failed to allocate " << tg_info.size << " bytes for "
                   << tg_info.fingerprint;
        return -1;
      }

      // 4.2 分割并标记为Loading状态
      auto allocated = to_load_region;
      allocated->status = RegionStatus::Loading;

      // 4.3 异步执行CPU->GPU的数据拷贝（带流同步）
      //   TODO: 使用其他同步方法代替手动轮询
      void* host_ptr = target_model->GetTensorGroupHostPtr()->get(idx);
      int pooling = 0;
      while (host_ptr == nullptr) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        pooling++;
        if (pooling > 10000) {
          break;
        }
        host_ptr = target_model->GetTensorGroupHostPtr()->get(idx);
      }
      if (!host_ptr) {
        LOG(ERROR) << "Host data not ready for " << tg_info.fingerprint;
        allocated->status = RegionStatus::Free;
        return -1;
      }
      auto start_load_data = std::chrono::high_resolution_clock::now();
      cudaError_t err = cudaMemcpyAsync(allocated->addr, host_ptr, tg_info.size,
                                        cudaMemcpyHostToDevice, stream_);
      if (err != cudaSuccess) {
        LOG(ERROR) << "cudaMemcpy error: " << cudaGetErrorString(err);
        allocated->status = RegionStatus::Free;
        return -1;
      }

      // 4.3 异步预分配下一个区域
      if (i < pending.size() - 1) {
        const auto next_size = tg_indexes[pending[i + 1]].size;
        auto start_next_region = std::chrono::high_resolution_clock::now();
        to_load_region = AllocateASAP(next_size);
        auto end_next_region = std::chrono::high_resolution_clock::now();
        prepare_free_region_time.push_back(
            std::chrono::duration<double, std::micro>(end_next_region -
                                                      start_next_region)
                .count());
      }

      // 4.5 等待当前流完成并更新状态
      cudaStreamSynchronize(stream_);
      auto end_load_data = std::chrono::high_resolution_clock::now();
      copy_time.push_back(std::chrono::duration<double, std::micro>(
                              end_load_data - start_load_data)
                              .count());

      allocated->status = RegionStatus::Allocated;
      allocated->fingerprint = tg_info.fingerprint;
      allocated->model_path = model_path;
      allocated_regions[allocated->fingerprint] = allocated;

      // LOG(INFO) << "Loaded tensor group" << tg_info.fingerprint << " ("
      //           << tg_info.size / 1024.0 << " KB)";
      // MemoryRegionView();
      // PrintAllocatedRegions();
      // LOG(INFO)<<"-------------------------------------";
      auto end_load = std::chrono::high_resolution_clock::now();
      total_load_time.push_back(
          std::chrono::duration<double, std::micro>(end_load - start_load)
              .count());
    }

    // 5. 更新模型访问计数
    total_model_request[model_path]++;
    return 0;
  }

    int LoadModelSync(const std::string& model_path) {
      // 1. 检查模型是否存在
      auto target_model = models[model_path];
      if (!target_model) {
          LOG(ERROR) << "Model " << model_path << " not found in pool";
          return -1;
      }
  
      // 2. 收集需要加载的 Tensor Group
      const auto& tg_indexes = target_model->GetTensorGroupIndexes();
      std::vector<size_t> pending;
      for (size_t i = 0; i < tg_indexes.size(); i++) {
          if (allocated_regions.find(tg_indexes[i].fingerprint) == allocated_regions.end()) {
              pending.push_back(i);
          }
      }
      if (pending.empty()) {
          LOG(INFO) << "All tensor groups already loaded for " << model_path;
          return 0;
      }
      LOG(INFO) << "Pending TG / Total TG: " << pending.size() << "/" << tg_indexes.size();
  
      // 3. 计算所需空间
      size_t required_space = 0;
      for (auto idx : pending) {
          required_space += tg_indexes[idx].size;
      }
  
      // 4. 如果空间不足，释放部分区域
      if (GetFreeSize() < required_space) {
          std::vector<std::tuple<std::string, size_t, double>> candidates;
          
          // 收集候选区域
          for (const auto& [fp, region] : allocated_regions) {
              const std::string& model_path = region->model_path;
              if (model_path.empty()) {
                  LOG(WARNING) << "Orphan region found: " << fp;
                  continue;
              }
  
              const double model_load_cost = load_cost.count(model_path) ? load_cost[model_path] : 1.0;
              const int model_request_count = total_model_request.count(model_path) ? 
                                           total_model_request[model_path] : 1;
  
              double cost = region->size * model_load_cost * (1 + model_request_count);
              candidates.emplace_back(fp, region->size, cost);
          }
  
          // 按成本排序并释放
          std::sort(candidates.begin(), candidates.end(),
                   [](const auto& a, const auto& b) { return std::get<2>(a) < std::get<2>(b); });
  
          size_t released = 0;
          size_t required_release = required_space - GetFreeSize();
          for (const auto& [fp, size, cost] : candidates) {
              if (released >= required_release) break;
              if (auto region = allocated_regions[fp]) {
                  FreeRegion(region);
                  released += size;
              }
          }
      }
  
      // 5. 将所有空闲空间合并到一起
      bool swapped;
      do {
          swapped = false;
          auto curr = memory_region_view;
          while (curr && curr->next) {
              if (curr->status == RegionStatus::Free && 
                  curr->next->status == RegionStatus::Allocated) {
                  curr->moveSwapAdjacent(curr->next);
                  swapped = true;
                  continue;
              }
              curr = curr->next;
          }
      } while (swapped);
  
      // 6. 合并所有相邻的空闲区域
      MergeFreeRegions();
  
      // 7. 检查合并后的空闲空间是否足够
      if (GetFreeSize() < required_space) {
          LOG(ERROR) << "Insufficient memory after consolidation (Need: " 
                    << required_space << ", Free: " << GetFreeSize() << ")";
          return -1;
      }
  
      // 8. 同步加载所有 Tensor Groups
      auto curr_free = memory_region_view;
      while (curr_free && curr_free->status != RegionStatus::Free) {
          curr_free = curr_free->next;
      }
  
      if (!curr_free || curr_free->size < required_space) {
          LOG(ERROR) << "Failed to find consolidated free region";
          return -1;
      }
  
      // 9. 按顺序加载每个 Tensor Group
      for (size_t i = 0; i < pending.size(); ++i) {
          const auto idx = pending[i];
          const auto& tg_info = tg_indexes[idx];
  
          // 分配空间
          auto allocated = curr_free->splitAllocated(tg_info.size);
          if (!allocated) {
              LOG(ERROR) << "Failed to allocate " << tg_info.size << " bytes";
              return -1;
          }
  
          // 同步复制数据
          void* host_ptr = target_model->GetTensorGroupHostPtr()->get(idx);
          int pooling = 0;
          while (host_ptr == nullptr && pooling < 10000) {
              std::this_thread::sleep_for(std::chrono::milliseconds(1));
              pooling++;
              host_ptr = target_model->GetTensorGroupHostPtr()->get(idx);
          }
          
          if (!host_ptr) {
              LOG(ERROR) << "Host data not ready for " << tg_info.fingerprint;
              allocated->status = RegionStatus::Free;
              return -1;
          }
  
          cudaError_t err = cudaMemcpy(allocated->addr, host_ptr, tg_info.size,
                                      cudaMemcpyHostToDevice);
          if (err != cudaSuccess) {
              LOG(ERROR) << "cudaMemcpy error: " << cudaGetErrorString(err);
              allocated->status = RegionStatus::Free;
              return -1;
          }
  
          // 更新区域信息
          allocated->status = RegionStatus::Allocated;
          allocated->fingerprint = tg_info.fingerprint;
          allocated->model_path = model_path;
          allocated_regions[allocated->fingerprint] = allocated;
  
          // 更新空闲区域指针
          curr_free = allocated->next;
      }
  
      // 10. 更新访问计数
      total_model_request[model_path]++;
      return 0;
  }

  int UnloadModel(const std::string& model_path) {
    auto it = models.find(model_path);
    if (it == models.end()) {
      LOG(ERROR) << "Model " << model_path << " not found in pool";
      return -1;
    }

    // 遍历所有已分配的区域，释放与模型路径匹配的区域
    for (auto it = allocated_regions.begin(); it != allocated_regions.end();) {
      if (it->second->model_path == model_path) {
        // FreeRegion(it->second);
        it->second->status = RegionStatus::Free;
        it = allocated_regions.erase(it);
      } else {
        ++it;
      }
    }
    MergeFreeRegions();
    return 0;
  }

  void MemoryRegionView() {
    auto mr = memory_region_view;
    while (mr) {
      std::cout << "[ " << mr->fingerprint << "]: ( " << mr->status << ", "
                << float(mr->size) / 1024 << "KB, "
                << reinterpret_cast<void*>(mr->addr) << " ) " << std::endl;

      mr = mr->next;
    }
    std::cout << std::endl;
  }
  void PrintAllocatedRegions() {
    for (const auto& [fingerprint, region] : allocated_regions) {
      std::cout << "Allocated Region: " << fingerprint << " -> "
                << region->toString() << std::endl;
    }
  }

  void CollectTimeMetrics() {
    double total_load_time_sum = 0;
    for (const auto& time : total_load_time) {
      total_load_time_sum += time;
    }
    double avg_load_time = total_load_time_sum / total_load_time.size();

    double total_prepare_free_region_time = 0;
    for (const auto& time : prepare_free_region_time) {
      total_prepare_free_region_time += time;
    }
    double avg_prepare_free_region_time =
        total_prepare_free_region_time / prepare_free_region_time.size();

    double total_copy_time = 0;
    for (const auto& time : copy_time) {
      total_copy_time += time;
    }
    double avg_copy_time = total_copy_time / copy_time.size();

    double total_first_region_time = 0;
    for (const auto& time : first_region_time) {
      total_first_region_time += time;
    }
    double avg_first_region_time =
        total_first_region_time / first_region_time.size();
    LOG(INFO) << "Average First Region Time: " << avg_first_region_time
              << " microseconds" << "Count: " << first_region_time.size();

    LOG(INFO) << "Average Load Time: " << avg_load_time
              << " microseconds, Average Prepare Free Region Time: "
              << avg_prepare_free_region_time
              << " microseconds, Average Copy Time: " << avg_copy_time
              << " microseconds"<< "Count: " << copy_time.size();
  }

  void CollectTimeMetrics2(){
    printf("prepare_free_region_time : copy_time \n ");
    for(int i=0;i<total_load_time.size();i++){
      printf("%f : %f \n",prepare_free_region_time[i],copy_time[i]);
    }
  }

  float GetFragmentation() {
    size_t used = 0;
    auto mr = memory_region_view;
    while (mr) {
      if (mr->status == RegionStatus::Allocated) {
        used += mr->size;
      }
      mr = mr->next;
    }
    return (double)(total_size - used) / total_size;
  }
};