#include <torch/extension.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

torch::ScalarType ParseDtype(const std::string& dtype) {
  if (dtype == "torch.float16") return torch::kFloat16;
  if (dtype == "torch.bfloat16") return torch::kBFloat16;
  if (dtype == "torch.float32") return torch::kFloat32;
  if (dtype == "torch.float64") return torch::kFloat64;
  if (dtype == "torch.int8") return torch::kInt8;
  if (dtype == "torch.uint8") return torch::kUInt8;
  if (dtype == "torch.int16") return torch::kInt16;
  if (dtype == "torch.int32") return torch::kInt32;
  if (dtype == "torch.int64") return torch::kInt64;
  if (dtype == "torch.bool") return torch::kBool;
  throw std::invalid_argument("Unsupported tensor dtype: " + dtype);
}

py::dict RestoreVmmTensors(const py::dict& tensor_meta,
                           const std::vector<uint64_t>& group_bases,
                           const std::vector<uint64_t>& tensor_offsets,
                           int device_id) {
  if (tensor_meta.size() != group_bases.size() ||
      tensor_meta.size() != tensor_offsets.size()) {
    throw std::invalid_argument("metadata, base, and offset counts differ");
  }
  py::dict result;
  size_t index = 0;
  for (const auto& item : tensor_meta) {
    const std::string name = py::cast<std::string>(item.first);
    const py::tuple meta = py::cast<py::tuple>(item.second);
    const std::vector<int64_t> shape =
        py::cast<std::vector<int64_t>>(meta[0]);
    const std::vector<int64_t> stride =
        py::cast<std::vector<int64_t>>(meta[1]);
    const std::string dtype = py::cast<std::string>(meta[2]);
    void* pointer = reinterpret_cast<void*>(group_bases[index] +
                                            tensor_offsets[index]);
    auto options = torch::TensorOptions()
                       .dtype(ParseDtype(dtype))
                       .device(torch::Device(torch::kCUDA, device_id));
    // VMM owns this storage. The no-op deleter prevents PyTorch from freeing
    // it; Python keeps TangramVmmSession attached to the vLLM model.
    result[py::str(name)] =
        torch::from_blob(pointer, shape, stride, [](void*) {}, options);
    ++index;
  }
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("restore_vmm_tensors", &RestoreVmmTensors,
             "Wrap Tangram VMM pointers as non-owning CUDA tensors");
}

