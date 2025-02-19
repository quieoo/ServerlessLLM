#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

class TensorIndex {
 public:
  // 默认构造函数
  TensorIndex() = default;

  // 带参数的构造函数
  TensorIndex(std::string name, size_t offset, size_t size,
              const std::vector<int64_t>& shape,
              const std::vector<int64_t>& strides, std::string dtype)
      : name(name),
        offset(offset),
        size(size),
        shape(shape),
        strides(strides),
        dtype(dtype) {}

  std::string name;
  size_t offset;
  size_t size;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  std::string dtype;
};

class TensorGroupIndex {
 public:
  // 默认构造函数
  TensorGroupIndex() = default;

  // 带参数的构造函数
  TensorGroupIndex(size_t file_offset, size_t size, std::string fingerprint,
                   const std::vector<TensorIndex>& tensor_indexes)
      : file_offset(file_offset),
        size(size),
        fingerprint(fingerprint),
        tensor_indexes(tensor_indexes) {}

  size_t file_offset;
  size_t size;
  std::string fingerprint;
  std::vector<TensorIndex> tensor_indexes;
};

void parse_line(std::string line, std::vector<int64_t>& dims) {
  std::regex re("\\d+");
  std::sregex_iterator it(line.begin(), line.end(), re);
  std::sregex_iterator end;

  while (it != end) {
    dims.push_back(std::stoi((*it)[0]));
    ++it;
  }
}

void ParseTensorGroupIndex(
    const std::string& file_path,
    std::unordered_map<std::string, TensorGroupIndex>& tensor_group_indexes_) {
  std::ifstream fin(file_path);
  if (!fin.is_open()) {
    std::cerr << "Failed to open tensor group index file: " << file_path
              << std::endl;
    return;
  }

  std::string line;
  std::string current_fingerprint;
  TensorGroupIndex current_tensor_group;
  bool reading_tensor_group = false;

  while (std::getline(fin, line)) {
    // Skip empty lines or comments
    if (line.empty() || line.find("#") == 0) continue;

    // Read the tensor group details
    if (line.find("Group Offset:") != std::string::npos) {
      if (reading_tensor_group) {
        // Store the current tensor group once it's finished
        tensor_group_indexes_[current_fingerprint] = current_tensor_group;
      }

      // Start reading a new tensor group
      reading_tensor_group = true;
      current_tensor_group = TensorGroupIndex();
      std::stringstream ss(line);
      std::string tmp;
      ss >> tmp >> tmp;  // "Group" "Offset:"
      ss >> current_tensor_group.file_offset;
    }

    if (line.find("Group Size:") != std::string::npos) {
      std::stringstream ss(line);
      std::string tmp;
      ss >> tmp >> tmp;  // "Group" "Size:"
      ss >> current_tensor_group.size;
    }

    if (line.find("Fingerprint:") != std::string::npos) {
      std::stringstream ss(line);
      std::string tmp;
      ss >> tmp;  // "Fingerprint:"
      ss >> current_tensor_group.fingerprint;
      current_fingerprint =
          current_tensor_group.fingerprint;  // Update the fingerprint
    }

    if (line.find("Tensor Name:") != std::string::npos) {
      // Read tensor index details
      TensorIndex tensor_index;
      std::stringstream ss(line);
      std::string tensor_name;
      std::string tmp;
      ss >> tmp >> tmp >> tensor_index.name;

      // Read offset
      std::getline(fin, line);
      ss.clear();
      ss.str(line);
      ss >> tmp >> tensor_index.offset;

      // Read size
      std::getline(fin, line);
      ss.clear();
      ss.str(line);
      ss >> tmp >> tensor_index.size;

      // Read shape
      std::getline(fin, line);
      parse_line(line, tensor_index.shape);

      // Read stride
      std::getline(fin, line);
      parse_line(line, tensor_index.strides);

      // Read dtype
      std::getline(fin, line);
      ss.clear();
      ss.str(line);
      ss >> tmp >> tensor_index.dtype;

      // Add tensor index to tensor group
      current_tensor_group.tensor_indexes.push_back(tensor_index);
    }
  }

  // Add the last tensor group to the map
  if (reading_tensor_group) {
    tensor_group_indexes_[current_fingerprint] = current_tensor_group;
  }

  fin.close();
}

template <typename T>
std::string Join(const std::vector<T>& vec, const std::string& delimiter) {
  std::ostringstream oss;
  for (size_t i = 0; i < vec.size(); ++i) {
    oss << vec[i];
    if (i != vec.size() - 1) {
      oss << delimiter;
    }
  }
  return oss.str();
}

void OutputTensorGroupIndex(
    std::unordered_map<std::string, TensorGroupIndex>& tensor_group_indexes_) {
  if (tensor_group_indexes_.empty()) {
    std::cout << "No tensor group indexes to output." << std::endl;
    return;
  }

  for (const auto& tg_entry : tensor_group_indexes_) {
    const TensorGroupIndex& tg = tg_entry.second;
    std::cout << "Tensor Group - Fingerprint: " << tg.fingerprint << std::endl;
    std::cout << "  File Offset: " << tg.file_offset << ", Size: " << tg.size
              << std::endl;
    std::cout << "  Tensor Group Contains " << tg.tensor_indexes.size()
              << " Tensors:" << std::endl;

    for (const TensorIndex& tensor : tg.tensor_indexes) {
      std::cout << "    Tensor Name: " << tensor.name << std::endl;
      std::cout << "      Offset: " << tensor.offset
                << ", Size: " << tensor.size << std::endl;
      std::cout << "      Shape: " << Join(tensor.shape, " ") << std::endl;
      std::cout << "      Strides: " << Join(tensor.strides, " ") << std::endl;
      std::cout << "      Dtype: " << tensor.dtype << std::endl;
    }
    std::cout << "" << std::endl;  // Blank line between tensor groups
  }
}

int main() {
  std::string file_path =
      "/mnt/n0/models/vllm/opt1.3_tmp/rank_0/tensor_group_index.txt";
  std::unordered_map<std::string, TensorGroupIndex> tensor_group_indexes;
  ParseTensorGroupIndex(file_path, tensor_group_indexes);
  OutputTensorGroupIndex(tensor_group_indexes);
}