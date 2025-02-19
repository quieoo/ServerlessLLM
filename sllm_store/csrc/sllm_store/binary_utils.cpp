// ----------------------------------------------------------------------------
//  ServerlessLLM
//  Copyright (c) ServerlessLLM Team 2024
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//
//   You may obtain a copy of the License at
//
//                   http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
//  ----------------------------------------------------------------------------
#include "binary_utils.h"

#include <iomanip>
#include <iostream>
#include <vector>

// Function to print the binary array in hexadecimal format
void PrintBinaryArrayInHex(const unsigned char* data, size_t size) {
  std::cout << "Data in Hex: ";
  for (size_t i = 0; i < size; ++i) {
    std::cout << std::hex << std::setw(2) << std::setfill('0')
              << static_cast<int>(data[i]) << " ";
  }
  std::cout << std::dec
            << std::endl;  // Switch back to decimal for any future output
}
std::string toHex(const std::vector<uint8_t>& data) {
  std::stringstream ss;
  for (uint8_t byte : data) {
      ss << std::setw(2) << std::setfill('0') << std::hex << (int)byte;
  }
  return ss.str();
}

std::vector<uint8_t> fromHex(const std::string& hexStr) {
  std::vector<uint8_t> data;
  for (size_t i = 0; i < hexStr.length(); i += 2) {
      uint8_t byte = (std::stoi(hexStr.substr(i, 2), nullptr, 16));
      data.push_back(byte);
  }
  return data;
}