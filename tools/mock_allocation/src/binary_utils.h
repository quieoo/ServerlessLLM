
#pragma once

#include <cstddef>
#include <string>
#include <vector>
// Function to print the binary array in hexadecimal format
void PrintBinaryArrayInHex(const unsigned char* data, size_t size);

std::string toHex(const std::vector<uint8_t>& data);
std::vector<uint8_t> fromHex(const std::string& hexStr);
void* allocateAlignedPinnedMemory(size_t size, size_t alignment);
void freeAlignedPinnedMemory(void* ptr);

template <typename T>
std::string Join(const std::vector<T>& vec, const std::string& delimiter);