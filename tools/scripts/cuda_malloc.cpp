#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>  // 添加这行

int main() {
    const int NUM_ITERATIONS = 100000;  // 测试次数
    const size_t size = 2*1024 * 1024; // 分配 1MB 内存
    std::vector<double> latencies;    // 存储每次调用的时延
    
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        float* d_ptr;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        cudaError_t err = cudaMalloc((void**)&d_ptr, size);
        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
            return 1;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        latencies.push_back(duration);
        
        // 释放内存
        cudaFree(d_ptr);
    }
    
    // 计算平均时延
    double avg_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0) / NUM_ITERATIONS;
    
    // 找出最大和最小时延
    double min_latency = *std::min_element(latencies.begin(), latencies.end());
    double max_latency = *std::max_element(latencies.begin(), latencies.end());
    
    // 输出统计信息
    std::cout << "测试次数: " << NUM_ITERATIONS << std::endl;
    std::cout << "平均时延: " << avg_latency << " 微秒" << std::endl;
    std::cout << "最小时延: " << min_latency << " 微秒" << std::endl;
    std::cout << "最大时延: " << max_latency << " 微秒" << std::endl;

    return 0;
}