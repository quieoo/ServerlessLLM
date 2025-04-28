#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

#define BYTES_PER_GB (1LL << 30)
#define BLOCK_SIZE 512

// 测试设备内存带宽的核心函数
__global__ void bandwidthTestKernel(double* src, double* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];  // 简单的内存拷贝操作
    }
}


double measureBandwidth(size_t dataSizeGB) {
    size_t dataSize = dataSizeGB * BYTES_PER_GB / sizeof(double);
    double *d_src, *d_dst;
    
    // 分配设备内存
    cudaMalloc((void**)&d_src, dataSize * sizeof(double));
    cudaMalloc((void**)&d_dst, dataSize * sizeof(double));

    // 预热，避免冷启动误差
    bandwidthTestKernel<<< (dataSize + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >>>(d_src, d_dst, dataSize);
    cudaDeviceSynchronize();

    // 使用 std::chrono 计时：注意 kernel 调用是异步的，
    // 因此在记录结束时间前需要调用 cudaDeviceSynchronize() 来确保所有任务完成。
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        bandwidthTestKernel<<< (dataSize + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >>>(d_src, d_dst, dataSize);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration_ms = end - start;
    double elapsedTime = duration_ms.count();  // 毫秒

    // 计算总传输字节数: 10 次拷贝
    double totalBytes = (double)dataSize * sizeof(double) * 10;  
    double bandwidth = (totalBytes / (elapsedTime / 1000.0f)) / BYTES_PER_GB;  // 单位：GB/s

    cudaFree(d_src);
    cudaFree(d_dst);
    
    return bandwidth;
}

int main() {
    // RTX 4090 官方标称带宽
    const double theoreticalBW = 1008.0f;  // 单位：GB/s (基于GDDR6X 21Gbps * 384位总线)

    printf("Starting HBM bandwidth test...\n");
    
    // 测试不同数据规模
    for (int sizeGB = 1; sizeGB <= 8; sizeGB += 1) {
        double measuredBW = measureBandwidth(sizeGB);
        printf("Data size: %2dGB | Measured BW: %6.1f GB/s | Efficiency: %.1f%%\n",
               sizeGB, measuredBW, (measuredBW/theoreticalBW)*100);
    }

    return 0;
}