#include <iostream>
#include <cuda_runtime.h>
#include <vector>

// CUDA Kernel function to swap data between a and c using b as temporary storage
// Each thread handles multiple elements and uses shared memory for faster access
__global__ void swapDataKernel(int* gpu_memory, size_t part_size) {
    extern __shared__ int shared_memory[];  // Declare shared memory
    
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Ensure that each thread works on 4 elements, while staying within the part_size bounds
    if (idx < part_size / 4) {
        int* a = gpu_memory;                      // a starts at the beginning of the memory
        int* c = gpu_memory + 2 * part_size;      // c starts after the second part of memory

        // Each thread handles 4 consecutive elements
        for (int i = 0; i < 4; i++) {
            size_t idx_shifted = idx * 4 + i;  // Access 4 consecutive elements
            if (idx_shifted < part_size) {
                // Load data into shared memory
                shared_memory[threadIdx.x + i] = a[idx_shifted];  // Load 'a' data into shared memory
                
                // Wait for all threads to load data
                __syncthreads();

                // Swap a[idx_shifted] and c[idx_shifted] using shared memory
                int temp = shared_memory[threadIdx.x + i];
                a[idx_shifted] = c[idx_shifted];
                c[idx_shifted] = temp;

                // Wait for all threads to complete the swap
                __syncthreads();
            }
        }
    }
}

int data_copy_with_kernel(int* gpu_memory, size_t part_size) {

    int blockSize = 256;  
    int numBlocks = (part_size + blockSize - 1) / blockSize;  // Calculate the number of blocks needed

    // Allocate shared memory size based on the number of threads
    size_t sharedMemorySize = blockSize * 4 * sizeof(int); // 4 integers per thread
    
    // Launch kernel to swap data between a and c using b as temporary storage
    swapDataKernel<<<numBlocks, blockSize, sharedMemorySize>>>(gpu_memory, part_size);

    // Synchronize to ensure the kernel finishes
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(gpu_memory);
        return 1;
    }
    return 0;
}

// Asynchronous memory copy with CUDA Streams
int data_copy_with_cudaAPI_streaming(int* gpu_memory, size_t part_size) {
    int* a = gpu_memory;                      // a starts at the beginning of the memory
    int* b = gpu_memory + part_size;          // b starts after the first part of memory
    int* c = gpu_memory + 2 * part_size;      // c starts after the second part of memory

    // Create three CUDA streams for asynchronous execution
    cudaStream_t stream1, stream2, stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);

    // Asynchronously copy data from a to b, from b to c, and from c to a using the streams
    cudaError_t err = cudaMemcpyAsync(b, a, part_size * sizeof(int), cudaMemcpyDeviceToDevice, stream1);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpyAsync failed for stream1: " << cudaGetErrorString(err) << std::endl;
        cudaFree(gpu_memory);
        return 1;
    }

    err = cudaMemcpyAsync(c, b, part_size * sizeof(int), cudaMemcpyDeviceToDevice, stream2);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpyAsync failed for stream2: " << cudaGetErrorString(err) << std::endl;
        cudaFree(gpu_memory);
        return 1;
    }

    err = cudaMemcpyAsync(a, c, part_size * sizeof(int), cudaMemcpyDeviceToDevice, stream3);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpyAsync failed for stream3: " << cudaGetErrorString(err) << std::endl;
        cudaFree(gpu_memory);
        return 1;
    }

    // Synchronize the streams to ensure all copy operations are completed
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamSynchronize(stream3);

    // Clean up streams
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);

    return 0;
}

int data_copy_with_cudaAPI(int* gpu_memory, size_t part_size) {
    int* a = gpu_memory;                      // a starts at the beginning of the memory
    int* b = gpu_memory + part_size;          // b starts after the first part of memory

    // Asynchronously copy data from a to b
    cudaError_t err = cudaMemcpyAsync(b, a, part_size * sizeof(int), cudaMemcpyDeviceToDevice, 0);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpyAsync failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(gpu_memory);
        return 1;
    }

    // Wait for the copy operation to complete
    cudaDeviceSynchronize();

    return 0;
}

int data_copy_with_multiple_streams(int* gpu_memory, size_t part_size, int num_streams) {
    // a为源数据，b为目标数据
    int* a = gpu_memory;
    int* b = gpu_memory + part_size;
    size_t total_bytes = part_size * sizeof(int);
    size_t chunk_bytes = total_bytes / num_streams;

    // 创建多个CUDA流
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // 分块拷贝：每个流异步拷贝一部分数据
    for (int i = 0; i < num_streams; i++) {
        // 计算当前块的源和目标地址（以字节为单位）
        char* src_ptr = reinterpret_cast<char*>(a) + i * chunk_bytes;
        char* dst_ptr = reinterpret_cast<char*>(b) + i * chunk_bytes;
        cudaError_t err = cudaMemcpyAsync(dst_ptr, src_ptr, chunk_bytes, cudaMemcpyDeviceToDevice, streams[i]);
        if (err != cudaSuccess) {
            std::cerr << "cudaMemcpyAsync failed on stream " << i << ": " << cudaGetErrorString(err) << std::endl;
            for (int j = 0; j < num_streams; j++) {
                cudaStreamDestroy(streams[j]);
            }
            cudaFree(gpu_memory);
            return 1;
        }
    }

    // 同步所有流
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return 0;
}

int main() {
    size_t total_size = 12LL * 1024 * 1024 * 1024; // 3GB total memory
    total_size= 4LL * 1024 * 1024 * 1024 * 3;
    // size_t total_size = 8LL * 1024 * 1024 / 30; // 3GB total memory
    
    size_t part_size = total_size / 3 / sizeof(int); // Divide into 3 parts, size in ints

    int* gpu_memory = nullptr;
    cudaError_t err = cudaMalloc(&gpu_memory, total_size);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Using asynchronous memory copy with CUDA API

    // data_copy_with_cudaAPI(gpu_memory, part_size);
    data_copy_with_multiple_streams(gpu_memory, part_size, 4);


    // data_copy_with_cudaAPI_streaming(gpu_memory, part_size);
    // slower than using CUDA API

    
    // Uncomment below line to use kernel for memory swap
    // data_copy_with_kernel(gpu_memory, part_size);
    // slower than using CUDA API

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;
    std::cout<<"Throughput: "<<(float)total_size/1024/1024/1024/(milliseconds/1000)/3<<" GB/s"<<std::endl;

    // Clean up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free the allocated GPU memory
    cudaFree(gpu_memory);

    return 0;
}
