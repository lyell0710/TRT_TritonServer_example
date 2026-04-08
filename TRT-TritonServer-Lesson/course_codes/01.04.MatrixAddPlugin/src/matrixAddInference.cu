#include <cuda_fp16.h>

#include "matrixAddInference.h"

using namespace nvinfer1;
using namespace nvinfer1::plugin;

// FP32（单精度浮点数）内联函数

float __device__ __inline__ add_mp(const float a, const float b) { return __fadd_rn(a, b); }

#if __CUDA_ARCH__ >= 530

// 如果CUDA架构版本大于等于530（支持半精度浮点数操作），则使用FP16内联函数

__half __device__ __inline__ add_mp(const __half a, const __half b) { return __hadd(a, b); }

#else

// 如果CUDA架构版本低于530，则使用FP16的回退实现

__half __device__ __inline__ add_mp(const __half a, const __half b) {
    return __float2half(add_mp(__half2float(a), __half2float(b))); // 将半精度浮点数转换为单精度浮点数进行计算，再转换回半精度浮点数
}

#endif

// 矩阵加法的CUDA内核函数模板
template <typename T>
__global__ void matrixAddKernel(const T* __restrict__ tensorInputA,
                                const T* __restrict__ tensorInputB, T* __restrict__ tensorOutput,
                                int numElements) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < numElements; i += stride) {
        tensorOutput[i] = add_mp(tensorInputA[i], tensorInputB[i]);
    }
}

// 矩阵加法的CUDA内核启动函数模板
template <typename T>
cudaError_t MatrixAddLauncher(const T* tensorInputA, const T* tensorInputB, T* tensorOutput,
                              int numElements, cudaStream_t stream) {
    constexpr int32_t kBLOCK_SIZE = 256;
    int32_t const blocksPerGrid = (numElements + kBLOCK_SIZE - 1) / kBLOCK_SIZE;

    matrixAddKernel<T><<<blocksPerGrid, kBLOCK_SIZE, 0, stream>>>(tensorInputA, tensorInputB,
                                                                  tensorOutput, numElements);

    return cudaGetLastError();
}

// 矩阵加法的调度函数模板
template <typename T>
pluginStatus_t MatrixAddDispatch(void const* tensorInputA, void const* tensorInputB,
                                 void* tensorOutput, int numElements, void* workspace,
                                 cudaStream_t stream) {
    // 清空输出缓冲区（因为并非所有元素都会被内核覆盖，所以需要先清空）
    CSC(cudaMemsetAsync(tensorOutput, 0x00, numElements * sizeof(T), stream), STATUS_FAILURE);

    // Other Buffers Workspace

    // 启动矩阵加法内核
    cudaError_t status = MatrixAddLauncher<T>((T*)tensorInputA, (T*)tensorInputB, (T*)tensorOutput,
                                              numElements, stream);
    CSC(status, STATUS_FAILURE);

    return STATUS_SUCCESS;
}

pluginStatus_t MatrixAddInference(DataType mType, void const* tensorInputA,
                                  void const* tensorInputB, void* tensorOutput, int numElements,
                                  void* workspace, cudaStream_t stream) {
    // 根据数据类型选择执行路径
    if (mType == DataType::kFLOAT) {
        return MatrixAddDispatch<float>(tensorInputA, tensorInputB, tensorOutput, numElements,
                                        workspace, stream);
    } else if (mType == DataType::kHALF) {
        return MatrixAddDispatch<__half>(tensorInputA, tensorInputB, tensorOutput, numElements,
                                         workspace, stream);
    } else {
        return STATUS_NOT_SUPPORTED;
    }
}