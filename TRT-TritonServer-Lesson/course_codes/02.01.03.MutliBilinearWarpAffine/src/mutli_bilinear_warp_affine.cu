#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

// 定义一个宏来检查CUDA调用是否成功
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


__device__ float3 operator*(float value0, float3 value1) {
    float3 result;
    result.x = value0 * value1.x;
    result.y = value0 * value1.y;
    result.z = value0 * value1.z;

    return result;
}

__device__ float3 operator+(float3& value0, float3& value1) {
    float3 result;
    result.x = value0.x + value1.x;
    result.y = value0.y + value1.y;
    result.z = value0.z + value1.z;

    return result;
}

__device__ void operator+=(float3& result, float3& value) {
    result.x += value.x;
    result.y += value.y;
    result.z += value.z;
}

__device__ float3 operator+=(float3& value0, const float3& value1) {
    value0.x += value1.x;
    value0.y += value1.y;
    value0.z += value1.z;
    return value0;
}

__device__ float3 uchar3_to_float3(uchar3 v) {
    return make_float3(v.x, v.y, v.z);
}

__device__ bool in_bounds(int x, int y, int cols, int rows) {
    return (x >= 0 && x < cols && y >= 0 && y < rows);
}

__device__ void warp_affine_bilinear(const uint8_t* src, const int src_cols, const int src_rows,
                                     float* dst, const int dst_cols, const int dst_rows,
                                     const float3 m0, const float3 m1, int element_x, int element_y) {
    if (element_x >= dst_cols || element_y >= dst_rows) {
        return;
    }

    float2 src_xy = make_float2(
        m0.x * element_x + m0.y * element_y + m0.z,
        m1.x * element_x + m1.y * element_y + m1.z);

    // 获取四个点的坐标 (src_x0, src_y0)、(src_x1, src_y0)、(src_x0, src_y1)、(src_x1, src_y1)
    int src_x0 = __float2int_rd(src_xy.x);
    int src_y0 = __float2int_rd(src_xy.y);
    int src_x1 = src_x0 + 1;
    int src_y1 = src_y0 + 1;

    // 提前计算好差值
    float wx0 = src_x1 - src_xy.x;
    float wx1 = src_xy.x - src_x0;
    float wy0 = src_y1 - src_xy.y;
    float wy1 = src_xy.y - src_y0;

    // 判断是否在范围内
    float3 src_value0, src_value1, value0, value1;
    bool   flag0 = in_bounds(src_x0, src_y0, src_cols, src_rows);
    bool   flag1 = in_bounds(src_x1, src_y0, src_cols, src_rows);
    bool   flag2 = in_bounds(src_x0, src_y1, src_cols, src_rows);
    bool   flag3 = in_bounds(src_x1, src_y1, src_cols, src_rows);

    // 计算四个 f * wx * wy 的值并求和
    float3 border_value = make_float3(114.0f, 114.0f, 114.0f);
    uchar3* input        = (uchar3*)((uint8_t*)src + src_y0 * src_cols * 3);
    src_value0           = flag0 ? uchar3_to_float3(input[src_x0]) : border_value;
    src_value1           = flag1 ? uchar3_to_float3(input[src_x1]) : border_value;
    value0               = wx0 * wy0 * src_value0;
    value1               = wx1 * wy0 * src_value1;
    float3 sum           = value0 + value1;

    input       = (uchar3*)((uint8_t*)src + src_y1 * src_cols * 3);
    src_value0  = flag2 ? uchar3_to_float3(input[src_x0]) : border_value;
    src_value1  = flag3 ? uchar3_to_float3(input[src_x1]) : border_value;
    value0      = wx0 * wy1 * src_value0;
    value1      = wx1 * wy1 * src_value1;
    sum        += value0 + value1;

    float3* dst_float3 = (float3*)dst;
    dst_float3[element_y * dst_cols + element_x] = sum;
}

__global__ void gpuBilinearWarpAffine(const void* src, const int src_cols, const int src_rows,
                                        void* dst, const int dst_cols, const int dst_rows,
                                        const float3 m0, const float3 m1) {
    int element_x = blockDim.x * blockIdx.x + threadIdx.x;
    int element_y = blockDim.y * blockIdx.y + threadIdx.y;

    warp_affine_bilinear(static_cast<const uint8_t*>(src), src_cols, src_rows,
                         static_cast<float*>(dst), dst_cols, dst_rows,
                         m0, m1, element_x, element_y);
}

__global__ void gpuMutliBilinearWarpAffine(const void* src, const int src_cols, const int src_rows,
                                            void* dst, const int dst_cols, const int dst_rows,
                                            const float3 m0, const float3 m1, int num_images) {
    int image_idx = blockIdx.z;
    if (image_idx >= num_images) {
        return;
    }

    int element_x = blockDim.x * blockIdx.x + threadIdx.x;
    int element_y = blockDim.y * blockIdx.y + threadIdx.y;

    warp_affine_bilinear(static_cast<const uint8_t*>(src) + image_idx * src_rows * src_cols * 3,
                         src_cols, src_rows,
                         static_cast<float*>(dst) + image_idx * dst_rows * dst_cols * 3,
                         dst_cols, dst_rows,
                         m0, m1, element_x, element_y);
}


int main() {
    // 定义图片路径
    std::vector<std::string> image_paths = {
        "./input_0.jpg",
        "./input_1.jpg",
        "./input_2.jpg",
        "./input_3.jpg",
        "./input_4.jpg"
    };

    // 获取源图像的宽度和高度
    int src_width = 720;
    int src_height = 1280;

    // 定义目标图像的尺寸
    int dst_width = 640;
    int dst_height = 640;

    // 计算仿射变换矩阵的参数
    double scale  = std::min(static_cast<double>(dst_width) / src_width, static_cast<double>(dst_height) / src_height);
    double offset = 0.5 * scale - 0.5;
    
    double scale_from_width  = -0.5 * scale * src_width;
    double scale_from_height = -0.5 * scale * src_height;
    double half_dst_width    = 0.5 * dst_width;
    double half_dst_height   = 0.5 * dst_height;

    double inv_d = (scale != 0.0) ? 1.0 / (scale * scale) : 0.0;
    double a     = scale * inv_d;

    // 定义仿射变换矩阵的两个行向量
    float3 m0 = make_float3(a, 0.0, -a * (scale_from_width + half_dst_width + offset));
    float3 m1 = make_float3(0.0, a, -a * (scale_from_height + half_dst_height + offset));

    // 分配设备内存用于源图像和目标图像
    uint8_t* d_src;
    float* d_dst;
    CHECK_CUDA(cudaMalloc((void**)&d_src, src_width * src_height * sizeof(uint8_t) * 3 * 5)); // 5张图片
    CHECK_CUDA(cudaMalloc((void**)&d_dst, dst_width * dst_height * sizeof(float) * 3 * 5)); // 5张图片

    // 将5张OpenCV图像转换为uchar3格式并复制到设备内存
    for (int i = 0; i < 5; ++i) {
        cv::Mat src_image = cv::imread(image_paths[i], cv::IMREAD_COLOR);
        if (src_image.empty()) {
            std::cerr << "无法打开或找到图像：" << image_paths[i] << std::endl;
            return -1;
        }
        uint8_t* h_src = (uint8_t*)src_image.data;
        CHECK_CUDA(cudaMemcpy(d_src + i * src_width * src_height * sizeof(uint8_t) * 3, h_src, src_width * src_height * sizeof(uint8_t) * 3, cudaMemcpyHostToDevice));
    }

    // 定义CUDA核函数的线程块和网格大小
    dim3 block(16, 16);
    dim3 grid((dst_width + block.x - 1) / block.x, (dst_height + block.y - 1) / block.y, 5);

    // 启动CUDA核函数
    gpuMutliBilinearWarpAffine<<<grid, block>>>(d_src, src_width, src_height, d_dst, dst_width, dst_height, m0, m1, 5);

    // 检查核函数是否成功启动
    CHECK_CUDA(cudaGetLastError());

    // 分配主机内存用于存储目标图像
    float* h_dst = (float*)malloc(dst_width * dst_height * sizeof(float) * 3 * 5); // 5张图片
    CHECK_CUDA(cudaMemcpy(h_dst, d_dst, dst_width * dst_height * sizeof(float) * 3 * 5, cudaMemcpyDeviceToHost));

    // 将目标图像转换回OpenCV格式并保存
    for (int i = 0; i < 5; ++i) {
        cv::Mat dst_image(dst_height, dst_width, CV_32FC3, h_dst + i * dst_width * dst_height * 3);
        std::string output_path = "output_" + std::to_string(i + 1) + ".jpg";
        cv::imwrite(output_path, dst_image);
    }

    // 释放设备内存
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));

    // 释放主机内存
    free(h_dst);

    return 0;
}