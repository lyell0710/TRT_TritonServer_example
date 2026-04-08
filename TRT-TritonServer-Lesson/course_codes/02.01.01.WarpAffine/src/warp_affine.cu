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

// 设备函数：将 uchar3 类型的像素值转换为 float3 类型
__device__ float3 uchar3_to_float3(uchar3 v) {
    return make_float3(v.x, v.y, v.z);
}

// CUDA核函数：使用最近邻插值进行仿射变换
__global__ void warp_affine_kernel(uchar3* src, int src_width, int src_height,
                                            float3* dst, int dst_width, int dst_height,
                                            float3 m0, float3 m1) {
    // 计算当前线程处理的像素坐标
    int dx = blockDim.x * blockIdx.x + threadIdx.x;
    int dy = blockDim.y * blockIdx.y + threadIdx.y;
    if (dx >= dst_width || dy >= dst_height) return;

    // 计算源图像中的对应坐标
    float src_x = m0.x * dx + m0.y * dy + m0.z;
    float src_y = m1.x * dx + m1.y * dy + m1.z;

    // 使用最近邻插值，直接取最接近的整数坐标
    int u = __float2int_rd(src_x);  // 向下取整
    int v = __float2int_rd(src_y);  // 向下取整

    float3 border_value = make_float3(114.0f, 114.0f, 114.0f);
    bool out_bounds = (u < 0 || u >= src_width || v < 0 || v >= src_height); // 检查是否超出源图像范围
    dst[dy * dst_width + dx] = out_bounds ? border_value: uchar3_to_float3(src[v * src_width + u]);
}

int main() {
    // 使用OpenCV加载图像
    cv::Mat src_image = cv::imread("./img_example.jpg", cv::IMREAD_COLOR);
    if (src_image.empty()) {
        std::cerr << "无法打开或找到图像！" << std::endl;
        return -1;
    }

    // 获取源图像的宽度和高度
    int src_width = src_image.cols;
    int src_height = src_image.rows;

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
    uchar3* d_src;
    float3* d_dst;
    CHECK_CUDA(cudaMalloc((void**)&d_src, src_width * src_height * sizeof(uchar3)));
    CHECK_CUDA(cudaMalloc((void**)&d_dst, dst_width * dst_height * sizeof(float3)));

    // 将OpenCV图像转换为uchar3格式并复制到设备内存
    uchar3* h_src = (uchar3*)src_image.data;
    CHECK_CUDA(cudaMemcpy(d_src, h_src, src_width * src_height * sizeof(uchar3), cudaMemcpyHostToDevice));

    // 定义CUDA核函数的线程块和网格大小
    dim3 block(16, 16);
    dim3 grid((dst_width + block.x - 1) / block.x, (dst_height + block.y - 1) / block.y);

    // 启动CUDA核函数
    warp_affine_kernel<<<grid, block>>>(d_src, src_width, src_height, d_dst, dst_width, dst_height, m0, m1);

    // 检查核函数是否成功启动
    CHECK_CUDA(cudaGetLastError());

    // 分配主机内存用于存储目标图像
    float3* h_dst = (float3*)malloc(dst_width * dst_height * sizeof(float3));
    CHECK_CUDA(cudaMemcpy(h_dst, d_dst, dst_width * dst_height * sizeof(float3), cudaMemcpyDeviceToHost));

    // 将目标图像转换回OpenCV格式
    cv::Mat dst_image(dst_height, dst_width, CV_32FC3, h_dst);

    // 保存结果图像
    cv::imwrite("output.jpg", dst_image);

    // 释放设备内存
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));

    // 释放主机内存
    free(h_dst);

    return 0;
}