/**
 * @file backend.hpp
 * @author laugh12321 (laugh12321@vip.qq.com)
 * @brief 推理后端定义
 * @version 0.1
 * @date 2025-02-23
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <memory>

#include "buffer.hpp"
#include "manager.hpp"
#include "result.hpp"
#include "warpaffine.hpp"

namespace deploy {

/**
 * @brief TensorInfo 结构体，表示张量的信息
 */
struct TensorInfo {
private:
    nvinfer1::DataType dtype_;           // 张量数据类型
    size_t             bytes_;           // 张量大小（字节数）

public:
    std::string                 name;    // 张量名称
    nvinfer1::Dims              shape;   // 张量形状
    bool                        input;   // 是否为输入张量
    std::unique_ptr<BaseBuffer> buffer;  // 张量对应的内存

    /**
     * @brief 构造函数，初始化张量信息
     *
     * @param name 张量名称
     * @param shape 张量形状
     * @param dtype 张量数据类型
     * @param input 是否为输入张量
     * @param buffer_type 缓冲区类型
     */
    TensorInfo(const std::string& name, const nvinfer1::Dims& shape, const nvinfer1::DataType dtype, const bool input, BufferType buffer_type);

    /**
     * @brief 更新张量的大小并分配内存
     */
    void update();

private:
    /**
     * @brief 将数据类型转换为字节数
     *
     * @param dtype 数据类型
     * @return size_t 数据类型对应的字节数
     */
    size_t dtype_to_bytes(nvinfer1::DataType dtype) const;
};

/**
 * @brief 从文件中读取二进制内容
 *
 * @param file 文件路径
 * @param contents 输出内容
 */
void ReadBinaryFromFile(const std::string& file, std::string* contents);

/**
 * @brief InferBackend 类，用于封装 TensorRT 推理逻辑
 */
class InferBackend {
public:
    InferBackend(const std::string& trt_engine_file);
    ~InferBackend();

    // 默认构造函数。
    InferBackend() = default;

    void                          infer(const std::vector<Image>& input_images);
    std::unique_ptr<InferBackend> clone();  // 克隆 InferBackend 对象，用于多上下文推理

    // 公开的成员变量
    cudaStream_t                 stream;             // CUDA 流
    std::vector<TensorInfo>      tensor_infos;       // 张量信息向量
    std::vector<AffineTransform> affine_transforms;  // 仿射变换向量
    int4                         min_shape;          // 最小形状
    int4                         max_shape;          // 最大形状
    bool                         dynamic;            // 是否为动态形状

protected:
    void getTensorInfo();
    void initialize();
    void captureGraph();                                        // *  新增：捕获 Cuda Graph，当模型是静态时
    void dynamicInfer(const std::vector<Image>& input_images);  // *  新增：用于 动态尺寸（batch） 推理
    void staticInfer(const std::vector<Image>& input_images);   // *  新增：用于 静态尺寸（batch） 推理，使用 CUDA Graph 加速

    // 私有成员变量
    std::unique_ptr<TRTManager>   manager_;        // TensorRT 管理器对象的智能指针
    std::unique_ptr<GraphManager> graph_manager_;  // * 新增：CUDA 图管理器智能指针
    std::unique_ptr<BaseBuffer>   inputs_buffer_;  // 输入缓冲区智能指针
    BufferType                    buffer_type_;    // 缓冲区类型
    int                           input_size_;     // 输入大小
    int                           infer_size_;     // 推理大小
};

}  // namespace deploy