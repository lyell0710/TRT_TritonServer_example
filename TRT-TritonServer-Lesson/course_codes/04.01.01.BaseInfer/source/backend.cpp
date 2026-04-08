/**
 * @file backend.cpp
 * @author laugh12321 (laugh12321@vip.qq.com)
 * @brief 推理后端实现
 * @version 0.1
 * @date 2025-02-23
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <algorithm>
#include <cstring>
#include <fstream>
#include <numeric>

#include "backend.hpp"
#include "manager.hpp"

namespace deploy {

// TensorInfo 的构造函数实现
TensorInfo::TensorInfo(const std::string& name, const nvinfer1::Dims& shape, const nvinfer1::DataType dtype, const bool input, BufferType buffer_type)
    : name(name), shape(shape), dtype_(dtype), input(input) {
    buffer = BufferFactory::createBuffer(buffer_type);
    update();
}

// 更新张量的大小并分配内存
void TensorInfo::update() {
    bytes_ = std::accumulate(shape.d, shape.d + shape.nbDims, 1, std::multiplies<int>()) * dtype_to_bytes(dtype_);
    buffer->allocate(bytes_);
}

// 将数据类型转换为字节数
size_t TensorInfo::dtype_to_bytes(nvinfer1::DataType dtype) const {
    switch (dtype) {
        case nvinfer1::DataType::kINT32:
        case nvinfer1::DataType::kFLOAT:
            return 4U;
        case nvinfer1::DataType::kHALF:
            return 2U;
        case nvinfer1::DataType::kBOOL:
        case nvinfer1::DataType::kUINT8:
        case nvinfer1::DataType::kINT8:
        case nvinfer1::DataType::kFP8:
            return 1U;
        default:
            throw std::runtime_error("Unsupported data type");
    }
}

// 从文件中读取二进制内容
void ReadBinaryFromFile(const std::string& file, std::string* contents) {
    std::ifstream fin(file, std::ios::in | std::ios::binary);
    if (!fin.is_open()) {
        throw std::runtime_error("Failed to open file: " + file + " to read.");
    }
    fin.seekg(0, std::ios::end);
    contents->clear();
    contents->resize(fin.tellg());
    fin.seekg(0, std::ios::beg);
    fin.read(&(contents->at(0)), contents->size());
    fin.close();
}

InferBackend::InferBackend(const std::string& trt_engine_file) {
    cudaSetDevice(0);                  // 设置 CUDA 设备为设备 0
    CHECK(cudaStreamCreate(&stream));  // 创建 CUDA 流

    // 获取 Engine Buffer
    std::string engine_buffer;
    ReadBinaryFromFile(trt_engine_file, &engine_buffer);

    // 反序列化 engine 并持有其引用
    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
        TRTManager::getRuntime()->deserializeCudaEngine(engine_buffer.data(), engine_buffer.size()),
        TRTManager::engineDeleter);
    if (!engine_) throw std::runtime_error("Failed to call deserializeCudaEngine().");

    // 创建 context
    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    if (!context_) throw std::runtime_error("Failed to call createExecutionContext().");

    // 获取 TensorInfo
    getTensorInfo();

    // 初始化相关变量
    initialize();
}

// *  新增：克隆 InferBackend 对象，用于多上下文推理
std::unique_ptr<InferBackend> InferBackend::clone() {
    auto clone_backend = std::make_unique<InferBackend>();                     // 创建一个新的InferBackend实例

    cudaSetDevice(0);                                                          // 设置CUDA设备为设备0
    CHECK(cudaStreamCreate(&clone_backend->stream));                           // 创建一个CUDA流

    clone_backend->engine_  = engine_;                                         // 共享当前引擎指针
    clone_backend->context_ = std::unique_ptr<nvinfer1::IExecutionContext>(
        clone_backend->engine_->createExecutionContext());                     // 为共享引擎创建一个新的上下文
    if (!clone_backend->context_) {
        throw std::runtime_error("Failed to call createExecutionContext().");  // 如果创建失败，抛出异常
    }

    // 获取Tensor信息
    clone_backend->getTensorInfo();

    // 初始化相关变量
    clone_backend->initialize();

    return clone_backend;  // 返回克隆后的InferBackend实例
}

InferBackend::~InferBackend() {
    std::vector<TensorInfo>().swap(tensor_infos);            // 释放 tensor_infos 的内存
    std::vector<AffineTransform>().swap(affine_transforms);  // 释放 affine_transforms 的内存
    CHECK(cudaStreamDestroy(stream));                        // 销毁 CUDA 流
}

void InferBackend::getTensorInfo() {
    std::vector<TensorInfo>().swap(tensor_infos);                                                           // 清空并释放 tensor_infos 的内存
    buffer_type_     = BufferType::Discrete;                                                                // 设置缓冲区类型为离散类型
    auto num_tensors = engine_->getNbIOTensors();                                                           // 获取引擎中输入输出张量的数量
    for (auto i = 0; i < num_tensors; ++i) {
        std::string name  = std::string(engine_->getIOTensorName(i));                                       // 获取张量名称
        auto        shape = engine_->getTensorShape(name.c_str());                                          // 获取张量形状
        auto        dtype = engine_->getTensorDataType(name.c_str());                                       // 获取张量数据类型
        bool        input = (engine_->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT);     // 判断张量是否为输入

        if (input) {
            dynamic = std::any_of(shape.d, shape.d + shape.nbDims, [](int val) { return val == -1; });      // 检查张量是否为动态形状
            if (dynamic) {
                shape     = engine_->getProfileShape(name.c_str(), 0, nvinfer1::OptProfileSelector::kMIN);  // 获取动态张量的最小形状
                min_shape = make_int4(shape.d[0], shape.d[1], shape.d[2], shape.d[3]);                      // 将最小形状转换为 int4 类型
                shape     = engine_->getProfileShape(name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);  // 获取动态张量的最大形状
                // 打印接受范围
            }
            max_shape = make_int4(shape.d[0], shape.d[1], shape.d[2], shape.d[3]);                        // 将最大形状转换为 int4 类型
        } else if (!input && dynamic) {
            shape.d[0] = max_shape.x;                                                                     // 对于动态输出张量，设置其形状为最大形状 // ! 为了提前按最大尺寸分配空间，防止频繁销毁创建
        }
        tensor_infos.emplace_back(name, shape, dtype, input, input ? BufferType::Device : buffer_type_);  // 将张量信息添加到 tensor_infos 中 // ! 如果是输入节点，则 buffer 为 BufferType::Device，不需要分配 Host 内存
    }
}

void InferBackend::initialize() {
    std::vector<AffineTransform>().swap(affine_transforms);      // 清空并释放 affine_transforms 的内存
    inputs_buffer_ = BufferFactory::createBuffer(buffer_type_);  // 创建输入缓冲区
    infer_size_    = max_shape.y * max_shape.w * max_shape.z;    // 计算推理时每个输入的大小
    affine_transforms.resize(max_shape.x, AffineTransform());    // 根据最大输入数量调整 affine_transforms 的大小
}

void InferBackend::infer(const std::vector<Image>& inputs) {
    auto num = inputs.size();  // 获取输入图像的数量

    if (num < (dynamic ? min_shape.x : 1) || num > max_shape.x) {
        throw std::invalid_argument("Number of inputs out of range");
    }

    // 更新 tensor_info 的 shape 和设备地址
    for (auto& tensor_info : tensor_infos) {
        tensor_info.shape.d[0] = num;                                                        // 更新张量的 batch 大小
        if (dynamic) tensor_info.update();                                                   // 如果张量是动态的，更新其形状
        context_->setTensorAddress(tensor_info.name.c_str(), tensor_info.buffer->device());  // 设置张量的设备地址
        if (tensor_info.input && dynamic) {
            context_->setInputShape(tensor_info.name.c_str(), tensor_info.shape);            // 如果张量是动态输入，设置输入形状
        }
    }

    int              total_size = 0;
    std::vector<int> input_sizes(num);

    // 计算输入大小，并累加总大小
    for (int idx = 0; idx < num; ++idx) {
        input_sizes[idx]  = inputs[idx].width * inputs[idx].height * max_shape.y;                              // 计算每个输入图像的大小
        total_size       += input_sizes[idx];                                                                  // 累加总大小
        affine_transforms[idx].updateMatrix(inputs[idx].width, inputs[idx].height, max_shape.w, max_shape.z);  // 更新仿射变换矩阵
    }

    // 在主机内存中分配空间并拷贝数据
    inputs_buffer_->allocate(total_size);                                 // 分配主机内存
    uint8_t* input_host = static_cast<uint8_t*>(inputs_buffer_->host());  // 获取主机内存指针

    // 拷贝输入数据到主机内存
    for (int idx = 0; idx < num; ++idx) {
        std::memcpy(input_host, inputs[idx].ptr, input_sizes[idx]);  // 将输入数据拷贝到主机内存
        input_host += input_sizes[idx];                              // 移动指针到下一个输入数据的位置
    }

    // 拷贝到设备内存
    inputs_buffer_->hostToDevice(stream);  // 将数据从主机内存拷贝到设备内存

    // 在设备内存中进行 WarpAffine 操作
    uint8_t* input_device = static_cast<uint8_t*>(inputs_buffer_->device());  // 获取设备内存指针
    for (int idx = 0; idx < num; ++idx) {
        cudaWarpAffine(
            input_device,
            inputs[idx].width,
            inputs[idx].height,
            static_cast<float*>(tensor_infos.front().buffer->device()) + idx * infer_size_,
            max_shape.w,
            max_shape.z,
            affine_transforms[idx].matrix,
            stream);                       // 在设备内存中执行仿射变换
        input_device += input_sizes[idx];  // 移动指针到下一个输入数据的位置
    }

    // 推理
    if (!context_->enqueueV3(stream)) {
        throw std::runtime_error("Infer Error.");  // 执行推理，如果失败则抛出异常
    }

    // 数据拷贝从设备到主机
    for (auto& tensor_info : tensor_infos) {
        if (!tensor_info.input) {
            tensor_info.buffer->deviceToHost(stream);  // 将输出数据从设备内存拷贝到主机内存
        }
    }

    // 同步流，确保所有 CUDA 操作完成
    CHECK(cudaStreamSynchronize(stream));  // 同步 CUDA 流，确保所有操作完成
}

}  // namespace deploy