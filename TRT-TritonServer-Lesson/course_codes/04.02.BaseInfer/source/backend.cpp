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

    // 创建 TRTManager 实例
    manager_ = std::make_unique<TRTManager>();

    // 获取 Engine Buffer
    std::string engine_buffer;
    ReadBinaryFromFile(trt_engine_file, &engine_buffer);

    // 调用 initialize 方法进行初始化
    manager_->initialize(engine_buffer.data(), engine_buffer.size());

    // 获取 TensorInfo
    getTensorInfo();

    // 初始化相关变量
    initialize();

    // 捕获 Cuda Graph，当模型是静态时
    if (!dynamic) captureGraph();
}

// 克隆 InferBackend 对象，用于多上下文推理
std::unique_ptr<InferBackend> InferBackend::clone() {
    auto clone_backend = std::make_unique<InferBackend>();  // 创建一个新的InferBackend实例

    cudaSetDevice(0);                                       // 设置CUDA设备为设备0
    CHECK(cudaStreamCreate(&clone_backend->stream));        // 创建一个CUDA流

    clone_backend->manager_ = manager_->clone();

    // 获取Tensor信息
    clone_backend->getTensorInfo();

    // 初始化相关变量
    clone_backend->initialize();

    // 捕获 Cuda Graph，当模型是静态时
    if (!clone_backend->dynamic) clone_backend->captureGraph();

    return clone_backend;  // 返回克隆后的InferBackend实例
}

InferBackend::~InferBackend() {
    std::vector<TensorInfo>().swap(tensor_infos);            // 释放 tensor_infos 的内存
    std::vector<AffineTransform>().swap(affine_transforms);  // 释放 affine_transforms 的内存
    if (!dynamic) graph_manager_->destroy();
    CHECK(cudaStreamDestroy(stream));                        // 销毁 CUDA 流
}

void InferBackend::getTensorInfo() {
    std::vector<TensorInfo>().swap(tensor_infos);                                                            // 清空并释放 tensor_infos 的内存
    buffer_type_     = BufferType::Discrete;                                                                 // 设置缓冲区类型为离散类型
    auto num_tensors = manager_->getNbIOTensors();                                                           // 获取引擎中输入输出张量的数量
    for (auto i = 0; i < num_tensors; ++i) {
        std::string name  = std::string(manager_->getIOTensorName(i));                                       // 获取张量名称
        auto        shape = manager_->getTensorShape(name.c_str());                                          // 获取张量形状
        auto        dtype = manager_->getTensorDataType(name.c_str());                                       // 获取张量数据类型
        bool        input = (manager_->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT);     // 判断张量是否为输入

        if (input) {
            dynamic = std::any_of(shape.d, shape.d + shape.nbDims, [](int val) { return val == -1; });       // 检查张量是否为动态形状
            if (dynamic) {
                shape     = manager_->getProfileShape(name.c_str(), 0, nvinfer1::OptProfileSelector::kMIN);  // 获取动态张量的最小形状
                min_shape = make_int4(shape.d[0], shape.d[1], shape.d[2], shape.d[3]);                       // 将最小形状转换为 int4 类型
                shape     = manager_->getProfileShape(name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);  // 获取动态张量的最大形状
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
    if (!dynamic) inputs_buffer_->allocate(max_shape.x * infer_size_);
}

// *  新增：捕获 Cuda Graph，当模型是静态时
void InferBackend::captureGraph() {
    // 步骤 1: 预推理，确保所有张量分配内存
    // 更新 tensor_info 的 shape 和设备地址
    for (auto& tensor_info : tensor_infos)
        manager_->setTensorAddress(tensor_info.name.c_str(), tensor_info.buffer->device());

    // 执行推理
    if (!manager_->enqueueV3(stream))
        // 执行推理，如果失败则抛出异常
        throw std::runtime_error("captureGraph: EnqueueV3 failed before graph creation.");

    // 同步流，确保所有 CUDA 操作完成
    CHECK(cudaStreamSynchronize(stream));  // 同步 CUDA 流，确保所有操作完成

    // 步骤 2: 定义辅助 Lambda 函数
    // Lambda: 计算输入数据大小和设备指针
    /**
     * @brief 计算指定索引和输入尺寸下的输入数据大小和设备指针。
     *
     * @param idx 输入数据的索引。
     * @param input_width 输入数据的宽度。
     * @param input_height 输入数据的高度。
     * @return std::pair<void*, void*> 输入设备指针和推理设备指针的 pair。
     */
    auto CalcInputDevPtrs = [&](int idx, int input_width, int input_height) {
        // 计算输入数据大小
        size_t input_size   = input_height * input_width * max_shape.y;
        // 计算输入设备指针
        void*  input_device = static_cast<uint8_t*>(inputs_buffer_->device()) + idx * input_size;
        // 计算推理设备指针
        void*  infer_device = static_cast<float*>(tensor_infos.front().buffer->device()) + idx * infer_size_;
        return std::make_pair(input_device, infer_device);
    };

    // Lambda: 执行仿射变换
    /**
     * @brief 对所有输入数据执行仿射变换。
     *
     * @param input_width 输入数据的宽度。
     * @param input_height 输入数据的高度。
     */
    auto ExecWarpAffine = [&](int input_width, int input_height) {
        for (int idx = 0; idx < max_shape.x; ++idx) {
            // 计算输入设备指针和推理设备指针
            auto [input_device, infer_device] = CalcInputDevPtrs(idx, input_width, input_height);
            // 执行仿射变换
            cudaWarpAffine(
                input_device,
                input_height,
                input_width,
                infer_device,
                max_shape.w,
                max_shape.z,
                affine_transforms[idx].matrix,
                stream);
        }
    };

    // 创建 GraphManager 实例
    graph_manager_ = std::make_unique<GraphManager>();

    // 步骤 3: 开始捕获 Cuda Graph
    graph_manager_->beginCapture(stream);

    // 步骤 4: 数据传输和处理
    // 将数据从主机内存拷贝到设备内存
    inputs_buffer_->hostToDevice(stream);

    // 执行仿射变换
    ExecWarpAffine(max_shape.w, max_shape.z);

    // 执行推理
    if (!manager_->enqueueV3(stream))
        // 执行推理，如果失败则抛出异常
        throw std::runtime_error("captureCudaGraph: EnqueueV3 failed when graph creation.");

    // 将输出数据从设备内存拷贝到主机内存
    for (auto& tensor_info : tensor_infos)
        if (!tensor_info.input) tensor_info.buffer->deviceToHost(stream);

    // 步骤 5: 结束捕获和实例化 Cuda Graph
    graph_manager_->endCapture(stream);
    graph_manager_->instantiate(stream);

    // 步骤 6: 获取 Cuda Graph 节点参数
    graph_manager_->getNodesParams(max_shape.x + 1);
}

// *  新增：静态推理函数
void InferBackend::staticInfer(const std::vector<Image>& input_images) {
    auto image_count = input_images.size();  // 获取输入图像的数量

    if (image_count < 1 || image_count > max_shape.x) {
        throw std::invalid_argument("Number of inputs out of range");
    }

    int              total_input_size = 0;
    std::vector<int> individual_input_sizes(image_count);

    // 计算输入大小，并累加总大小
    for (int image_index = 0; image_index < image_count; ++image_index) {
        individual_input_sizes[image_index]  = input_images[image_index].width * input_images[image_index].height * max_shape.y;  // 计算每个输入图像的大小
        total_input_size                    += individual_input_sizes[image_index];                                               // 累加总大小
    }

    // 在主机内存中分配空间并拷贝数据
    inputs_buffer_->allocate(total_input_size);                               // 分配主机内存
    uint8_t* host_input_ptr = static_cast<uint8_t*>(inputs_buffer_->host());  // 获取主机内存指针

    // 拷贝输入数据到主机内存
    for (int image_index = 0; image_index < image_count; ++image_index) {
        std::memcpy(host_input_ptr, input_images[image_index].ptr, individual_input_sizes[image_index]);  // 将输入数据拷贝到主机内存
        host_input_ptr += individual_input_sizes[image_index];                                            // 移动指针到下一个输入数据的位置
    }

    // 更新 Memcpy 节点
    graph_manager_->setMemcpyNodeParams(0, inputs_buffer_->host(), inputs_buffer_->device(), total_input_size);

    // 提前计算 infer_device 基础指针
    float*   base_inference_device_ptr = static_cast<float*>(tensor_infos.front().buffer->device());
    uint8_t* device_input_ptr          = static_cast<uint8_t*>(inputs_buffer_->device());

    for (int image_index = 0; image_index < image_count; ++image_index) {
        affine_transforms[image_index].updateMatrix(input_images[image_index].width, input_images[image_index].height, max_shape.w, max_shape.z);
        // 计算 infer_device_ptr，避免重复计算
        auto inference_device_ptr = base_inference_device_ptr + image_index * infer_size_;

        void* kernel_params[] = {
            (void*)&device_input_ptr,
            (void*)&input_images[image_index].width,
            (void*)&input_images[image_index].height,
            (void*)&inference_device_ptr,
            (void*)&max_shape.w,
            (void*)&max_shape.z,
            (void*)&affine_transforms[image_index].matrix[0],
            (void*)&affine_transforms[image_index].matrix[1]};

        // 判断 idx 更新 kernel 参数
        graph_manager_->setKernelNodeParams(image_index + 1, kernel_params);

        // 更新 input_ptr
        device_input_ptr += individual_input_sizes[image_index];
    }

    // 执行 Cuda Graph
    graph_manager_->launch(stream);

    // 同步流，确保所有 CUDA 操作完成
    CHECK(cudaStreamSynchronize(stream));
}

// *  新增：动态推理函数
void InferBackend::dynamicInfer(const std::vector<Image>& input_images) {
    auto image_count = input_images.size();  // 获取输入图像的数量

    if (image_count < min_shape.x || image_count > max_shape.x) {
        throw std::invalid_argument("Number of inputs out of range");
    }

    // 更新 tensor_info 的 shape 和设备地址
    for (auto& tensor_info : tensor_infos) {
        tensor_info.shape.d[0] = image_count;                                                         // 更新张量的 batch 大小
        tensor_info.update();                                                                         // 如果张量是动态的，更新其形状
        manager_->setTensorAddress(tensor_info.name.c_str(), tensor_info.buffer->device());           // 设置张量的设备地址
        if (tensor_info.input) manager_->setInputShape(tensor_info.name.c_str(), tensor_info.shape);  // 如果张量是动态输入，设置输入形状
    }

    int              total_input_size = 0;
    std::vector<int> individual_input_sizes(image_count);

    // 计算输入大小，并累加总大小
    for (int image_index = 0; image_index < image_count; ++image_index) {
        individual_input_sizes[image_index]  = input_images[image_index].width * input_images[image_index].height * max_shape.y;  // 计算每个输入图像的大小
        total_input_size                    += individual_input_sizes[image_index];                                               // 累加总大小
    }

    // 在主机内存中分配空间并拷贝数据
    inputs_buffer_->allocate(total_input_size);                               // 分配主机内存
    uint8_t* host_input_ptr = static_cast<uint8_t*>(inputs_buffer_->host());  // 获取主机内存指针

    // 拷贝输入数据到主机内存
    for (int image_index = 0; image_index < image_count; ++image_index) {
        std::memcpy(host_input_ptr, input_images[image_index].ptr, individual_input_sizes[image_index]);  // 将输入数据拷贝到主机内存
        host_input_ptr += individual_input_sizes[image_index];                                            // 移动指针到下一个输入数据的位置
    }

    // 拷贝到设备内存
    inputs_buffer_->hostToDevice(stream);  // 将数据从主机内存拷贝到设备内存

    // 提前计算 infer_device 基础指针
    float*   base_inference_device_ptr = static_cast<float*>(tensor_infos.front().buffer->device());
    uint8_t* device_input_ptr          = static_cast<uint8_t*>(inputs_buffer_->device());
    // 在设备内存中进行 WarpAffine 操作
    for (int image_index = 0; image_index < image_count; ++image_index) {
        affine_transforms[image_index].updateMatrix(input_images[image_index].width, input_images[image_index].height, max_shape.w, max_shape.z);
        // 计算 infer_device_ptr，避免重复计算
        auto inference_device_ptr = base_inference_device_ptr + image_index * infer_size_;

        // 在设备内存中执行仿射变换
        cudaWarpAffine(
            device_input_ptr,
            input_images[image_index].width,
            input_images[image_index].height,
            inference_device_ptr,
            max_shape.w,
            max_shape.z,
            affine_transforms[image_index].matrix,
            stream);

        // 更新 input_ptr
        device_input_ptr += individual_input_sizes[image_index];
    }

    // 执行推理
    if (!manager_->enqueueV3(stream))
        // 执行推理，如果失败则抛出异常
        throw std::runtime_error("dynamicInfer: EnqueueV3 failed.");

    // 将输出数据从设备内存拷贝到主机内存
    for (auto& tensor_info : tensor_infos)
        if (!tensor_info.input) tensor_info.buffer->deviceToHost(stream);

    // 同步流，确保所有 CUDA 操作完成
    CHECK(cudaStreamSynchronize(stream));
}

void InferBackend::infer(const std::vector<Image>& input_images) {
    dynamic ? dynamicInfer(input_images) : staticInfer(input_images);
}

}  // namespace deploy