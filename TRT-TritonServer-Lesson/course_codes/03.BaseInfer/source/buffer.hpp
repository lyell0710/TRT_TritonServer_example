/**
 * @file buffer.hpp
 * @author laugh12321 (laugh12321@vip.qq.com)
 * @brief 管理内存-显存操作
 * @version 0.1
 * @date 2025-02-23
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <cuda_runtime.h>

#include <memory>

/**
 * @brief 检查 CUDA 错误并处理，通过打印错误消息。
 *
 * 此函数用于验证 CUDA API 调用的结果。如果发生错误，它将输出错误详情，包括文件名、行号和错误描述。
 * 如果检测到 CUDA 错误，程序将终止。
 *
 * @param code CUDA API 调用返回的 CUDA 错误码。
 * @param file 发生错误的文件名。
 * @param line 发生错误的行号。
 */
void checkCudaError(cudaError_t code, const char* file, int line);

/**
 * @brief 宏，用于简化 CUDA 错误检查。
 *
 * 此宏封装了 `checkCudaError` 函数，使检查 CUDA API 调用错误更加方便。如果 CUDA 调用返回错误，
 * 宏会捕获发生错误的文件和行号，并输出错误消息。
 *
 * @param code 要检查错误的 CUDA API 调用。
 */
#define CHECK(code) checkCudaError((code), __FILE__, __LINE__)

namespace deploy {

/**
 * @brief 抽象基类 Buffer，用于管理内存操作
 *
 */
class BaseBuffer {
public:
    virtual ~BaseBuffer() = default;

    /**
     * @brief 分配内存
     *
     * @param size 要分配的内存大小
     */
    virtual void allocate(size_t size) = 0;

    /**
     * @brief 释放内存
     *
     */
    virtual void free() = 0;

    /**
     * @brief 获取设备内存指针
     *
     * @return void* 设备内存指针
     */
    virtual void* device() = 0;

    /**
     * @brief 获取主机内存指针
     *
     * @return void* 主机内存指针
     */
    virtual void* host() = 0;

    /**
     * @brief 获取内存大小
     *
     * @return size_t 内存大小
     */
    virtual size_t size() const = 0;

    /**
     * @brief 从主机到设备拷贝数据
     *
     * @param stream CUDA流
     */
    virtual void hostToDevice(cudaStream_t stream = nullptr) = 0;

    /**
     * @brief 从设备到主机拷贝数据
     *
     * @param stream CUDA流
     */
    virtual void deviceToHost(cudaStream_t stream = nullptr) = 0;
};

/**
 * @brief DeviceBuffer 类，表示设备内存
 *
 */
class DeviceBuffer : public BaseBuffer {
public:
    DeviceBuffer() : size_(0), device_(nullptr) {}
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;
    ~DeviceBuffer() { free(); }

    void   allocate(size_t size) override;
    void   free() override;
    void*  device() override;
    void*  host() override;
    size_t size() const override;
    void   hostToDevice(cudaStream_t stream = nullptr) override;
    void   deviceToHost(cudaStream_t stream = nullptr) override;

private:
    void*  device_;  // 设备内存指针
    size_t size_;    // 内存大小
};

/**
 * @brief DiscreteBuffer 类，表示具有主机和设备内存的分离内存
 *
 */
class DiscreteBuffer : public BaseBuffer {
public:
    DiscreteBuffer() : size_(0), host_(nullptr), device_(nullptr) {}
    DiscreteBuffer(const DiscreteBuffer&)            = delete;
    DiscreteBuffer& operator=(const DiscreteBuffer&) = delete;
    DiscreteBuffer(DiscreteBuffer&& other) noexcept;
    DiscreteBuffer& operator=(DiscreteBuffer&& other) noexcept;
    ~DiscreteBuffer() { free(); }

    void   allocate(size_t size) override;
    void   free() override;
    void*  device() override;
    void*  host() override;
    size_t size() const override;
    void   hostToDevice(cudaStream_t stream = nullptr) override;
    void   deviceToHost(cudaStream_t stream = nullptr) override;

private:
    void*  host_;    // 主机内存指针
    void*  device_;  // 设备内存指针
    size_t size_;    // 内存大小
};

/**
 * @brief Buffer 类型枚举，用于选择不同类型的 Buffer
 *
 */
enum class BufferType {
    Device,    // 设备内存
    Discrete,  // 分离内存（主机和设备都有内存）
};

/**
 * @brief Buffer 工厂类，根据类型创建不同的 Buffer
 *
 */
class BufferFactory {
public:
    /**
     * @brief 创建指定类型的 Buffer
     *
     * @param type 要创建的 Buffer 类型
     * @return 指定类型的 Buffer 智能指针
     */
    static std::unique_ptr<BaseBuffer> createBuffer(BufferType type);
};

}  // namespace deploy
