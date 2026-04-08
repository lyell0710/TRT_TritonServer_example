import ctypes

import numpy as np
from cuda import cudart
from dataclasses import dataclass
from typing import Tuple
from time import perf_counter_ns
import tensorrt as trt

__all__ = ["BaseModel"]


def cuda_assert(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t"
        )
    if len(cuda_ret) > 1:
        return cuda_ret[1]
    return None


class HostDeviceMem:
    """主机和设备内存的配对，其中主机内存被包装在 numpy 数组中"""

    def __init__(self, size: int, dtype: np.dtype):
        nbytes = size * dtype.itemsize                                                    # 计算需要分配的字节数
        host_mem = cuda_assert(cudart.cudaMallocHost(nbytes))                             # 分配主机内存
        pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))                 # 定义指向数据类型的指针类型

        self._host = np.ctypeslib.as_array(ctypes.cast(host_mem, pointer_type), (size,))  # 将主机内存包装成 numpy 数组
        self._device = cuda_assert(cudart.cudaMalloc(nbytes))                             # 分配设备内存
        self._nbytes = nbytes                                                             # 保存分配的字节数

    @property
    def host(self) -> np.ndarray:
        return self._host                                                                 # 返回主机内存包装的 numpy 数组

    @host.setter
    def host(self, arr: np.ndarray):
        if arr.size > self.host.size:                                                     # 如果尝试复制的数组大小超过主机内存大小，则抛出错误
            raise ValueError(f"Tried to fit an array of size {arr.size} into host memory of size {self.host.size}")
        np.copyto(self.host[: arr.size], arr.flat, casting='safe')                        # 将输入数组复制到主机内存

    @property
    def device(self) -> int:
        return self._device                                                               # 返回设备内存的指针

    @property
    def nbytes(self) -> int:
        return self._nbytes                                                               # 返回分配的字节数

    def __str__(self):
        return f"Host:\n{self.host}\nDevice:\n{self.device}\nSize:\n{self.nbytes}\n"

    def __repr__(self):
        return self.__str__()

    def free(self):
        cuda_assert(cudart.cudaFreeHost(self.host.ctypes.data))                           # 释放主机内存
        cuda_assert(cudart.cudaFree(self.device))                                         # 释放设备内存


@dataclass
class TensorInfo:
    """
    一个表示有关张量信息的数据类。

    属性：
        name (str): 张量的名称。
        shape (Tuple[int]): 张量的形状。
        input (bool): 表示该张量是否为输入张量。
        memory (HostDeviceMem): 张量的内存信息，包含主机和设备内存。

    注意：
        HostDeviceMem 是内存分配对象类型的占位符。
    """

    name: str
    input: bool
    shape: Tuple[int]
    memory: HostDeviceMem

class BaseModel:
    def __init__(self, engine_file: str) -> None:
        try:
            # 创建一个TensorRT日志记录器用于处理运行时日志消息
            logger = trt.Logger(trt.Logger.WARNING)
            # 初始化TensorRT插件并加载所需的插件库
            trt.init_libnvinfer_plugins(logger, namespace="")
            # 从序列化文件加载TensorRT引擎
            with open(engine_file, 'rb') as f, trt.Runtime(logger) as runtime:
                assert runtime
                engine = runtime.deserialize_cuda_engine(f.read())
            assert engine

            # 创建一个CUDA流用于异步CUDA操作
            self.stream = cuda_assert(cudart.cudaStreamCreate())
            # 创建一个执行上下文用于运行推理操作
            self.context = engine.create_execution_context()

            # 设置输入和输出绑定，准备推理执行的张量信息
            self.tensors = []
            for binding in engine:
                # 检查当前绑定是否为输入张量
                input = engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT
                # 获取张量的形状
                shape = engine.get_tensor_shape(binding)
                # 计算张量的大小（元素数量）
                size = trt.volume(shape)
                # 获取张量的numpy数据类型
                dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(binding)))
                # 分配主机和设备内存缓冲区
                bindingMemory = HostDeviceMem(size, dtype)
                # 添加到tensors列表
                self.tensors.append(
                    TensorInfo(
                        name=binding,
                        shape=shape,
                        input=input,
                        memory=bindingMemory,
                    )
                )

        except Exception as e:
            print(f"Error initializing BaseModel: {e}")
            raise

    def __del__(self) -> None:
        """释放资源"""
        for tensor in self.tensors:
            tensor.memory.free()
        cuda_assert(cudart.cudaStreamDestroy(self.stream))

    def inference(self) -> None:
        """执行推理"""
        try:
            for tensor in self.tensors:
                if tensor.input:
                    cuda_assert(cudart.cudaMemcpyAsync(
                        tensor.memory.device, 
                        tensor.memory.host, 
                        tensor.memory.nbytes, 
                        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, 
                        self.stream
                    ))
                self.context.set_tensor_address(tensor.name, int(tensor.memory.device))

            self.context.execute_async_v3(self.stream)

            for tensor in self.tensors:
                if not tensor.input:
                    cuda_assert(cudart.cudaMemcpyAsync(
                        tensor.memory.host, 
                        tensor.memory.device, 
                        tensor.memory.nbytes, 
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, 
                        self.stream
                    ))

            cuda_assert(cudart.cudaStreamSynchronize(self.stream))

        except Exception as e:
            print(f"Error during inference: {e}")
            raise

    def warmup(self, iters: int = 10) -> None:
        """预热模型"""
        start_time_ns = perf_counter_ns()
        for _ in range(iters):
            self.inference()
        end_time_ns = perf_counter_ns()
        elapsed_time_ms = (end_time_ns - start_time_ns) / 1e6
        print(f"warmup {iters} iters cost {elapsed_time_ms:.2f} ms.")