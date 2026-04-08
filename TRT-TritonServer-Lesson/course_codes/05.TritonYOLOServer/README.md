# Triton Client for YOLO OBB Demo

## 进入 Docker 容器

假设当前课程正在运行的容器名为 lesson ，你可以通过以下命令打开一个交互式 Bash 终端：

```bash
docker exec -it lesson /bin/bash
```

## 在容器中安装相关依赖

```bash
pip install -r requirements.txt
```

## 构造模型仓库 (`model_repository`)

### 注意事项

- **动态批处理 (Dynamic Batch) 和静态批处理 (Static Batch) 客户端要求**：ONNX 模型的输入 `batchsize` 维度必须为 `-1`（在使用 `torch.onnx.export` 导出模型时，需指定 `dynamic_axes` 参数），即支持动态批处理大小。
- **使用 `trtexec` 构建引擎时**：需要指定最小、最优和最大输入形状。例如：

```bash
/usr/src/tensorrt/bin/trtexec --onnx=yolo11n-obb.onnx --saveEngine=model.plan --staticPlugins=./libcustom_plugins.so --setPluginsToSerialize=./libcustom_plugins.so --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:8x3x640x640
```

### 模型部署目录结构

1. **单请求 (Single Request) 客户端**：
   - 将本地构建的引擎文件重命名为 `model.plan`，并放置在目录 `trt_models/yolo11obb/1/` 中。
   - **注意**：该引擎文件必须在本地构建，且输入 `batchsize` 维度大小任意。

2. **静态批处理 (Static Batch) 客户端**：
   - 将本地构建的引擎文件重命名为 `model.plan`，并放置在目录 `trt_models/yolo11obb/1/` 中。
   - **注意**：该引擎文件必须在本地构建，且输入 `batchsize` 维度为 `-1`。

3. **动态批处理 (Dynamic Batch) 客户端**：
   - 将本地构建的引擎文件重命名为 `model.plan`，并放置在目录 `trt_dy_bs_models/yolo11obb/1/` 中。
   - **注意**：该引擎文件必须在本地构建，且输入 `batchsize` 维度为 `-1`。

综上，无论是运行单请求、静态批处理还是动态批处理，建议基于输入 `batchsize=-1` 的 ONNX 模型构建引擎，并按照上述命令支持动态批处理大小。

## 启动 Triton Server

```bash
/opt/tritonserver/bin/tritonserver --model-repository=/path/to/trt_models
```

## 客户端运行命令

### 注意事项

- **单请求和静态批处理客户端**：共用同一个 `config.pbtxt` 文件。
- **动态批处理客户端**：需要使用不同的 `config.pbtxt` 文件。运行动态批处理时，需重新启动 Triton 服务：

```bash
/opt/tritonserver/bin/tritonserver --model-repository=/path/to/trt_dy_bs_models
```

### 客户端运行命令

1. **单请求客户端**：

```bash
python3 single_request_client.py image /path/to/obb_images/ --model yolo11obb --labels labels-obb.txt
```

2. **静态批处理客户端**：

```bash
python3 static_batch_client.py image /path/to/obb_images/ --model yolo11obb --labels labels-obb.txt
```

3. **动态批处理客户端**：

```bash
python3 dynamic_batch_client.py image /path/to/obb_images/ --model yolo11obb --labels labels-obb.txt
```

---

## TensorRT 模型优化工具的 Fake Quant 日志

```bash
root@machine:/home/TensorRT-Model-Optimizer/examples/onnx_ptq# python -m modelopt.onnx.quantization \
    --onnx_path=./yolo11n-obb.onnx \
    --trt_plugins=/home/chapter3/lib/plugin/libcustom_plugins.so
```

### 日志内容

```
INFO:root:No output path specified, save quantized model to ./yolo11n-obb-fake.quant.onnx
INFO:root:Model with ORT support is saved to ./yolo11n-obb_ort_support.onnx. Model contains custom ops: ['EfficientRotatedNMS_TRT'].
INFO:root:Model ./yolo11n-obb_ort_support.onnx with opset_version 11 is loaded.
INFO:root:Model is cloned to ./yolo11n-obb_opset13.onnx with opset_version 13.
INFO:root:Model is cloned to ./yolo11n-obb_named.onnx after naming the nodes.
INFO:root:Successfully imported the `tensorrt` python package with version 10.8.0.43.
INFO:root:libcudnn*.so* is accessible in /usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib/libcudnn_engines_runtime_compiled.so.9! Please check that this is the correct version needed for your ORT version at [NVIDIA - CUDA](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements).
INFO:root:Quantization Mode: int8
INFO:root:Successfully imported the `tensorrt` python package with version 10.8.0.43.
INFO:root:libcudnn*.so* is accessible in /usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib/libcudnn_engines_runtime_compiled.so.9! Please check that this is the correct version needed for your ORT version at [NVIDIA - CUDA](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements).
INFO:root:Quantizable op types in the model: ['Resize', 'Add', 'Mul', 'MaxPool', 'MatMul', 'Conv']
INFO:root:Building non-residual Add input map ...
INFO:root:Searching for hard-coded patterns like MHA, LayerNorm, etc. to avoid quantization.
INFO:root:Building KGEN/CASK targeted partitions ...
INFO:root:Classifying the partition nodes ...
INFO:root:Successfully imported the `tensorrt` python package with version 10.8.0.43.
INFO:root:libcudnn*.so* is accessible in /usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib/libcudnn_engines_runtime_compiled.so.9! Please check that this is the correct version needed for your ORT version at [NVIDIA - CUDA](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements).
INFO:root:Total number of nodes: 491
WARNING:root:Please consider running pre-processing before quantization. Refer to example: [ONNX Runtime Quantization Example](https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/ReadMe.md).
Collecting tensor data and making histogram ...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 116/116 [00:00<00:00, 385.00it/s]
Finding optimal threshold for each tensor using 'entropy' algorithm ...
Number of tensors : 116
Number of histogram bins : 128 (The number may increase depending on the data collected)
Number of quantized bins : 128
WARNING:root:Please consider pre-processing before quantization. See [ONNX Runtime Quantization Example](https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/ReadMe.md).
INFO:root:Deleting QDQ nodes from marked inputs to make certain operations fusible ...
INFO:root:Total number of quantized nodes: 126
INFO:root:Quantized type counts: {'Conv': 97, 'Add': 15, 'MaxPool': 3, 'Concat': 5, 'Reshape': 1, 'MatMul': 2, 'Resize': 2, 'Sub': 1}
INFO:root:Quantized onnx model is saved as ./yolo11n-obb-fake.quant.onnx
```