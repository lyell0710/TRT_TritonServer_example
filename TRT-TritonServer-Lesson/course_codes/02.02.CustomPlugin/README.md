# CustomPlugin: 基于 Efficient NMS 改进的 TensorRT 自定义插件

## Efficient Rotated NMS Plugin 主要改进

- **输出节点维度调整**：`detection_boxes` 的维度由 `[batch_size, num_boxes, 4]` 调整为 `[batch_size, num_boxes, 5]`。
- **IoU 计算方式更新**：采用 prob IoU 计算方式，代码实现参考自 [ultralytics - probiou](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L198)。

## Efficient NMS Plugin With Indices 主要改进

- **新增输出节点**：增加了 `detection_indices` 输出节点，用于获取 NMS 处理后的索引值。

## 编译指南

按照以下步骤完成编译：

```bash
cmake -S . -B build -DTENSORRT_PATH="/your/tensorrt/dir" # 在本docker 环境中， TensorRT 的安装路径在 /usr 目录下
cmake --build build -j8 --config Release
```

编译完成后，生成的插件动态库 `custom_plugins` 将位于 `libs` 文件夹中。

## 参考文档

> [TensorRT 8.x - Extending TensorRT with Custom Layers](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/developer-guide/index.html#extending)
