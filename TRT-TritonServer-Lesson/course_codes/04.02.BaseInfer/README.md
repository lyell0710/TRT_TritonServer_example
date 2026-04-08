# 模块编译

使用 CMake，可以按照以下步骤操作：

```bash
cmake -S . -B build -DTENSORRT_PATH=/usr/local/tensorrt # 在本docker 环境中， TensorRT 的安装路径在 /usr 目录下
cmake --build build -j$(nproc) --config Release
```

编译完成后，根目录下将生成一个名为 `lib` 的文件夹。`lib` 文件夹下包含一个名为 `deploy` 的动态库文件。

在调用动态库时，只需要引用 `include` 中的 `model.hpp` 和 `result.hpp` 头文件即可。

## 改动

- `manager.hpp` 与 `manager.cpp` 新增 `GraphManager` 类，用于管理 CUDA 图的创建、执行、捕获等操作，可以显著提高一些重复性操作的性能。

- `backend.hpp` 与 `backend.cpp` 进行了对应的修改，新增了 `captureGraph`、 `dynamicInfer` 与 `staticInfer` 私有方法。

请注意以下几点：

- CUDA Graph 仅在静态模型（`staticInfer`）场景下使用时，才能充分发挥其性能优势。虽然在动态模型中也可应用 CUDA Graph，但其性能提升效果较为有限（具体可参考 PPT 内容，因为在动态模型中，要么需要每次重新实例化图，要么需捕获多个不同 batch 的图）。

- 当 Batch 数量为 1 时，使用 CUDA Graph 进行优化的效果并不明显，故建议 Batch 数量至少为 4，以更好地体现 CUDA Graph 的优化价值。