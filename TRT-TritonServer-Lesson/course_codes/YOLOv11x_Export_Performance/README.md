# YOLO 不同导出方式性能分析

本次分析基于官方提供的 [YOLOv11x](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt) 模型。其输入尺寸为 **1x3x640x640**。

### 模型导出方式

1. **yolo11x-official.onnx**  
   使用 Ultralytics 提供的 `yolo` CLI 工具导出的不带 NMS 的 ONNX 模型。导出指令为：
   ```bash
   yolo export model=yolo11x.pt format=onnx
   ```

2. **yolo11x-official-nms.onnx**  
   使用 Ultralytics 提供的 `yolo` CLI 工具导出的带 NMS 的 ONNX 模型。导出指令为：
   ```bash
   yolo export model=yolo11x.pt format=onnx nms=True conf=0.25 iou=0.45 max_det=100
   ```

3. **yolo11x-nms-plugin.onnx**  
   使用课程 **【02-03-YOLO_Detection】** 中的方法导出的带 NMS 插件的 ONNX 模型。

### 模型配置

导出模型均采用以下统一的 NMS 参数：
- **置信度阈值 (Confidence Threshold)**: 0.25  
- **交并比阈值 (IoU Threshold)**: 0.45  
- **最大检测数量 (Max Detections)**: 100  

> [!NOTE]
>
> `YOLOv11x_Export_Performance.zip.00x` 是预先导出的三种 ONNX 模型的压缩包。由于 GitHub 对文件上传大小有限制，因此该文件被分卷压缩。