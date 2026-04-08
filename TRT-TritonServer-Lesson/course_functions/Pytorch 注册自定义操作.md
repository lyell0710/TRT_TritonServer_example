# 注册自定义操作

`torch.autograd.Function` 是 PyTorch 中用于定义自定义操作的核心类。它允许用户定义以下关键方法：

- **`forward`**：定义操作的前向逻辑。
- **`backward`**：定义操作的反向逻辑（如果需要自动求导）。
- **`symbolic`**：将自定义操作映射为 ONNX 操作，以便在其他框架中使用。

## 映射自定义操作

在 PyTorch 中，将模型转换为 ONNX 格式时，需要将 PyTorch 的操作转换为 ONNX 的操作。这个过程称为**符号化（Symbolic）**。符号化操作的作用是将 PyTorch 的操作映射为 ONNX 的操作，从而使得模型可以在其他框架中运行（如 TensorRT）。

在 `torch.autograd.Function` 的 `symbolic` 方法中，`g.op` 是一个非常重要的工具。`g` 是一个符号化图（Symbolic Graph）的引用，`g.op` 用于在符号化图中添加自定义的操作。

### `g.op` 语法

**源码**：[jit_utils.py#L57](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/onnx/_internal/jit_utils.py#L57)

```python
def op(
    self,
    opname: str,
    *raw_args: torch.Tensor | _C.Value,
    outputs: int = 1,
    **kwargs,
):
    """Creates an ONNX operator "opname", taking "raw_args" as inputs and "kwargs" as attributes.

    The set of operators and the inputs/attributes they take
    is documented at https://github.com/onnx/onnx/blob/master/docs/Operators.md

    Args:
        opname: The ONNX operator name, e.g., `Abs` or `Add`, or an operator qualified
            with a namespace, e.g., `aten::add`.
        raw_args: The inputs to the operator; usually provided
            as arguments to the `symbolic` definition.
        outputs: The number of outputs this operator returns.
            By default an operator is assumed to return a single output.
            If `outputs` is greater than one, this functions returns a tuple
            of output `Value`, representing each output of the ONNX operator
            in order.
        kwargs: The attributes of the ONNX operator, whose keys are named
            according to the following convention: `alpha_f` indicates
            the `alpha` attribute with type `f`.  The valid type specifiers are
            `f` (float), `i` (int), `s` (string) or `t` (Tensor).  An attribute
            specified with type float accepts either a single float, or a
            list of floats (e.g., you would say `dims_i` for a `dims` attribute
            that takes a list of integers).

    Returns:
        The value representing the single output of this operator (see the `outputs`
        keyword argument for multi-return nodes).
    """
    return _add_op(self, opname, *raw_args, outputs=outputs, **kwargs)
```

**示例**：

```python
g.op(
    'OperatorName',
    input1,
    input2,
    outputs=2,
    attribute1_f=value1,
    attribute2_i=value2,
    ...
)
```

- **`OperatorName`**：ONNX 中的操作名称（如 `Add`、`MatMul` 等）。
- **`input1`, `input2`, `...`**：操作的输入。
- **`outputs=2`**：操作的输出个数。
- **`attribute1_f=value1`, `attribute2_i=value2`, ...**：操作的属性，后缀 `_f` 表示浮点数，`_i` 表示整数，`_s` 表示字符串，`_t` 表示张量。

### `g.op` 的作用

- **映射操作**：将 PyTorch 的操作映射为 ONNX 的操作。
- **自定义操作**：允许用户定义不在 PyTorch 标准操作集中的自定义操作。

## 示例：注册 EfficientNMS_TRT 插件

以注册 [Efficient NMS Plugin](https://github.com/NVIDIA/TensorRT/tree/release/8.5/plugin/efficientNMSPlugin) 为例。

首先定义 `EfficientNMS_TRT` 操作类，插件有两个输入 `boxes` 和 `scores`，代码如下：

```python
class EfficientNMS_TRT(torch.autograd.Function):
    """NMS block for YOLO-fused model for TensorRT."""

    @staticmethod
    def forward(ctx, boxes, scores):
        pass

    @staticmethod
    def symbolic(g, boxes, scores):
        pass
```

### 实现 `forward` 方法

由于该操作类仅用于映射自定义操作而不进行训练，因此可以使用随机数替代实际计算。根据[插件文档](https://github.com/NVIDIA/TensorRT/blob/release/10.7/plugin/efficientNMSPlugin/README.md)，我们获取输入和输出的维度信息，并将插件中的参数作为 `forward` 的参数。

```python
@staticmethod
def forward(
    ctx,
    boxes,
    scores,
    score_threshold: float = 0.25,
    iou_threshold: float = 0.65,
    max_output_boxes: float = 100,
    background_class: int = -1,
    score_activation: bool = False,
    class_agnostic: bool = True,
    box_coding: int = 1,
):
    batch_size, num_boxes, num_classes = scores.shape
    num_dets = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
    det_boxes = torch.randn(batch_size, max_output_boxes, 4, dtype=torch.float32)
    det_scores = torch.randn(batch_size, max_output_boxes, dtype=torch.float32)
    det_classes = torch.randint(0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)

    return num_dets, det_boxes, det_scores, det_classes
```

### 实现 `symbolic` 方法

在 `symbolic` 方法中，我们定义自定义操作。由于插件的返回节点有四个，因此设置 `outputs=4`。插件名称前需加上 `TRT::` 表示这是 TensorRT 的插件。

```python
@staticmethod
def symbolic(
    g,
    boxes,
    scores,
    score_threshold: float = 0.25,
    iou_threshold: float = 0.65,
    max_output_boxes: float = 100,
    background_class: int = -1,
    score_activation: bool = False,
    class_agnostic: bool = True,
    box_coding: int = 1,
):
    return g.op(
        'TRT::EfficientNMS_TRT',
        boxes,
        scores,
        outputs=4,
        score_threshold_f=score_threshold,
        iou_threshold_f=iou_threshold,
        max_output_boxes_i=max_output_boxes,
        background_class_i=background_class,
        score_activation_i=int(score_activation),
        class_agnostic_i=int(class_agnostic),
        box_coding_i=box_coding,
    )
```

### 添加插件版本

插件中还有一个参数 `plugin_version`，用于管理插件的版本。目前 `EfficientNMS_TRT` 只有一个版本，版本号为 `1`。

最终的完整代码如下：

```python
class EfficientNMS_TRT(torch.autograd.Function):
    """NMS block for YOLO-fused model for TensorRT."""

    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        score_threshold: float = 0.25,
        iou_threshold: float = 0.65,
        max_output_boxes: float = 100,
        background_class: int = -1,
        score_activation: int = 0,
        class_agnostic: int = 1,
        box_coding: int = 1,
        plugin_version: str = '1',
    ):
        batch_size, num_boxes, num_classes = scores.shape
        num_dets = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4, dtype=torch.float32)
        det_scores = torch.randn(batch_size, max_output_boxes, dtype=torch.float32)
        det_classes = torch.randint(0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)

        return num_dets, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(
        g,
        boxes,
        scores,
        score_threshold: float = 0.25,
        iou_threshold: float = 0.65,
        max_output_boxes: float = 100,
        background_class: int = -1,
        score_activation: int = 0,
        class_agnostic: int = 1,
        box_coding: int = 1,
        plugin_version: str = '1',
    ):
        return g.op(
            'TRT::EfficientNMS_TRT',
            boxes,
            scores,
            outputs=4,
            score_threshold_f=score_threshold,
            iou_threshold_f=iou_threshold,
            max_output_boxes_i=max_output_boxes,
            background_class_i=background_class,
            score_activation_i=score_activation,
            class_agnostic_i=class_agnostic,
            box_coding_i=box_coding,
            plugin_version_s=plugin_version,
        )
```

### 简化版本

如果实际只需要输入 `boxes`、`scores`、`score_threshold`、`iou_threshold` 和 `max_output_boxes`，可以简化为以下形式：

```python
class EfficientNMS_TRT(torch.autograd.Function):
    """NMS block for YOLO-fused model for TensorRT."""

    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        score_threshold: float = 0.25,
        iou_threshold: float = 0.65,
        max_output_boxes: float = 100,
    ):
        batch_size, num_boxes, num_classes = scores.shape
        num_dets = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4, dtype=torch.float32)
        det_scores = torch.randn(batch_size, max_output_boxes, dtype=torch.float32)
        det_classes = torch.randint(0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)

        return num_dets, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(
        g,
        boxes,
        scores,
        score_threshold: float = 0.25,
        iou_threshold: float = 0.65,
        max_output_boxes: float = 100,
    ):
        return g.op(
            'TRT::EfficientNMS_TRT',
            boxes,
            scores,
            outputs=4,
            score_threshold_f=score_threshold,
            iou_threshold_f=iou_threshold,
            max_output_boxes_i=max_output_boxes,
            background_class_i=-1,
            score_activation_i=0,
            class_agnostic_i=1,
            box_coding_i=1,
            plugin_version_s='1',
        )
```

## Detect 检测头加 NMS 插件

**[Ultralytics Detect](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L21) 精简版源码**：

```python
class Detect(nn.Module):
    """YOLO Detect head for detection models."""

    dynamic = False  # 是否强制重建网格
    export = False  # 是否处于导出模式
    shape = None  # 输入特征图的形状
    anchors = torch.empty(0)  # 初始化锚点
    strides = torch.empty(0)  # 初始化步幅
    legacy = False  # 是否兼容 v3/v5/v8/v9 模型

    def __init__(self, nc=80, ch=()):
        """初始化 YOLO 检测层，指定类别数和通道数。"""
        super().__init__()
        self.nc = nc  # 类别数
        self.nl = len(ch)  # 检测层数
        self.reg_max = 16  # DFL 通道数（ch[0] // 16 用于缩放 4/8/12/16/20，对应 n/s/m/l/x 模型）
        self.no = nc + self.reg_max * 4  # 每个锚点的输出数
        self.stride = torch.zeros(self.nl)  # 步幅在构建时计算
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # 通道数
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """拼接并返回预测的边界框和类别概率。"""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        y = self._inference(x)
        return y if self.export else (y, x)

    def _inference(self, x):
        """基于多级特征图解码预测的边界框和类别概率。"""
        # 推理路径
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """初始化 Detect() 的偏置，注意：需要步幅可用。"""
        m = self  # self.model[-1]  # Detect() 模块
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # 名义类别频率
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # 从
            a[-1].bias.data[:] = 1.0  # 边界框
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # 类别 (.01 物体, 80 类别, 640 图像)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # 从
                a[-1].bias.data[:] = 1.0  # 边界框
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # 类别 (.01 物体, 80 类别, 640 图像)

    def decode_bboxes(self, bboxes, anchors, xywh=True):
        """解码边界框。"""
        return dist2bbox(bboxes, anchors, xywh=xywh, dim=1)
```

### 输出维度分析

我们已经知道输出维度为 `(batch, 4+classes, predicts)`，其中：
- `4` 表示锚点坐标 `(x, y, w, h)`。
- `classes` 表示类别的置信度。

这与 `torch.cat((dbox, cls.sigmoid()), 1)` 的输出维度一致。同时，插件要求输入 `boxes` 和 `scores` 的维度分别为：
- `boxes`: `(batch_size, number_boxes, 4)`
- `scores`: `(batch_size, number_boxes, number_classes)`

而 `dbox` 的维度为 `(batch_size, 4, number_boxes)`，`cls.sigmoid()` 的维度为 `(batch_size, number_classes, number_boxes)`。因此，我们需要对维度进行调整。

### 修改后的代码

```python
    ...

    max_det = 100  # 最大检测框数
    iou_thres = 0.45  # IoU 阈值
    conf_thres = 0.25  # 置信度阈值

    def forward(self, x):
        """拼接并返回预测的边界框和类别概率。"""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        dbox, cls = self._inference(x)

        # 使用 transpose 调整维度以适配 EfficientNMS_TRT
        return EfficientNMS_TRT.apply(
            dbox.transpose(1, 2),  # 调整 boxes 维度
            cls.transpose(1, 2),  # 调整 scores 维度
            self.iou_thres,
            self.conf_thres,
            self.max_det,
        )

    def _inference(self, x):
        """基于多级特征图解码预测的边界框和类别概率。"""
        # 推理路径
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return dbox, cls.sigmoid()

    ...
```

## OBB 检测头加 NMS 插件

[Ultralytics OBB 源码](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L200)

OBB 检测头也是同理：

```python
    ...

    max_det = 100  # 最大检测框数
    iou_thres = 0.45  # IoU 阈值
    conf_thres = 0.25  # 置信度阈值

    def forward(self, x):
        """拼接并返回预测的边界框和类别概率。"""
        
        # 与 x = Detect.forward(self, x) 替换
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        dbox, cls = self._inference(x)
        rotated_box = torch.cat([x, angle], 1).transpose(1, 2)
        # 使用 transpose 调整维度以适配 EfficientRotatedNMS_TRT
        return EfficientRotatedNMS_TRT.apply(
            rotated_box,
            cls.sigmoid().transpose(1, 2),
            self.iou_thres,
            self.conf_thres,
            self.max_det,
        )

    def _inference(self, x):
        """基于多级特征图解码预测的边界框和类别概率。"""
        # 推理路径
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return dbox, cls.sigmoid()

    ...
```

## Pose 检测头加 NMS 插件

[Ultralytics Pose 源码](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L230)

Pose 检测头也是同理，只不过我们要根据索引找对应的关键点。

```python
    ...

    max_det = 100  # 最大检测框数
    iou_thres = 0.45  # IoU 阈值
    conf_thres = 0.25  # 置信度阈值

    def forward(self, x):
        """拼接并返回预测的边界框和类别概率。"""
        
        # 与 x = Detect.forward(self, x) 替换
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        dbox, cls = self._inference(x)
        rotated_box = torch.cat([x, angle], 1).transpose(1, 2)
        # 使用 transpose 调整维度以适配 EEfficientIdxNMS_TRT
        num_dets, det_boxes, det_scores, det_classes, det_indices = EfficientIdxNMS_TRT.apply(
            dbox.transpose(1, 2),
            cls.transpose(1, 2),
            self.iou_thres,
            self.conf_thres,
            self.max_det,
        )

        # 生成 batch 维度的索引
        batch_indices = (
            torch.arange(bs, device=det_classes.device, dtype=det_classes.dtype).unsqueeze(1).expand(-1, self.max_det).reshape(-1)
        )
        det_indices = det_indices.view(-1)

        pred_kpt = self.kpts_decode(bs, kpt) # 不变，获取解码后的关键点坐标

        # pred_kpt[batch_indices, det_indices] 即为 NMS 筛选后的关键点信息。 我们在reshape 为 (batch_size, number_boxes, kpt_num, 2 or 3)
        return (
            num_dets,
            det_boxes,
            det_scores,
            det_classes,
            pred_kpt[batch_indices, det_indices].view(bs, self.max_det, self.kpt_shape[0], self.kpt_shape[1]),
        )


    def _inference(self, x):
        """基于多级特征图解码预测的边界框和类别概率。"""
        # 推理路径
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return dbox, cls.sigmoid()

    ...
```


## Segment 检测头加 NMS 插件

[Ultralytics Segment 源码](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L175)

Segment 检测头的实现与 Detect 检测头类似，但稍微复杂一些。以下是具体步骤：

### 步骤概述
1. **注册插件**：与 Detect 检测头相同，首先需要注册插件。
2. **筛选掩码系数**：在获取到 `det_indices` 后，筛选 NMS 后的掩码系数，并将其 reshape 为 `(batch_size, max_boxes, num=32)`。
3. **调整掩码原型图**：将掩码原型图进行 reshape，以便与掩码系数进行矩阵乘法，得到最终的分割掩码。
4. **还原掩码尺寸**：最终的分割掩码尺寸为输入尺寸的 `1/4`，需要将其还原为与输入尺寸一致，方便用户操作。
5. **掩码二值化**：对分割掩码进行二值化处理。

### 实现代码

```python
    ...

    max_det = 100  # 最大检测框数
    iou_thres = 0.45  # IoU 阈值
    conf_thres = 0.25  # 置信度阈值

    def forward(self, x):
        """拼接并返回预测的边界框、类别概率和分割掩码。"""
        p = self.proto(x[0])  # 掩码原型图
        bs, _, mask_h, mask_w = p.shape  # 获取批次大小和掩码尺寸
        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2).permute(0, 2, 1)  # 掩码系数

        # 替换 Detect.forward(self, x) 的逻辑
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        dbox, cls = self._inference(x)
        rotated_box = torch.cat([x, angle], 1).transpose(1, 2)

        # 使用 transpose 调整维度以适配 EfficientIdxNMS_TRT
        num_dets, det_boxes, det_scores, det_classes, det_indices = EfficientIdxNMS_TRT.apply(
            dbox.transpose(1, 2),
            cls.transpose(1, 2),
            self.iou_thres,
            self.conf_thres,
            self.max_det,
        )

        # 生成批次维度的索引
        batch_indices = (
            torch.arange(bs, device=det_classes.device, dtype=det_classes.dtype)
            .unsqueeze(1)
            .expand(-1, self.max_det)
            .reshape(-1)
        )
        det_indices = det_indices.view(-1)

        # 筛选掩码系数并调整形状
        selected_mc = mc[batch_indices, det_indices].view(bs, self.max_det, self.nm)
        masks_protos = p.view(bs, self.nm, mask_h * mask_w)

        # 计算最终的分割掩码
        det_masks = torch.matmul(selected_mc, masks_protos).sigmoid().view(bs, self.max_det, mask_h, mask_w)

        return (
            num_dets,
            det_boxes,
            det_scores,
            det_classes,
            F.interpolate(
                det_masks, size=(mask_h * 4, mask_w * 4), mode="bilinear", align_corners=False
            ).gt_(0.5).to(torch.uint8),  # 还原尺寸并二值化
        )

    def _inference(self, x):
        """基于多级特征图解码预测的边界框和类别概率。"""
        # 推理路径
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return dbox, cls.sigmoid()

    ...
```