

# nms plugin

nms中的RotatedBox，体现rotated的地方仅仅在于多了一个r这个角度成员，其它任何成员函数比如算iou都没有用到r，都和不带Rotated的一个逻辑

`box.decode(anchor)` 是一个用于解码检测框信息的操作，在目标检测任务中，模型预测的通常是相对于锚框（anchor）的**偏移量或缩放比例**等信息，**需要通过解码操作将这些预测信息转换为真实的检测框坐标**。下面详细解释不同结构体中 `decode` 方法的含义：

### `RotatedBoxCorner` 结构体的 `decode` 方法

```cpp
__device__ RotatedBoxCorner<T> decode(RotatedBoxCorner<T> anchor) const
{
    return {add_mp(y1, anchor.y1), add_mp(x1, anchor.x1), add_mp(y2, anchor.y2), add_mp(x2, anchor.x2), r};
}
```

- **功能**：该方法用于将预测的 `RotatedBoxCorner` 类型的框信息与锚框信息相加，从而得到真实的检测框坐标。**`RotatedBoxCorner` 结构体表示的是旋转检测框的四个角点坐标（这里可能简化为两个对角点 `(y1, x1)` 和 `(y2, x2)` 以及旋转角度 `r`）**。
- 参数：
  - `anchor`：`RotatedBoxCorner<T>` 类型的锚框对象，包含锚框的四个角点坐标和旋转角度。
- **返回值**：返回一个新的 `RotatedBoxCorner<T>` 对象，**其四个角点坐标是预测框的对应坐标与锚框对应坐标相加的结果，旋转角度保持不变。**

### `RotatedBoxCenterSize` 结构体的 `decode` 方法

```cpp
__device__ RotatedBoxCenterSize<T> decode(RotatedBoxCenterSize<T> anchor) const
{ //此处的x y h w全是模型预测出来的
    return {add_mp(mul_mp(y, anchor.h), anchor.y), add_mp(mul_mp(x, anchor.w), anchor.x),
        mul_mp(anchor.h, exp_mp(h)), mul_mp(anchor.w, exp_mp(w)), r};
}
```

- **功能**：该方法用于将预测的 `RotatedBoxCenterSize` 类型的框信息与锚框信息进行解码操作，得到真实的检测框信息。**`RotatedBoxCenterSize` 结构体表示的是旋转检测框的中心点坐标 `(y, x)`、高度 `h`、宽度 `w` 以及旋转角度 `r`。**

- 参数：

  - `anchor`：`RotatedBoxCenterSize<T>` 类型的锚框对象，包含**锚框的中心点坐标、高度、宽度和旋转角度。**

- 返回值：返回一个新的

  ```
  RotatedBoxCenterSize<T>
  ```

  对象，其中心点坐标、高度和宽度是经过解码计算得到的，旋转角度保持不变。具体计算方式如下：

  - 新的中心点 `y` 坐标：预测的 `y` 值乘以锚框的高度 `anchor.h` 后加上锚框的 `y` 坐标。
  - 新的中心点 `x` 坐标：预测的 `x` 值乘以锚框的宽度 `anchor.w` 后加上锚框的 `x` 坐标。
  - 新的高度：锚框的高度 `anchor.h` 乘以预测高度的指数 `exp_mp(h)`。
  - 新的宽度：锚框的宽度 `anchor.w` 乘以预测宽度的指数 `exp_mp(w)`。

### 上面俩的总结

`box.decode(anchor)` 的作用是将模型预测的相对信息（相对于锚框）转换为真实的检测框信息，通过结合锚框的初始信息和预测的偏移量或缩放比例等信息，得到最终的检测框坐标和尺寸。**不同的框编码方式（如角点坐标和中心点坐标）对应不同的解码方法。**

### nms的输入 `boxesInput`

- 当 `param.shareLocation` 为 `true` 时，形状为 `[batchSize, numAnchors, 1, 4]`，这里的 `4` 就是每个检测框的坐标信息。由于 `shareLocation` 为 `true`，意味着所有类别的检测框共享同一组位置信息，所以第三个维度是 `1`。
- 当 `param.shareLocation` 为 `false` 时，形状为 `[batchSize, numAnchors, numClasses, 4]`，每个类别都有独立的检测框位置信息，所以第三个维度是 `numClasses`，而最后的 `4` 依旧表示每个检测框的坐标信息。

### `anchorsInput`

- 当 `param.shareAnchors` 为 `true` 时，形状为 `[1, numAnchors, 4]`，这里的 `4` 同样表示每个锚框（anchor box）的坐标信息。因为 `shareAnchors` 为 `true`，所有批次共享同一组锚框，所以第一个维度是 `1`。
- 当 `param.shareAnchors` 为 `false` 时，形状为 `[batchSize, numAnchors, 4]`，每个批次都有自己独立的锚框，第一个维度是 `batchSize`，最后的 `4` 还是表示每个锚框的坐标信息。

综上所述，在这些输入形状里，`4` 代表每个检测框或锚框的坐标信息所占用的维度数量，具体坐标表示方式取决于模型的实现和设计。

**anchors用于在decodebox中结合boxesinput根据rotatedBoxCorner或rotatedBoxCenterSize解码出实际的box**

疑问：boxesInput和anchorsInput都是yolo模型的输出？

### 是否分类求nms or iou

EfficientRotatedNMS这个kernel，里面求nms是对同一类别的框求nms，还是对这张图片里的所有框求nms?

https://github.com/laugh12321/TensorRT-YOLO/blob/v6.0.0/plugin/efficientRotatedNMSPlugin/efficientRotatedNMSInference.cu#L271-L273，tensorrt把这个留给了用户来选择

两种情况分析：

1. `param.classAgnostic` 为 `true`（类别无关）

当 `param.classAgnostic` 为 `true` 时，`ignoreClass` 始终为 `true`。这意味着在进行 NMS 计算时，不考虑检测框的类别信息，会对当前批次图片里的所有检测框求 NMS。也就是说，无论这些框属于什么类别，只要它们的交并比（IoU）超过阈值，就会按照 NMS 规则进行筛选，去除重叠度高的框。

优点

- **计算效率高**：由于不考虑类别信息，只对所有框统一进行 NMS 操作，避免了按类别分组处理的额外开销，减少了计算量，从而提高了推理速度。在对实时性要求较高的场景中，如实时视频流的目标检测，这种方法可以显著提升处理效率。
- **简单直接**：实现逻辑相对简单，代码复杂度低，易于理解和维护。

缺点

- **可能误删不同类别重叠框**：当不同类别的检测框存在重叠时，类别无关的 NMS 可能会错误地删除一些实际上有效的框。例如，在一个图像中，一辆汽车和一个行人的检测框部分重叠，如果采用类别无关的 NMS，可能会因为 IoU 超过阈值而误删其中一个框，即使它们属于不同的类别。

适用场景

- **实时性要求高且不同类别重叠较少的场景**：比如一些工业检测场景，不同类型的目标在空间上分布较为分散，很少出现不同类别检测框重叠的情况，此时使用类别无关的 NMS 可以在保证检测精度的前提下，大幅提高检测速度。
- **类别区分不明显或对类别精度要求不高的场景**：例如在一些简单的物体计数任务中，只需要统计目标的总数，而不需要精确区分每个目标的类别，类别无关的 NMS 可以满足需求。

2. `param.classAgnostic` 为 `false`（类别相关）

当 `param.classAgnostic` 为 `false` 时，`ignoreClass` 的值由 `threadClass[tile] == testClass` 决定。这表明在进行 NMS 计算时，只会对同一类别的检测框求 NMS。只有当两个检测框属于同一类别时，才会计算它们之间的 IoU 并进行筛选操作；不同类别的检测框不会相互影响，各自独立进行 NMS 处理。

优点

- **保留不同类别重叠框**：可以避免误删不同类别之间重叠的检测框，因为它只对同一类别的框进行 NMS 操作。这在目标密集且不同类别容易重叠的场景中非常重要，能够更准确地保留所有有效的检测框，提高检测的召回率和精度。
- **更符合实际需求**：在大多数实际的目标检测任务中，不同类别的目标是相互独立的，需要分别进行处理。类别相关的 NMS 更符合这种实际需求，能够提供更准确的检测结果。

缺点

- **计算开销大**：需要按类别对检测框进行分组，然后分别对每个类别进行 NMS 操作，增加了计算复杂度和时间开销。在处理大规模目标检测任务时，可能会导致推理速度明显下降。

适用场景

- **目标密集且不同类别易重叠的场景**：例如城市街道的目标检测，图像中可能包含大量的行人、车辆、自行车等不同类别的目标，它们之间很容易出现重叠。此时使用类别相关的 NMS 可以有效地保留所有有效的检测框，提高检测精度。
- **对类别精度要求高的场景**：如安防监控、自动驾驶等领域，准确区分不同类别的目标至关重要，类别相关的 NMS 能够更好地满足这种需求。

### rotatedNMS中用RotatedBoxCorner还是RotatedBoxCenterSize

这个也是用户选项param.boxCoding来控制，为0的时候是前者，为1的时候是后者

### filter、sort、nms各kernel的输入输出shape和grid block size

1. `EfficientRotatedNMS`

   ```cpp
   unsigned int tileSize = param.numSelectedBoxes / NMS_TILES;
   if (param.numSelectedBoxes <= 512)
   {
       tileSize = 512;
   }
   if (param.numSelectedBoxes <= 256)
   {
       tileSize = 256;// tilesize个box为一个tile
   }
   
   const dim3 blockSize = {tileSize, 1, 1};
   const dim3 gridSize = {1, (unsigned int) param.batchSize, 1};
   ```

- 输入形状：

  - `topNumData`：`[batchSize]`，存储每个批次中选中框的数量。

  - `outputIndexData`：未明确在输入形状中有特殊要求，推测可能是 `[batchSize * numOutputBoxes]`。

  - `outputClassData`：`[batchSize * numClasses]`，用于记录每个批次中每个类别的框数量。

  - `sortedIndexData`：`[batchSize * numScoreElements]`，排序后的索引数据。

  - `sortedScoresData`：`[batchSize * numScoreElements]`，排序后的分数数据。

  - `topClassData`：`[batchSize * numScoreElements]`，每个分数对应的类别。

  - `topAnchorsData`：`[batchSize * numScoreElements]`，每个分数对应的锚点。

  - ```
    boxesInput：
    ```

    - 若 `param.shareLocation` 为 `true`，形状为 `[batchSize, numAnchors, 1, 4]`。4表示x1 y1 x2 y2或者cx cy h w
    - 若 `param.shareLocation` 为 `false`，形状为 `[batchSize, numAnchors, numClasses, 4]`。

  - **疑问：boxesinput和anchorsinput都是yolo模型预测出来的？**

  - ```
    anchorsInput：
    ```

    - 若 `param.shareAnchors` 为 `true`，形状为 `[1, numAnchors, 4]`。
    - 若 `param.shareAnchors` 为 `false`，形状为 `[batchSize, numAnchors, 4]`。

  - `numDetectionsOutput`：`[batchSize]`，存储每个批次中最终检测到的框数量。

  - `nmsScoresOutput`：`[batchSize * numOutputBoxes/*numOutputBoxes为一个预设的超参，表示想要输出多少个框，只用在了nms打头的变量里面，只是nms的过程中，满足条件的框的数量超过了numOutputBoxes*/]`，存储最终筛选后的分数。

  - `nmsClassesOutput`：`[batchSize * numOutputBoxes]`，存储最终筛选后的类别。

  - `nmsBoxesOutput`：`[batchSize * numOutputBoxes]` 个 `RotatedBoxCorner<T>` 对象，存储最终筛选后的框。

- 输出形状：

  - `numDetectionsOutput`：`[batchSize]`，更新每个批次中最终检测到的框数量。
  - `nmsScoresOutput`：`[batchSize * numOutputBoxes]`，存储最终筛选后的分数。
  - `nmsClassesOutput`：`[batchSize * numOutputBoxes]`，存储最终筛选后的类别。
  - `nmsBoxesOutput`：`[batchSize * numOutputBoxes]` 个 `RotatedBoxCorner<T>` 对象，存储最终筛选后的框。

2. `EfficientRotatedNMSFilterSegments`

```cpp
<<<1, param.batchSize, 0, stream>>>
```

- 输入形状：
  - `topNumData`：`[batchSize]`，存储每个批次中选中框的数量。
- 输出形状：
  - `topOffsetsStartData`：`[batchSize]`，存储每个批次的起始偏移量。
  - `topOffsetsEndData`：`[batchSize]`，存储每个批次的结束偏移量。

3. `EfficientRotatedNMSFilter`

```c++
const unsigned int elementsPerBlock = 512;
const unsigned int imagesPerBlock = 1;
const unsigned int elementBlocks = (param.numScoreElements + elementsPerBlock - 1) / elementsPerBlock;
const unsigned int imageBlocks = (param.batchSize + imagesPerBlock - 1) / imagesPerBlock;
const dim3 blockSize = {elementsPerBlock/*一个x thread负责一个score*/, imagesPerBlock/*一个y thread负责一个img*/, 1};
const dim3 gridSize = {elementBlocks/*一个x block负责elementsPerBlock个score*/, imageBlocks/*一个y block负责imagesPerblock个img*/, 1};
```

- 输入形状：
  - `scoresInput`：`[batchSize, numAnchors, numClasses]`，输入的分数数据。
- 输出形状：
  - `topNumData`：`[batchSize]`，更新每个批次中选中框的数量。
  - `topIndexData`：`[batchSize * numScoreElements]`，存储选中框的索引。
  - `topAnchorsData`：`[batchSize * numScoreElements]`，存储选中框对应的锚点。
  - `topScoresData`：`[batchSize * numScoreElements]`，存储选中框的分数。
  - `topClassData`：`[batchSize * numScoreElements]`，存储选中框对应的类别。

4.sort

输入和输出都是scoresDB（topScoresData）和indexDB（topIndexData），shape都是[batchSize * numScoreElements]；`param.batchSize` 是分段数量，`topOffsetsStartData` 和 `topOffsetsEndData` 分别是分段的起始和结束偏移量，即每个图片为一段，即每个图片内部排自己的scores和index

### efficientIdxNMSplugin和efficientRotatedNMSplugin的区别

[TensorRT-YOLO/plugin/efficientIdxNMSPlugin/efficientIdxNMSInference.cu at v6.0.0 · laugh12321/TensorRT-YOLO](https://github.com/laugh12321/TensorRT-YOLO/blob/v6.0.0/plugin/efficientIdxNMSPlugin/efficientIdxNMSInference.cu)

1.在定义box的时候不再是RotatedBoxcorner或者RotatedBoxCentersize，而是BoxCorner和BoxCentersize

### 整个plugin的输入输出

```c++
pluginStatus_t EfficientRotatedNMSInference(EfficientRotatedNMSParameters param, const void* boxesInput, const void* scoresInput,
    const void* anchorsInput, void* numDetectionsOutput, void* nmsBoxesOutput, void* nmsScoresOutput,
    void* nmsClassesOutput, void* workspace, cudaStream_t stream)
```

plugin函数名：EfficientRotatedNMSInference，在xxxplugin.cpp/enqueue里面被call

输入：boxesinput、scoresInput、anchorsInput

输出：numDetectionsOutput、nmsBoxesOutput、nmsScoresOutput、nmsClassesOutput

## 理一下输入输出shape：

1.filter

作用：过滤出所有图像里面大于score_thres的所有score，并且把这些score的anchorId，classId存到了buffer里面去

输入：`scoresinput`，`[batchSize * numScoreElements]=[batchsize * numBoxes * num_classes]`，输入的检测框分数。

输出：`topNumData`(虽然shape是bs，但是每一个值表示一张image filter过滤出来的边界框分数数量) `topOffsetsStartData topOffsetsEndData` (前三个都是bs)` topIndexData  topScoresData topClassData  topAnchorsData`（后四个都是`[bs，num boxes, num classes]`）

豆包回答：

- `topNumData`：形状为 `[batchSize]`，更新每个图像中筛选出的累计检测框分数数量。
- `topIndexData`：形状为 `[batchSize * numScoreElements]`，筛选出的检测框的索引。
- `topAnchorsData`：形状为 `[batchSize * numScoreElements]`，筛选出的检测框对应的锚框索引。
- `topScoresData`：形状为 `[batchSize * numScoreElements]`，筛选出的检测框的分数。
- `topClassData`：形状为 `[batchSize * numScoreElements]`，筛选出的检测框的类别。

1.1 filterSegment

- 输入
  - `topNumData`：形状为 `[batchSize]`，每个元素表示对应图像中筛选出的检测框数量。
- 输出
  - `topOffsetsStartData`：形状为 `[batchSize]`，每个元素表示对应图像筛选出的检测框在 `topIndexData` 等缓冲区中的起始偏移量。
  - `topOffsetsEndData`：形状为 `[batchSize]`，每个元素表示对应图像筛选出的检测框在 `topIndexData` 等缓冲区中的结束偏移量。

2.sort

`scoresDB`（以`topScoresData`来初始化）和`indexDB`（以`topIndexData`来初始化）是存储分数和索引的缓冲区，`param.batchSize * param.numScoreElements` 是要排序的项的总数，`param.batchSize` 是段的数量，

3.nms：排序后，假设两个框，求他们的交并比IOU，如果IOU>iou_thres，那么分数低的框将会被

- 输入

  - `topNumData`：形状为 `[batchSize]`，每个元素表示对应图像中筛选出的检测框数量。
  - `sortedIndexData(indexDB)`：形状为 `[batchSize * numScoreElements]`，排序后的检测框索引。
  - `sortedScoresData(scoresDB)`：形状为 `[batchSize * numScoreElements]`，排序后的检测框分数。
  - `topClassData`：形状为 `[batchSize * numScoreElements]`，筛选出的检测框的类别。
  - `topAnchorsData`：形状为 `[batchSize * numScoreElements]`，筛选出的检测框对应的锚框索引。
  - `boxesInput`，graph上可见的boxinput，filter用到了scoresInput
    - 如果 `param.shareLocation` 为 `true`，形状为 `[batchSize, numAnchors, 1, 4]`。
    - 如果 `param.shareLocation` 为 `false`，形状为 `[batchSize, numAnchors, numClasses, 4]`。
  - `anchorsInput`
    - 如果 `param.shareAnchors` 为 `true`，形状为 `[1, numAnchors, 4]`。
    - 如果 `param.shareAnchors` 为 `false`，形状为 `[batchSize, numAnchors, 4]`。
- 输出

  - `numDetectionsOutput`：形状为 `[batchSize]`，每个元素表示对应图像中最终保留的检测框数量。
  - `nmsScoresOutput`：形状为 `[batchSize * param.numOutputBoxes]`，最终保留的检测框的分数。
  - `nmsClassesOutput`：形状为 `[batchSize * param.numOutputBoxes]`，最终保留的检测框的类别。
  - `nmsBoxesOutput`：形状为 `[batchSize * param.numOutputBoxes]`，最终保留的检测框的信息，以 `RotatedBoxCorner<T>` 格式存储。
  - `outputIndexData(optional,when param.numOutputBoxesPerClass >= 0)`：形状为 `[batchSize * numScoreElements]`，用于存储最终输出检测框的索引。
  - `outputClassData(optional，param.numOutputBoxesPerClass >= 0)`：形状为 `[batchSize * numClasses]`，用于记录每个图像中每个类别的输出检测框数量。

# 注册自定义操作

`torch.autograd.Function` 是 PyTorch 中用于定义自定义操作的核心类。通过继承这个类并重写特定的方法，你可以定义自己的op的前向传播和反向传播逻辑。这种方式允许你创建新的算子，并且让 PyTorch 的自动求导系统能够处理这些自定义操作。它允许用户定义以下关键方法：

- **`forward`**：定义操作的前向逻辑。
- **`backward`**：定义操作的反向逻辑（如果需要自动求导）。
- **`symbolic`**：将自定义操作映射为 ONNX 操作，以便在其他框架中使用。

## 映射自定义操作

在 PyTorch 中，将模型转换为 ONNX 格式时，需要将 PyTorch 的操作转换为 ONNX 的操作。这个过程称为**符号化（Symbolic）**。符号化操作的作用是将 PyTorch 的操作映射为 ONNX 的操作，从而使得模型可以在其他框架中运行（如 TensorRT）。

在 `torch.autograd.Function` 的 `symbolic` 方法中，`g` 是 ONNX  graph生成器，`g.op` 方法用于在 ONNX 图中插入一个新的操作节点

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
    # 添加op到onnx graph
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

由于该操作类仅用于映射自定义操作，因此可以使用随机数替代实际计算。根据[插件文档](https://github.com/NVIDIA/TensorRT/blob/release/10.7/plugin/efficientNMSPlugin/README.md)，我们获取输入和输出的维度信息，并将插件中的参数作为 `forward` 的参数。具体的forwad逻辑后续用 TensorRT plugin来实现

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