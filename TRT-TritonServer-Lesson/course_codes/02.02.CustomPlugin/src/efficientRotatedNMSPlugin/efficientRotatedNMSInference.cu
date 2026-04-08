/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "common/bboxUtils.h"
#include "cub/cub.cuh"
#include "cuda_runtime_api.h"

#include "efficientRotatedNMSInference.cuh"
#include "efficientRotatedNMSInference.h"

#define NMS_TILES 5

using namespace nvinfer1;
using namespace nvinfer1::plugin;

// * 改动：更改 iou 计算方式，使用 probiou
template <typename T>
__device__ float IOU(EfficientRotatedNMSParameters param, RotatedBoxCorner<T> box1, RotatedBoxCorner<T> box2)
{
    // ! 无论选择哪种边界框编码方式，IOU始终在 RotatedBoxCorner 编码中进行。
    // 复制边界框，以便可以重新排序而不影响原始边界框。
    RotatedBoxCorner<T> b1 = box1;
    RotatedBoxCorner<T> b2 = box2;
    b1.reorder();
    b2.reorder();
    return RotatedBoxCorner<T>::probiou(b1, b2); // 计算 probiou
}

// 解码边界框
template <typename T, typename Tb>
__device__ RotatedBoxCorner<T> DecodeBoxes(EfficientRotatedNMSParameters param, int boxIdx, int anchorIdx,
    const Tb* __restrict__ boxesInput, const Tb* __restrict__ anchorsInput)
{
    // ! 输入将使用选定的编码格式，解码后的边界框将始终以 RotatedBoxCorner 格式返回
    Tb box = boxesInput[boxIdx];
    // 如果不需要解码，直接返回 RotatedBoxCorner 格式的边界框
    if (!param.boxDecoder)
    {
        return RotatedBoxCorner<T>(box);
    }
    Tb anchor = anchorsInput[anchorIdx];             // 获取对应的锚框
    box.reorder();                                   // 重新排序边界框坐标
    anchor.reorder();                                // 重新排序锚框坐标
    return RotatedBoxCorner<T>(box.decode(anchor));  // 解码边界框并返回
}

// * 与 MapNMSData 完全一致
// 映射 RotatedNMS 数据
template <typename T, typename Tb>
__device__ void MapRotatedNMSData(EfficientRotatedNMSParameters param, int idx, int imageIdx, const Tb* __restrict__ boxesInput,
    const Tb* __restrict__ anchorsInput, const int* __restrict__ topClassData, const int* __restrict__ topAnchorsData,
    const int* __restrict__ topNumData, const T* __restrict__ sortedScoresData, const int* __restrict__ sortedIndexData,
    T& scoreMap, int& classMap, RotatedBoxCorner<T>& boxMap, int& boxIdxMap)
{
    // idx: 当前批次中的NMS边界框索引
    // idxSort: 批处理后的NMS边界框索引，用于索引（已过滤但已排序）的分数缓冲区
    // scoreMap: 当前处理的边界框对应的分数
    // 如果索引超出当前图像的边界框数量，直接返回
    if (idx >= topNumData[imageIdx])
    {
        return;
    }
    int idxSort = imageIdx * param.numScoreElements + idx; // 计算排序后的索引
    scoreMap = sortedScoresData[idxSort];                  // 获取对应的分数

    // idxMap: 重新映射的索引，用于索引（已过滤但未排序）的缓冲区
    // classMap: 当前处理的边界框对应的类别
    // anchorMap: 当前处理的边界框对应的锚框
    int idxMap = imageIdx * param.numScoreElements + sortedIndexData[idxSort]; // 计算重新映射的索引
    classMap = topClassData[idxMap];                                           // 获取对应的类别
    int anchorMap = topAnchorsData[idxMap];                                    // 获取对应的锚框

    // boxIdxMap: 重新映射的索引，用于索引（未过滤且未排序）的边界框输入缓冲区
    boxIdxMap = -1;
    if (param.shareLocation) // 如果共享位置信息，边界框输入的形状为 [batchSize, numAnchors, 1, 4]
    {
        boxIdxMap = imageIdx * param.numAnchors + anchorMap;
    }
    else // 否则，边界框输入的形状为 [batchSize, numAnchors, numClasses, 4]
    {
        int batchOffset = imageIdx * param.numAnchors * param.numClasses;
        int anchorOffset = anchorMap * param.numClasses;
        boxIdxMap = batchOffset + anchorOffset + classMap;
    }
    // anchorIdxMap: 重新映射的索引，用于索引（未过滤且未排序）的锚框输入缓冲区
    int anchorIdxMap = -1;
    if (param.shareAnchors) // 如果共享锚框信息，锚框输入的形状为 [1, numAnchors, 4]
    {
        anchorIdxMap = anchorMap;
    }
    else // 否则，锚框输入的形状为 [batchSize, numAnchors, 4]
    {
        anchorIdxMap = imageIdx * param.numAnchors + anchorMap;
    }
    // boxMap: 当前处理的边界框
    boxMap = DecodeBoxes<T, Tb>(param, boxIdxMap, anchorIdxMap, boxesInput, anchorsInput);
}

// * 与 WriteNMSResult 基本一致
// 写入 RotatedNMS 结果
template <typename T>
__device__ void WriteRotatedNMSResult(EfficientRotatedNMSParameters param, int* __restrict__ numDetectionsOutput,
    T* __restrict__ nmsScoresOutput, int* __restrict__ nmsClassesOutput, RotatedBoxCorner<T>* __restrict__ nmsBoxesOutput,
    T threadScore, int threadClass, RotatedBoxCorner<T> threadBox, int imageIdx, unsigned int resultsCounter)
{
    int outputIdx = imageIdx * param.numOutputBoxes + resultsCounter - 1; // 计算输出索引
    if (param.scoreSigmoid)                                               // 如果分数经过 Sigmoid 处理
    {
        nmsScoresOutput[outputIdx] = sigmoid_mp(threadScore);             // 应用 Sigmoid 函数
    }
    else if (param.scoreBits > 0)                                         // 如果分数经过比特优化处理
    {
        nmsScoresOutput[outputIdx] = add_mp(threadScore, (T) -1);         // 应用比特优化
    }
    else                                                                  // 否则直接使用原始分数
    {
        nmsScoresOutput[outputIdx] = threadScore;
    }
    nmsClassesOutput[outputIdx] = threadClass;                            // 写入类别
    if (param.clipBoxes)                                                  // 如果需要裁剪边界框
    {
        nmsBoxesOutput[outputIdx] = threadBox.clip((T) 0, (T) 1);         // 裁剪边界框
    }
    else                                                                  // 否则直接写入边界框
    {
        nmsBoxesOutput[outputIdx] = threadBox;
    }
    numDetectionsOutput[imageIdx] = resultsCounter;                       // 更新检测到的边界框数量
}

// * 与 EfficientNMS 完全一致
// * 改动：删除 ONNX 插件相关部分
// EfficientRotatedNMS CUDA 内核函数
template <typename T, typename Tb>
__global__ void EfficientRotatedNMS(EfficientRotatedNMSParameters param, const int* topNumData, int* outputIndexData,
    int* outputClassData, const int* sortedIndexData, const T* __restrict__ sortedScoresData,
    const int* __restrict__ topClassData, const int* __restrict__ topAnchorsData, const Tb* __restrict__ boxesInput,
    const Tb* __restrict__ anchorsInput, int* __restrict__ numDetectionsOutput, T* __restrict__ nmsScoresOutput,
    int* __restrict__ nmsClassesOutput, RotatedBoxCorner<T>* __restrict__ nmsBoxesOutput)
{
    unsigned int thread = threadIdx.x;  // 当前线程的索引
    unsigned int imageIdx = blockIdx.y; // 当前图像在批次中的索引
    unsigned int tileSize = blockDim.x; // Tile 大小
    if (imageIdx >= param.batchSize) // 如果图像索引超出批次大小，直接返回
    {
        return;
    }

    int numSelectedBoxes = min(topNumData[imageIdx], param.numSelectedBoxes); // 计算当前图像中选定的边界框数量
    int numTiles = (numSelectedBoxes + tileSize - 1) / tileSize;              // 计算 Tile 的数量
    if (thread >= numSelectedBoxes)   
    {
        return;
    }

    __shared__ int blockState;              // 共享内存中的块状态
    __shared__ unsigned int resultsCounter; // 共享内存中的结果计数器
    if (thread == 0)                        // 如果当前线程是第一个线程，初始化共享变量
    {
        blockState = 0;
        resultsCounter = 0;
    }

    int threadState[NMS_TILES];                 // 每个 Tile 的线程状态
    unsigned int boxIdx[NMS_TILES];             // 每个 Tile 的边界框索引
    T threadScore[NMS_TILES];                   // 每个 Tile 的分数
    int threadClass[NMS_TILES];                 // 每个 Tile 的类别
    RotatedBoxCorner<T> threadBox[NMS_TILES];   // 每个 Tile 的边界框
    int boxIdxMap[NMS_TILES];                   // 每个 Tile 的边界框索引映射
    for (int tile = 0; tile < numTiles; tile++) // 遍历每个 Tile
    {
        threadState[tile] = 0;
        boxIdx[tile] = thread + tile * blockDim.x;
        MapRotatedNMSData<T, Tb>(param, boxIdx[tile], imageIdx, boxesInput, anchorsInput, topClassData, topAnchorsData,
            topNumData, sortedScoresData, sortedIndexData, threadScore[tile], threadClass[tile], threadBox[tile],
            boxIdxMap[tile]);
    }

    // 遍历所有边界框进行 NMS
    for (int i = 0; i < numSelectedBoxes; i++)
    {
        int tile = i / tileSize;

        if (boxIdx[tile] == i)
        {
            // 当前线程是迭代的引导线程，决定其他线程的操作
            if (threadState[tile] == -1)
            {
                // 线程已经死亡，跳过当前迭代
                blockState = -1; // -1 => 通知所有线程跳过迭代
            }
            else if (threadState[tile] == 0)
            {
                // 当前线程仍然存活，需要检查是否满足条件
                if (resultsCounter >= param.numOutputBoxes)
                {
                    blockState = -2; // -2 => 通知所有线程提前退出循环
                }
                else
                {
                    // Thread is still alive, because it has not had a large enough IOU overlap with
                    // any other kept box previously. Therefore, this box will be kept for sure. However,
                    // we need to check against all other subsequent boxes from this position onward,
                    // to see how those other boxes will behave in future iterations.
                    // 当前边界框将被保留，并写入结果
                    blockState = 1;        // +1 => 通知所有线程计算IOU
                    threadState[tile] = 1; // +1 => 标记当前边界框将被保留

                    // If the numOutputBoxesPerClass check is enabled, write the result only if the limit for this
                    // class on this image has not been reached yet. Other than (possibly) skipping the write, this
                    // won't affect anything else in the NMS threading.
                    // 如果启用了每个类别的最大输出框数检查，仅在未达到限制时写入结果
                    bool write = true;
                    if (param.numOutputBoxesPerClass >= 0)
                    {
                        int classCounterIdx = imageIdx * param.numClasses + threadClass[tile];
                        write = (outputClassData[classCounterIdx] < param.numOutputBoxesPerClass);
                        outputClassData[classCounterIdx]++;
                    }
                    if (write)
                    {
                        // This branch is visited by one thread per iteration, so it's safe to do non-atomic increments.
                        resultsCounter++;
                        WriteRotatedNMSResult<T>(param, numDetectionsOutput, nmsScoresOutput, nmsClassesOutput,
                            nmsBoxesOutput, threadScore[tile], threadClass[tile], threadBox[tile], imageIdx,
                            resultsCounter);
                    }
                }
            }
            else
            {
                // 此状态不应被达到，但为了安全起见...
                blockState = 0; // 0 => 通知所有线程不进行任何更新
            }
        }

        __syncthreads();

        if (blockState == -2)
        {
            // This is the signal to exit from the loop.
            // 所有线程提前退出循环
            return;
        }

        if (blockState == -1)
        {
            // This is the signal for all threads to just skip this iteration, as no IOU's need to be checked.
            // 所有线程跳过当前迭代
            continue;
        }

        // Grab a box and class to test the current box against. The test box corresponds to iteration i,
        // therefore it will have a lower index than the current thread box, and will therefore have a higher score
        // than the current box because it's located "before" in the sorted score list.
        T testScore;
        int testClass;
        RotatedBoxCorner<T> testBox;
        int testBoxIdxMap;
        MapRotatedNMSData<T, Tb>(param, i, imageIdx, boxesInput, anchorsInput, topClassData, topAnchorsData, topNumData,
            sortedScoresData, sortedIndexData, testScore, testClass, testBox, testBoxIdxMap);

        for (int tile = 0; tile < numTiles; tile++)
        {
            bool ignoreClass = true;
            if (!param.classAgnostic)
            {
                ignoreClass = threadClass[tile] == testClass;
            }

            // IOU
            if (boxIdx[tile] > i && // 确保测试两个不同的边界框，且当前边界框索引大于测试边界框索引
                boxIdx[tile] < numSelectedBoxes && // 确保当前边界框在选定的边界框范围内
                blockState == 1 &&                 // 允许进行IOU检查
                threadState[tile] == 0 &&          // 确保当前边界框尚未被丢弃或保留
                ignoreClass &&                     // 仅在类别匹配时比较边界框
                lte_mp(threadScore[tile], testScore) && // 确保分数排序顺序正确
                IOU<T>(param, threadBox[tile], testBox) >= param.iouThreshold) // 计算IOU并检查是否超过阈值
            {
                // 当前边界框与测试边界框重叠，丢弃当前边界框
                threadState[tile] = -1; // -1 => 标记当前边界框将被丢弃
            }
        }
    }
}

// * 与 EfficientNMSLauncher 完全一致
// * 改动：删除 ONNX 插件相关部分
// EfficientRotatedNMS 启动函数
template <typename T>
cudaError_t EfficientRotatedNMSLauncher(EfficientRotatedNMSParameters& param, int* topNumData, int* outputIndexData,
    int* outputClassData, int* sortedIndexData, T* sortedScoresData, int* topClassData, int* topAnchorsData,
    const void* boxesInput, const void* anchorsInput, int* numDetectionsOutput, T* nmsScoresOutput,
    int* nmsClassesOutput, void* nmsBoxesOutput, cudaStream_t stream)
{
    unsigned int tileSize = param.numSelectedBoxes / NMS_TILES;
    if (param.numSelectedBoxes <= 512)
    {
        tileSize = 512;
    }
    if (param.numSelectedBoxes <= 256)
    {
        tileSize = 256;
    }

    const dim3 blockSize = {tileSize, 1, 1};
    const dim3 gridSize = {1, (unsigned int) param.batchSize, 1};

    if (param.boxCoding == 0)
    {
        EfficientRotatedNMS<T, RotatedBoxCorner<T>><<<gridSize, blockSize, 0, stream>>>(param, topNumData, outputIndexData,
            outputClassData, sortedIndexData, sortedScoresData, topClassData, topAnchorsData,
            (RotatedBoxCorner<T>*) boxesInput, (RotatedBoxCorner<T>*) anchorsInput, numDetectionsOutput, nmsScoresOutput,
            nmsClassesOutput, (RotatedBoxCorner<T>*) nmsBoxesOutput);
    }
    else if (param.boxCoding == 1)
    {
        // Note that nmsBoxesOutput is always coded as RotatedBoxCorner<T>, regardless of the input coding type.
        EfficientRotatedNMS<T, RotatedBoxCenterSize<T>><<<gridSize, blockSize, 0, stream>>>(param, topNumData, outputIndexData,
            outputClassData, sortedIndexData, sortedScoresData, topClassData, topAnchorsData,
            (RotatedBoxCenterSize<T>*) boxesInput, (RotatedBoxCenterSize<T>*) anchorsInput, numDetectionsOutput, nmsScoresOutput,
            nmsClassesOutput, (RotatedBoxCorner<T>*) nmsBoxesOutput);
    }

    return cudaGetLastError();
}

// * 与 EfficientNMSFilterSegments 完全一致
__global__ void EfficientRotatedNMSFilterSegments(EfficientRotatedNMSParameters param, const int* __restrict__ topNumData,
    int* __restrict__ topOffsetsStartData, int* __restrict__ topOffsetsEndData)
{
    int imageIdx = threadIdx.x;     // 当前图像的索引
    if (imageIdx > param.batchSize) // 如果图像索引超出批次大小，直接返回
    {
        return;
    }
    topOffsetsStartData[imageIdx] = imageIdx * param.numScoreElements;  // 计算起始偏移
    topOffsetsEndData[imageIdx] = imageIdx * param.numScoreElements + topNumData[imageIdx]; // 计算结束偏移
}

// * 与 EfficientNMSFilter 完全一致
// EfficientRotatedNMS 过滤的内核函数
template <typename T>
__global__ void EfficientRotatedNMSFilter(EfficientRotatedNMSParameters param, const T* __restrict__ scoresInput,
    int* __restrict__ topNumData, int* __restrict__ topIndexData, int* __restrict__ topAnchorsData,
    T* __restrict__ topScoresData, int* __restrict__ topClassData)
{
    int elementIdx = blockDim.x * blockIdx.x + threadIdx.x; // 当前元素的索引
    int imageIdx = blockDim.y * blockIdx.y + threadIdx.y;   // 当前图像的索引

    // 边界条件检查
    if (elementIdx >= param.numScoreElements || imageIdx >= param.batchSize)
    {
        return;
    }

    // 分数输入的形状为 [batchSize, numAnchors, numClasses]
    int scoresInputIdx = imageIdx * param.numScoreElements + elementIdx;

    // 对于每个类别，检查其对应的分数是否超过阈值，如果超过则选择该锚框
    T score = scoresInput[scoresInputIdx];
    if (gte_mp(score, (T) param.scoreThreshold))
    {
        // 从元素索引中解包类别和锚框索引
        int classIdx = elementIdx % param.numClasses;
        int anchorIdx = elementIdx / param.numClasses;

        // 如果当前类别是背景类别，忽略它
        if (classIdx == param.backgroundClass)
        {
            return;
        }

        // 使用原子操作找到一个空闲的槽位来写入选定的锚框数据
        if (topNumData[imageIdx] >= param.numScoreElements)
        {
            return;
        }
        int selectedIdx = atomicAdd((unsigned int*) &topNumData[imageIdx], 1);
        if (selectedIdx >= param.numScoreElements)
        {
            topNumData[imageIdx] = param.numScoreElements;
            return;
        }

        // topScoresData / topClassData 的形状为 [batchSize, numScoreElements]
        int topIdx = imageIdx * param.numScoreElements + selectedIdx;

        if (param.scoreBits > 0) // 如果分数经过比特优化处理
        {
            score = add_mp(score, (T) 1);
            if (gt_mp(score, (T) (2.f - 1.f / 1024.f)))
            {
                // 确保递增后的分数适合尾数而不改变指数
                score = (2.f - 1.f / 1024.f);
            }
        }

        topIndexData[topIdx] = selectedIdx; // 写入索引
        topAnchorsData[topIdx] = anchorIdx; // 写入锚框索引
        topScoresData[topIdx] = score;      // 写入分数
        topClassData[topIdx] = classIdx;    // 写入类别
    }
}

// * 与 EfficientNMSDenseIndex 完全一致
// EfficientRotatedNMS 密集索引的内核函数
template <typename T>
__global__ void EfficientRotatedNMSDenseIndex(EfficientRotatedNMSParameters param, int* __restrict__ topNumData,
    int* __restrict__ topIndexData, int* __restrict__ topAnchorsData, int* __restrict__ topOffsetsStartData,
    int* __restrict__ topOffsetsEndData, T* __restrict__ topScoresData, int* __restrict__ topClassData)
{
    int elementIdx = blockDim.x * blockIdx.x + threadIdx.x; // 当前元素的索引
    int imageIdx = blockDim.y * blockIdx.y + threadIdx.y;   // 当前图像的索引

    if (elementIdx >= param.numScoreElements || imageIdx >= param.batchSize)
    {
        return;
    }

    int dataIdx = imageIdx * param.numScoreElements + elementIdx; // 计算数据索引
    int anchorIdx = elementIdx / param.numClasses;                // 计算锚框索引
    int classIdx = elementIdx % param.numClasses;                 // 计算类别索引
    if (param.scoreBits > 0)                                      // 如果分数经过比特优化处理
    {
        T score = topScoresData[dataIdx];
        if (lt_mp(score, (T) param.scoreThreshold))
        {
            score = (T) 1;
        }
        else if (classIdx == param.backgroundClass)
        {
            score = (T) 1;
        }
        else
        {
            score = add_mp(score, (T) 1);
            if (gt_mp(score, (T) (2.f - 1.f / 1024.f)))
            {
                // 确保递增后的分数适合尾数而不改变指数
                score = (2.f - 1.f / 1024.f);
            }
        }
        topScoresData[dataIdx] = score; // 写入分数
    }
    else
    {
        T score = topScoresData[dataIdx];
        if (lt_mp(score, (T) param.scoreThreshold))
        {
            topScoresData[dataIdx] = -(1 << 15);
        }
        else if (classIdx == param.backgroundClass)
        {
            topScoresData[dataIdx] = -(1 << 15);
        }
    }

    topIndexData[dataIdx] = elementIdx;  // 写入索引
    topAnchorsData[dataIdx] = anchorIdx; // 写入锚框索引
    topClassData[dataIdx] = classIdx;    // 写入类别

    if (elementIdx == 0)
    {
        // 饱和计数器
        topNumData[imageIdx] = param.numScoreElements;
        topOffsetsStartData[imageIdx] = imageIdx * param.numScoreElements;
        topOffsetsEndData[imageIdx] = (imageIdx + 1) * param.numScoreElements;
    }
}

// * 与 EfficientNMSFilterLauncher 完全一致
// EfficientRotatedNMS 过滤启动函数
template <typename T>
cudaError_t EfficientRotatedNMSFilterLauncher(EfficientRotatedNMSParameters& param, const T* scoresInput, int* topNumData,
    int* topIndexData, int* topAnchorsData, int* topOffsetsStartData, int* topOffsetsEndData, T* topScoresData,
    int* topClassData, cudaStream_t stream)
{
    const unsigned int elementsPerBlock = 512;
    const unsigned int imagesPerBlock = 1;
    const unsigned int elementBlocks = (param.numScoreElements + elementsPerBlock - 1) / elementsPerBlock;
    const unsigned int imageBlocks = (param.batchSize + imagesPerBlock - 1) / imagesPerBlock;
    const dim3 blockSize = {elementsPerBlock, imagesPerBlock, 1};
    const dim3 gridSize = {elementBlocks, imageBlocks, 1};

    float kernelSelectThreshold = 0.007f;
    if (param.scoreSigmoid)
    {
        // Inverse Sigmoid
        if (param.scoreThreshold <= 0.f)
        {
            param.scoreThreshold = -(1 << 15);
        }
        else
        {
            param.scoreThreshold = logf(param.scoreThreshold / (1.f - param.scoreThreshold));
        }
        kernelSelectThreshold = logf(kernelSelectThreshold / (1.f - kernelSelectThreshold));
        // Disable Score Bits Optimization
        param.scoreBits = -1;
    }

    if (param.scoreThreshold < kernelSelectThreshold)
    {
        // A full copy of the buffer is necessary because sorting will scramble the input data otherwise.
        PLUGIN_CHECK_CUDA(cudaMemcpyAsync(topScoresData, scoresInput,
            param.batchSize * param.numScoreElements * sizeof(T), cudaMemcpyDeviceToDevice, stream));

        EfficientRotatedNMSDenseIndex<T><<<gridSize, blockSize, 0, stream>>>(param, topNumData, topIndexData, topAnchorsData,
            topOffsetsStartData, topOffsetsEndData, topScoresData, topClassData);
    }
    else
    {
        EfficientRotatedNMSFilter<T><<<gridSize, blockSize, 0, stream>>>(
            param, scoresInput, topNumData, topIndexData, topAnchorsData, topScoresData, topClassData);

        EfficientRotatedNMSFilterSegments<<<1, param.batchSize, 0, stream>>>(
            param, topNumData, topOffsetsStartData, topOffsetsEndData);
    }

    return cudaGetLastError();
}

// * 与 EfficientNMSSortWorkspaceSize 完全一致
// 计算 EfficientRotatedNMS 排序工作空间大小
template <typename T>
size_t EfficientRotatedNMSSortWorkspaceSize(int batchSize, int numScoreElements)
{
    size_t sortedWorkspaceSize = 0;
    cub::DoubleBuffer<T> keysDB(nullptr, nullptr);
    cub::DoubleBuffer<int> valuesDB(nullptr, nullptr);
    cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr, sortedWorkspaceSize, keysDB, valuesDB,
        numScoreElements, batchSize, (const int*) nullptr, (const int*) nullptr);
    return sortedWorkspaceSize;
}

// * 与 EfficientIdxNMSWorkspaceSize 完全一致
// 计算 EfficientRotatedNMSWorkspace 空间大小
size_t EfficientRotatedNMSWorkspaceSize(int batchSize, int numScoreElements, int numClasses, DataType datatype)
{
    size_t total = 0;
    const size_t align = 256;
    // 计数器
    // 3 用于过滤
    // 1 用于输出索引
    // C 用于每个类别的最大限制
    size_t size = (3 + 1 + numClasses) * batchSize * sizeof(int);
    total += size + (size % align ? align - (size % align) : 0);
    // 整数缓冲区
    for (int i = 0; i < 4; i++)
    {
        size = batchSize * numScoreElements * sizeof(int);
        total += size + (size % align ? align - (size % align) : 0);
    }
    // 浮点数缓冲区
    for (int i = 0; i < 2; i++)
    {
        size = batchSize * numScoreElements * dataTypeSize(datatype);
        total += size + (size % align ? align - (size % align) : 0);
    }
    // 排序工作空间
    if (datatype == DataType::kHALF)
    {
        size = EfficientRotatedNMSSortWorkspaceSize<__half>(batchSize, numScoreElements);
        total += size + (size % align ? align - (size % align) : 0);
    }
    else if (datatype == DataType::kFLOAT)
    {
        size = EfficientRotatedNMSSortWorkspaceSize<float>(batchSize, numScoreElements);
        total += size + (size % align ? align - (size % align) : 0);
    }

    return total;
}

// * 与 EfficientNMSWorkspace 完全一致
// 获取 EfficientRotatedNMS 工作空间
template <typename T>
T* EfficientRotatedNMSWorkspace(void* workspace, size_t& offset, size_t elements)
{
    T* buffer = (T*) ((size_t) workspace + offset);
    size_t align = 256;
    size_t size = elements * sizeof(T);
    size_t sizeAligned = size + (size % align ? align - (size % align) : 0);
    offset += sizeAligned;
    return buffer;
}

// * 改动：删除 ONNX 插件相关部分
// * 与 EfficientNMSDispatch 完全一致
// EfficientRotatedNMS 调度函数
template <typename T>
pluginStatus_t EfficientRotatedNMSDispatch(EfficientRotatedNMSParameters param, const void* boxesInput, const void* scoresInput,
    const void* anchorsInput, void* numDetectionsOutput, void* nmsBoxesOutput, void* nmsScoresOutput,
    void* nmsClassesOutput, void* workspace, cudaStream_t stream)
{
    // Clear Outputs (not all elements will get overwritten by the kernels, so safer to clear everything out)
    CSC(cudaMemsetAsync(numDetectionsOutput, 0x00, param.batchSize * sizeof(int), stream), STATUS_FAILURE);
    CSC(cudaMemsetAsync(nmsScoresOutput, 0x00, param.batchSize * param.numOutputBoxes * sizeof(T), stream), STATUS_FAILURE);
    CSC(cudaMemsetAsync(nmsBoxesOutput, 0x00, param.batchSize * param.numOutputBoxes * 5 * sizeof(T), stream), STATUS_FAILURE);
    CSC(cudaMemsetAsync(nmsClassesOutput, 0x00, param.batchSize * param.numOutputBoxes * sizeof(int), stream), STATUS_FAILURE);

    // Empty Inputs
    if (param.numScoreElements < 1)
    {
        return STATUS_SUCCESS;
    }

    // Counters Workspace
    size_t workspaceOffset = 0;
    int countersTotalSize = (3 + 1 + param.numClasses) * param.batchSize;
    int* topNumData = EfficientRotatedNMSWorkspace<int>(workspace, workspaceOffset, countersTotalSize);
    int* topOffsetsStartData = topNumData + param.batchSize;
    int* topOffsetsEndData = topNumData + 2 * param.batchSize;
    int* outputIndexData = topNumData + 3 * param.batchSize;
    int* outputClassData = topNumData + 4 * param.batchSize;
    CSC(cudaMemsetAsync(topNumData, 0x00, countersTotalSize * sizeof(int), stream), STATUS_FAILURE);
    cudaError_t status = cudaGetLastError();
    CSC(status, STATUS_FAILURE);

    // Other Buffers Workspace
    int* topIndexData
        = EfficientRotatedNMSWorkspace<int>(workspace, workspaceOffset, param.batchSize * param.numScoreElements);
    int* topClassData
        = EfficientRotatedNMSWorkspace<int>(workspace, workspaceOffset, param.batchSize * param.numScoreElements);
    int* topAnchorsData
        = EfficientRotatedNMSWorkspace<int>(workspace, workspaceOffset, param.batchSize * param.numScoreElements);
    int* sortedIndexData
        = EfficientRotatedNMSWorkspace<int>(workspace, workspaceOffset, param.batchSize * param.numScoreElements);
    T* topScoresData = EfficientRotatedNMSWorkspace<T>(workspace, workspaceOffset, param.batchSize * param.numScoreElements);
    T* sortedScoresData
        = EfficientRotatedNMSWorkspace<T>(workspace, workspaceOffset, param.batchSize * param.numScoreElements);
    size_t sortedWorkspaceSize = EfficientRotatedNMSSortWorkspaceSize<T>(param.batchSize, param.numScoreElements);
    char* sortedWorkspaceData = EfficientRotatedNMSWorkspace<char>(workspace, workspaceOffset, sortedWorkspaceSize);
    cub::DoubleBuffer<T> scoresDB(topScoresData, sortedScoresData);
    cub::DoubleBuffer<int> indexDB(topIndexData, sortedIndexData);

    // Kernels
    status = EfficientRotatedNMSFilterLauncher<T>(param, (T*) scoresInput, topNumData, topIndexData, topAnchorsData,
        topOffsetsStartData, topOffsetsEndData, topScoresData, topClassData, stream);
    CSC(status, STATUS_FAILURE);

    status = cub::DeviceSegmentedRadixSort::SortPairsDescending(sortedWorkspaceData, sortedWorkspaceSize, scoresDB,
        indexDB, param.batchSize * param.numScoreElements, param.batchSize, topOffsetsStartData, topOffsetsEndData,
        param.scoreBits > 0 ? (10 - param.scoreBits) : 0, param.scoreBits > 0 ? 10 : sizeof(T) * 8, stream);
    CSC(status, STATUS_FAILURE);

    status = EfficientRotatedNMSLauncher<T>(param, topNumData, outputIndexData, outputClassData, indexDB.Current(),
        scoresDB.Current(), topClassData, topAnchorsData, boxesInput, anchorsInput, (int*) numDetectionsOutput,
        (T*) nmsScoresOutput, (int*) nmsClassesOutput, nmsBoxesOutput, stream);
    CSC(status, STATUS_FAILURE);

    return STATUS_SUCCESS;
}

// * 改动：删除 ONNX 插件相关部分，删除了 nmsIndicesOutput （onnx 插件只返回索引）
// * 与 EfficientIdxNMSInference 基本一致
// EfficientRotatedNMSInference 推理函数
pluginStatus_t EfficientRotatedNMSInference(EfficientRotatedNMSParameters param, const void* boxesInput, const void* scoresInput,
    const void* anchorsInput, void* numDetectionsOutput, void* nmsBoxesOutput, void* nmsScoresOutput,
    void* nmsClassesOutput, void* workspace, cudaStream_t stream)
{
    if (param.datatype == DataType::kFLOAT)
    {
        param.scoreBits = -1;
        return EfficientRotatedNMSDispatch<float>(param, boxesInput, scoresInput, anchorsInput, numDetectionsOutput,
            nmsBoxesOutput, nmsScoresOutput, nmsClassesOutput, workspace, stream);
    }
    else if (param.datatype == DataType::kHALF)
    {
        if (param.scoreBits <= 0 || param.scoreBits > 10)
        {
            param.scoreBits = -1;
        }
        return EfficientRotatedNMSDispatch<__half>(param, boxesInput, scoresInput, anchorsInput, numDetectionsOutput,
            nmsBoxesOutput, nmsScoresOutput, nmsClassesOutput, workspace, stream);
    }
    else
    {
        return STATUS_NOT_SUPPORTED;
    }
}
