#ifndef TRT_MATRIX_ADD_INFERENCE_H
#define TRT_MATRIX_ADD_INFERENCE_H

#include <NvInferPlugin.h>

#include "pluginUtils.h"

pluginStatus_t MatrixAddInference(nvinfer1::DataType mType, void const* tensorInputA,
                                  void const* tensorInputB, void* tensorOutput, int numElements,
                                  void* workspace, cudaStream_t stream);

#endif