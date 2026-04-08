#ifndef TRT_PLUGIN_REGISTRATION_H
#define TRT_PLUGIN_REGISTRATION_H
#include <NvInferRuntime.h>

#ifdef _MSC_VER
#define TENSORRTAPI __declspec(dllexport)
#else
#define TENSORRTAPI __attribute__((visibility("default")))
#endif

// ! 创建插件共享库时，必须定义以下公共接口

extern "C" TENSORRTAPI void setLoggerFinder(nvinfer1::ILoggerFinder* finder);

extern "C" TENSORRTAPI nvinfer1::IPluginCreatorInterface* const* getCreators(int32_t& nbCreators);

#endif // TRT_PLUGIN_REGISTRATION_H