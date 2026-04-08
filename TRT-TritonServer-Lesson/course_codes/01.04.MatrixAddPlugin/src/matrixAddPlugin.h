#ifndef TRT_MATRIX_ADD_PLUGIN_H
#define TRT_MATRIX_ADD_PLUGIN_H

#include <NvInferPlugin.h>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "pluginUtils.h"

namespace nvinfer1 {
namespace plugin {

// 定义 矩阵加法插件类
// ! 由于一个 IPluginV3 插件必须具备多个功能，每个功能由一个单独的接口定义
// ! 可以使用组合原则或多重继承来实现插件。
// ! 对于大多数用例来说，多重继承方法更简单
// ! 尤其是在将构建和运行时能力耦合在单个类中是可以容忍的情况下
class MatrixAddPlugin : public IPluginV3,
                        public IPluginV3OneCore,
                        public IPluginV3OneBuild,
                        public IPluginV3OneRuntime {
   public:
    // 默认构造函数
    MatrixAddPlugin() = default;

    // 默认析构函数
    ~MatrixAddPlugin() override = default;

    // IPluginV3 Methods
    // 提供一个指向实现了指定 PluginCapabilityType 的插件对象的指针
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override;
    // 克隆插件实例
    IPluginV3* clone() noexcept override;
    // end of IPluginV3 Methods

    // IPluginV3OneCore Methods
    // 获取插件名称
    char const* getPluginName() const noexcept override;
    // 获取插件版本
    char const* getPluginVersion() const noexcept override;
    // 获取插件命名空间
    char const* getPluginNamespace() const noexcept override;
    // 设置插件命名空间
    void setPluginNamespace(char const* pluginNamespace) noexcept;
    // end of IPluginV3OneCore Methods

    // IPluginV3Build Methods
    // 获取输出节点数量
    int32_t getNbOutputs() const noexcept override;
    // 检查是否支持格式组合
    bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut,
                                   int32_t nbInputs, int32_t nbOutputs) noexcept override;
    // 获取输出形状
    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
                            int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs,
                            IExprBuilder& exprBuilder) noexcept override;
    // 配置插件
    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
                            DynamicPluginTensorDesc const* out,
                            int32_t nbOutputs) noexcept override;
    // 获取工作空间大小 (中间变量的显存大小)
    size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
                            DynamicPluginTensorDesc const* outputs,
                            int32_t nbOutputs) const noexcept override;
    // 获取输出数据类型
    int32_t getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes,
                               int32_t nbInputs) const noexcept override;
    // end IPluginV3Build Methods

    // IPluginV3Runtime Methods
    // 执行插件
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs,
                    void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    // 形状变化时的处理
    int32_t onShapeChange(PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out,
                          int32_t nbOutputs) noexcept override;
    // 附加到上下文
    IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override;
    // 获取要序列化的字段
    PluginFieldCollection const* getFieldsToSerialize() noexcept override;
    // end IPluginV3Runtime Methods

   private:
    // metadata
    std::string mNamespace;   // 插件命名空间
    nvinfer1::DataType mType; // 数据类型

    // serialization data structures
    std::vector<nvinfer1::PluginField> mDataToSerialize; // 要序列化的数据
    nvinfer1::PluginFieldCollection mFCToSerialize;      // 要序列化的字段集合
};

// 定义 矩阵加法插件创建器类
class MatrixAddPluginCreator : public nvinfer1::IPluginCreatorV3One {
   public:
    MatrixAddPluginCreator();
    ~MatrixAddPluginCreator() override = default;
    // 获取插件名称
    char const* getPluginName() const noexcept override;
    // 获取插件版本
    char const* getPluginVersion() const noexcept override;
    // 获取字段名称集合
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    // 创建插件实例
    IPluginV3* createPlugin(char const* name, PluginFieldCollection const* fc,
                            TensorRTPhase phase) noexcept override;
    // 设置插件命名空间
    void setPluginNamespace(char const* libNamespace) noexcept;
    // 获取插件命名空间
    char const* getPluginNamespace() const noexcept override;

   private:
    static PluginFieldCollection mFC;                    // 静态字段集合
    static std::vector<PluginField> mPluginAttributes;   // 插件属性
    std::string mNamespace;                              // 插件命名空间
};

}  // namespace plugin
}  // namespace nvinfer1

#endif  // TRT_MATRIX_ADD_PLUGIN_H