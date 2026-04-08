#include "matrixAddInference.h"
#include "matrixAddPlugin.h"

using namespace nvinfer1;
using namespace nvinfer1::plugin;

// Clip plugin specific constants
namespace {
char const* const kMATRIX_ADD_PLUGIN_VERSION{"1"};
char const* const kMATRIX_ADD_PLUGIN_NAME{"MatrixAdd"};
}  // namespace

// Static class fields initialization
PluginFieldCollection MatrixAddPluginCreator::mFC{};
std::vector<PluginField> MatrixAddPluginCreator::mPluginAttributes;

// 注册插件创建器类
REGISTER_TENSORRT_PLUGIN(MatrixAddPluginCreator);

// 提供一个指向实现了指定 PluginCapabilityType 的插件对象的指针
// ! 对于每个 PluginCapabilityType ，必须通过对应的能力接口进行类型转换，以消除编译器的歧义
IPluginCapability* MatrixAddPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept {
    try {
        if (type == PluginCapabilityType::kBUILD) {
            return static_cast<IPluginV3OneBuild*>(this);
        }
        if (type == PluginCapabilityType::kRUNTIME) {
            return static_cast<IPluginV3OneRuntime*>(this);
        }
        PLUGIN_ASSERT(type == PluginCapabilityType::kCORE);
        return static_cast<IPluginV3OneCore*>(this);
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return nullptr;
}

// 克隆插件实例
IPluginV3* MatrixAddPlugin::clone() noexcept {
    try {
        auto* p = new MatrixAddPlugin();
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return nullptr;
}

// 获取插件名称
char const* MatrixAddPlugin::getPluginName() const noexcept { return kMATRIX_ADD_PLUGIN_NAME; }

// 获取插件版本
char const* MatrixAddPlugin::getPluginVersion() const noexcept {
    return kMATRIX_ADD_PLUGIN_VERSION;
}

// 获取插件命名空间
char const* MatrixAddPlugin::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

// 设置插件命名空间
void MatrixAddPlugin::setPluginNamespace(char const* libNamespace) noexcept {
    try {
        mNamespace = libNamespace;
    } catch (std::exception const& e) {
        caughtError(e);
    }
}

// ** 获取输出节点数量
int32_t MatrixAddPlugin::getNbOutputs() const noexcept { return 1; }

// ** 检查是否支持格式组合
// ** 如果插件支持由 pos 索引的 input/output 的格式和数据类型，则返回 true。
// ! TensorRT 使用 supportsFormatCombination 来询问插件是否接受在给定位置 pos 以及给定格式/类型组合的连接，以及较低索引连接的格式/类型。
// ! 接口将输入/输出统一索引为连接，从 0 开始索引第一个输入，然后按顺序索引其余输入，接着编号输出。在示例中，输入是连接 0, 1 ，输出是连接 2 。
bool MatrixAddPlugin::supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut,
                                                int32_t nbInputs, int32_t nbOutputs) noexcept {
    try {
        PLUGIN_VALIDATE(inOut != nullptr);
        PLUGIN_VALIDATE(nbInputs == 2);
        PLUGIN_VALIDATE(nbOutputs == 1);
        PLUGIN_VALIDATE(pos >= 0 && pos < (nbInputs + nbOutputs));

        // all inputs/outputs: fp32 or fp16
        return (inOut[pos].desc.type == DataType::kHALF ||
                inOut[pos].desc.type == DataType::kFLOAT) &&
               (inOut[0].desc.type == inOut[pos].desc.type);
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return false;
}

// ** 获取输出形状
int32_t MatrixAddPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
                                         DimsExprs const* shapeInputs, int32_t nbShapeInputs,
                                         DimsExprs* outputs, int32_t nbOutputs,
                                         IExprBuilder& exprBuilder) noexcept {
    try {
        PLUGIN_VALIDATE(inputs != nullptr);
        PLUGIN_VALIDATE(nbInputs == 2);
        PLUGIN_VALIDATE(inputs[0].nbDims == inputs[1].nbDims);
        outputs[0] = inputs[0];
        return pluginStatus_t::STATUS_SUCCESS;
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return pluginStatus_t::STATUS_FAILURE;
}

// ** 配置插件
int32_t MatrixAddPlugin::configurePlugin(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
                                         DynamicPluginTensorDesc const* outputs,
                                         int32_t nbOutputs) noexcept {
    try {
        PLUGIN_ASSERT(nbInputs == 2);
        PLUGIN_ASSERT(nbOutputs == 1);

        mType = inputs[0].desc.type;
        return pluginStatus_t::STATUS_SUCCESS;
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return pluginStatus_t::STATUS_FAILURE;
}

// ** 获取工作空间大小 (中间变量的显存大小)
size_t MatrixAddPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
                                         DynamicPluginTensorDesc const* outputs,
                                         int32_t nbOutputs) const noexcept {
    return 0;
}

// ** 获取输出数据类型
int32_t MatrixAddPlugin::getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs,
                                            DataType const* inputTypes,
                                            int32_t nbInputs) const noexcept {
    try {
        PLUGIN_VALIDATE(outputTypes != nullptr);
        PLUGIN_VALIDATE(nbOutputs == 1);
        PLUGIN_VALIDATE(inputTypes != nullptr);
        PLUGIN_VALIDATE(nbInputs == 2);
        outputTypes[0] = inputTypes[0];
        return pluginStatus_t::STATUS_SUCCESS;
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return pluginStatus_t::STATUS_FAILURE;
}

// ** 执行插件
int32_t MatrixAddPlugin::enqueue(PluginTensorDesc const* inputDesc,
                                 PluginTensorDesc const* outputDesc, void const* const* inputs,
                                 void* const* outputs, void* workspace,
                                 cudaStream_t stream) noexcept {
    int32_t status = -1;
    try {
        PLUGIN_VALIDATE(inputDesc != nullptr && outputDesc != nullptr && inputs != nullptr &&
                        outputs != nullptr);

        int32_t const inputVolume = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];
        void const* const tensorInputA = inputs[0];
        void const* const tensorInputB = inputs[1];
        void* tensorOutput = outputs[0];

        return MatrixAddInference(mType, tensorInputA, tensorInputB, tensorOutput, inputVolume,
                                  workspace, stream);
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return status;
}

// ** 形状变化时的处理
int32_t MatrixAddPlugin::onShapeChange(PluginTensorDesc const* inputs, int32_t nbInputs,
                                       PluginTensorDesc const* outputs,
                                       int32_t nbOutputs) noexcept {
    try {
        // Validate input arguments
        PLUGIN_VALIDATE(inputs != nullptr);
        PLUGIN_VALIDATE(outputs != nullptr);
        PLUGIN_VALIDATE(nbOutputs == 1);
        PLUGIN_VALIDATE(nbInputs == 2);
        if (mType == DataType::kFLOAT || mType == DataType::kHALF) {
            PLUGIN_VALIDATE(mType == inputs[0].type);
            PLUGIN_VALIDATE(mType == inputs[1].type);
        } else {
            PLUGIN_VALIDATE(mType == inputs[0].type || DataType::kFLOAT == inputs[0].type);
            PLUGIN_VALIDATE(mType == inputs[1].type || DataType::kFLOAT == inputs[1].type);
        }
        auto const& inDims0 = inputs[0].dims;
        auto const& inDims1 = inputs[1].dims;
        PLUGIN_VALIDATE(inDims0.nbDims == inDims1.nbDims);

        return pluginStatus_t::STATUS_SUCCESS;
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return pluginStatus_t::STATUS_FAILURE;
}

// 附加到上下文
IPluginV3* MatrixAddPlugin::attachToContext(IPluginResourceContext* context) noexcept {
    return clone();
}

// 获取要序列化的字段
PluginFieldCollection const* MatrixAddPlugin::getFieldsToSerialize() noexcept {
    mDataToSerialize.clear();
    mFCToSerialize.nbFields = mDataToSerialize.size();
    mFCToSerialize.fields = mDataToSerialize.data();

    return &mFCToSerialize;
}

////////////////////////// MatrixAddPlugin Creator ///////////////////////////////

MatrixAddPluginCreator::MatrixAddPluginCreator() {
    static std::mutex sMutex;
    std::lock_guard<std::mutex> guard(sMutex);
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

// 获取插件名称
char const* MatrixAddPluginCreator::getPluginName() const noexcept {
    return kMATRIX_ADD_PLUGIN_NAME;
}

// 获取插件版本
char const* MatrixAddPluginCreator::getPluginVersion() const noexcept {
    return kMATRIX_ADD_PLUGIN_VERSION;
}

// 获取字段名称集合
PluginFieldCollection const* MatrixAddPluginCreator::getFieldNames() noexcept { return &mFC; }

// 创建插件实例
IPluginV3* MatrixAddPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc,
                                                TensorRTPhase phase) noexcept {
    try {
        PLUGIN_VALIDATE(fc != nullptr);
        // PLUGIN_VALIDATE(fc->fields != nullptr);

        return new MatrixAddPlugin();
    } catch (std::exception const& e) {
        caughtError(e);
    }
    return nullptr;
}

// 设置插件命名空间
void MatrixAddPluginCreator::setPluginNamespace(char const* libNamespace) noexcept {
    try {
        mNamespace = libNamespace;
    } catch (std::exception const& e) {
        caughtError(e);
    }
}

// 获取插件命名空间
char const* MatrixAddPluginCreator::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}