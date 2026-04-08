#include <iostream>
#include <mutex>

#include <NvInferRuntime.h>

#include "pluginRegistration.h"
#include "matrixAddPlugin.h"

using nvinfer1::plugin::MatrixAddPluginCreator;


class ThreadSafeLoggerFinder
{
private:
    nvinfer1::ILoggerFinder* mLoggerFinder{nullptr};
    std::mutex mMutex;

public:
    ThreadSafeLoggerFinder() = default;

    //! Set the logger finder.
    void setLoggerFinder(nvinfer1::ILoggerFinder* finder)
    {
        std::lock_guard<std::mutex> lk(mMutex);
        if (mLoggerFinder == nullptr && finder != nullptr)
        {
            mLoggerFinder = finder;
        }
    }

    //! Get the logger.
    nvinfer1::ILogger* getLogger() noexcept
    {
        std::lock_guard<std::mutex> lk(mMutex);
        if (mLoggerFinder != nullptr)
        {
            return mLoggerFinder->findLogger();
        }
        return nullptr;
    }
};

ThreadSafeLoggerFinder gLoggerFinder;

extern "C" nvinfer1::IPluginCreatorInterface* const* getCreators(int32_t& nbCreators)
{
    nbCreators = 1;
    static MatrixAddPluginCreator sMatrixAddPluginCreator;
    static nvinfer1::IPluginCreatorInterface* const kPLUGIN_CREATOR_LIST[] = {&sMatrixAddPluginCreator};
    return kPLUGIN_CREATOR_LIST;
}

extern "C" void setLoggerFinder(nvinfer1::ILoggerFinder* finder)
{
    gLoggerFinder.setLoggerFinder(finder);
}