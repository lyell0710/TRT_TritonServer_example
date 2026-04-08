#ifndef TRT_PLUGIN_UTILS_H
#define TRT_PLUGIN_UTILS_H

#include <cstring>
#include <sstream>

// Enumerator for status
typedef enum {
    STATUS_SUCCESS = 0,
    STATUS_FAILURE = 1,
    STATUS_BAD_PARAM = 2,
    STATUS_NOT_SUPPORTED = 3,
    STATUS_NOT_INITIALIZED = 4
} pluginStatus_t;

#define CSC(call, err)                   \
    do {                                 \
        cudaError_t cudaStatus = call;   \
        if (cudaStatus != cudaSuccess) { \
            return err;                  \
        }                                \
    } while (0)

void caughtError(std::exception const& e);

#define PLUGIN_ASSERT(val) reportAssertion((val), #val, __FILE__, __LINE__)
void reportAssertion(bool success, char const* msg, char const* file, int32_t line);

#define PLUGIN_VALIDATE(val) reportValidation((val), #val, __FILE__, __LINE__)
void reportValidation(bool success, char const* msg, char const* file, int32_t line);

#endif  // TRT_PLUGIN_UTILS_H