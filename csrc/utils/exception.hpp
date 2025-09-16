#pragma once

#include <exception>
#include <string>

namespace MiraOperator {

class MOException final : public std::exception {
    std::string message = {};

public:
    explicit MOException(const char *name, const char* file, const int line, const std::string& error) {
        message = std::string("Failed: ") + name + " error " + file + ":" + std::to_string(line) + " '" + error + "'";
    }

    const char *what() const noexcept override {
        return message.c_str();
    }
};

#ifndef MO_STATIC_ASSERT
#define MO_STATIC_ASSERT(cond, ...) static_assert(cond, __VA_ARGS__)// 定义该宏，接受条件表达式和可变参数
#endif

#ifndef MO_HOST_ASSERT
#define MO_HOST_ASSERT(cond) \
do { \
    if (not (cond)) { \
        throw MOException("Assertion", __FILE__, __LINE__, #cond); \
    } \
} while (0)
#endif

#ifndef MO_HOST_UNREACHABLE
#define MO_HOST_UNREACHABLE(reason) (throw MOException("Assertion", __FILE__, __LINE__, reason))
#endif

#ifndef MO_CUDA_DRIVER_CHECK
#define MO_CUDA_DRIVER_CHECK(cmd) \
do { \
    const auto& e = (cmd); \
    if (e != CUDA_SUCCESS) { \
        throw MOException("CUDA driver", __FILE__, __LINE__, ""); \
    } \
} while (0)
#endif

#ifndef MO_CUDA_RUNTIME_CHECK
#define MO_CUDA_RUNTIME_CHECK(cmd) \
do { \
    const auto& e = (cmd); \
    if (e != cudaSuccess) { \
        throw MOException("CUDA runtime", __FILE__, __LINE__, std::to_string(static_cast<int>(e))); \
    } \
} while (0)
#endif

} // namespace MiraOperator
