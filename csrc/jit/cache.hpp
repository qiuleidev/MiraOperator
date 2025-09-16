#pragma once  // 头文件保护，防止重复包含
#include <filesystem>   // 文件路径操作库
#include <memory>       // 智能指针支持
#include <unordered_map> // 哈希表容器
#include "kernel_runtime.hpp"  // 自定义头文件（核心功能）

namespace MiraOperator {
class KernelRuntimeCache {
    std::unordered_map<std::filesystem::path, 
                      std::shared_ptr<KernelRuntime>> cache;  // 核心缓存结构
public:
    KernelRuntimeCache() = default;  // 默认构造函数（TODO提示考虑缓存容量）
    
    // 缓存获取/创建方法
    std::shared_ptr<KernelRuntime> get(const std::filesystem::path& dir_path) {
        // 缓存命中检查
        if (const auto& iterator = cache.find(dir_path); iterator != cache.end())//C++17写法，这样iterator的作用域只在if-else语句中
            return iterator->second;

        // 缓存未命中时的处理逻辑
        if (KernelRuntime::check_validity(dir_path))  // 验证路径有效性
            return cache[dir_path] = std::make_shared<KernelRuntime>(dir_path);  // 创建并缓存新实例
        return nullptr;  // 无效路径返回空指针
    }
};

static auto kernel_runtime_cache = std::make_shared<KernelRuntimeCache>();

}