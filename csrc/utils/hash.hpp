#pragma once

#include <string>
#include <iomanip>

namespace MiraOperator {

//生成hash函数映射值
static uint64_t fnv1a(const std::string& data, const uint64_t& seed) {
    uint64_t h = seed;
    const uint64_t& prime = 0x100000001b3ull;
    for (const char& c: data) {
        h ^= static_cast<uint8_t>(c);// 异或
        h *= prime;// 乘以大质数（取模隐含在64位溢出中）
    }
    return h;
}

//返回的是128位十六进制哈希值
static std::string get_hex_digest(const std::string& data) {
    const auto& state_0 = fnv1a(data, 0xc6a4a7935bd1e995ull);
    const auto& state_1 = fnv1a(data, 0x9e3779b97f4a7c15ull);

    // Split-mix 64
    const auto& split_mix = [](uint64_t z) {
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    };

    std::ostringstream oss;
    oss << std::hex << std::setfill('0')
        << std::setw(16) << split_mix(state_0)
        << std::setw(16) << split_mix(state_1);
    return oss.str();
}

} // namespace MiraOperator
