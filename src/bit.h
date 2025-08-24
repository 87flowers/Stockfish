#ifndef BIT_H_INCLUDED
#define BIT_H_INCLUDED

#include <cstdint>

namespace Stockfish {
namespace Bit {

inline unsigned int ctz(uint32_t x) {
#if defined(__GNUC__)  // GCC, Clang, ICX
    return __builtin_ctzl(x);
#elif defined(_MSC_VER)
    unsigned int idx;
    _BitScanForward(&idx, x);
    return idx;
#else  // Compiler is neither GCC nor MSVC compatible
    #error "Compiler not supported."
#endif
}

inline unsigned int ctz(uint64_t x) {
#if defined(__GNUC__)  // GCC, Clang, ICX
    return __builtin_ctzll(x);
#elif defined(_MSC_VER)
    #ifdef _WIN64
    unsigned int idx;
    _BitScanForward64(&idx, x);
    return idx;
    #else  // MSVC, WIN32
    unsigned long idx;

    if (b & 0xffffffff)
    {
        _BitScanForward(&idx, int32_t(b));
        return idx;
    }
    else
    {
        _BitScanForward(&idx, int32_t(b >> 32));
        return idx + 32;
    }
    #endif
#else  // Compiler is neither GCC nor MSVC compatible
    #error "Compiler not supported."
#endif
}

}
}

#endif
