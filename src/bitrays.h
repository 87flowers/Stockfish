/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2025 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef BITRAYS_H_INCLUDED
#define BITRAYS_H_INCLUDED

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>

#include "bitboard.h"
#include "types.h"

#if defined(USE_AVX512ICL) || defined(USE_AVX2)
    #include <immintrin.h>
    #define USE_BITRAYS
#endif

namespace Stockfish {

#if defined(USE_BITRAYS)

    #if defined(USE_AVX512ICL)

using BitraysPermutation = __m512i;
using Rays               = __m512i;
using RaysMask           = __mmask64;

    #else

using BitraysPermutation = std::array<__m256i, 2>;
using Rays               = std::array<__m256i, 2>;
using RaysMask           = std::array<__m256i, 2>;

namespace internal {

inline __m256i permute8(__m256i index, __m256i a, __m256i b) {
    __m256i mask1 = _mm256_slli_epi16(index, 3);
    __m256i x =
      _mm256_blendv_epi8(_mm256_shuffle_epi8(_mm256_permute2x128_si256(a, a, 0x00), index),
                         _mm256_shuffle_epi8(_mm256_permute2x128_si256(a, a, 0x11), index), mask1);
    __m256i y =
      _mm256_blendv_epi8(_mm256_shuffle_epi8(_mm256_permute2x128_si256(b, b, 0x00), index),
                         _mm256_shuffle_epi8(_mm256_permute2x128_si256(b, b, 0x11), index), mask1);
    __m256i mask0 = _mm256_slli_epi16(index, 2);
    return _mm256_blendv_epi8(x, y, mask0);
}

inline uint64_t concat(uint32_t lo, uint32_t hi) {
    return static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
}

}

    #endif

inline std::tuple<BitraysPermutation, RaysMask> bitrays_permuation(Square focus) {
    // We use the 0x88 board representation here for intermediate calculations.
    // We convert to and from this representation to avoid a 4KiB LUT.

    alignas(64) static constexpr std::array<uint8_t, 64> OFFSETS{{
      0xDF, 0xF0, 0xE0, 0xD0, 0xC0, 0xB0, 0xA0, 0x90,  // S
      0xE1, 0xFF, 0xFE, 0xFD, 0xFC, 0xFB, 0xFA, 0xF9,  // W
      0xEE, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,  // E
      0xF2, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70,  // N
      0x0E, 0xEF, 0xDE, 0xCD, 0xBC, 0xAB, 0x9A, 0x89,  // SW
      0x12, 0x0F, 0x1E, 0x2D, 0x3C, 0x4B, 0x5A, 0x69,  // NW
      0x1F, 0xF1, 0xE2, 0xD3, 0xC4, 0xB5, 0xA6, 0x97,  // SE
      0x21, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,  // NE
    }};

    uint8_t f = static_cast<uint8_t>(focus);
    f         = f + (f & 0x38);

    #if defined(USE_AVX512ICL)
    __m512i  coords = _mm512_add_epi8(_mm512_load_si512(OFFSETS.data()), _mm512_set1_epi8(f));
    __m512i  perm = _mm512_gf2p8affine_epi64_epi8(coords, _mm512_set1_epi64(0x0102041020400000), 0);
    RaysMask mask = _mm512_testn_epi8_mask(coords, _mm512_set1_epi8(static_cast<int8_t>(0x88)));
    return {perm, mask};
    #else
    __m256i offsets0 = _mm256_load_si256(reinterpret_cast<__m256i const*>(OFFSETS.data()) + 0);
    __m256i offsets1 = _mm256_load_si256(reinterpret_cast<__m256i const*>(OFFSETS.data()) + 1);
    __m256i fVec     = _mm256_set1_epi8(f);
    __m256i coords0  = _mm256_add_epi8(offsets0, fVec);
    __m256i coords1  = _mm256_add_epi8(offsets1, fVec);
    __m256i x0F      = _mm256_set1_epi8(static_cast<int8_t>(0x0F));
    __m256i xF0      = _mm256_set1_epi8(static_cast<int8_t>(0xF0));
    __m256i perm0    = _mm256_or_si256(_mm256_and_si256(coords0, x0F),
                                       _mm256_srli_epi16(_mm256_and_si256(coords0, xF0), 1));
    __m256i perm1    = _mm256_or_si256(_mm256_and_si256(coords1, x0F),
                                       _mm256_srli_epi16(_mm256_and_si256(coords1, xF0), 1));
    __m256i x88      = _mm256_set1_epi8(static_cast<int8_t>(0x88));
    __m256i mask0    = _mm256_cmpeq_epi8(_mm256_and_si256(coords0, x88), _mm256_setzero_si256());
    __m256i mask1    = _mm256_cmpeq_epi8(_mm256_and_si256(coords1, x88), _mm256_setzero_si256());
    return {{perm0, perm1}, {mask0, mask1}};
    #endif
}

inline Rays board_to_rays(BitraysPermutation perm, RaysMask mask, const Piece board[]) {
    alignas(16) static constexpr std::array<uint8_t, 16> T{{
      //          p           n           b           r           q           k
      0, 0b00000001, 0b00000100, 0b00001000, 0b00010000, 0b00100000, 0b01000000, 0,  // White
      0, 0b10000010, 0b10000100, 0b10001000, 0b10010000, 0b10100000, 0b11000000, 0,  // Black
    }};

    #if defined(USE_AVX512ICL)
    __m512i t = _mm512_broadcast_i32x4(_mm_load_si128(reinterpret_cast<__m128i const*>(T.data())));
    __m512i boardVec = _mm512_loadu_epi8(board);
    __m512i res      = _mm512_permutexvar_epi8(perm, boardVec);
    res              = _mm512_shuffle_epi8(t, res);
    return _mm512_maskz_mov_epi8(mask, res);
    #else
    __m256i t =
      _mm256_broadcastsi128_si256(_mm_load_si128(reinterpret_cast<__m128i const*>(T.data())));
    __m256i board0 = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(&board[0]));
    __m256i board1 = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(&board[32]));
    __m256i res0   = internal::permute8(perm[0], board0, board1);
    __m256i res1   = internal::permute8(perm[1], board0, board1);
    res0           = _mm256_shuffle_epi8(t, res0);
    res1           = _mm256_shuffle_epi8(t, res1);
    return {_mm256_and_si256(res0, mask[0]), _mm256_and_si256(res1, mask[1])};
    #endif
}

inline Bitrays bitrays_from_bb(BitraysPermutation perm, RaysMask mask, Bitboard bb) {
    #if defined(USE_AVX512ICL)
    return _mm512_mask_bitshuffle_epi64_mask(mask, _mm512_set1_epi64(bb), perm);
    #else
    if (!bb)
        return 0;

    __m256i bits  = _mm256_set1_epi64x(0x8040201008040201);
    __m256i shuf  = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101,
                                      0x0000000000000000);
    __m256i x0    = _mm256_shuffle_epi8(_mm256_set1_epi32(static_cast<int32_t>(bb)), shuf);
    __m256i x1    = _mm256_shuffle_epi8(_mm256_set1_epi32(static_cast<int32_t>(bb >> 32)), shuf);
    x0            = _mm256_cmpeq_epi8(_mm256_and_si256(x0, bits), bits);
    x1            = _mm256_cmpeq_epi8(_mm256_and_si256(x1, bits), bits);
    __m256i  y0   = _mm256_and_si256(internal::permute8(perm[0], x0, x1), mask[0]);
    __m256i  y1   = _mm256_and_si256(internal::permute8(perm[1], x0, x1), mask[1]);
    uint32_t res0 = _mm256_movemask_epi8(y0);
    uint32_t res1 = _mm256_movemask_epi8(y1);
    return internal::concat(res0, res1);
    #endif
}

inline Bitrays bitrays_occupied(Rays rays) {
    #if defined(USE_AVX512ICL)
    return _mm512_cmpneq_epu8_mask(rays, _mm512_setzero_si512());
    #else
    uint32_t res0 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(rays[0], _mm256_setzero_si256()));
    uint32_t res1 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(rays[1], _mm256_setzero_si256()));
    return ~internal::concat(res0, res1);
    #endif
}

inline Bitrays bitrays_attackers(Rays rays) {
    //                               ckqrbnpp
    constexpr uint8_t horse      = 0b00000100;  // knight
    constexpr uint8_t orth       = 0b00110000;  // rook and queen
    constexpr uint8_t diag       = 0b00101000;  // bishop and queen
    constexpr uint8_t orth_near  = 0b01110000;  // king, rook and queen
    constexpr uint8_t wpawn_near = 0b01101001;  // wp, king, bishop, queen
    constexpr uint8_t bpawn_near = 0b01101010;  // bp, king, bishop, queen

    alignas(64) static constexpr std::array<uint8_t, 64> MASK{{
      horse, orth_near,  orth, orth, orth, orth, orth, orth,  // S
      horse, orth_near,  orth, orth, orth, orth, orth, orth,  // W
      horse, orth_near,  orth, orth, orth, orth, orth, orth,  // E
      horse, orth_near,  orth, orth, orth, orth, orth, orth,  // N
      horse, wpawn_near, diag, diag, diag, diag, diag, diag,  // SW
      horse, bpawn_near, diag, diag, diag, diag, diag, diag,  // NW
      horse, wpawn_near, diag, diag, diag, diag, diag, diag,  // SE
      horse, bpawn_near, diag, diag, diag, diag, diag, diag,  // NE
    }};

    #if defined(USE_AVX512ICL)
    return _mm512_test_epi8_mask(rays, _mm512_load_si512(MASK.data()));
    #else
    __m256i  mask0 = _mm256_load_si256(reinterpret_cast<__m256i const*>(MASK.data()) + 0);
    __m256i  mask1 = _mm256_load_si256(reinterpret_cast<__m256i const*>(MASK.data()) + 1);
    __m256i  x0    = _mm256_cmpeq_epi8(_mm256_and_si256(rays[0], mask0), _mm256_setzero_si256());
    __m256i  x1    = _mm256_cmpeq_epi8(_mm256_and_si256(rays[1], mask1), _mm256_setzero_si256());
    uint32_t res0  = _mm256_movemask_epi8(x0);
    uint32_t res1  = _mm256_movemask_epi8(x1);
    return ~internal::concat(res0, res1);
    #endif
}

inline Bitrays bitrays_test(Rays rays, uint8_t x) {
    #if defined(USE_AVX512ICL)
    return _mm512_test_epi8_mask(rays, _mm512_set1_epi8(x));
    #else
    __m256i  xVec = _mm256_set1_epi8(x);
    __m256i  y0   = _mm256_cmpeq_epi8(_mm256_and_si256(rays[0], xVec), _mm256_setzero_si256());
    __m256i  y1   = _mm256_cmpeq_epi8(_mm256_and_si256(rays[1], xVec), _mm256_setzero_si256());
    uint32_t res0 = _mm256_movemask_epi8(y0);
    uint32_t res1 = _mm256_movemask_epi8(y1);
    return ~internal::concat(res0, res1);
    #endif
}

template<PieceType piece>
inline Bitrays bitrays_for(Rays rays) {
    switch (piece)
    {
    case PAWN :
        return bitrays_test(rays, 0b00000011);
    case KNIGHT :
        return bitrays_test(rays, 0b00000100);
    case BISHOP :
        return bitrays_test(rays, 0b00001000);
    case ROOK :
        return bitrays_test(rays, 0b00010000);
    case QUEEN :
        return bitrays_test(rays, 0b00100000);
    case KING :
        return bitrays_test(rays, 0b01000000);
    }
}

inline Bitrays bitrays_color(Rays rays) {
    #if defined(USE_AVX512ICL)
    return _mm512_movepi8_mask(rays);
    #else
    uint32_t res0 = _mm256_movemask_epi8(rays[0]);
    uint32_t res1 = _mm256_movemask_epi8(rays[1]);
    return internal::concat(res0, res1);
    #endif
}

inline Bitrays least_significant_square_br(Bitrays b) {
    assert(b);
    return b & -b;
}

inline Bitrays bitrays_from_sq(BitraysPermutation perm, RaysMask mask, Square sq) {
    #if defined(USE_AVX512ICL)
    return _mm512_cmpeq_epu8_mask(perm, _mm512_set1_epi8(sq)) & mask;
    #else
    __m256i  sqVec = _mm256_set1_epi8(sq);
    __m256i  x0    = _mm256_and_si256(_mm256_cmpeq_epi8(perm[0], sqVec), mask[0]);
    __m256i  x1    = _mm256_and_si256(_mm256_cmpeq_epi8(perm[1], sqVec), mask[1]);
    uint32_t res0  = _mm256_movemask_epi8(x0);
    uint32_t res1  = _mm256_movemask_epi8(x1);
    return internal::concat(res0, res1);
    #endif
}

inline Square bitray_bit_to_sq(BitraysPermutation perm, Bitrays b) {
    #if defined(USE_AVX512ICL)
    return static_cast<Square>(
      _mm_cvtsi128_si32(_mm512_castsi512_si128(_mm512_maskz_compress_epi8(b, perm))));
    #else
    return static_cast<Square>(reinterpret_cast<char*>(&perm)[__builtin_ctzll(b)]);
    #endif
}

inline Bitrays bitrays_closest(Bitrays occupied) {
    Bitrays o = occupied | 0x8181818181818181;
    Bitrays x = o ^ (o - 0x0303030303030303);
    return x & occupied;
}

inline PieceType bitrays_see_next(const std::array<Bitrays, 8>& pieces, Bitrays attackers) {
    #if defined(USE_AVX512ICL)
    uint8_t mask =
      _mm512_test_epi64_mask(_mm512_load_epi64(pieces.data()), _mm512_set1_epi64(attackers));
    return static_cast<PieceType>(__builtin_ctz(mask));
    #else
    __m256i attackersVec = _mm256_set1_epi64x(attackers);
    __m256i pieces0      = _mm256_load_si256(reinterpret_cast<__m256i const*>(pieces.data()) + 0);
    __m256i pieces1      = _mm256_load_si256(reinterpret_cast<__m256i const*>(pieces.data()) + 1);
    pieces0 = _mm256_cmpeq_epi64(_mm256_and_si256(pieces0, attackersVec), _mm256_setzero_si256());
    pieces1 = _mm256_cmpeq_epi64(_mm256_and_si256(pieces1, attackersVec), _mm256_setzero_si256());
    uint8_t res0 = _mm256_movemask_pd(reinterpret_cast<__m256d>(pieces0)) ^ 0x0F;
    uint8_t res1 = _mm256_movemask_pd(reinterpret_cast<__m256d>(pieces1)) ^ 0x0F;
    return static_cast<PieceType>(__builtin_ctz(res0 | (res1 << 4)));
    #endif
}

#else

inline Bitboard pick_one_from(Bitboard first, Bitboard second) {
    #if defined(__GNUC__) && defined(IS_64BIT)
    __extension__ using uint128_t = unsigned __int128;
    uint128_t x = static_cast<uint128_t>(first) | (static_cast<uint128_t>(second) << 64);
    x           = x & -x;
    return static_cast<uint64_t>(x) | static_cast<uint64_t>(x >> 64);
    #else
    Bitboard a = -first;
    Bitboard b = -second - (a > 0);
    return (first & a) | (second & b);
    #endif
}

template<PieceType piece>
inline Bitboard see_pick_a_piece(Bitboard bb, Square to) {
    switch (piece)
    {
    case PAWN :
    case ROOK :
    case KNIGHT :
        return least_significant_square_bb(bb);
    case BISHOP : {
        Bitboard mask = file_bb(to) - FileABB;
        return pick_one_from(bb & mask, bb);
    }
    case QUEEN :
    case KING : {
        Bitboard orth = attacks_bb<ROOK>(to);
        return pick_one_from(bb & orth, see_pick_a_piece<BISHOP>(bb, to));
    }
    }
}

#endif
}

#endif
