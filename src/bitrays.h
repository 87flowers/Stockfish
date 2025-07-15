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
#include <cstdint>
#include <cassert>
#include <xmmintrin.h>

#include "types.h"

#if defined(USE_VNNI) && !defined(USE_AVXVNNI)
    #define USE_BITRAYS
#endif

#if defined(USE_BITRAYS)

namespace Stockfish {

using BitraysPermutation = alignas(64) std::array<__m256i, 2>;
using Rays               = alignas(64) std::array<__m256i, 2>;

inline Bitrays concat_bitrays(uint32_t a, uint32_t b) { return _mm512_kunpackd(b, a); }

inline std::tuple<BitraysPermutation, uint64_t> bitrays_permuation(Square focus) {
    // We use the 0x88 board representation here for intermediate calculations.
    // We convert to and from this representation to avoid a 4KiB LUT.

    alignas(64) static constexpr std::array<uint8_t, 64> OFFSETS{{
      0x1F, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70,  // N
      0x21, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,  // NE
      0x12, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,  // E
      0xF2, 0xF1, 0xE2, 0xD3, 0xC4, 0xB5, 0xA6, 0x97,  // SE
      0xE1, 0xF0, 0xE0, 0xD0, 0xC0, 0xB0, 0xA0, 0x90,  // S
      0xDF, 0xEF, 0xDE, 0xCD, 0xBC, 0xAB, 0x9A, 0x89,  // SW
      0xEE, 0xFF, 0xFE, 0xFD, 0xFC, 0xFB, 0xFA, 0xF9,  // W
      0x0E, 0x0F, 0x1E, 0x2D, 0x3C, 0x4B, 0x5A, 0x69,  // NW
    }};

    uint8_t f             = static_cast<uint8_t>(focus);
    uint8_t expandedFocus = f + (f & 0x38);
    __m256i focusVec      = _mm256_set1_epi8(expandedFocus);
    __m256i offsets0      = _mm256_loadu_epi8(&OFFSETS[0]);
    __m256i offsets1      = _mm256_loadu_epi8(&OFFSETS[32]);
    __m256i uncompPerm0   = _mm256_add_epi8(offsets0, focusVec);
    __m256i uncompPerm1   = _mm256_add_epi8(offsets1, focusVec);

    __m256i x0F = _mm256_set1_epi8(0x0F);
    __m256i xF0 = _mm256_set1_epi8(0xF0);

    __m256i perm0 = _mm256_or_si256(_mm256_and_si256(uncompPerm0, x0F),
                                    _mm256_slli_epi16(_mm256_and_si256(uncompPerm0, xF0), 1));
    __m256i perm1 = _mm256_or_si256(_mm256_and_si256(uncompPerm1, x0F),
                                    _mm256_slli_epi16(_mm256_and_si256(uncompPerm0, xF0), 1));

    __m256i   x88   = _mm256_set1_epi8(0x88);
    __mmask32 mask0 = _mm256_testn_epi16_mask(uncompPerm0, x88);
    __mmask32 mask1 = _mm256_testn_epi16_mask(uncompPerm1, x88);

    return {{perm0, perm1}, concat_bitrays(mask0, mask1)};
}

inline Rays board_to_rays(const BitraysPermutation& perm, uint64_t mask, const Piece board[]) {
    __m256i board0 = _mm256_loadu_epi8(&board[0]);
    __m256i board1 = _mm256_loadu_epi8(&board[32]);
    __m256i res0   = _mm256_permutex2var_epi8(board0, perm[0], board1);
    __m256i res1   = _mm256_permutex2var_epi8(board0, perm[1], board1);

    //                                                 p     n     b     r     q     k
    __m256i convert =
      _mm256_broadcastsi128_si256(_mm_setr_epi8(0x00, 0x01, 0x04, 0x08, 0x10, 0x20, 0x40, 0x00,
                                                0x00, 0x82, 0x84, 0x88, 0x80, 0xA0, 0xC0, 0x00));
    res0 = _mm256_shuffle_epi8(convert, res0);
    res1 = _mm256_shuffle_epi8(convert, res1);

    uint32_t mask0 = static_cast<uint32_t>(mask >> 0);
    uint32_t mask1 = static_cast<uint32_t>(mask >> 32);

    return {_mm256_maskz_mov_epi8(mask0, res0), _mm256_maskz_mov_epi8(mask1, res1)};
}

inline Bitrays bitrays_from_bb(const BitraysPermutation& perm, uint64_t mask, Bitboard bb) {
    uint32_t mask0 = static_cast<uint32_t>(mask >> 0);
    uint32_t mask1 = static_cast<uint32_t>(mask >> 32);

    __m256i bbVec = _mm256_set1_epi64x(bb);

    return concat_bitrays(_mm256_mask_bitshuffle_epi64_mask(mask0, bbVec, perm[0]),
                          _mm256_mask_bitshuffle_epi64_mask(mask1, bbVec, perm[1]));
}

inline Bitrays bitrays_occupied(const Rays& rays) {
    return concat_bitrays(_mm256_test_epi8_mask(rays[0], rays[0]),
                          _mm256_test_epi8_mask(rays[1], rays[1]));
}

inline Rays bitrays_attackers(const Rays& rays) {
    alignas(64) static constexpr std::array<uint8_t, 64> MASK{{
      0x04, 0x70, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30,  // N
      0x04, 0x6A, 0x28, 0x28, 0x28, 0x28, 0x28, 0x28,  // NE
      0x04, 0x70, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30,  // E
      0x04, 0x69, 0x28, 0x28, 0x28, 0x28, 0x28, 0x28,  // SE
      0x04, 0x70, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30,  // S
      0x04, 0x69, 0x28, 0x28, 0x28, 0x28, 0x28, 0x28,  // SW
      0x04, 0x70, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30,  // W
      0x04, 0x6A, 0x28, 0x28, 0x28, 0x28, 0x28, 0x28,  // NW
    }};

    __m256i mask0 = _mm256_loadu_epi8(&MASK[0]);
    __m256i mask1 = _mm256_loadu_epi8(&MASK[32]);
    return {_mm256_and_si256(rays[0], mask0), _mm256_and_si256(rays[0], mask1)};
}

inline Bitrays bitrays_test(const Rays& rays, uint8_t x) {
    __m256i xVec = _mm256_set1_epi8(x);
    return concat_bitrays(_mm256_test_epi8_mask(rays[0], xVec),
                          _mm256_test_epi8_mask(rays[1], xVec));
}

template<PieceType piece>
inline Bitrays bitrays_with(const Rays& rays) {
    switch (piece)
    {
    case PAWN :
        return bitrays_test(rays, 0x03);
    case KNIGHT :
        return bitrays_test(rays, 0x04);
    case BISHOP :
        return bitrays_test(rays, 0x08);
    case ROOK :
        return bitrays_test(rays, 0x10);
    case QUEEN :
        return bitrays_test(rays, 0x20);
    case KING :
        return bitrays_test(rays, 0x40);
    }
}

inline Bitrays bitrays_color(const Rays& rays) {
    return concat_bitrays(_mm256_movepi8_mask(rays[0]), _mm256_movepi8_mask(rays[1]));
}

inline Bitrays least_significant_square_br(Bitrays b) {
    assert(b);
    return b & -b;
}

inline Bitrays bitrays_closest(Bitrays occupied) {
    Bitrays o = occupied | 0x8181818181818181;
    Bitrays x = o ^ (o - 0x0303030303030303);
    return x & occupied;
}

inline PieceType bitrays_see_next(const std::array<Bitrays, 8>& pieceRays, Bitrays attackers) {
    __m256i attackersVec = _mm256_set1_epi64x(attackers);
    __m256i pieces0      = _mm256_loadu_epi8(&pieceRays[0]);
    __m256i pieces1      = _mm256_loadu_epi8(&pieceRays[4]);

    uint8_t mask = 0;
    mask |= _mm256_test_epi64_mask(pieces0, attackersVec) << 0;
    mask |= _mm256_test_epi64_mask(pieces1, attackersVec) << 4;
    mask &= 0x7E;

    return static_cast<PieceType>(__builtin_ctz(mask));
}

}

#endif

#endif
