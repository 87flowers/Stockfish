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
#include <xmmintrin.h>

#include "types.h"

#if defined(USE_AVX512ICL)
    #define USE_BITRAYS
#endif

#if defined(USE_BITRAYS)

namespace Stockfish {

using BitraysPermutation = __m512i;
using Rays               = __m512i;
using RaysMask           = __mmask64;

inline std::tuple<BitraysPermutation, RaysMask> bitrays_permuation(Square focus) {
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

    uint8_t f      = static_cast<uint8_t>(focus);
    f              = f + (f & 0x38);
    __m512i coords = _mm512_add_epi8(_mm512_loadu_epi8(OFFSETS.data()), _mm512_set1_epi8(f));

    __m512i  perm = _mm512_gf2p8affine_epi64_epi8(coords, _mm512_set1_epi64(0x0102041020400000), 0);
    RaysMask mask = _mm512_testn_epi8_mask(coords, _mm512_set1_epi8(static_cast<int8_t>(0x88)));
    return {perm, mask};
}

inline Rays board_to_rays(BitraysPermutation perm, RaysMask mask, const Piece board[]) {
    alignas(16) static constexpr std::array<uint8_t, 16> T{{
      //          p           n           b           r           q           k
      0, 0b00000001, 0b00000100, 0b00001000, 0b00010000, 0b00100000, 0b01000000, 0,  // White
      0, 0b10000010, 0b10000100, 0b10001000, 0b10010000, 0b10100000, 0b11000000, 0,  // Black
    }};

    __m512i boardVec = _mm512_loadu_epi8(board);
    __m512i res      = _mm512_permutexvar_epi8(perm, boardVec);
    res              = _mm512_shuffle_epi8(_mm512_broadcast_i32x4(_mm_loadu_epi8(T.data())), res);
    return _mm512_maskz_mov_epi8(mask, res);
}

inline Bitrays bitrays_from_bb(BitraysPermutation perm, RaysMask mask, Bitboard bb) {
    return _mm512_mask_bitshuffle_epi64_mask(mask, _mm512_set1_epi64(bb), perm);
}

inline Bitrays bitrays_occupied(Rays rays) {
    return _mm512_cmpneq_epu8_mask(rays, _mm512_setzero_si512());
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
      horse, orth_near,  orth, orth, orth, orth, orth, orth,  // N
      horse, bpawn_near, diag, diag, diag, diag, diag, diag,  // NE
      horse, orth_near,  orth, orth, orth, orth, orth, orth,  // E
      horse, wpawn_near, diag, diag, diag, diag, diag, diag,  // SE
      horse, orth_near,  orth, orth, orth, orth, orth, orth,  // S
      horse, wpawn_near, diag, diag, diag, diag, diag, diag,  // SW
      horse, orth_near,  orth, orth, orth, orth, orth, orth,  // W
      horse, bpawn_near, diag, diag, diag, diag, diag, diag,  // NW
    }};

    return _mm512_test_epi8_mask(rays, _mm512_loadu_epi8(MASK.data()));
}

inline Bitrays bitrays_test(Rays rays, uint8_t x) {
    return _mm512_test_epi8_mask(rays, _mm512_set1_epi8(x));
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

inline Bitrays bitrays_color(Rays rays) { return _mm512_movepi8_mask(rays); }

inline Bitrays least_significant_square_br(Bitrays b) {
    assert(b);
    return b & -b;
}

inline Square bitray_bit_to_sq(BitraysPermutation perm, Bitrays b) {
    return static_cast<Square>(
      _mm_cvtsi128_si32(_mm512_castsi512_si128(_mm512_maskz_compress_epi8(b, perm))));
}

inline Bitrays bitrays_closest(Bitrays occupied) {
    Bitrays o = occupied | 0x8181818181818181;
    Bitrays x = o ^ (o - 0x0303030303030303);
    return x & occupied;
}

inline PieceType bitrays_see_next(const std::array<Bitrays, 8>& pieces, Bitrays attackers) {
    uint8_t mask =
      _mm512_test_epi64_mask(_mm512_load_epi64(pieces.data()), _mm512_set1_epi64(attackers));
    return static_cast<PieceType>(__builtin_ctz(mask));
}

}

#endif

#endif
