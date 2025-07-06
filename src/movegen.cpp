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

#include "movegen.h"

#include <cassert>
#include <initializer_list>

#include "bitboard.h"
#include "position.h"

#if defined(USE_AVX512) || defined(USE_VNNI)
    #include <array>
    #include <immintrin.h>
#endif

namespace Stockfish {

namespace {

#if defined(USE_AVX512) || defined(USE_VNNI)
alignas(64) constexpr std::array<ExtMove, 64> SPLAT_PAWN_TABLE = [] {
    // Ensure the move format hasn't changed.
    static_assert(Move(SQ_D4 - Up, SQ_D4).raw() == Move(SQ_D4, SQ_D4).raw() - (Up << 6));

    std::array<ExtMove, 64> pawn_table{};
    for (int i = 0; i < 64; i++)
        pawn_table[i] = Move(i, i);
    return pawn_table;
}();

static_assert(sizeof(SPLAT_PAWN_TABLE) == 256);

alignas(64) constexpr std::array<ExtMove, 64> SPLAT_TABLE = [] {
    // Ensure the move format hasn't changed.
    static_assert(Move(SQ_E7, SQ_D4).raw() == Move(0, SQ_D4).raw() | (SQ_E7 << 6));

    std::array<ExtMove, 64> table{};
    for (int i = 0; i < 64; i++)
        table[i] = Move(0, i);
    return table;
}();

static_assert(sizeof(SPLAT_TABLE) == 256);

inline void write_moves(ExtMove* moveList, uint16_t mask, __m512i vector) {
    __m512i toWrite = _mm512_maskz_compress_epi32(mask, vector);
    _mm_storeu_si128((__m512i*) moveList, toWrite);
    moveList += __builtin_popcount(mask);
}
#endif

template<Direction offset>
inline void splat_pawn_moves(ExtMove* moveList, Bitboard to_bb) {
#if defined(USE_AVX512) || defined(USE_VNNI)
    __m512i offsetVec = _mm512_set1_epi32(offset << 6);
    write_moves(static_cast<u16>(to_bb >> 0),
                _mm512_sub_epi32(_mm512_load_si512(SPLAT_PAWN_TABLE.data() + 0), offsetVec));
    write_moves(static_cast<u16>(to_bb >> 16),
                _mm512_sub_epi32(_mm512_load_si512(SPLAT_PAWN_TABLE.data() + 16), offsetVec));
    write_moves(static_cast<u16>(to_bb >> 32),
                _mm512_sub_epi32(_mm512_load_si512(SPLAT_PAWN_TABLE.data() + 32), offsetVec));
    write_moves(static_cast<u16>(to_bb >> 48),
                _mm512_sub_epi32(_mm512_load_si512(SPLAT_PAWN_TABLE.data() + 48), offsetVec));
#else
    while (to_bb)
    {
        Square to   = pop_lsb(to_bb);
        *moveList++ = Move(to - offset, to);
    }
#endif
}

inline void splat_moves(ExtMove* moveList, Square from, Bitboard to_bb) {
#if defined(USE_AVX512) || defined(USE_VNNI)
    __m512i fromVec = _mm512_set1_epi32(from << 6);
    write_moves(static_cast<u16>(to_bb >> 0),
                _mm512_or_epi32(_mm512_load_si512(SPLAT_TABLE.data() + 0), fromVec));
    write_moves(static_cast<u16>(to_bb >> 16),
                _mm512_or_epi32(_mm512_load_si512(SPLAT_TABLE.data() + 16), fromVec));
    write_moves(static_cast<u16>(to_bb >> 32),
                _mm512_or_epi32(_mm512_load_si512(SPLAT_TABLE.data() + 32), fromVec));
    write_moves(static_cast<u16>(to_bb >> 48),
                _mm512_or_epi32(_mm512_load_si512(SPLAT_TABLE.data() + 48), fromVec));
#else
    while (to_bb)
        *moveList++ = Move(from, pop_lsb(to_bb));
#endif
}

template<GenType Type, Direction D, bool Enemy>
ExtMove* make_promotions(ExtMove* moveList, [[maybe_unused]] Square to) {

    constexpr bool all = Type == EVASIONS || Type == NON_EVASIONS;

    if constexpr (Type == CAPTURES || all)
        *moveList++ = Move::make<PROMOTION>(to - D, to, QUEEN);

    if constexpr ((Type == CAPTURES && Enemy) || (Type == QUIETS && !Enemy) || all)
    {
        *moveList++ = Move::make<PROMOTION>(to - D, to, ROOK);
        *moveList++ = Move::make<PROMOTION>(to - D, to, BISHOP);
        *moveList++ = Move::make<PROMOTION>(to - D, to, KNIGHT);
    }

    return moveList;
}


template<Color Us, GenType Type>
ExtMove* generate_pawn_moves(const Position& pos, ExtMove* moveList, Bitboard target) {

    constexpr Color     Them     = ~Us;
    constexpr Bitboard  TRank7BB = (Us == WHITE ? Rank7BB : Rank2BB);
    constexpr Bitboard  TRank3BB = (Us == WHITE ? Rank3BB : Rank6BB);
    constexpr Direction Up       = pawn_push(Us);
    constexpr Direction UpRight  = (Us == WHITE ? NORTH_EAST : SOUTH_WEST);
    constexpr Direction UpLeft   = (Us == WHITE ? NORTH_WEST : SOUTH_EAST);

    const Bitboard emptySquares = ~pos.pieces();
    const Bitboard enemies      = Type == EVASIONS ? pos.checkers() : pos.pieces(Them);

    Bitboard pawnsOn7    = pos.pieces(Us, PAWN) & TRank7BB;
    Bitboard pawnsNotOn7 = pos.pieces(Us, PAWN) & ~TRank7BB;

    // Single and double pawn pushes, no promotions
    if constexpr (Type != CAPTURES)
    {
        Bitboard b1 = shift<Up>(pawnsNotOn7) & emptySquares;
        Bitboard b2 = shift<Up>(b1 & TRank3BB) & emptySquares;

        if constexpr (Type == EVASIONS)  // Consider only blocking squares
        {
            b1 &= target;
            b2 &= target;
        }

        splat_pawn_moves<Up>(moveList, b1);
        splat_pawn_moves<Up + Up>(moveList, b2);
    }

    // Promotions and underpromotions
    if (pawnsOn7)
    {
        Bitboard b1 = shift<UpRight>(pawnsOn7) & enemies;
        Bitboard b2 = shift<UpLeft>(pawnsOn7) & enemies;
        Bitboard b3 = shift<Up>(pawnsOn7) & emptySquares;

        if constexpr (Type == EVASIONS)
            b3 &= target;

        while (b1)
            moveList = make_promotions<Type, UpRight, true>(moveList, pop_lsb(b1));

        while (b2)
            moveList = make_promotions<Type, UpLeft, true>(moveList, pop_lsb(b2));

        while (b3)
            moveList = make_promotions<Type, Up, false>(moveList, pop_lsb(b3));
    }

    // Standard and en passant captures
    if constexpr (Type == CAPTURES || Type == EVASIONS || Type == NON_EVASIONS)
    {
        Bitboard b1 = shift<UpRight>(pawnsNotOn7) & enemies;
        Bitboard b2 = shift<UpLeft>(pawnsNotOn7) & enemies;

        splat_pawn_moves<UpRight>(moveList, b1);
        splat_pawn_moves<UpLeft>(moveList, b2);

        if (pos.ep_square() != SQ_NONE)
        {
            assert(rank_of(pos.ep_square()) == relative_rank(Us, RANK_6));

            // An en passant capture cannot resolve a discovered check
            if (Type == EVASIONS && (target & (pos.ep_square() + Up)))
                return moveList;

            b1 = pawnsNotOn7 & attacks_bb<PAWN>(pos.ep_square(), Them);

            assert(b1);

            while (b1)
                *moveList++ = Move::make<EN_PASSANT>(pop_lsb(b1), pos.ep_square());
        }
    }

    return moveList;
}


template<Color Us, PieceType Pt>
ExtMove* generate_moves(const Position& pos, ExtMove* moveList, Bitboard target) {

    static_assert(Pt != KING && Pt != PAWN, "Unsupported piece type in generate_moves()");

    Bitboard bb = pos.pieces(Us, Pt);

    while (bb)
    {
        Square   from = pop_lsb(bb);
        Bitboard b    = attacks_bb<Pt>(from, pos.pieces()) & target;

        splat_moves(moveList, from, b);
    }

    return moveList;
}


template<Color Us, GenType Type>
ExtMove* generate_all(const Position& pos, ExtMove* moveList) {

    static_assert(Type != LEGAL, "Unsupported type in generate_all()");

    const Square ksq = pos.square<KING>(Us);
    Bitboard     target;

    // Skip generating non-king moves when in double check
    if (Type != EVASIONS || !more_than_one(pos.checkers()))
    {
        target = Type == EVASIONS     ? between_bb(ksq, lsb(pos.checkers()))
               : Type == NON_EVASIONS ? ~pos.pieces(Us)
               : Type == CAPTURES     ? pos.pieces(~Us)
                                      : ~pos.pieces();  // QUIETS

        moveList = generate_pawn_moves<Us, Type>(pos, moveList, target);
        moveList = generate_moves<Us, KNIGHT>(pos, moveList, target);
        moveList = generate_moves<Us, BISHOP>(pos, moveList, target);
        moveList = generate_moves<Us, ROOK>(pos, moveList, target);
        moveList = generate_moves<Us, QUEEN>(pos, moveList, target);
    }

    Bitboard b = attacks_bb<KING>(ksq) & (Type == EVASIONS ? ~pos.pieces(Us) : target);

    splat_moves(moveList, ksq, b);

    if ((Type == QUIETS || Type == NON_EVASIONS) && pos.can_castle(Us & ANY_CASTLING))
        for (CastlingRights cr : {Us & KING_SIDE, Us & QUEEN_SIDE})
            if (!pos.castling_impeded(cr) && pos.can_castle(cr))
                *moveList++ = Move::make<CASTLING>(ksq, pos.castling_rook_square(cr));

    return moveList;
}

}  // namespace


// <CAPTURES>     Generates all pseudo-legal captures plus queen promotions
// <QUIETS>       Generates all pseudo-legal non-captures and underpromotions
// <EVASIONS>     Generates all pseudo-legal check evasions
// <NON_EVASIONS> Generates all pseudo-legal captures and non-captures
//
// Returns a pointer to the end of the move list.
template<GenType Type>
ExtMove* generate(const Position& pos, ExtMove* moveList) {

    static_assert(Type != LEGAL, "Unsupported type in generate()");
    assert((Type == EVASIONS) == bool(pos.checkers()));

    Color us = pos.side_to_move();

    return us == WHITE ? generate_all<WHITE, Type>(pos, moveList)
                       : generate_all<BLACK, Type>(pos, moveList);
}

// Explicit template instantiations
template ExtMove* generate<CAPTURES>(const Position&, ExtMove*);
template ExtMove* generate<QUIETS>(const Position&, ExtMove*);
template ExtMove* generate<EVASIONS>(const Position&, ExtMove*);
template ExtMove* generate<NON_EVASIONS>(const Position&, ExtMove*);


// generate<LEGAL> generates all the legal moves in the given position

template<>
ExtMove* generate<LEGAL>(const Position& pos, ExtMove* moveList) {

    Color    us     = pos.side_to_move();
    Bitboard pinned = pos.blockers_for_king(us) & pos.pieces(us);
    Square   ksq    = pos.square<KING>(us);
    ExtMove* cur    = moveList;

    moveList =
      pos.checkers() ? generate<EVASIONS>(pos, moveList) : generate<NON_EVASIONS>(pos, moveList);
    while (cur != moveList)
        if (((pinned & cur->from_sq()) || cur->from_sq() == ksq || cur->type_of() == EN_PASSANT)
            && !pos.legal(*cur))
            *cur = *(--moveList);
        else
            ++cur;

    return moveList;
}

}  // namespace Stockfish
