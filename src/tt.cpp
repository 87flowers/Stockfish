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

#include "tt.h"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "bit.h"
#include "memory.h"
#include "misc.h"
#include "syzygy/tbprobe.h"
#include "thread.h"

#if defined(USE_AVX512)
    #include <immintrin.h>
#elif defined(USE_SSE2)
    #include <emmintrin.h>
#elif defined(USE_NEON)
    #include <arm_neon.h>
#endif


namespace Stockfish {

// A transposition table entry is made up of 10 bytes, split into two parts:
//
// Part A:
//
// key        16 bit
//
// Part B:
//
// move       16 bit
// value      16 bit
// evaluation 16 bit
// generation  5 bit
// pv node     1 bit
// bound type  2 bit
// depth       8 bit

// `genBound8` is where most of the details are. We use the following constants to manipulate 5 leading generation bits
// and 3 trailing miscellaneous bits.

// These bits are reserved for other things.
static constexpr unsigned GENERATION_BITS = 3;
// increment for generation field
static constexpr int GENERATION_DELTA = (1 << GENERATION_BITS);
// cycle length
static constexpr int GENERATION_CYCLE = 255 + GENERATION_DELTA;
// mask to pull out generation number
static constexpr int GENERATION_MASK = (0xFF << GENERATION_BITS) & 0xFF;

struct TTEntryB {
    Move    move16;
    int16_t value16;
    int16_t eval16;
    uint8_t genBound8;
    int8_t  depth8;

    static TTEntryB from_raw(uint64_t raw) {
        TTEntryB e;
        std::memcpy(&e, &raw, sizeof(uint64_t));
        return e;
    }

    Depth depth() const { return Depth(depth8 + DEPTH_ENTRY_OFFSET); }
    Bound bound() const { return Bound(genBound8 & 0x3); }
    bool  is_pv() const { return bool(genBound8 & 0x4); }

    bool is_occupied() const {
        // DEPTH_ENTRY_OFFSET exists because 1) we use `bool(depth8)` as the occupancy check, but
        // 2) we need to store negative depths for QS. (`depth8` is the only field with "spare bits":
        // we sacrifice the ability to store depths greater than 1<<8 less the offset, as asserted in `save`.)
        return bool(depth8);
    }

    uint8_t relative_age(const uint8_t generation8) const {
        // Due to our packed storage format for generation and its cyclic
        // nature we add GENERATION_CYCLE (256 is the modulus, plus what
        // is needed to keep the unrelated lowest n bits from affecting
        // the result) to calculate the entry age correctly even after
        // generation8 overflows into the next cycle.
        return (GENERATION_CYCLE + generation8 - genBound8) & GENERATION_MASK;
    }

    int replace_score(const uint8_t generation8) const {
        return depth8 - relative_age(generation8);
    }

    uint64_t to_raw() const {
        uint64_t raw;
        std::memcpy(&raw, this, sizeof(uint64_t));
        return raw;
    }
};

static_assert(sizeof(TTEntryB) == sizeof(uint64_t));

// A TranspositionTable is an array of Cluster, of size clusterCount. Each cluster consists of ClusterSize number
// of TTEntry. Each non-empty TTEntry contains information on exactly one position. The size of a Cluster should
// divide the size of a cache line for best performance, as the cacheline is prefetched when possible.

static constexpr int ClusterSize = 6;

struct Cluster {
    uint64_t entry[ClusterSize];
    uint16_t key[ClusterSize];
    char     padding[4];

    void
    save(int i, Key k, Value v, bool pv, Bound b, Depth d, Move m, Value ev, uint8_t generation8);

    int replace_score(int i, const uint8_t generation8) const {
        return TTEntryB::from_raw(entry[i]).replace_score(generation8);
    }

#if defined(USE_SSE2) || defined(USE_AVX512)
    __m128i key_vec() const { return _mm_load_si128(reinterpret_cast<__m128i const*>(&key[0])); }
#endif
};

static_assert(sizeof(Cluster) == 64, "Suboptimal Cluster size");

// Populates the TTEntry with a new node's data, possibly
// overwriting an old position. The update is not atomic and can be racy.
void Cluster::save(
  int i, Key k, Value v, bool pv, Bound b, Depth d, Move m, Value ev, uint8_t generation8) {
    const uint16_t oldkey16 = key[i];
    TTEntryB       e        = TTEntryB::from_raw(entry[i]);

    // Preserve the old ttmove if we don't have a new one
    if (m || uint16_t(k) != oldkey16)
        e.move16 = m;

    // Overwrite less valuable entries (cheapest checks first)
    if (b == BOUND_EXACT || uint16_t(k) != oldkey16
        || d - DEPTH_ENTRY_OFFSET + 2 * pv > e.depth8 - 4 || e.relative_age(generation8))
    {
        assert(d > DEPTH_ENTRY_OFFSET);
        assert(d < 256 + DEPTH_ENTRY_OFFSET);

        e.depth8    = uint8_t(d - DEPTH_ENTRY_OFFSET);
        e.genBound8 = uint8_t(generation8 | uint8_t(pv) << 2 | b);
        e.value16   = int16_t(v);
        e.eval16    = int16_t(ev);

        key[i] = uint16_t(k);
    }
    else if (e.depth() >= 5 && e.bound() != BOUND_EXACT)
        e.depth8--;

    entry[i] = e.to_raw();
}


// TTWriter is but a very thin wrapper around the pointer
TTWriter::TTWriter(Cluster* cl_, int i_) :
    cl(cl_),
    i(i_) {}

void TTWriter::write(
  Key k, Value v, bool pv, Bound b, Depth d, Move m, Value ev, uint8_t generation8) {
    cl->save(i, k, v, pv, b, d, m, ev, generation8);
}


// Sets the size of the transposition table,
// measured in megabytes. Transposition table consists
// of clusters and each cluster consists of ClusterSize number of TTEntry.
void TranspositionTable::resize(size_t mbSize, ThreadPool& threads) {
    aligned_large_pages_free(table);

    clusterCount = mbSize * 1024 * 1024 / sizeof(Cluster);

    table = static_cast<Cluster*>(aligned_large_pages_alloc(clusterCount * sizeof(Cluster)));

    if (!table)
    {
        std::cerr << "Failed to allocate " << mbSize << "MB for transposition table." << std::endl;
        exit(EXIT_FAILURE);
    }

    clear(threads);
}


// Initializes the entire transposition table to zero,
// in a multi-threaded way.
void TranspositionTable::clear(ThreadPool& threads) {
    generation8              = 0;
    const size_t threadCount = threads.num_threads();

    for (size_t i = 0; i < threadCount; ++i)
    {
        threads.run_on_thread(i, [this, i, threadCount]() {
            // Each thread will zero its part of the hash table
            const size_t stride = clusterCount / threadCount;
            const size_t start  = stride * i;
            const size_t len    = i + 1 != threadCount ? stride : clusterCount - start;

            std::memset(&table[start], 0, len * sizeof(Cluster));
        });
    }

    for (size_t i = 0; i < threadCount; ++i)
        threads.wait_on_thread(i);
}


// Returns an approximation of the hashtable
// occupation during a search. The hash is x permill full, as per UCI protocol.
// Only counts entries which match the current generation.
int TranspositionTable::hashfull(int maxAge) const {
    int maxAgeInternal = maxAge << GENERATION_BITS;
    int cnt            = 0;
    for (int i = 0; i < 1000; ++i)
        for (int j = 0; j < ClusterSize; ++j)
        {
            const TTEntryB e = TTEntryB::from_raw(table[i].entry[j]);
            cnt += e.is_occupied() && e.relative_age(generation8) <= maxAgeInternal;
        }

    return cnt / ClusterSize;
}


void TranspositionTable::new_search() {
    // increment by delta to keep lower bits as is
    generation8 += GENERATION_DELTA;
}


uint8_t TranspositionTable::generation() const { return generation8; }


// Looks up the current position in the transposition
// table. It returns true if the position is found.
// Otherwise, it returns false and a pointer to an empty or least valuable TTEntry
// to be replaced later. The replace value of an entry is calculated as its depth
// minus 8 times its relative age. TTEntry t1 is considered more valuable than
// TTEntry t2 if its replace value is greater than that of t2.
std::tuple<bool, TTData, TTWriter> TranspositionTable::probe(const Key key) const {
    Cluster* const cl    = cluster(key);
    const uint16_t key16 = uint16_t(key);  // Use the low 16 bits as key inside the cluster

#if defined(USE_AVX512)
    uint8_t mask = _mm_cmpeq_epi16_mask(cl->key_vec(), _mm_set1_epi16(key16));
    mask &= 0x3F;
    if (mask)
        return read(cl, Bit::ctz(static_cast<uint32_t>(mask)));
#elif defined(USE_SSE2)
    uint32_t mask = _mm_movemask_epi8(_mm_cmpeq_epi16(cl->key_vec(), _mm_set1_epi16(key16)));
    mask &= 0x00AAAAAA;
    if (mask)
        return read(cl, Bit::ctz(mask) / 2);
#else
    for (int i = 0; i < ClusterSize; ++i)
        if (cl->key[i] == key16)
            return read(cl, i);
#endif

    // Find an entry to be replaced according to the replacement strategy
    int replacei     = 0;
    int replacescore = cl->replace_score(replacei, generation8);
    for (int i = 1; i < ClusterSize; ++i)
    {
        int currentscore = cl->replace_score(i, generation8);
        if (replacescore > currentscore)
        {
            replacei     = i;
            replacescore = currentscore;
        }
    }

    return {false,
            TTData{Move::none(), VALUE_NONE, VALUE_NONE, DEPTH_ENTRY_OFFSET, BOUND_NONE, false},
            TTWriter(cl, replacei)};
}

std::tuple<bool, TTData, TTWriter> TranspositionTable::read(Cluster* cl, int i) const {
    const TTEntryB e = TTEntryB::from_raw(cl->entry[i]);
    return {e.is_occupied(),
            TTData{
              Move(e.move16),
              Value(e.value16),
              Value(e.eval16),
              e.depth(),
              e.bound(),
              e.is_pv(),
            },
            TTWriter(cl, i)};
}

Cluster* TranspositionTable::cluster(const Key key) const {
    return &table[mul_hi64(key, clusterCount)];
}

}  // namespace Stockfish
