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

#include "position.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string_view>
#include <utility>

#include "bitboard.h"
#include "misc.h"
#include "movegen.h"
#include "syzygy/tbprobe.h"
#include "tt.h"
#include "uci.h"

using std::string;

namespace Stockfish {

namespace Zobrist {

std::array<std::array<Key, SQUARE_NB>, PIECE_NB> psq;
std::array<Key, FILE_NB>                         enpassant;
std::array<Key, CASTLING_RIGHT_NB>               castling;
Key                                              side, noPawns;

}

namespace {

constexpr std::string_view PieceToChar(" PNBRQK  pnbrqk");

static constexpr Piece Pieces[] = {W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
                                   B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING};
}  // namespace


// Returns an ASCII representation of the position
std::ostream& operator<<(std::ostream& os, const Position& pos) {

    os << "\n +---+---+---+---+---+---+---+---+\n";

    for (Rank r = RANK_8; r >= RANK_1; --r)
    {
        for (File f = FILE_A; f <= FILE_H; ++f)
            os << " | " << PieceToChar[pos.piece_on(make_square(f, r))];

        os << " | " << (1 + r) << "\n +---+---+---+---+---+---+---+---+\n";
    }

    os << "   a   b   c   d   e   f   g   h\n"
       << "\nFen: " << pos.fen() << "\nKey: " << std::hex << std::uppercase << std::setfill('0')
       << std::setw(16) << pos.key() << std::setfill(' ') << std::dec << "\nCheckers: ";

    for (Bitboard b = pos.checkers(); b;)
        os << UCIEngine::square(pop_lsb(b)) << " ";

    if (Tablebases::MaxCardinality >= popcount(pos.pieces()) && !pos.can_castle(ANY_CASTLING))
    {
        StateInfo st;

        Position p;
        p.set(pos.fen(), pos.is_chess960(), &st);
        Tablebases::ProbeState s1, s2;
        Tablebases::WDLScore   wdl = Tablebases::probe_wdl(p, &s1);
        int                    dtz = Tablebases::probe_dtz(p, &s2);
        os << "\nTablebases WDL: " << std::setw(4) << wdl << " (" << s1 << ")"
           << "\nTablebases DTZ: " << std::setw(4) << dtz << " (" << s2 << ")";
    }

    return os;
}


// Implements Marcel van Kervinck's cuckoo algorithm to detect repetition of positions
// for 3-fold repetition draws. The algorithm uses two hash tables with Zobrist hashes
// to allow fast detection of recurring positions. For details see:
// http://web.archive.org/web/20201107002606/https://marcelk.net/2013-04-06/paper/upcoming-rep-v2.pdf

// First and second hash functions for indexing the cuckoo tables
inline int H1(Key h) { return (h >> 51) & 0x1fff; }
inline int H2(Key h) { return (h >> 35) & 0x1fff; }

// Cuckoo tables with Zobrist hashes of valid reversible moves, and the moves themselves
std::array<Key, 8192>  cuckoo;
std::array<Move, 8192> cuckooMove;

// Initializes at startup the various arrays used to compute hash keys
void Position::init() {

    PRNG rng(1070372);

    Zobrist::psq[W_PAWN] = {
      0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
      0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
      0x0080000000000000, 0x0040000000000000, 0x8120000000000000, 0x4090000000000000,
      0x2048000000000000, 0x9124000000000000, 0x4892000000000000, 0x2449000000000000,
      0x9324800000000000, 0x4992400000000000, 0x24c9200000000000, 0x9364900000000000,
      0xc892380000000000, 0x64696c0000000000, 0xb334b60000000000, 0x59ba2b0000000000,
      0x2cdd158000000000, 0x974efac000000000, 0xca870d6000000000, 0x654386b000000000,
      0xb3a1c35800000000, 0x59d0e1ac00000000, 0xade87ace00000000, 0xd7d4477f00000000,
      0xeaea29a780000000, 0xf4751ecbc0000000, 0xfb1aff65e0000000, 0x7dad05aaf0000000,
      0x3ed682d578000000, 0x9e6b4b72bc000000, 0xce35a5b95e000000, 0x671ad2dcaf000000,
      0x33ad1375ab800000, 0x19f6f3a129c00000, 0x8ddb09d368e00000, 0xc7ed84e9b4700000,
      0xe2f6c274da380000, 0xf07b6b226d1c0000, 0xf93db592ca8e0000, 0x7cbeaac965470000,
      0x3e5f55674ea38000, 0x9e0fd0aba751c000, 0x4f2798562f3d2000, 0x27b3b630eb9e9000,
      0x13f9ab1b89cf4800, 0x88fcd58e38726400, 0xc55e10df1cacf200, 0xe38f72778ec3b900,
      0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
      0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    };

    Zobrist::psq[W_KNIGHT] = {
      0x3f3eae66760fc55e, 0x1f9f5d2b3b92030f, 0x0fefde9661c90187, 0x86d79550cce480c3,
      0xc24bbaa866724061, 0xe005ad5433ace030, 0x7002dcb1e54391b0, 0x38211e5b0ea1e970,
      0x1c30ff2d87c51510, 0x0e38058d3fe2ab28, 0x071c08dd63f1559c, 0x038e0e754d6d4b6e,
      0x01c70d215a2365bf, 0x81e38690ad84537f, 0xc1f1c34baa57c81f, 0xe1f8e1a5d5be05af,
      0xf1fc7ac916df02df, 0xf9de4d648b6fa0cf, 0xfdef2ca9b9b7d067, 0xfff79657204e2833,
      0xfedbb133902735b1, 0xfe4da899c813bb70, 0xfe26d44ce409fc18, 0x7f33103e72911fa4,
      0x3fb9f20739488fda, 0x1ffc831860a4664d, 0x8efe418c30c7f32e, 0xc67f2ade18f63997,
      0xe23f9f770c7b3d6b, 0x713fb5a3863d9ebd, 0x38bfaad1c31eeefe, 0x1c5fd56b1d1a96df,
      0x8f0f9ab6728d6ac7, 0x47a7b74339d3756b, 0x23f3aba2607c7ab5, 0x90f9d5d130abdcfa,
      0xc95c9ae898c00fdd, 0xe58e3d744c602646, 0x72e764a226301323, 0xb873b25113180999,
      0xdd19a92b758c2564, 0x6e8cd49646c612ba, 0x37661053236328fd, 0x9a93782a6db1b5de,
      0x4d69cc16cad8fb4f, 0xa7b4ec1365f9bdaf, 0x53da760a4efcded7, 0xa8cd4b0527eb8ecb,
      0xd566af996ff5c76d, 0xebb357cf4b6f23b6, 0x75f9d1fc59b7b07b, 0xbbdc92e5d0dbf995,
      0xdcce336ae8f81d6a, 0xef4769b5747c0ebd, 0xf6a3bec2ba3e26fe, 0x7b71af615d1f32df,
      0xbcb8d7b3528fb8cf, 0x5e7c1bd9a947dc67, 0xae1e7def28362e33, 0x572f44ef941b36b1,
      0x2b97a86fca0dbaf8, 0x15cbde2fe506fcd4, 0x8bc5950c0e169fc2, 0xc4c2b09e070b4fe9,
    };

    Zobrist::psq[W_BISHOP] = {
      0xe3412254ff1067fc, 0xf0a09b31831dd256, 0x78703d9b3d1b0883, 0xbd186ece628d8449,
      0x5e8c3d7f3146c22c, 0x2f6664a46436a11e, 0x96b3384a321b712f, 0x4b79ec2519987897,
      0xa4bcfc097059dde3, 0x527e0e04b8b92ef1, 0xa83f0d1a5c5cb6d8, 0x541f868d2e2e7acc,
      0x2a0fc95e9782fd66, 0x1507eeb4b754beb3, 0x0a83fd41a7aa7ef9, 0x05618ea32f40ded4,
      0x83b0c7526b358eca, 0x41d863aac99ac76d, 0xa1ec31d69858a3b6, 0xd1d662f34c2c7073,
      0xe9eb3179a683d999, 0xf5d5e8bcd3d40d64, 0xfbeafe4595ea06ba, 0xfcd5053936f522f5,
      0xff6a829c9b7ab0d2, 0xfeb54b55b1bd79c9, 0x7f5aafb1244b7ce4, 0xbead57d892b07e72,
      0x5f76dbec49cdde99, 0xae9b17edd8730eec, 0x576df1eeec398776, 0xaa9682ef761ce21b,
      0x554b4b6fbb9b90a5, 0xaba5afac21cdc85a, 0x55d2ddcdec73058d, 0xabc914fef63982ce,
      0xd4c4f0677b1cc167, 0xeb420228411ba0b3, 0xf4a10117dc8df1f9, 0x7a508a93eed3195c,
      0xbc284549f769ad06, 0x5e1422a707b4d68b, 0x2f0a1b487f4fab45, 0x17a57da7c3a7d5aa,
      0x0bf2c4c81d462add, 0x84f96267f2a334ce, 0xc37cbb2bf9519a67, 0xe09e2d9600a8cd33,
      0x704f1cd300c18739, 0xb907fe698060e234, 0x5ca38f34c0a590ba, 0x2e51cd826052e9fd,
      0x9628e6c130bc9556, 0x4b147360985e6b0b, 0x25aa49b04cbaf58d, 0x12d524d826c8bac6,
      0x886a926c13645d6b, 0xc515332df527eebd, 0x62aae38d060616fe, 0x31557bde83032ad7,
      0x998ac7f4bd81b4cb, 0x4cc569e1a2c0da6d, 0xa762b4f0d1f5ad36, 0xd2912a7b94faf733,
    };

    Zobrist::psq[W_ROOK] = {
      0x6948953dcae89a39, 0xb5843086e5746cbc, 0x5ae262588e2fd7f6, 0xac71312c47820a53,
      0xd718e28ddfc12489, 0x6b8c7b5d13e0924c, 0x35e647b575f0492e, 0x9bf329c1466de497,
      0xccf994e0a3a313eb, 0xe75cba73add189f5, 0xf28e2d3a2ae8e55a, 0x7947169d15e19305,
      0xbd83f15576f0c982, 0x5ee182b2bbeda4c9, 0xae70c15aa1f6d26c, 0xd63860aeac6ea93e,
      0x6b1c3a4f56377537, 0x35ae673fab8e5b3b, 0x1ad739842952cc3d, 0x8c4be6d9e83c87b6,
      0xc725f36cf41e627b, 0x63b283ae7a9ad095, 0x31d94bcf3dd889e2, 0x99ecaffc62ec44f9,
      0xcdf65de631e3c3d4, 0xe7db54e8e4640042, 0xf2cdda7472320021, 0xf846972239190018,
      0xfd033b92e08ca1a4, 0x7ea1edc9704650d2, 0x3f50f6e4b8b6c9c1, 0x9e88016a5ccea4e8,
      0x4f4400b52e67527c, 0xa6a20a4297a6489e, 0x53510522b746c5e7, 0xa8a88292a736a2fb,
      0x5454414aaf9b70dd, 0xab2a20a6ab5859c6, 0x55951a48a939eceb, 0x2aeafd27a809367d,
      0x9455048bd404ba96, 0xcb2a8245ea027ce3, 0x65954b3af594fe79, 0x32caa59e865f9e9c,
      0x986558d7432feeee, 0xcd12d6705d023777, 0x66a91b3bd2813a13, 0x3374fd9de940bca1,
      0x989a04d508a05e50, 0x4c4d087284c5ce80, 0xa726843942f72740, 0x5393421ca17b93a0,
      0x29c9ab15acbdc9d0, 0x95e4df92d6cb0548, 0xcbd21fc96b6582a4, 0xe4c97fe749b2c15a,
      0xf344c5e8584c8105, 0xf8a262f42c264082, 0x7c513b621686e041, 0xbf08edb10bd6b028,
      0x5f8476db79eb5814, 0x2fe24b6e40604da2, 0x96f12faf20a5e6d9, 0xca789dcf90c712c4,
    };

    Zobrist::psq[W_QUEEN] = {
      0xe41c34ffc8638962, 0x722e6067e431c4b1, 0x39173a2bf28d03f8, 0x1cabed15f946a05c,
      0x0e55fc9100a3718e, 0x860a8e4880c478c7, 0x4305472440f7fc63, 0x2182a98a20ee3e31,
      0x10c154c510773eb0, 0x8940d07a883bbef0, 0xc580183d441dfed8, 0x62e07606a29b1ecc,
      0xb050411b514d8f66, 0xd928208e54a6c7b3, 0x6c941a5f2ac68271, 0x366a773795636098,
      0x9a154180362451e4, 0x4d0aa0c01b87e8f2, 0x26855063f15615d1, 0x1362d83204ab2b40,
      0x88911c19025595a0, 0x4468fe0c812acad0, 0xa314051dbc0084c8, 0x518a0896de00426c,
      0x28c50e536f95e13e, 0x9562872a4b5f113f, 0xcbb14396d9afa93f, 0x65d8abd090d7f537,
      0xb3ec55e848fe1b3b, 0xd8d65af4247f2c35, 0xed4b5762123fb7b2, 0xf785dbb1091ffa79,
      0x7be29ddb781a1c9c, 0xbcd13eedbc0d2fe6, 0xdf48e56ede0697fb, 0xeea478af6f036a55,
      0x7772464c4b145482, 0xbab9293dd91fea49, 0x5d5c9e85101a352c, 0xaf8e355a880d1a9e,
      0x57e76aad4406ace7, 0xaaf3bf4ea2035673, 0xd459a5bf51944a91, 0xeb2cd8c4545fc4e8,
      0x75b6167a2aba2274, 0x3afb7b3d155d3092, 0x9c5dc78576aeb9e9, 0xcf2ee9dabb575cfc,
      0x679774eea13e4fde, 0x33ebc06cac0ac64f, 0x98f5ea2e5690a327, 0xcd7aff0f2b48519b,
      0xe79d059c6931c96d, 0x73ce88d5c80d24be, 0xb8e74e72e406b3ff, 0xdd73a73972037857,
      0xefb9d39cb9945d8b, 0x77fc93d5a05feecd, 0xbade33f2d0ba3766, 0xdc4f69f9685d1bbb,
      0xef27b4fcb42eac75, 0x77b3a0665a17779a, 0x3bd9da2b2d9e5a6d, 0x9ccc9d166a5aed3e,
    };

    Zobrist::psq[W_KING] = {
      0xcf46349335b89737, 0xe6836a4a66dc6a3b, 0x7341b52533fbd4bd, 0xb880a08965680bfe,
      0x5c4050474eb42457, 0xaf00523ba75a338b, 0x57a0591e2fad19c5, 0x2bf05694ebd68ce2,
      0x94d8515189eb4679, 0xcb4c58ab38604294, 0xe4865c559ca5c0e2, 0x72635432cec72071,
      0xb811da196763b190, 0x5c289d0f4fb1f968, 0x2e34349c5b4d3cbc, 0x173a6055d1a6bff6,
      0x0b9d3a3114d37e53, 0x05eeed188afc5e89, 0x83f7768c45ebef44, 0xc0dbc15dde6037a2,
      0xe16deab6ef301bd1, 0xf1b6ff408b982c48, 0x78fb0fa3b9cc162c, 0xbd7d87d220e60b16,
      0xdfbec3e91073242b, 0x6fdf61f48839921d, 0xb6efbae2441ce8a6, 0xda57ad71229bb453,
      0xec2bd6b8914dfb81, 0x76359b5fb4333dc0, 0x3b3ab7b7da199ee0, 0x1dbd21c3ed0ccf70,
      0x0ede90e20a138618, 0x864f38710509e2a4, 0xc207ec3b7e11315a, 0x6103f61dbf08b905,
      0x30a1811523845c8a, 0x1850ca916d57ee4d, 0x8d28654b4a3e3726, 0x469432a5a51f1b93,
      0x236a63492e8fac61, 0x90b531a49747d638, 0x487ae2c9b7360ab4, 0xa53d7167279b24fa,
      0x52bec2a86fcdb3d5, 0x295f6157cbe6f842, 0x95afbab01966bc29, 0x4af7ad5bf0b35e14,
      0xa47bd6adf8cc4ea2, 0xd31d914efcf3e759, 0x69aeb2bf7eec120c, 0x34f72347bf76090e,
      0x9b7b9bb823bb048f, 0xcc9dbddfeddd8247, 0x666ea4f40aeec123, 0xb237586205e2a091,
      0x593bdc32fef171e8, 0x2cbd9e197fed78f4, 0x167ebf0f43635dd2, 0x8a1f259c5d244f49,
      0x450f98d5d207e7a4, 0x22a7b672e99633d2, 0x1173ab3a88cb3841, 0x89b9d59d44659c20,
    };

    Zobrist::psq[B_PAWN] = {
      0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
      0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
      0x4be8655c1424ba09, 0xa4f438b60a125d0c, 0xd35a6643059cee8e, 0xe8ad33227e5bb747,
      0x7476e9913f2ddba3, 0xbb3b74cb63032dd9, 0x5dbdca664d81b744, 0x2edeef28dac0dbaa,
      0x966f77946df5add5, 0xca17c1d1cafaf74a, 0x650be0e8e5e8bbad, 0x3285f0778ef45dd6,
      0x19628223c7efcf4b, 0x8db141121f6227a5, 0x46d8a08af3b113d2, 0xa26c504685d8a849,
      0xd0165238beec542c, 0x682b591c5fe3ea16, 0x3435d695d36414a3, 0x1a3a915115b20a59,
      0x0d3d38ab76d92484, 0x06beec55bb6c9242, 0x035f7c3121b64921, 0x808fce1b6c4ee490,
      0x4047e70db6b293e8, 0xa123f99edb5949fc, 0x50b186d491394556, 0x2858c971b409430b,
      0x952c64b8da916185, 0x4a96325c6ddd70c2, 0x256b6335ca7b7869, 0x93b5bb82e5a87c3c,
      0x49faadc28e41dfb6, 0xa5fd56e147b50e73, 0xd3dedb735fdaa699, 0xe8cf1dba53ed72ec,
      0xf547feded563797e, 0xfb83857496245d1f, 0x7dc1c8a24b87cf27, 0xbfe0e452d956279b,
      0xdef0722a90ab3265, 0xee5849154855993a, 0xf62c2e92a42aed3d, 0x7b16174952809736,
      0x3dab7ba4a9406a33, 0x1ef5c7c9a835d4b1, 0x8e7ae3e4d48f0bf8, 0xc63d7bea6a47a45c,
      0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
      0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
    };

    Zobrist::psq[B_KNIGHT] = {
      0xeda6149e1f83af21, 0x76f37054f3c1d790, 0xba59c23185750a68, 0xdc2ce11b3eba8534,
      0x6e16708d9f5d633a, 0x372b425d333b503d, 0x1b95ab35650849b6, 0x0dcadf814e11c57b,
      0x87c51fc0a79d0315, 0xc2c2ffe3afcea02a, 0xe0410ff22be7501d, 0xf12087fae96649a6,
      0x789043fe8826e4d3, 0x3c482be7448693c1, 0x9f241feba24349e0, 0x4fb27ff5d1b464f8,
      0x27f945e1144fd3dc, 0x92fca2f08ab20846, 0xc87e51784559042b, 0xe51f58bfdeac8215,
      0x72afd647ef5660aa, 0x397791380b3ef055, 0x9d9bb89ff90a9982, 0x4eeda65400854cc9,
      0xa676d93200d7666c, 0xd21b1c9900fe7336, 0x692dfe4c80ead833, 0x34b6853e40e08db1,
      0x1a5b488720706770, 0x8c2dae5b90add210, 0x4616d72dc8c308a0, 0x232b118ee4618450,
      0x11b5f2df7230e388, 0x08fa8377b98db1c4, 0x857d4ba020c6d8ea, 0xc3bea5d010f6ac7d,
      0x61df52e8087b7796, 0xb1cfd97404a85a63, 0xd9c796a202c1ed39, 0xedc3bb5101f5173c,
      0xf7c1adab7cfaaa3e, 0xfae0d6d5be7d74b7, 0xfc501172dfab5bfb, 0xff0878ba93404c55,
      0x7fa44c5eb535c78a, 0x3fd22c34a60f23c5, 0x9ee91c02530791e2, 0xce54fe02d583c8f9,
      0xe60a0f02965405d4, 0x730507814b2a234a, 0x398283c3599511ad, 0x1cc141e250ca88d6,
      0x8f60a0f1286565c3, 0xc6b0507894a772e1, 0x6378583c4ac67978, 0xb09c560625f6dd14,
      0x586e5118ee6e8f22, 0xad17588c77374799, 0x56abd65dc70e426c, 0x2b7591351f12e136,
      0x949ab281731c913b, 0x4a6d2943458e693d, 0xa43694a25e52d53e, 0x523b3a512fbc8b37,
    };

    Zobrist::psq[B_BISHOP] = {
      0x293ded2b6bde643b, 0x149ef696497ad3b5, 0x0a6f0150d828887a, 0x843780a86c14659d,
      0x421bc054369fd366, 0x210dea321bda29b3, 0x1086f51af1ed3571, 0x08630a8e84f6bb18,
      0x85318f5f427b7c2c, 0x4298cdb7a1a87e16, 0x214c6cc02c41deab, 0x91a6366016b52f5d,
      0x48f36b300b5ab606, 0xa579b59bf9ad5b0b, 0xd39caace00436d8d, 0x69ce5f7f00b476ce,
      0xb5c755a780cffb67, 0xdbc3d0cbc0f23dbb, 0xecc19865e0793f75, 0xf740b62af03cbe1a,
      0xfa802b15781e7ea5, 0x7d401f92bc9aff52, 0xbf807fc95ed89e09, 0x5fe04fe4af6c4f04,
      0xaef02de9ab23e782, 0xd67816f7290433c1, 0xea1c7160688219e8, 0x752e48b034410cf4,
      0x3a9724581a20a7d2, 0x1d4b922c0d107241, 0x8f85b30dfa1df920, 0x47e2a39efd9b3c98,
      0xa2f15bd482cdbfec, 0xd058d7f24166dff6, 0xe90c1bfadc268e53, 0x74a67dfd6e136689,
      0x3a7344e6b79c734c, 0x9c39a868a75bf9ae, 0x4e1cd437af383cd7, 0x272e10002b9c3fc3,
      0x13b77803e9ce1fe9, 0x09fbcc0208e70ff4, 0x85fde6010473a652, 0xc3fef3008239f281,
      0xe0df098041893948, 0xf16f84c3dcc49ca4, 0xf9b7c261ee624e52, 0x7cdbe130f7a4c681,
      0xbf6df09b8747a348, 0xde96884e3fa3d1a4, 0x6f4b4e3ce34428d2, 0xb6a5ad058da235c1,
      0x5b52dc993ad11ae0, 0xac891e4c9d688d78, 0x5664f53db2b4671c, 0xaa120086d9cfd226,
      0x55090a589072291b, 0x2a84852c48393525, 0x1542488e241c9a9a, 0x8ba12e5f120e6ced,
      0x45d09d378992f67e, 0xa3c83480385c9a9f, 0xd0c46a401c2e6ce7, 0xe96235200e82f67b,
    };

    Zobrist::psq[B_ROOK] = {
      0xf5916a9007d49a9d, 0x7ac8b54bffea6ce6, 0xbc442aa60360f673, 0xdf221f48fd259a99,
      0x6fb17fa78292ecec, 0x37f8c5cbc1dcb676, 0x9afc62e61cee7a9b, 0xcc7e3b6b0ee2dced,
      0xe71f6db587e4ae76, 0x738fbcc13ff27693, 0x39e7ae63636cdae9, 0x9df3d7324d23ad7c,
      0xcfd99b9ada91f71e, 0xe6ccbdcd6ddd1a2f, 0xf24624fd4aee8d1f, 0xf8231866a577672f,
      0x7c31f628ae2e739f, 0x3e388b145782d86f, 0x1f1c4f91d754ac3f, 0x0f8e27cb17aa77b7,
      0x07c713e67740da7b, 0x82c3f3e8c7358c9d, 0xc04189f79f9ae7e6, 0xe120cee03358b3fb,
      0x70906773e5ac7855, 0x384833ba0e43dd82, 0x9d0469dd07b42ec1, 0x4e823ef57fda1760,
      0x2761656143ed0bb0, 0x92b0b2b35df6a478, 0x4978295a52fb7394, 0xa5bc14ad29e85862,
      0x52fe704d6861ec31, 0xa85f423eb4a517b0, 0xd52fab075a52aa78, 0x6a97df9bad29749c,
      0x356b9fce2a015be6, 0x9b95b5ff15956df3, 0x4deaa0e4765f5759, 0xa7f55a6a3bba4a04,
      0xd2dadd36e148e50a, 0xe84d14808c31b28d, 0xf506fa404618d946, 0x7aa30d202399aca3,
      0x3d518693edccd659, 0x9fa8c34a0a738a8c, 0x4fd461a50539c54e, 0xa6ea3ac97e0922af,
      0xd2556d64bf04915f, 0xe82abca9a382690f, 0x74352e572d54f487, 0x3a1a9d306a3fba4b,
      0x1d2d3e98351fdd2d, 0x0eb6ef4fe61a2e96, 0x075b7dbff30d36eb, 0x828dc4c405869b75,
      0x4146e879fec36c1a, 0xa1a3743cfff457a5, 0x50f1c005836febda, 0xa978ea193d221445,
      0xd5bc750f62910a2a, 0x6afe409fb1488515, 0xb47f2a5424a4632a, 0xdb3f9f3212c7f19d,
    };

    Zobrist::psq[B_QUEEN] = {
      0x6dbfbf9909f6196e, 0x36ffafcf78fb0cbf, 0x9a7fddffbc7da7ff, 0xcc1f94e7de3ef257,
      0x662fb06bef8a988b, 0x3337a8360bc54c45, 0x199bde00f977662a, 0x0ced9f03802e7315,
      0x8756bf81c082d822, 0xc28b2fc0e0d4ac11, 0x614597e0706a77a8, 0xb182bbf038a0fbdc,
      0x58e12df81cc59c46, 0xad7096fc0e62ce2b, 0xd798316607a4a71d, 0x6bec62a8ffd27226,
      0xb4f63157837cf91b, 0xdb5b62b03d2b9d25, 0xecadb15be295ce9a, 0x7676a8adf1df06e5,
      0xba3b5e4d04ef8372, 0x5d3dd53e8277e011, 0x2ebe908741ae11a8, 0x177f32585cd708d4,
      0x8a9fe92c2e6ba5ca, 0x454ffe8e1735d2ed, 0xa387855cf70f297e, 0x51c3c8b58787b517,
      0xa9e1ee413fc3fb2b, 0xd5f0f72363743d95, 0xebd80b924dba3f6a, 0xf4ec05cadadd1fb5,
      0xfb7602e56d6eae72, 0xfcbb0b694ab77699, 0x7e5d85b4a5ce5aec, 0xbe2ec8c1ae72ed76,
      0x5f176460d7ac9713, 0x2f8bb23397d66a29, 0x17e5a91a377ef51c, 0x8af2d48ee72a9b2e,
      0xc459105c8f954d97, 0xe30cf235bb5f476b, 0x71a60301213a63bd, 0x38d301836c08d076,
      0x9d6980c1b691899b, 0xcfb4c060db48e56d, 0x67da60339131b2be, 0xb2ed301a3498f8f7,
      0xd856e80d1ad99dd3, 0xed2b7e1e8d6cef41, 0x76b5c514ba23b7a8, 0x3b5ae8925d11dbdc,
      0x9cad744ad21d0c4e, 0x4e76ca25690e8627, 0xa63b6f0948874313, 0x531db784a4d64021,
      0x29aea1da52fee010, 0x14d750ed29ea91a8, 0x8b4bd26d68f548dc, 0xc485932eb4ef45c6,
      0x6262b38f5ae262e3, 0xb01123dfade4f171, 0x58089bf42a679918, 0x2c2437e21533ed24};

    Zobrist::psq[B_KING] = {
      0x16326bf2f60c369a, 0x0b1935f97b063ae5, 0x05aceaff41831d7a, 0x02d67f645cc1af1d,
      0x804b45aa2e60f62e, 0xc125a2d517a5bb1f, 0x6092db7177d2fc27, 0x30691dbb477cbe13,
      0x9914fede5fbe7ea9, 0x4caa0574d34aff5c, 0x265508a195309e06, 0x922a845336984f03,
      0x491542299bd9e781, 0x248aa117317933c0, 0x12455a9064bc99e0, 0x8802dd48325e4cf8,
      0x44211ea419bac7dc, 0x2230f549f0488246, 0x11380aa4f824412b, 0x089c0f4a7c87e09d,
      0x044e07a53ed611ee, 0x832709ca9f6b08ff, 0x419384e6b3b5a5d7, 0x20c9c868a5daf34b,
      0x9164e437ae78b9a5, 0xc9b27803d73c5cda, 0x64f94c02170bcfc5, 0xb37ca602f71027e2,
      0xd8be5302878813f1, 0x6c7f5982bfc42858, 0xb71fdcc2a3e2358c, 0x5baf9e62adf11ace,
      0x2df7bf32aaf88d67, 0x97dbaf99557c46b3, 0xcaedd7cf562bc2f1, 0xe45691ffab8000d8,
      0xf30b32e429c021cc, 0x79a5e369e8e010e6, 0x3cd2f1b4f4700873, 0x9f4902c27a382599,
      0xcea481613d1c336c, 0x675240b3628e19be, 0xb2a92059b1472d77, 0x5954902f24a3b71b,
      0xad8a320f9251fa2d, 0x56e5631fc9bd3d16, 0xaa72bb9418debf2b, 0xd41927d20c6f5f95,
      0x6a0c93e906a24e62, 0x352639f483c4e731, 0x1ab366e1bd779238, 0x0d59b37322bbe8bc,
      0x878ca9b991c815fe, 0x43c654df34e42b5f, 0xa0c350779a72340f, 0xd141d223cd391a07,
      0xe98099121a9c8d0b, 0x74e03c890d4e468d, 0xbb506e477a32e346, 0xdca83d3bbd8cb1a3,
      0x6e746e9e22c658d1, 0xb63a3d5711f6cdc8, 0x5b3d64b0746ea6ec, 0x2d9eb2583a37537e,
    };

    Zobrist::enpassant = {
      0x16ef292c1d8e4817, 0x8a779e8df252c5a3, 0xc41bb55ef9bca2d1, 0x622da0b480de70c8,
      0x3116da4240faf86c, 0x18ab1d2120e8bc3e, 0x0c75fe9090747fbf, 0x871a8f4848afde7f,
    };

    Zobrist::castling = {
      0x0000000000000000, 0x438d47a424c20e9f, 0x21c6a9ca126126ef, 0x624bee6e36a32870,
      0x91e354e509309377, 0xd26e13412df29de8, 0xb025fd2f1b51b598, 0xf3a8ba8b3f93bb07,
      0xc9d1d06978986813, 0x8a5c97cd5c5a668c, 0xe81779a36af94efc, 0xab9a3e074e3b4063,
      0x5832848c71a8fb64, 0x1bbfc328556af5fb, 0x79f42d4663c9dd8b, 0x3a796ae2470bd314,
    };

    Zobrist::side    = 0xe5c89834bcd9d5a1;
    Zobrist::noPawns = 0x7e3dbccb153449fd;

    // Prepare the cuckoo tables
    cuckoo.fill(0);
    cuckooMove.fill(Move::none());
    [[maybe_unused]] int count = 0;
    for (Piece pc : Pieces)
        for (Square s1 = SQ_A1; s1 <= SQ_H8; ++s1)
            for (Square s2 = Square(s1 + 1); s2 <= SQ_H8; ++s2)
                if ((type_of(pc) != PAWN) && (attacks_bb(type_of(pc), s1, 0) & s2))
                {
                    Move move = Move(s1, s2);
                    Key  key  = Zobrist::psq[pc][s1] ^ Zobrist::psq[pc][s2] ^ Zobrist::side;
                    int  i    = H1(key);
                    while (true)
                    {
                        std::swap(cuckoo[i], key);
                        std::swap(cuckooMove[i], move);
                        if (move == Move::none())  // Arrived at empty slot?
                            break;
                        i = (i == H1(key)) ? H2(key) : H1(key);  // Push victim to alternative slot
                    }
                    count++;
                }
    assert(count == 3668);
}


// Initializes the position object with the given FEN string.
// This function is not very robust - make sure that input FENs are correct,
// this is assumed to be the responsibility of the GUI.
Position& Position::set(const string& fenStr, bool isChess960, StateInfo* si) {
    /*
   A FEN string defines a particular position using only the ASCII character set.

   A FEN string contains six fields separated by a space. The fields are:

   1) Piece placement (from white's perspective). Each rank is described, starting
      with rank 8 and ending with rank 1. Within each rank, the contents of each
      square are described from file A through file H. Following the Standard
      Algebraic Notation (SAN), each piece is identified by a single letter taken
      from the standard English names. White pieces are designated using upper-case
      letters ("PNBRQK") whilst Black uses lowercase ("pnbrqk"). Blank squares are
      noted using digits 1 through 8 (the number of blank squares), and "/"
      separates ranks.

   2) Active color. "w" means white moves next, "b" means black.

   3) Castling availability. If neither side can castle, this is "-". Otherwise,
      this has one or more letters: "K" (White can castle kingside), "Q" (White
      can castle queenside), "k" (Black can castle kingside), and/or "q" (Black
      can castle queenside).

   4) En passant target square (in algebraic notation). If there's no en passant
      target square, this is "-". If a pawn has just made a 2-square move, this
      is the position "behind" the pawn. Following X-FEN standard, this is recorded
      only if there is a pawn in position to make an en passant capture, and if
      there really is a pawn that might have advanced two squares.

   5) Halfmove clock. This is the number of halfmoves since the last pawn advance
      or capture. This is used to determine if a draw can be claimed under the
      fifty-move rule.

   6) Fullmove number. The number of the full move. It starts at 1, and is
      incremented after Black's move.
*/

    unsigned char      col, row, token;
    size_t             idx;
    Square             sq = SQ_A8;
    std::istringstream ss(fenStr);

    std::memset(this, 0, sizeof(Position));
    std::memset(si, 0, sizeof(StateInfo));
    st = si;

    ss >> std::noskipws;

    // 1. Piece placement
    while ((ss >> token) && !isspace(token))
    {
        if (isdigit(token))
            sq += (token - '0') * EAST;  // Advance the given number of files

        else if (token == '/')
            sq += 2 * SOUTH;

        else if ((idx = PieceToChar.find(token)) != string::npos)
        {
            put_piece(Piece(idx), sq);
            ++sq;
        }
    }

    // 2. Active color
    ss >> token;
    sideToMove = (token == 'w' ? WHITE : BLACK);
    ss >> token;

    // 3. Castling availability. Compatible with 3 standards: Normal FEN standard,
    // Shredder-FEN that uses the letters of the columns on which the rooks began
    // the game instead of KQkq and also X-FEN standard that, in case of Chess960,
    // if an inner rook is associated with the castling right, the castling tag is
    // replaced by the file letter of the involved rook, as for the Shredder-FEN.
    while ((ss >> token) && !isspace(token))
    {
        Square rsq;
        Color  c    = islower(token) ? BLACK : WHITE;
        Piece  rook = make_piece(c, ROOK);

        token = char(toupper(token));

        if (token == 'K')
            for (rsq = relative_square(c, SQ_H1); piece_on(rsq) != rook; --rsq)
            {}

        else if (token == 'Q')
            for (rsq = relative_square(c, SQ_A1); piece_on(rsq) != rook; ++rsq)
            {}

        else if (token >= 'A' && token <= 'H')
            rsq = make_square(File(token - 'A'), relative_rank(c, RANK_1));

        else
            continue;

        set_castling_right(c, rsq);
    }

    // 4. En passant square.
    // Ignore if square is invalid or not on side to move relative rank 6.
    bool enpassant = false;

    if (((ss >> col) && (col >= 'a' && col <= 'h'))
        && ((ss >> row) && (row == (sideToMove == WHITE ? '6' : '3'))))
    {
        st->epSquare = make_square(File(col - 'a'), Rank(row - '1'));

        // En passant square will be considered only if
        // a) side to move have a pawn threatening epSquare
        // b) there is an enemy pawn in front of epSquare
        // c) there is no piece on epSquare or behind epSquare
        enpassant = attacks_bb<PAWN>(st->epSquare, ~sideToMove) & pieces(sideToMove, PAWN)
                 && (pieces(~sideToMove, PAWN) & (st->epSquare + pawn_push(~sideToMove)))
                 && !(pieces() & (st->epSquare | (st->epSquare + pawn_push(sideToMove))));
    }

    if (!enpassant)
        st->epSquare = SQ_NONE;

    // 5-6. Halfmove clock and fullmove number
    ss >> std::skipws >> st->rule50 >> gamePly;

    // Convert from fullmove starting from 1 to gamePly starting from 0,
    // handle also common incorrect FEN with fullmove = 0.
    gamePly = std::max(2 * (gamePly - 1), 0) + (sideToMove == BLACK);

    chess960 = isChess960;
    set_state();

    assert(pos_is_ok());

    return *this;
}


// Helper function used to set castling
// rights given the corresponding color and the rook starting square.
void Position::set_castling_right(Color c, Square rfrom) {

    Square         kfrom = square<KING>(c);
    CastlingRights cr    = c & (kfrom < rfrom ? KING_SIDE : QUEEN_SIDE);

    st->castlingRights |= cr;
    castlingRightsMask[kfrom] |= cr;
    castlingRightsMask[rfrom] |= cr;
    castlingRookSquare[cr] = rfrom;

    Square kto = relative_square(c, cr & KING_SIDE ? SQ_G1 : SQ_C1);
    Square rto = relative_square(c, cr & KING_SIDE ? SQ_F1 : SQ_D1);

    castlingPath[cr] = (between_bb(rfrom, rto) | between_bb(kfrom, kto)) & ~(kfrom | rfrom);
}


// Sets king attacks to detect if a move gives check
void Position::set_check_info() const {

    update_slider_blockers(WHITE);
    update_slider_blockers(BLACK);

    Square ksq = square<KING>(~sideToMove);

    st->checkSquares[PAWN]   = attacks_bb<PAWN>(ksq, ~sideToMove);
    st->checkSquares[KNIGHT] = attacks_bb<KNIGHT>(ksq);
    st->checkSquares[BISHOP] = attacks_bb<BISHOP>(ksq, pieces());
    st->checkSquares[ROOK]   = attacks_bb<ROOK>(ksq, pieces());
    st->checkSquares[QUEEN]  = st->checkSquares[BISHOP] | st->checkSquares[ROOK];
    st->checkSquares[KING]   = 0;
}


// Computes the hash keys of the position, and other
// data that once computed is updated incrementally as moves are made.
// The function is only used when a new position is set up
void Position::set_state() const {

    st->key = st->materialKey = 0;
    st->minorPieceKey         = 0;
    st->nonPawnKey[WHITE] = st->nonPawnKey[BLACK] = 0;
    st->pawnKey                                   = Zobrist::noPawns;
    st->nonPawnMaterial[WHITE] = st->nonPawnMaterial[BLACK] = VALUE_ZERO;
    st->checkersBB = attackers_to(square<KING>(sideToMove)) & pieces(~sideToMove);

    set_check_info();

    for (Bitboard b = pieces(); b;)
    {
        Square s  = pop_lsb(b);
        Piece  pc = piece_on(s);
        st->key ^= Zobrist::psq[pc][s];

        if (type_of(pc) == PAWN)
            st->pawnKey ^= Zobrist::psq[pc][s];

        else
        {
            st->nonPawnKey[color_of(pc)] ^= Zobrist::psq[pc][s];

            if (type_of(pc) != KING)
            {
                st->nonPawnMaterial[color_of(pc)] += PieceValue[pc];

                if (type_of(pc) <= BISHOP)
                    st->minorPieceKey ^= Zobrist::psq[pc][s];
            }
        }
    }

    if (st->epSquare != SQ_NONE)
        st->key ^= Zobrist::enpassant[file_of(st->epSquare)];

    if (sideToMove == BLACK)
        st->key ^= Zobrist::side;

    st->key ^= Zobrist::castling[st->castlingRights];

    for (Piece pc : Pieces)
        for (int cnt = 0; cnt < pieceCount[pc]; ++cnt)
            st->materialKey ^= Zobrist::psq[pc][8 + cnt];
}


// Overload to initialize the position object with the given endgame code string
// like "KBPKN". It's mainly a helper to get the material key out of an endgame code.
Position& Position::set(const string& code, Color c, StateInfo* si) {

    assert(code[0] == 'K');

    string sides[] = {code.substr(code.find('K', 1)),                                // Weak
                      code.substr(0, std::min(code.find('v'), code.find('K', 1)))};  // Strong

    assert(sides[0].length() > 0 && sides[0].length() < 8);
    assert(sides[1].length() > 0 && sides[1].length() < 8);

    std::transform(sides[c].begin(), sides[c].end(), sides[c].begin(), tolower);

    string fenStr = "8/" + sides[0] + char(8 - sides[0].length() + '0') + "/8/8/8/8/" + sides[1]
                  + char(8 - sides[1].length() + '0') + "/8 w - - 0 10";

    return set(fenStr, false, si);
}


// Returns a FEN representation of the position. In case of
// Chess960 the Shredder-FEN notation is used. This is mainly a debugging function.
string Position::fen() const {

    int                emptyCnt;
    std::ostringstream ss;

    for (Rank r = RANK_8; r >= RANK_1; --r)
    {
        for (File f = FILE_A; f <= FILE_H; ++f)
        {
            for (emptyCnt = 0; f <= FILE_H && empty(make_square(f, r)); ++f)
                ++emptyCnt;

            if (emptyCnt)
                ss << emptyCnt;

            if (f <= FILE_H)
                ss << PieceToChar[piece_on(make_square(f, r))];
        }

        if (r > RANK_1)
            ss << '/';
    }

    ss << (sideToMove == WHITE ? " w " : " b ");

    if (can_castle(WHITE_OO))
        ss << (chess960 ? char('A' + file_of(castling_rook_square(WHITE_OO))) : 'K');

    if (can_castle(WHITE_OOO))
        ss << (chess960 ? char('A' + file_of(castling_rook_square(WHITE_OOO))) : 'Q');

    if (can_castle(BLACK_OO))
        ss << (chess960 ? char('a' + file_of(castling_rook_square(BLACK_OO))) : 'k');

    if (can_castle(BLACK_OOO))
        ss << (chess960 ? char('a' + file_of(castling_rook_square(BLACK_OOO))) : 'q');

    if (!can_castle(ANY_CASTLING))
        ss << '-';

    ss << (ep_square() == SQ_NONE ? " - " : " " + UCIEngine::square(ep_square()) + " ")
       << st->rule50 << " " << 1 + (gamePly - (sideToMove == BLACK)) / 2;

    return ss.str();
}

// Calculates st->blockersForKing[c] and st->pinners[~c],
// which store respectively the pieces preventing king of color c from being in check
// and the slider pieces of color ~c pinning pieces of color c to the king.
void Position::update_slider_blockers(Color c) const {

    Square ksq = square<KING>(c);

    st->blockersForKing[c] = 0;
    st->pinners[~c]        = 0;

    // Snipers are sliders that attack 's' when a piece and other snipers are removed
    Bitboard snipers = ((attacks_bb<ROOK>(ksq) & pieces(QUEEN, ROOK))
                        | (attacks_bb<BISHOP>(ksq) & pieces(QUEEN, BISHOP)))
                     & pieces(~c);
    Bitboard occupancy = pieces() ^ snipers;

    while (snipers)
    {
        Square   sniperSq = pop_lsb(snipers);
        Bitboard b        = between_bb(ksq, sniperSq) & occupancy;

        if (b && !more_than_one(b))
        {
            st->blockersForKing[c] |= b;
            if (b & pieces(c))
                st->pinners[~c] |= sniperSq;
        }
    }
}


// Computes a bitboard of all pieces which attack a given square.
// Slider attacks use the occupied bitboard to indicate occupancy.
Bitboard Position::attackers_to(Square s, Bitboard occupied) const {

    return (attacks_bb<ROOK>(s, occupied) & pieces(ROOK, QUEEN))
         | (attacks_bb<BISHOP>(s, occupied) & pieces(BISHOP, QUEEN))
         | (attacks_bb<PAWN>(s, BLACK) & pieces(WHITE, PAWN))
         | (attacks_bb<PAWN>(s, WHITE) & pieces(BLACK, PAWN))
         | (attacks_bb<KNIGHT>(s) & pieces(KNIGHT)) | (attacks_bb<KING>(s) & pieces(KING));
}

bool Position::attackers_to_exist(Square s, Bitboard occupied, Color c) const {

    return ((attacks_bb<ROOK>(s) & pieces(c, ROOK, QUEEN))
            && (attacks_bb<ROOK>(s, occupied) & pieces(c, ROOK, QUEEN)))
        || ((attacks_bb<BISHOP>(s) & pieces(c, BISHOP, QUEEN))
            && (attacks_bb<BISHOP>(s, occupied) & pieces(c, BISHOP, QUEEN)))
        || (((attacks_bb<PAWN>(s, ~c) & pieces(PAWN)) | (attacks_bb<KNIGHT>(s) & pieces(KNIGHT))
             | (attacks_bb<KING>(s) & pieces(KING)))
            & pieces(c));
}

// Tests whether a pseudo-legal move is legal
bool Position::legal(Move m) const {

    assert(m.is_ok());

    Color  us   = sideToMove;
    Square from = m.from_sq();
    Square to   = m.to_sq();

    assert(color_of(moved_piece(m)) == us);
    assert(piece_on(square<KING>(us)) == make_piece(us, KING));

    // En passant captures are a tricky special case. Because they are rather
    // uncommon, we do it simply by testing whether the king is attacked after
    // the move is made.
    if (m.type_of() == EN_PASSANT)
    {
        Square   ksq      = square<KING>(us);
        Square   capsq    = to - pawn_push(us);
        Bitboard occupied = (pieces() ^ from ^ capsq) | to;

        assert(to == ep_square());
        assert(moved_piece(m) == make_piece(us, PAWN));
        assert(piece_on(capsq) == make_piece(~us, PAWN));
        assert(piece_on(to) == NO_PIECE);

        return !(attacks_bb<ROOK>(ksq, occupied) & pieces(~us, QUEEN, ROOK))
            && !(attacks_bb<BISHOP>(ksq, occupied) & pieces(~us, QUEEN, BISHOP));
    }

    // Castling moves generation does not check if the castling path is clear of
    // enemy attacks, it is delayed at a later time: now!
    if (m.type_of() == CASTLING)
    {
        // After castling, the rook and king final positions are the same in
        // Chess960 as they would be in standard chess.
        to             = relative_square(us, to > from ? SQ_G1 : SQ_C1);
        Direction step = to > from ? WEST : EAST;

        for (Square s = to; s != from; s += step)
            if (attackers_to_exist(s, pieces(), ~us))
                return false;

        // In case of Chess960, verify if the Rook blocks some checks.
        // For instance an enemy queen in SQ_A1 when castling rook is in SQ_B1.
        return !chess960 || !(blockers_for_king(us) & m.to_sq());
    }

    // If the moving piece is a king, check whether the destination square is
    // attacked by the opponent.
    if (type_of(piece_on(from)) == KING)
        return !(attackers_to_exist(to, pieces() ^ from, ~us));

    // A non-king move is legal if and only if it is not pinned or it
    // is moving along the ray towards or away from the king.
    return !(blockers_for_king(us) & from) || line_bb(from, to) & pieces(us, KING);
}


// Takes a random move and tests whether the move is
// pseudo-legal. It is used to validate moves from TT that can be corrupted
// due to SMP concurrent access or hash position key aliasing.
bool Position::pseudo_legal(const Move m) const {

    Color  us   = sideToMove;
    Square from = m.from_sq();
    Square to   = m.to_sq();
    Piece  pc   = moved_piece(m);

    // Use a slower but simpler function for uncommon cases
    // yet we skip the legality check of MoveList<LEGAL>().
    if (m.type_of() != NORMAL)
        return checkers() ? MoveList<EVASIONS>(*this).contains(m)
                          : MoveList<NON_EVASIONS>(*this).contains(m);

    // Is not a promotion, so the promotion piece must be empty
    assert(m.promotion_type() - KNIGHT == NO_PIECE_TYPE);

    // If the 'from' square is not occupied by a piece belonging to the side to
    // move, the move is obviously not legal.
    if (pc == NO_PIECE || color_of(pc) != us)
        return false;

    // The destination square cannot be occupied by a friendly piece
    if (pieces(us) & to)
        return false;

    // Handle the special case of a pawn move
    if (type_of(pc) == PAWN)
    {
        // We have already handled promotion moves, so destination cannot be on the 8th/1st rank
        if ((Rank8BB | Rank1BB) & to)
            return false;

        // Check if it's a valid capture, single push, or double push
        const bool isCapture    = bool(attacks_bb<PAWN>(from, us) & pieces(~us) & to);
        const bool isSinglePush = (from + pawn_push(us) == to) && empty(to);
        const bool isDoublePush = (from + 2 * pawn_push(us) == to)
                               && (relative_rank(us, from) == RANK_2) && empty(to)
                               && empty(to - pawn_push(us));

        if (!(isCapture || isSinglePush || isDoublePush))
            return false;
    }
    else if (!(attacks_bb(type_of(pc), from, pieces()) & to))
        return false;

    // Evasions generator already takes care to avoid some kind of illegal moves
    // and legal() relies on this. We therefore have to take care that the same
    // kind of moves are filtered out here.
    if (checkers())
    {
        if (type_of(pc) != KING)
        {
            // Double check? In this case, a king move is required
            if (more_than_one(checkers()))
                return false;

            // Our move must be a blocking interposition or a capture of the checking piece
            if (!(between_bb(square<KING>(us), lsb(checkers())) & to))
                return false;
        }
        // In case of king moves under check we have to remove the king so as to catch
        // invalid moves like b1a1 when opposite queen is on c1.
        else if (attackers_to_exist(to, pieces() ^ from, ~us))
            return false;
    }

    return true;
}


// Tests whether a pseudo-legal move gives a check
bool Position::gives_check(Move m) const {

    assert(m.is_ok());
    assert(color_of(moved_piece(m)) == sideToMove);

    Square from = m.from_sq();
    Square to   = m.to_sq();

    // Is there a direct check?
    if (check_squares(type_of(piece_on(from))) & to)
        return true;

    // Is there a discovered check?
    if (blockers_for_king(~sideToMove) & from)
        return !(line_bb(from, to) & pieces(~sideToMove, KING)) || m.type_of() == CASTLING;

    switch (m.type_of())
    {
    case NORMAL :
        return false;

    case PROMOTION :
        return attacks_bb(m.promotion_type(), to, pieces() ^ from) & pieces(~sideToMove, KING);

    // En passant capture with check? We have already handled the case of direct
    // checks and ordinary discovered check, so the only case we need to handle
    // is the unusual case of a discovered check through the captured pawn.
    case EN_PASSANT : {
        Square   capsq = make_square(file_of(to), rank_of(from));
        Bitboard b     = (pieces() ^ from ^ capsq) | to;

        return (attacks_bb<ROOK>(square<KING>(~sideToMove), b) & pieces(sideToMove, QUEEN, ROOK))
             | (attacks_bb<BISHOP>(square<KING>(~sideToMove), b)
                & pieces(sideToMove, QUEEN, BISHOP));
    }
    default :  //CASTLING
    {
        // Castling is encoded as 'king captures the rook'
        Square rto = relative_square(sideToMove, to > from ? SQ_F1 : SQ_D1);

        return check_squares(ROOK) & rto;
    }
    }
}


// Makes a move, and saves all information necessary
// to a StateInfo object. The move is assumed to be legal. Pseudo-legal
// moves should be filtered out before this function is called.
// If a pointer to the TT table is passed, the entry for the new position
// will be prefetched
DirtyPiece Position::do_move(Move                      m,
                             StateInfo&                newSt,
                             bool                      givesCheck,
                             const TranspositionTable* tt = nullptr) {

    assert(m.is_ok());
    assert(&newSt != st);

    Key k = st->key ^ Zobrist::side;

    // Copy some fields of the old state to our new StateInfo object except the
    // ones which are going to be recalculated from scratch anyway and then switch
    // our state pointer to point to the new (ready to be updated) state.
    std::memcpy(&newSt, st, offsetof(StateInfo, key));
    newSt.previous = st;
    st             = &newSt;

    // Increment ply counters. In particular, rule50 will be reset to zero later on
    // in case of a capture or a pawn move.
    ++gamePly;
    ++st->rule50;
    ++st->pliesFromNull;

    Color  us       = sideToMove;
    Color  them     = ~us;
    Square from     = m.from_sq();
    Square to       = m.to_sq();
    Piece  pc       = piece_on(from);
    Piece  captured = m.type_of() == EN_PASSANT ? make_piece(them, PAWN) : piece_on(to);

    bool checkEP = false;

    DirtyPiece dp;
    dp.pc     = pc;
    dp.from   = from;
    dp.to     = to;
    dp.add_sq = SQ_NONE;

    assert(color_of(pc) == us);
    assert(captured == NO_PIECE || color_of(captured) == (m.type_of() != CASTLING ? them : us));
    assert(type_of(captured) != KING);

    if (m.type_of() == CASTLING)
    {
        assert(pc == make_piece(us, KING));
        assert(captured == make_piece(us, ROOK));

        Square rfrom, rto;
        do_castling<true>(us, from, to, rfrom, rto, &dp);

        k ^= Zobrist::psq[captured][rfrom] ^ Zobrist::psq[captured][rto];
        st->nonPawnKey[us] ^= Zobrist::psq[captured][rfrom] ^ Zobrist::psq[captured][rto];
        captured = NO_PIECE;
    }
    else if (captured)
    {
        Square capsq = to;

        // If the captured piece is a pawn, update pawn hash key, otherwise
        // update non-pawn material.
        if (type_of(captured) == PAWN)
        {
            if (m.type_of() == EN_PASSANT)
            {
                capsq -= pawn_push(us);

                assert(pc == make_piece(us, PAWN));
                assert(to == st->epSquare);
                assert(relative_rank(us, to) == RANK_6);
                assert(piece_on(to) == NO_PIECE);
                assert(piece_on(capsq) == make_piece(them, PAWN));
            }

            st->pawnKey ^= Zobrist::psq[captured][capsq];
        }
        else
        {
            st->nonPawnMaterial[them] -= PieceValue[captured];
            st->nonPawnKey[them] ^= Zobrist::psq[captured][capsq];

            if (type_of(captured) <= BISHOP)
                st->minorPieceKey ^= Zobrist::psq[captured][capsq];
        }

        dp.remove_pc = captured;
        dp.remove_sq = capsq;

        // Update board and piece lists
        remove_piece(capsq);

        k ^= Zobrist::psq[captured][capsq];
        st->materialKey ^= Zobrist::psq[captured][8 + pieceCount[captured]];

        // Reset rule 50 counter
        st->rule50 = 0;
    }
    else
        dp.remove_sq = SQ_NONE;

    // Update hash key
    k ^= Zobrist::psq[pc][from] ^ Zobrist::psq[pc][to];

    // Reset en passant square
    if (st->epSquare != SQ_NONE)
    {
        k ^= Zobrist::enpassant[file_of(st->epSquare)];
        st->epSquare = SQ_NONE;
    }

    // Update castling rights if needed
    if (st->castlingRights && (castlingRightsMask[from] | castlingRightsMask[to]))
    {
        k ^= Zobrist::castling[st->castlingRights];
        st->castlingRights &= ~(castlingRightsMask[from] | castlingRightsMask[to]);
        k ^= Zobrist::castling[st->castlingRights];
    }

    // Move the piece. The tricky Chess960 castling is handled earlier
    if (m.type_of() != CASTLING)
        move_piece(from, to);

    // If the moving piece is a pawn do some special extra work
    if (type_of(pc) == PAWN)
    {
        // Check later if the en passant square needs to be set
        if ((int(to) ^ int(from)) == 16)
            checkEP = true;

        else if (m.type_of() == PROMOTION)
        {
            Piece     promotion     = make_piece(us, m.promotion_type());
            PieceType promotionType = type_of(promotion);

            assert(relative_rank(us, to) == RANK_8);
            assert(type_of(promotion) >= KNIGHT && type_of(promotion) <= QUEEN);

            remove_piece(to);
            put_piece(promotion, to);

            dp.add_pc = promotion;
            dp.add_sq = to;
            dp.to     = SQ_NONE;

            // Update hash keys
            // Zobrist::psq[pc][to] is zero, so we don't need to clear it
            k ^= Zobrist::psq[promotion][to];
            st->materialKey ^= Zobrist::psq[promotion][8 + pieceCount[promotion] - 1]
                             ^ Zobrist::psq[pc][8 + pieceCount[pc]];

            if (promotionType <= BISHOP)
                st->minorPieceKey ^= Zobrist::psq[promotion][to];

            // Update material
            st->nonPawnMaterial[us] += PieceValue[promotion];
        }

        // Update pawn hash key
        st->pawnKey ^= Zobrist::psq[pc][from] ^ Zobrist::psq[pc][to];

        // Reset rule 50 draw counter
        st->rule50 = 0;
    }

    else
    {
        st->nonPawnKey[us] ^= Zobrist::psq[pc][from] ^ Zobrist::psq[pc][to];

        if (type_of(pc) <= BISHOP)
            st->minorPieceKey ^= Zobrist::psq[pc][from] ^ Zobrist::psq[pc][to];
    }

    // Set capture piece
    st->capturedPiece = captured;

    // Calculate checkers bitboard (if move gives check)
    st->checkersBB = givesCheck ? attackers_to(square<KING>(them)) & pieces(us) : 0;

    sideToMove = ~sideToMove;

    // Update king attacks used for fast check detection
    set_check_info();

    // Accurate e.p. info is needed for correct zobrist key generation and 3-fold checking
    while (checkEP)
    {
        auto updateEpSquare = [&] {
            st->epSquare = to - pawn_push(us);
            k ^= Zobrist::enpassant[file_of(st->epSquare)];
        };

        Bitboard pawns = attacks_bb<PAWN>(to - pawn_push(us), us) & pieces(them, PAWN);

        // If there are no pawns attacking the ep square, ep is not possible
        if (!pawns)
            break;

        // If there are checkers other than the to be captured pawn, ep is never legal
        if (checkers() & ~square_bb(to))
            break;

        if (more_than_one(pawns))
        {
            // If there are two pawns potentially being abled to capture and at least one
            // is not pinned, ep is legal as there are no horizontal exposed checks
            if (!more_than_one(blockers_for_king(them) & pawns))
            {
                updateEpSquare();
                break;
            }

            // If there is no pawn on our king's file, and thus both pawns are pinned
            // by bishops, ep is not legal as the king square must be in front of the to square.
            // And because the ep square and the king are not on a common diagonal, either ep capture
            // would expose the king to a check from one of the bishops
            if (!(file_bb(square<KING>(them)) & pawns))
                break;

            // Otherwise remove the pawn on the king file, as an ep capture by it can never be legal and the
            // check below relies on there only being one pawn
            pawns &= ~file_bb(square<KING>(them));
        }

        Square   ksq      = square<KING>(them);
        Square   capsq    = to;
        Bitboard occupied = (pieces() ^ lsb(pawns) ^ capsq) | (to - pawn_push(us));

        // If our king is not attacked after making the move, ep is legal.
        if (!(attacks_bb<ROOK>(ksq, occupied) & pieces(us, QUEEN, ROOK))
            && !(attacks_bb<BISHOP>(ksq, occupied) & pieces(us, QUEEN, BISHOP)))
            updateEpSquare();

        break;
    }

    // Update the key with the final value
    st->key = k;
    if (tt)
        prefetch(tt->first_entry(key()));

    // Calculate the repetition info. It is the ply distance from the previous
    // occurrence of the same position, negative in the 3-fold case, or zero
    // if the position was not repeated.
    st->repetition = 0;
    int end        = std::min(st->rule50, st->pliesFromNull);
    if (end >= 4)
    {
        StateInfo* stp = st->previous->previous;
        for (int i = 4; i <= end; i += 2)
        {
            stp = stp->previous->previous;
            if (stp->key == st->key)
            {
                st->repetition = stp->repetition ? -i : i;
                break;
            }
        }
    }

    assert(pos_is_ok());

    assert(dp.pc != NO_PIECE);
    assert(!(bool(captured) || m.type_of() == CASTLING) ^ (dp.remove_sq != SQ_NONE));
    assert(dp.from != SQ_NONE);
    assert(!(dp.add_sq != SQ_NONE) ^ (m.type_of() == PROMOTION || m.type_of() == CASTLING));
    return dp;
}


// Unmakes a move. When it returns, the position should
// be restored to exactly the same state as before the move was made.
void Position::undo_move(Move m) {

    assert(m.is_ok());

    sideToMove = ~sideToMove;

    Color  us   = sideToMove;
    Square from = m.from_sq();
    Square to   = m.to_sq();
    Piece  pc   = piece_on(to);

    assert(empty(from) || m.type_of() == CASTLING);
    assert(type_of(st->capturedPiece) != KING);

    if (m.type_of() == PROMOTION)
    {
        assert(relative_rank(us, to) == RANK_8);
        assert(type_of(pc) == m.promotion_type());
        assert(type_of(pc) >= KNIGHT && type_of(pc) <= QUEEN);

        remove_piece(to);
        pc = make_piece(us, PAWN);
        put_piece(pc, to);
    }

    if (m.type_of() == CASTLING)
    {
        Square rfrom, rto;
        do_castling<false>(us, from, to, rfrom, rto);
    }
    else
    {
        move_piece(to, from);  // Put the piece back at the source square

        if (st->capturedPiece)
        {
            Square capsq = to;

            if (m.type_of() == EN_PASSANT)
            {
                capsq -= pawn_push(us);

                assert(type_of(pc) == PAWN);
                assert(to == st->previous->epSquare);
                assert(relative_rank(us, to) == RANK_6);
                assert(piece_on(capsq) == NO_PIECE);
                assert(st->capturedPiece == make_piece(~us, PAWN));
            }

            put_piece(st->capturedPiece, capsq);  // Restore the captured piece
        }
    }

    // Finally point our state pointer back to the previous state
    st = st->previous;
    --gamePly;

    assert(pos_is_ok());
}


// Helper used to do/undo a castling move. This is a bit
// tricky in Chess960 where from/to squares can overlap.
template<bool Do>
void Position::do_castling(
  Color us, Square from, Square& to, Square& rfrom, Square& rto, DirtyPiece* const dp) {

    bool kingSide = to > from;
    rfrom         = to;  // Castling is encoded as "king captures friendly rook"
    rto           = relative_square(us, kingSide ? SQ_F1 : SQ_D1);
    to            = relative_square(us, kingSide ? SQ_G1 : SQ_C1);

    assert(!Do || dp);

    if (Do)
    {
        dp->to        = to;
        dp->remove_pc = dp->add_pc = make_piece(us, ROOK);
        dp->remove_sq              = rfrom;
        dp->add_sq                 = rto;
    }

    // Remove both pieces first since squares could overlap in Chess960
    remove_piece(Do ? from : to);
    remove_piece(Do ? rfrom : rto);
    board[Do ? from : to] = board[Do ? rfrom : rto] =
      NO_PIECE;  // remove_piece does not do this for us
    put_piece(make_piece(us, KING), Do ? to : from);
    put_piece(make_piece(us, ROOK), Do ? rto : rfrom);
}


// Used to do a "null move": it flips
// the side to move without executing any move on the board.
void Position::do_null_move(StateInfo& newSt, const TranspositionTable& tt) {

    assert(!checkers());
    assert(&newSt != st);

    std::memcpy(&newSt, st, sizeof(StateInfo));

    newSt.previous = st;
    st             = &newSt;

    if (st->epSquare != SQ_NONE)
    {
        st->key ^= Zobrist::enpassant[file_of(st->epSquare)];
        st->epSquare = SQ_NONE;
    }

    st->key ^= Zobrist::side;
    prefetch(tt.first_entry(key()));

    st->pliesFromNull = 0;

    sideToMove = ~sideToMove;

    set_check_info();

    st->repetition = 0;

    assert(pos_is_ok());
}


// Must be used to undo a "null move"
void Position::undo_null_move() {

    assert(!checkers());

    st         = st->previous;
    sideToMove = ~sideToMove;
}


// Tests if the SEE (Static Exchange Evaluation)
// value of move is greater or equal to the given threshold. We'll use an
// algorithm similar to alpha-beta pruning with a null window.
bool Position::see_ge(Move m, int threshold) const {

    assert(m.is_ok());

    // Only deal with normal moves, assume others pass a simple SEE
    if (m.type_of() != NORMAL)
        return VALUE_ZERO >= threshold;

    Square from = m.from_sq(), to = m.to_sq();

    int swap = PieceValue[piece_on(to)] - threshold;
    if (swap < 0)
        return false;

    swap = PieceValue[piece_on(from)] - swap;
    if (swap <= 0)
        return true;

    assert(color_of(piece_on(from)) == sideToMove);
    Bitboard occupied  = pieces() ^ from ^ to;  // xoring to is important for pinned piece logic
    Color    stm       = sideToMove;
    Bitboard attackers = attackers_to(to, occupied);
    Bitboard stmAttackers, bb;
    int      res = 1;

    while (true)
    {
        stm = ~stm;
        attackers &= occupied;

        // If stm has no more attackers then give up: stm loses
        if (!(stmAttackers = attackers & pieces(stm)))
            break;

        // Don't allow pinned pieces to attack as long as there are
        // pinners on their original square.
        if (pinners(~stm) & occupied)
        {
            stmAttackers &= ~blockers_for_king(stm);

            if (!stmAttackers)
                break;
        }

        res ^= 1;

        // Locate and remove the next least valuable attacker, and add to
        // the bitboard 'attackers' any X-ray attackers behind it.
        if ((bb = stmAttackers & pieces(PAWN)))
        {
            if ((swap = PawnValue - swap) < res)
                break;
            occupied ^= least_significant_square_bb(bb);

            attackers |= attacks_bb<BISHOP>(to, occupied) & pieces(BISHOP, QUEEN);
        }

        else if ((bb = stmAttackers & pieces(KNIGHT)))
        {
            if ((swap = KnightValue - swap) < res)
                break;
            occupied ^= least_significant_square_bb(bb);
        }

        else if ((bb = stmAttackers & pieces(BISHOP)))
        {
            if ((swap = BishopValue - swap) < res)
                break;
            occupied ^= least_significant_square_bb(bb);

            attackers |= attacks_bb<BISHOP>(to, occupied) & pieces(BISHOP, QUEEN);
        }

        else if ((bb = stmAttackers & pieces(ROOK)))
        {
            if ((swap = RookValue - swap) < res)
                break;
            occupied ^= least_significant_square_bb(bb);

            attackers |= attacks_bb<ROOK>(to, occupied) & pieces(ROOK, QUEEN);
        }

        else if ((bb = stmAttackers & pieces(QUEEN)))
        {
            swap = QueenValue - swap;
            //  implies that the previous recapture was done by a higher rated piece than a Queen (King is excluded)
            assert(swap >= res);
            occupied ^= least_significant_square_bb(bb);

            attackers |= (attacks_bb<BISHOP>(to, occupied) & pieces(BISHOP, QUEEN))
                       | (attacks_bb<ROOK>(to, occupied) & pieces(ROOK, QUEEN));
        }

        else  // KING
              // If we "capture" with the king but the opponent still has attackers,
              // reverse the result.
            return (attackers & ~pieces(stm)) ? res ^ 1 : res;
    }

    return bool(res);
}

// Tests whether the position is drawn by 50-move rule
// or by repetition. It does not detect stalemates.
bool Position::is_draw(int ply) const {

    if (st->rule50 > 99 && (!checkers() || MoveList<LEGAL>(*this).size()))
        return true;

    return is_repetition(ply);
}

// Return a draw score if a position repeats once earlier but strictly
// after the root, or repeats twice before or at the root.
bool Position::is_repetition(int ply) const { return st->repetition && st->repetition < ply; }

// Tests whether there has been at least one repetition
// of positions since the last capture or pawn move.
bool Position::has_repeated() const {

    StateInfo* stc = st;
    int        end = std::min(st->rule50, st->pliesFromNull);
    while (end-- >= 4)
    {
        if (stc->repetition)
            return true;

        stc = stc->previous;
    }
    return false;
}


// Tests if the position has a move which draws by repetition.
// This function accurately matches the outcome of is_draw() over all legal moves.
bool Position::upcoming_repetition(int ply) const {

    int j;

    int end = std::min(st->rule50, st->pliesFromNull);

    if (end < 3)
        return false;

    Key        originalKey = st->key;
    StateInfo* stp         = st->previous;
    Key        other       = originalKey ^ stp->key ^ Zobrist::side;

    for (int i = 3; i <= end; i += 2)
    {
        stp = stp->previous;
        other ^= stp->key ^ stp->previous->key ^ Zobrist::side;
        stp = stp->previous;

        if (other != 0)
            continue;

        Key moveKey = originalKey ^ stp->key;
        if ((j = H1(moveKey), cuckoo[j] == moveKey) || (j = H2(moveKey), cuckoo[j] == moveKey))
        {
            Move   move = cuckooMove[j];
            Square s1   = move.from_sq();
            Square s2   = move.to_sq();

            if (!((between_bb(s1, s2) ^ s2) & pieces()))
            {
                if (ply > i)
                    return true;

                // For nodes before or at the root, check that the move is a
                // repetition rather than a move to the current position.
                if (stp->repetition)
                    return true;
            }
        }
    }
    return false;
}


// Flips position with the white and black sides reversed. This
// is only useful for debugging e.g. for finding evaluation symmetry bugs.
void Position::flip() {

    string            f, token;
    std::stringstream ss(fen());

    for (Rank r = RANK_8; r >= RANK_1; --r)  // Piece placement
    {
        std::getline(ss, token, r > RANK_1 ? '/' : ' ');
        f.insert(0, token + (f.empty() ? " " : "/"));
    }

    ss >> token;                        // Active color
    f += (token == "w" ? "B " : "W ");  // Will be lowercased later

    ss >> token;  // Castling availability
    f += token + " ";

    std::transform(f.begin(), f.end(), f.begin(),
                   [](char c) { return char(islower(c) ? toupper(c) : tolower(c)); });

    ss >> token;  // En passant square
    f += (token == "-" ? token : token.replace(1, 1, token[1] == '3' ? "6" : "3"));

    std::getline(ss, token);  // Half and full moves
    f += token;

    set(f, is_chess960(), st);

    assert(pos_is_ok());
}


// Performs some consistency checks for the position object
// and raise an assert if something wrong is detected.
// This is meant to be helpful when debugging.
bool Position::pos_is_ok() const {

    constexpr bool Fast = true;  // Quick (default) or full check?

    if ((sideToMove != WHITE && sideToMove != BLACK) || piece_on(square<KING>(WHITE)) != W_KING
        || piece_on(square<KING>(BLACK)) != B_KING
        || (ep_square() != SQ_NONE && relative_rank(sideToMove, ep_square()) != RANK_6))
        assert(0 && "pos_is_ok: Default");

    if (Fast)
        return true;

    if (pieceCount[W_KING] != 1 || pieceCount[B_KING] != 1
        || attackers_to_exist(square<KING>(~sideToMove), pieces(), sideToMove))
        assert(0 && "pos_is_ok: Kings");

    if ((pieces(PAWN) & (Rank1BB | Rank8BB)) || pieceCount[W_PAWN] > 8 || pieceCount[B_PAWN] > 8)
        assert(0 && "pos_is_ok: Pawns");

    if ((pieces(WHITE) & pieces(BLACK)) || (pieces(WHITE) | pieces(BLACK)) != pieces()
        || popcount(pieces(WHITE)) > 16 || popcount(pieces(BLACK)) > 16)
        assert(0 && "pos_is_ok: Bitboards");

    for (PieceType p1 = PAWN; p1 <= KING; ++p1)
        for (PieceType p2 = PAWN; p2 <= KING; ++p2)
            if (p1 != p2 && (pieces(p1) & pieces(p2)))
                assert(0 && "pos_is_ok: Bitboards");


    for (Piece pc : Pieces)
        if (pieceCount[pc] != popcount(pieces(color_of(pc), type_of(pc)))
            || pieceCount[pc] != std::count(board, board + SQUARE_NB, pc))
            assert(0 && "pos_is_ok: Pieces");

    for (Color c : {WHITE, BLACK})
        for (CastlingRights cr : {c & KING_SIDE, c & QUEEN_SIDE})
        {
            if (!can_castle(cr))
                continue;

            if (piece_on(castlingRookSquare[cr]) != make_piece(c, ROOK)
                || castlingRightsMask[castlingRookSquare[cr]] != cr
                || (castlingRightsMask[square<KING>(c)] & cr) != cr)
                assert(0 && "pos_is_ok: Castling");
        }

    return true;
}

}  // namespace Stockfish
