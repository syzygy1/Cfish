#include "misc.h"

Bitboard  RookMasks  [64];
Bitboard  RookMagics [64];
Bitboard *RookAttacks[64];
uint8_t   RookShifts [64];

Bitboard  BishopMasks  [64];
Bitboard  BishopMagics [64];
Bitboard *BishopAttacks[64];
uint8_t   BishopShifts [64];

static Bitboard RookTable[0x19000];  // To store rook attacks
static Bitboard BishopTable[0x1480]; // To store bishop attacks

typedef unsigned (Fn)(Square, Bitboard);

static void init_magics(Bitboard table[], Bitboard *attacks[],
                        Bitboard magics[], Bitboard masks[], uint8_t shifts[],
                        int deltas[], Fn index)
{
  int seeds[][8] = { { 8977, 44560, 54343, 38998,  5731, 95205, 104912, 17020 },
                     {  728, 10316, 55013, 32803, 12281, 15100,  16645,   255 } };

  Bitboard occupancy[4096], reference[4096], edges, b;
  int age[4096] = {0}, current = 0, i, size;

  // attacks[s] is a pointer to the beginning of the attacks table for square 's'
  attacks[0] = table;

  for (Square s = 0; s < 64; s++) {
    // Board edges are not considered in the relevant occupancies
    edges = ((Rank1BB | Rank8BB) & ~rank_bb_s(s)) | ((FileABB | FileHBB) & ~file_bb_s(s));

    // Given a square 's', the mask is the bitboard of sliding attacks from
    // 's' computed on an empty board. The index must be big enough to contain
    // all the attacks for each possible subset of the mask and so is 2 power
    // the number of 1s of the mask. Hence we deduce the size of the shift to
    // apply to the 64 or 32 bits word to get the index.
    masks[s]  = sliding_attack(deltas, s, 0) & ~edges;
    shifts[s] = (Is64Bit ? 64 : 32) - popcount(masks[s]);

    // Use Carry-Rippler trick to enumerate all subsets of masks[s] and
    // store the corresponding sliding attack bitboard in reference[].
    b = size = 0;
    do {
      occupancy[size] = b;
      reference[size] = sliding_attack(deltas, s, b);

      if (HasPext)
        attacks[s][pext(b, masks[s])] = reference[size];

      size++;
      b = (b - masks[s]) & masks[s];
    } while (b);

    // Set the offset for the table of the next square. We have individual
    // table sizes for each square with "Fancy Magic Bitboards".
    if (s < 63)
      attacks[s + 1] = attacks[s] + size;

    if (HasPext)
      continue;

    PRNG rng;
    prng_init(&rng, seeds[Is64Bit][rank_of(s)]);

    // Find a magic for square 's' picking up an (almost) random number
    // until we find the one that passes the verification test.
    do {
      do
        magics[s] = prng_sparse_rand(&rng);
      while (popcount((magics[s] * masks[s]) >> 56) < 6);

      // A good magic must map every possible occupancy to an index that
      // looks up the correct sliding attack in the attacks[s] database.
      // Note that we build up the database for square 's' as a side
      // effect of verifying the magic.
      for (current++, i = 0; i < size; i++) {
        unsigned idx = index(s, occupancy[i]);

        if (age[idx] < current) {
          age[idx] = current;
          attacks[s][idx] = reference[i];
        }
        else if (attacks[s][idx] != reference[i])
          break;
      }
    } while (i < size);
  }
}

static void init_sliding_attacks(void)
{
  init_magics(RookTable, RookAttacks, RookMagics, RookMasks,
              RookShifts, RookDirs, magic_index_rook);
  init_magics(BishopTable, BishopAttacks, BishopMagics, BishopMasks,
              BishopShifts, BishopDirs, magic_index_bishop);
}

