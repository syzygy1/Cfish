Bitboard RookMasks[64];
Bitboard *RookAttacks[64];

Bitboard BishopMasks[64];
Bitboard *BishopAttacks[64];

Bitboard BishopTable[5248];
Bitboard RookTable[102400];

typedef unsigned (Fn)(Square, Bitboard);

static void init_bmi2(Bitboard table[], Bitboard *attacks[], Bitboard masks[],
                      int deltas[], Fn index)
{
  Bitboard edges, b;

  for (int s = 0; s < 64; s++) {
    attacks[s] = table;

    // Board edges are not considered in the relevant occupancies
    edges = ((Rank1BB | Rank8BB) & ~rank_bb_s(s)) | ((FileABB | FileHBB) & ~file_bb_s(s));

    masks[s] = sliding_attack(deltas, s, 0) & ~edges;

    // Use Carry-Rippler trick to enumerate all subsets of masks[s] and
    // fill the attacks table.
    b = 0;
    do {
      attacks[s][index(s, b)] = sliding_attack(deltas, s, b);
      b = (b - masks[s]) & masks[s];
      table++;
    } while (b);
  }
}

static void init_sliding_attacks(void)
{
  init_bmi2(RookTable, RookAttacks, RookMasks, RookDirs, bmi2_index_rook);
  init_bmi2(BishopTable, BishopAttacks, BishopMasks, BishopDirs,
            bmi2_index_bishop);
}

