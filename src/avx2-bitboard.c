__m256i queen_mask_v4[64][2];
__m256i bishop_mask_v4[64];
__m128i rook_mask_NS[64];
uint8_t rook_attacks_EW[64 * 8];

static void init_sliding_attacks(void)
{
  static const int dirs[2][4] = {{ EAST, NORTH, NORTH_EAST, NORTH_WEST }, { WEST, SOUTH, SOUTH_WEST, SOUTH_EAST }};
  Bitboard attacks[4];
  int i, j, occ8;
  Square sq, s;
  uint8_t s8, att8;

  // pseudo attacks for Queen 8 directions
  for (sq = SQ_A1; sq <= SQ_H8; ++sq)
    for (j = 0; j < 2; ++j) {
      for (i = 0; i < 4; ++i) {
        attacks[i] = 0;
        for (s = sq + dirs[j][i];
            square_is_ok(s) && distance(s, s - dirs[j][i]) == 1; s += dirs[j][i])
        {
          attacks[i] |= sq_bb(s);
        }
      }
      queen_mask_v4[sq][j] = _mm256_set_epi64x(attacks[3], attacks[2], attacks[1], attacks[0]);
    }

  // pseudo attacks for Rook (NORTH-SOUTH) and Bishop
  for (sq = SQ_A1; sq <= SQ_H8; ++sq) {
    rook_mask_NS[sq] = _mm_set_epi64x(
        _mm256_extract_epi64(queen_mask_v4[SQUARE_FLIP(sq)][0], 1),	// SOUTH (vertically flipped)
        _mm256_extract_epi64(queen_mask_v4[sq][0], 1));	// NORTH
    bishop_mask_v4[sq] = _mm256_set_epi64x(
        _mm256_extract_epi64(queen_mask_v4[SQUARE_FLIP(sq)][0], 2),	// SOUTH_EAST (vertically flipped)
        _mm256_extract_epi64(queen_mask_v4[sq][0], 3),	// NORTH_WEST
        _mm256_extract_epi64(queen_mask_v4[SQUARE_FLIP(sq)][0], 3),	// SOUTH_WEST (vertically flipped)
        _mm256_extract_epi64(queen_mask_v4[sq][0], 2));	// NORTH_EAST
  }

  // sliding attacks for Rook EAST-WEST
  for (occ8 = 0; occ8 < 128; occ8 += 2)	// inner 6 bits
    for (sq = 0; sq < 8; ++sq) {
      att8 = 0;
      for (s8 = (1 << sq) << 1; s8; s8 <<= 1) {
        att8 |= s8;
        if (occ8 & s8)
          break;
      }
      for (s8 = (1 << sq) >> 1; s8; s8 >>= 1) {
        att8 |= s8;
        if (occ8 & s8)
          break;
      }
      rook_attacks_EW[occ8 * 4 + sq] = att8;
    }
}
