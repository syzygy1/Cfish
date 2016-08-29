#include <immintrin.h>

extern Bitboard RookMasks[64], RookMasks2[64];
extern Bitboard BishopMasks[64], BishopMasks2[64];
extern uint16_t *RookAttacks[64];
extern uint16_t *BishopAttacks[64];

INLINE unsigned bmi2_index_bishop(Square s, Bitboard occupied)
{
  return (unsigned)_pext_u64(occupied, BishopMasks[s]);
}

INLINE unsigned bmi2_index_rook(Square s, Bitboard occupied)
{
  return (unsigned)_pext_u64(occupied, RookMasks[s]);
}

INLINE Bitboard attacks_bb_bishop(Square s, Bitboard occupied)
{
  return _pdep_u64(BishopAttacks[s][bmi2_index_bishop(s, occupied)],
                   BishopMasks2[s]);
}

INLINE Bitboard attacks_bb_rook(Square s, Bitboard occupied)
{
  return _pdep_u64(RookAttacks[s][bmi2_index_rook(s, occupied)],
                   RookMasks2[s]);
}

