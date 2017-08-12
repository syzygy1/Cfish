extern Bitboard RookMasks[64];
extern Bitboard RookMagics[64];
extern Bitboard BishopMasks[64];
extern Bitboard BishopMagics[64];
extern Bitboard *RookAttacks[64];
extern Bitboard *BishopAttacks[64];

INLINE unsigned magic_index_bishop(Square s, Bitboard occupied)
{
  return ((occupied | BishopMasks[s]) * BishopMagics[s]) >> (64-9);
}

INLINE unsigned magic_index_rook(Square s, Bitboard occupied)
{
  return ((occupied | RookMasks[s]) * RookMagics[s]) >> (64-12);
}

INLINE Bitboard attacks_bb_bishop(Square s, Bitboard occupied)
{
  return BishopAttacks[s][magic_index_bishop(s, occupied)];
}

INLINE Bitboard attacks_bb_rook(Square s, Bitboard occupied)
{
  return RookAttacks[s][magic_index_rook(s, occupied)];
}

