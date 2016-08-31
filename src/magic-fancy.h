extern Bitboard RookMasks[64];
extern Bitboard RookMagics[64];
extern uint8_t  RookShifts[64];
extern Bitboard BishopMasks[64];
extern Bitboard BishopMagics[64];
extern uint8_t  BishopShifts[64];
extern Bitboard *RookAttacks[64];
extern Bitboard *BishopAttacks[64];

// attacks_bb() returns a bitboard representing all the squares attacked
// by a // piece of type Pt (bishop or rook) placed on 's'. The helper
// magic_index() looks up the index using the 'magic bitboards' approach.

INLINE unsigned magic_index_bishop(Square s, Bitboard occupied)
{
  if (Is64Bit)
      return (unsigned)(((occupied & BishopMasks[s]) * BishopMagics[s])
                           >> BishopShifts[s]);

  unsigned lo = (unsigned)(occupied) & (unsigned)(BishopMasks[s]);
  unsigned hi = (unsigned)(occupied >> 32) & (unsigned)(BishopMasks[s] >> 32);
  return (lo * (unsigned)(BishopMagics[s]) ^ hi * (unsigned)(BishopMagics[s] >> 32)) >> BishopShifts[s];
}

INLINE unsigned magic_index_rook(Square s, Bitboard occupied)
{
  if (Is64Bit)
      return (unsigned)(((occupied & RookMasks[s]) * RookMagics[s])
                           >> RookShifts[s]);

  unsigned lo = (unsigned)(occupied) & (unsigned)(RookMasks[s]);
  unsigned hi = (unsigned)(occupied >> 32) & (unsigned)(RookMasks[s] >> 32);
  return (lo * (unsigned)(RookMagics[s]) ^ hi * (unsigned)(RookMagics[s] >> 32)) >> RookShifts[s];
}

INLINE Bitboard attacks_bb_bishop(Square s, Bitboard occupied)
{
  return BishopAttacks[s][magic_index_bishop(s, occupied)];
}

INLINE Bitboard attacks_bb_rook(Square s, Bitboard occupied)
{
  return RookAttacks[s][magic_index_rook(s, occupied)];
}

