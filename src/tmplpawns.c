#if Us == WHITE
#define Them BLACK
#define func(name) name##_white
#define Up    DELTA_N
#define Right DELTA_NE
#define Left  DELTA_NW
#define shift_bb_Up    shift_bb_N
#define shift_bb_Right shift_bb_NE
#define shift_bb_Left  shift_bb_NW
#else
#define Them WHITE
#define func(name) name##_black
#define Up    DELTA_S
#define Right DELTA_SW
#define Left  DELTA_SE
#define shift_bb_Up    shift_bb_S
#define shift_bb_Right shift_bb_SW
#define shift_bb_Left  shift_bb_SE
#endif

static Score func(pawn_evaluate)(Pos *pos, PawnEntry *e)
{
  Bitboard b, neighbours, stoppers, doubled, supported, phalanx;
  Square s;
  int opposed, lever, connected, backward;
  Score score = SCORE_ZERO;
  const Square* pl = pos->pieceList[Us][PAWN];
  const Bitboard* pawnAttacksBB = StepAttacksBB[make_piece(Us, PAWN)];

  Bitboard ourPawns   = pieces_cp(Us, PAWN);
  Bitboard theirPawns = pieces_p(PAWN) ^ ourPawns;

  e->passedPawns[Us] = e->pawnAttacksSpan[Us] = 0;
  e->kingSquares[Us] = SQ_NONE;
  e->semiopenFiles[Us] = 0xFF;
  e->pawnAttacks[Us] = shift_bb_Right(ourPawns) | shift_bb_Left(ourPawns);
  e->pawnsOnSquares[Us][BLACK] = popcount(ourPawns & DarkSquares);
  e->pawnsOnSquares[Us][WHITE] = popcount(ourPawns & LightSquares);

  // Loop through all pawns of the current color and score each pawn
  while ((s = *pl++) != SQ_NONE) {
    assert(piece_on(s) == make_piece(Us, PAWN));

    int f = file_of(s);

    e->semiopenFiles[Us] &= ~(1 << f);
    e->pawnAttacksSpan[Us] |= pawn_attack_span(Us, s);

    // Flag the pawn
    opposed    = theirPawns & forward_bb(Us, s);
    stoppers   = theirPawns & passed_pawn_mask(Us, s);
    lever      = theirPawns & pawnAttacksBB[s];
    doubled    = ourPawns   & sq_bb(s + Up);
    neighbours = ourPawns   & adjacent_files_bb(f);
    phalanx    = neighbours & rank_bb_s(s);
    supported  = neighbours & rank_bb_s(s - Up);
    connected  = supported | phalanx;

    // A pawn is backward when it is behind all pawns of the same color on the
    // adjacent files and cannot be safely advanced.
    if (!neighbours || lever || relative_rank_s(Us, s) >= RANK_5)
      backward = 0;
    else {
      // Find the backmost rank with neighbours or stoppers
      b = rank_bb_s(backmost_sq(Us, neighbours | stoppers));

      // The pawn is backward when it cannot safely progress to that rank:
      // either there is a stopper in the way on this rank, or there is a
      // stopper on adjacent file which controls the way to that rank.
      backward = (b | shift_bb_Up(b & adjacent_files_bb(f))) & stoppers;

      assert(!backward || !(pawn_attack_span(Them, s + Up) & neighbours));
    }

    // Passed pawns will be properly scored in evaluation because we need
    // full attack info to evaluate them.
    if (!stoppers && !(ourPawns & forward_bb(Us, s)))
      e->passedPawns[Us] |= sq_bb(s);

    // Score this pawn
    if (!neighbours)
      score -= Isolated[opposed];

    else if (backward)
      score -= Backward[opposed];

    else if (!supported)
      score -= Unsupported[more_than_one(neighbours & pawnAttacksBB[s])];

    if (connected)
      score += Connected[opposed][!!phalanx][more_than_one(supported)][relative_rank_s(Us, s)];

    if (doubled)
      score -= Doubled;

    if (lever)
      score += Lever[relative_rank_s(Us, s)];
  }

  b = e->semiopenFiles[Us] ^ 0xFF;
  e->pawnSpan[Us] = b ? (msb(b) - lsb(b)) : 0;

  return score;
}

// shelter_storm() calculates shelter and storm penalties for the file
// the king is on, as well as the two adjacent files.

Value func(shelter_storm)(Pos *pos, Square ksq)
{
  enum { NoFriendlyPawn, Unblocked, BlockedByPawn, BlockedByKing };

  Bitboard b = pieces_p(PAWN) & (in_front_bb(Us, rank_of(ksq)) | rank_bb_s(ksq));
  Bitboard ourPawns = b & pieces_c(Us);
  Bitboard theirPawns = b & pieces_c(Them);
  Value safety = MaxSafetyBonus;
  int center = max(FILE_B, min(FILE_G, file_of(ksq)));

  for (int f = center - 1; f <= center + 1; f++) {
    b = ourPawns & file_bb(f);
    int rkUs = b ? relative_rank_s(Us, backmost_sq(Us, b)) : RANK_1;

    b  = theirPawns & file_bb(f);
    int rkThem = b ? relative_rank_s(Us, frontmost_sq(Them, b)) : RANK_1;

    safety -=  ShelterWeakness[min(f, FILE_H - f)][rkUs]
             + StormDanger
               [f == file_of(ksq) && rkThem == relative_rank_s(Us, ksq) + 1 ? BlockedByKing  :
                rkUs   == RANK_1                                            ? NoFriendlyPawn :
                rkThem == rkUs + 1                                          ? BlockedByPawn  : Unblocked]
               [min(f, FILE_H - f)][rkThem];
  }

  return safety;
}


// do_king_safety() calculates a bonus for king safety. It is called only
// when king square changes, which is about 20% of total king_safety() calls.

Score func(do_king_safety)(PawnEntry *pe, Pos *pos, Square ksq)
{
  pe->kingSquares[Us] = ksq;
  pe->castlingRights[Us] = can_castle_c(Us);
  int minKingPawnDistance = 0;

  Bitboard pawns = pieces_cp(Us, PAWN);
  if (pawns)
    while (!(DistanceRingBB[ksq][minKingPawnDistance++] & pawns)) {}

  Value bonus = func(shelter_storm)(pos, ksq);

  // If we can castle use the bonus after the castling if it is bigger
  if (can_castle_cr(make_castling_right(Us, KING_SIDE)))
    bonus = max(bonus, func(shelter_storm)(pos, relative_square(Us, SQ_G1)));

  if (can_castle_cr(make_castling_right(Us, QUEEN_SIDE)))
    bonus = max(bonus, func(shelter_storm)(pos, relative_square(Us, SQ_C1)));

  return make_score(bonus, -16 * minKingPawnDistance);
}

#undef Them
#undef func
#undef Up
#undef Right
#undef Left
#undef shift_bb_Up
#undef shift_bb_Right
#undef shift_bb_Left

