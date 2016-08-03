#if Us == WHITE
#define Them BLACK
#define func(name) name##_white
#define shift_bb_Up    shift_bb_N
#define shift_bb_Down  shift_bb_S
#define shift_bb_Left  shift_bb_NW
#define shift_bb_Right shift_bb_NE
#else
#define Them WHITE
#define func(name) name##_black
#define shift_bb_Up    shift_bb_S
#define shift_bb_Down  shift_bb_N
#define shift_bb_Left  shift_bb_SE
#define shift_bb_Right shift_bb_SW
#endif

// eval_init() initializes king and attack bitboards for a given color
// adding pawn attacks. To be done at the beginning of the evaluation.

void func(eval_init)(Pos *pos, EvalInfo *ei)
{
  ei->pinnedPieces[Us] = pinned_pieces(pos, Us);
  Bitboard b = ei->attackedBy[Them][KING];
  ei->attackedBy[Them][0] |= b;
  ei->attackedBy[Us][0] |= ei->attackedBy[Us][PAWN] = ei->pi->pawnAttacks[Us];
  ei->attackedBy2[Us] = ei->attackedBy[Us][PAWN] & ei->attackedBy[Us][KING];

  // Init king safety tables only if we are going to use them
  if (pos_non_pawn_material(Us) >= QueenValueMg) {
    ei->kingRing[Them] = b | shift_bb_Down(b);
    b &= ei->attackedBy[Us][PAWN];
    ei->kingAttackersCount[Us] = popcount(b);
    ei->kingAdjacentZoneAttacksCount[Us] = ei->kingAttackersWeight[Us] = 0;
  }
  else
      ei->kingRing[Them] = ei->kingAttackersCount[Us] = 0;
}


// evaluate_pieces() assigns bonuses and penalties to the pieces of a
// given color and type.

//  template<bool DoTrace, Color Us = WHITE, PieceType Pt = KNIGHT>
Score func(evaluate_pieces)(Pos *pos, EvalInfo *ei, Score* mobility,
                            Bitboard *mobilityArea)
{
  Bitboard b, bb;
  Square s;
  Score score = SCORE_ZERO;

  Bitboard OutpostRanks = (Us == WHITE ? Rank4BB | Rank5BB | Rank6BB
                                       : Rank5BB | Rank4BB | Rank3BB);

  Square *pl;

#define LOOP(Pt) \
  pl = piece_list(Us, Pt); \
  ei->attackedBy[Us][Pt] = 0; \
  while ((s = *pl++) != SQ_NONE)

#define GENERAL(Pt) \
  if (ei->pinnedPieces[Us] & sq_bb(s)) \
    b &= LineBB[square_of(Us, KING)][s]; \
\
  ei->attackedBy2[Us] |= ei->attackedBy[Us][0] & b; \
  ei->attackedBy[Us][0] |= ei->attackedBy[Us][Pt] |= b; \
\
  if (b & ei->kingRing[Them]) { \
    ei->kingAttackersCount[Us]++; \
    ei->kingAttackersWeight[Us] += KingAttackWeights[Pt]; \
    ei->kingAdjacentZoneAttacksCount[Us] += popcount(b & ei->attackedBy[Them][KING]); \
  }

#define PIECE_MOBILITY(Pt) \
  int mob = popcount(b & mobilityArea[Us]); \
  mobility[Us] += MobilityBonus[Pt][mob];

#define MINOR(Pt) \
  bb = OutpostRanks & ~ei->pi->pawnAttacksSpan[Them]; \
  if (bb & sq_bb(s)) \
    score += Outpost[Pt == BISHOP][!!(ei->attackedBy[Us][PAWN] & sq_bb(s))]; \
  else { \
    bb &= b & ~pieces_c(Us); \
    if (bb) \
      score += ReachableOutpost[Pt == BISHOP][!!(ei->attackedBy[Us][PAWN] & bb)]; \
  } \
\
  if (    relative_rank(Us, rank_of(s)) < RANK_5 \
      && (pieces_c(PAWN) & sq_bb(s + pawn_push(Us)))) \
    score += MinorBehindPawn;


  // KNIGHT
  LOOP(KNIGHT) {
    b = attacks_from_knight(s);

    GENERAL(KNIGHT);
    PIECE_MOBILITY(KNIGHT);

    MINOR(KNIGHT);
  }

  // BISHOP
  LOOP(BISHOP) {
    b = attacks_bb_bishop(s, pieces() ^ pieces_cpp(Us, QUEEN, ROOK));

    GENERAL(BISHOP);
    PIECE_MOBILITY(BISHOP);

    MINOR(BISHOP);

    // Penalty for pawns on the same color square as the bishop
    score -= BishopPawns * pawns_on_same_color_squares(ei->pi, Us, s);

    // An important Chess960 pattern: A cornered bishop blocked by a friendly
    // pawn diagonally in front of it is a very serious problem, especially
    // when that pawn is also blocked.
    if (   is_chess960()
        && (s == relative_square(Us, SQ_A1) || s == relative_square(Us, SQ_H1)))
    {
      Square d = pawn_push(Us) + (file_of(s) == FILE_A ? DELTA_E : DELTA_W);
      if (piece_on(s + d) == make_piece(Us, PAWN))
        score -=  piece_on(s + d + pawn_push(Us))             ? TrappedBishopA1H1 * 4
                : piece_on(s + d + d) == make_piece(Us, PAWN) ? TrappedBishopA1H1 * 2
                                                              : TrappedBishopA1H1;
    }
  }

  // ROOK
  LOOP(ROOK) {
    b = attacks_bb_rook(s, pieces() ^ pieces_cpp(Us, QUEEN, ROOK));

    GENERAL(ROOK);
    PIECE_MOBILITY(ROOK);

    // Bonus for aligning with enemy pawns on the same rank/file
    if (relative_rank(Us, rank_of(s)) >= RANK_5)
      score += RookOnPawn * popcount(pieces_c(Them) & PseudoAttacks[ROOK][s]);

    // Bonus when on an open or semi-open file
    if (semiopen_file(ei->pi, Us, file_of(s)))
      score += RookOnFile[!!semiopen_file(ei->pi, Them, file_of(s))];

    // Penalize when trapped by the king, even more if the king cannot castle
    else if (mob <= 3) {
      Square ksq = square_of(Us, KING);

      if (   ((file_of(ksq) < FILE_E) == (file_of(s) < file_of(ksq)))
          && (rank_of(ksq) == rank_of(s) || relative_rank(Us, rank_of(ksq)) == RANK_1)
          && !semiopen_side(ei->pi, Us, file_of(ksq), file_of(s) < file_of(ksq)))
        score -= (TrappedRook - make_score(mob * 22, 0)) * (1 + !can_castle_c(Us));
    }
  }

  // QUEEN
  LOOP(QUEEN) {
    b = attacks_from_queen(s);

    GENERAL(QUEEN);

    b &= ~(  ei->attackedBy[Them][KNIGHT]
	| ei->attackedBy[Them][BISHOP]
	| ei->attackedBy[Them][ROOK]);

    PIECE_MOBILITY(QUEEN);

    // Penalty if any relative pin or discovered attack against the queen
    if (slider_blockers(pos, pieces(), pieces_cpp(Them, ROOK, BISHOP), s))
      score -= WeakQueen;
  }

  return score;
}

#if 0
Score func(evaluate_pieces)(Pos *pos, EvalInfo *ei, Score* mobility,
                            Bitboard *mobilityArea)
{
  Bitboard b, bb;
  Square s;
  Score score = SCORE_ZERO;

  int NextPt = (Us == WHITE ? Pt : PieceType(Pt + 1));
  Bitboard OutpostRanks = (Us == WHITE ? Rank4BB | Rank5BB | Rank6BB
                                       : Rank5BB | Rank4BB | Rank3BB);
  Square* pl = piece_list(us, pt);

  ei->attackedBy[Us][Pt] = 0;

  while ((s = *pl++) != SQ_NONE) {
    // Find attacked squares, including x-ray attacks for bishops and rooks
    b = Pt == BISHOP ? attacks_bb_bishop(s, pieces() ^ pieces_cpp(Us, QUEEN, ROOK))
      : Pt ==   ROOK ? attacks_bb_rook(s, pieces() ^ pieces_cpp(Us, QUEEN, ROOK))
                     : pos_attacks_from<Pt>(s);

    if (ei->pinnedPieces[Us] & sq_bb(s))
      b &= LineBB[square_of(Us, KING)][s];

    ei->attackedBy2[Us] |= ei->attackedBy[Us][0] & b;
    ei->attackedBy[Us][0] |= ei->attackedBy[Us][Pt] |= b;

    if (b & ei->kingRing[Them]) {
      ei->kingAttackersCount[Us]++;
      ei->kingAttackersWeight[Us] += KingAttackWeights[Pt];
      ei->kingAdjacentZoneAttacksCount[Us] += popcount(b & ei->attackedBy[Them][KING]);
    }

    if (Pt == QUEEN)
      b &= ~(  ei->attackedBy[Them][KNIGHT]
             | ei->attackedBy[Them][BISHOP]
             | ei->attackedBy[Them][ROOK]);

    int mob = popcount(b & mobilityArea[Us]);

    mobility[Us] += MobilityBonus[Pt][mob];

    if (Pt == BISHOP || Pt == KNIGHT) {
      // Bonus for outpost squares
      bb = OutpostRanks & ~ei->pi->pawn_attacks_span(Them);
      if (bb & sq_bb(s))
        score += Outpost[Pt == BISHOP][!!(ei->attackedBy[Us][PAWN] & s)];
      else {
        bb &= b & ~pieces_c(Us);
        if (bb)
          score += ReachableOutpost[Pt == BISHOP][!!(ei->attackedBy[Us][PAWN] & bb)];
      }

      // Bonus when behind a pawn
      if (    relative_rank(Us, rank_of(s)) < RANK_5
          && (pieces_c(PAWN) & sq_bb(s + pawn_push(Us))))
        score += MinorBehindPawn;

      // Penalty for pawns on the same color square as the bishop
      if (Pt == BISHOP)
        score -= BishopPawns * ei->pi->pawns_on_same_color_squares(Us, s);

      // An important Chess960 pattern: A cornered bishop blocked by a friendly
      // pawn diagonally in front of it is a very serious problem, especially
      // when that pawn is also blocked.
      if (   Pt == BISHOP
          && is_chess960()
          && (s == relative_square(Us, SQ_A1) || s == relative_square(Us, SQ_H1))) {
        Square d = pawn_push(Us) + (file_of(s) == FILE_A ? DELTA_E : DELTA_W);
        if (pos.piece_on(s + d) == make_piece(Us, PAWN))
          score -=  piece_on(s + d + pawn_push(Us))             ? TrappedBishopA1H1 * 4
                  : piece_on(s + d + d) == make_piece(Us, PAWN) ? TrappedBishopA1H1 * 2
                                                                : TrappedBishopA1H1;
      }
    }

    if (Pt == ROOK) {
      // Bonus for aligning with enemy pawns on the same rank/file
      if (relative_rank(Us, rank_of(s)) >= RANK_5)
        score += RookOnPawn * popcount(pieces_c(Them) & PseudoAttacks[ROOK][s]);

      // Bonus when on an open or semi-open file
      if (ei->pi->semiopen_file(Us, file_of(s)))
        score += RookOnFile[!!ei->pi->semiopen_file(Them, file_of(s))];

      // Penalize when trapped by the king, even more if the king cannot castle
      else if (mob <= 3) {
        Square ksq = square_of(Us, KING);

        if (   ((file_of(ksq) < FILE_E) == (file_of(s) < file_of(ksq)))
            && (rank_of(ksq) == rank_of(s) || relative_rank(Us, rank_of(ksq)) == RANK_1)
            && !ei->pi->semiopen_side(Us, file_of(ksq), file_of(s) < file_of(ksq)))
          score -= (TrappedRook - make_score(mob * 22, 0)) * (1 + !can_castle(Us));
      }
    }

    if (Pt == QUEEN) {
      // Penalty if any relative pin or discovered attack against the queen
      if (slider_blockers(pos, pieces(), pieces_cpp(Them, ROOK, BISHOP), s))
          score -= WeakQueen;
    }
  }

  if (DoTrace)
    Trace::add(Pt, Us, score);

  // Recursively call evaluate_pieces() of next piece type until KING is excluded
  return score - evaluate_pieces<DoTrace, Them, NextPt>(pos, ei, mobility, mobilityArea);
}
#endif

// evaluate_king() assigns bonuses and penalties to a king of a given color

//  template<Color Us, bool DoTrace>
Score func(evaluate_king)(Pos *pos, EvalInfo *ei)
{
  Bitboard undefended, b, b1, b2, safe, other;
  int attackUnits;
  const Square ksq = square_of(Us, KING);

  // King shelter and enemy pawns storm
  Score score = func(king_safety)(ei->pi, pos, ksq);

  // Main king safety evaluation
  if (ei->kingAttackersCount[Them]) {
    // Find the attacked squares which are defended only by the king...
    undefended =   ei->attackedBy[Them][0]
                &  ei->attackedBy[Us][KING]
                & ~ei->attackedBy2[Us];

    // ... and those which are not defended at all in the larger king ring
    b =  ei->attackedBy[Them][0] & ~ei->attackedBy[Us][0]
       & ei->kingRing[Us] & ~pieces_c(Them);

    // Initialize the 'attackUnits' variable, which is used later on as an
    // index into the KingDanger[] array. The initial value is based on the
    // number and types of the enemy's attacking pieces, the number of
    // attacked and undefended squares around our king and the quality of
    // the pawn shelter (current 'score' value).
    attackUnits =  min(72, ei->kingAttackersCount[Them] * ei->kingAttackersWeight[Them])
                 +  9 * ei->kingAdjacentZoneAttacksCount[Them]
                 + 21 * popcount(undefended)
                 + 12 * (popcount(b) + !!ei->pinnedPieces[Us])
                 - 64 * !pos->pieceCount[Them][QUEEN]
                 - mg_value(score) / 8;

    // Analyse the enemy's safe queen contact checks. Firstly, find the
    // undefended squares around the king reachable by the enemy queen...
    b = undefended & ei->attackedBy[Them][QUEEN] & ~pieces_c(Them);

    // ...and keep squares supported by another enemy piece
    attackUnits += QueenContactCheck * popcount(b & ei->attackedBy2[Them]);

    // Analyse the safe enemy's checks which are possible on next move...
    safe  = ~(ei->attackedBy[Us][0] | pieces_c(Them));

    // ... and some other potential checks, only requiring the square to be
    // safe from pawn-attacks, and not being occupied by a blocked pawn.
    other = ~(   ei->attackedBy[Us][PAWN]
              | (pieces_cp(Them, PAWN) & shift_bb_Up(pieces_c(PAWN))));

    b1 = attacks_from_rook(ksq);
    b2 = attacks_from_bishop(ksq);

    // Enemy queen safe checks
    if ((b1 | b2) & ei->attackedBy[Them][QUEEN] & safe) {
      attackUnits += QueenCheck;
      score -= SafeCheck;
    }

    // For other pieces, also consider the square safe if attacked twice,
    // and only defended by a queen.
    safe |=  ei->attackedBy2[Them]
           & ~(ei->attackedBy2[Us] | pieces_c(Them))
           & ei->attackedBy[Us][QUEEN];

    // Enemy rooks safe and other checks
    if (b1 & ei->attackedBy[Them][ROOK] & safe) {
      attackUnits += RookCheck;
      score -= SafeCheck;
    }

    else if (b1 & ei->attackedBy[Them][ROOK] & other)
      score -= OtherCheck;

    // Enemy bishops safe and other checks
    if (b2 & ei->attackedBy[Them][BISHOP] & safe) {
      attackUnits += BishopCheck;
      score -= SafeCheck;
    }

    else if (b2 & ei->attackedBy[Them][BISHOP] & other)
      score -= OtherCheck;

    // Enemy knights safe and other checks
    b = attacks_from_knight(ksq) & ei->attackedBy[Them][KNIGHT];
    if (b & safe) {
      attackUnits += KnightCheck;
      score -= SafeCheck;
    }

    else if (b & other)
      score -= OtherCheck;

    // Finally, extract the king danger score from the KingDanger[]
    // array and subtract the score from the evaluation.
    score -= KingDanger[max(min(attackUnits, 399), 0)];
  }

#if 0
  if (DoTrace)
    trace_add(KING, Us, score);
#endif

  return score;
}


// evaluate_threats() assigns bonuses according to the types of the attacking
// and the attacked pieces.

//  template<Color Us, bool DoTrace>
Score func(evaluate_threats)(Pos *pos, EvalInfo *ei)
{
  const Bitboard TRank2BB = (Us == WHITE ? Rank2BB  : Rank7BB);
  const Bitboard TRank7BB = (Us == WHITE ? Rank7BB  : Rank2BB);

  const Bitboard TheirCamp = (Us == WHITE ? Rank4BB | Rank5BB | Rank6BB | Rank7BB | Rank8BB
                                          : Rank5BB | Rank4BB | Rank3BB | Rank2BB | Rank1BB);

  const Bitboard QueenSide   = TheirCamp & (FileABB | FileBBB | FileCBB | FileDBB);
  const Bitboard CenterFiles = TheirCamp & (FileCBB | FileDBB | FileEBB | FileFBB);
  const Bitboard KingSide    = TheirCamp & (FileEBB | FileFBB | FileGBB | FileHBB);

  Bitboard KingFlank[8] = {
    QueenSide, QueenSide, QueenSide, CenterFiles,
    CenterFiles, KingSide, KingSide, KingSide
  };

  enum { Minor, Rook };

  Bitboard b, weak, defended, safeThreats;
  Score score = SCORE_ZERO;

  // Small bonus if the opponent has loose pawns or pieces
  if (   (pieces_c(Them) ^ pieces_cpp(Them, QUEEN, KING))
      & ~(ei->attackedBy[Us][0] | ei->attackedBy[Them][0]))
    score += LooseEnemies;

  // Non-pawn enemies attacked by a pawn
  weak = (pieces_c(Them) ^ pieces_cp(Them, PAWN)) & ei->attackedBy[Us][PAWN];

  if (weak) {
    b = pieces_cp(Us, PAWN) & ( ~ei->attackedBy[Them][0]
                               | ei->attackedBy[Us][0]);

    safeThreats = (shift_bb_Right(b) | shift_bb_Left(b)) & weak;

    if (weak ^ safeThreats)
      score += ThreatByHangingPawn;

    while (safeThreats)
      score += ThreatBySafePawn[type_of_p(piece_on(pop_lsb(&safeThreats)))];
  }

  // Non-pawn enemies defended by a pawn
  defended = (pieces_c(Them) ^ pieces_cp(Them, PAWN)) & ei->attackedBy[Them][PAWN];

  // Enemies not defended by a pawn and under our attack
  weak =   pieces_c(Them)
        & ~ei->attackedBy[Them][PAWN]
        &  ei->attackedBy[Us][0];

  // Add a bonus according to the kind of attacking pieces
  if (defended | weak) {
    b = (defended | weak) & (ei->attackedBy[Us][KNIGHT] | ei->attackedBy[Us][BISHOP]);
    while (b)
      score += Threat[Minor][type_of_p(piece_on(pop_lsb(&b)))];

    b = (pieces_cp(Them, QUEEN) | weak) & ei->attackedBy[Us][ROOK];
    while (b)
      score += Threat[Rook ][type_of_p(piece_on(pop_lsb(&b)))];

    score += Hanging * popcount(weak & ~ei->attackedBy[Them][0]);

    b = weak & ei->attackedBy[Us][KING];
    if (b)
      score += ThreatByKing[more_than_one(b)];
  }

  // Bonus if some pawns can safely push and attack an enemy piece
  b = pieces_cp(Us, PAWN) & ~TRank7BB;
  b = shift_bb_Up(b | (shift_bb_Up(b & TRank2BB) & ~pieces()));

  b &=  ~pieces()
      & ~ei->attackedBy[Them][PAWN]
      & (ei->attackedBy[Us][0] | ~ei->attackedBy[Them][0]);

  b =  (shift_bb_Left(b) | shift_bb_Right(b))
     &  pieces_c(Them)
     & ~ei->attackedBy[Us][PAWN];

  score += ThreatByPawnPush * popcount(b);

  // King tropism: firstly, find squares that we attack in the enemy king flank
  b = ei->attackedBy[Us][0] & KingFlank[file_of(square_of(Them, KING))];

  // Secondly, add to the bitboard the squares which we attack twice in that flank
  // but which are not protected by a enemy pawn. Note the trick to shift away the
  // previous attack bits to the empty part of the bitboard.
  b =  (b & ei->attackedBy2[Us] & ~ei->attackedBy[Them][PAWN])
     | (Us == WHITE ? b >> 4 : b << 4);

  // Count all these squares with a single popcount
  score += make_score(7 * popcount(b), 0);

//  if (DoTrace)
//    Trace::add(THREAT, Us, score);

  return score;
}


// evaluate_passed_pawns() evaluates the passed pawns of the given color

//  template<Color Us, bool DoTrace>
Score func(evaluate_passed_pawns)(Pos *pos, EvalInfo *ei)
{
  Bitboard b, squaresToQueen, defendedSquares, unsafeSquares;
  Score score = SCORE_ZERO;

  b = ei->pi->passedPawns[Us];

  while (b) {
    Square s = pop_lsb(&b);

    assert(pawn_passed(pos, Us, s));
    assert(!(pieces_p(PAWN) & forward_bb(Us, s)));

    int r = relative_rank_s(Us, s) - RANK_2;
    int rr = r * (r - 1);

    Value mbonus = Passed[MG][r], ebonus = Passed[EG][r];

    if (rr) {
      Square blockSq = s + pawn_push(Us);

      // Adjust bonus based on the king's proximity
      ebonus +=  distance(square_of(Them, KING), blockSq) * 5 * rr
               - distance(square_of(Us, KING), blockSq) * 2 * rr;

      // If blockSq is not the queening square then consider also a second push
      if (relative_rank_s(Us, blockSq) != RANK_8)
        ebonus -= distance(square_of(Us, KING), blockSq + pawn_push(Us)) * rr;

      // If the pawn is free to advance, then increase the bonus
      if (!piece_on(blockSq)) {
        // If there is a rook or queen attacking/defending the pawn from behind,
        // consider all the squaresToQueen. Otherwise consider only the squares
        // in the pawn's path attacked or occupied by the enemy.
        defendedSquares = unsafeSquares = squaresToQueen = forward_bb(Us, s);

        Bitboard bb = forward_bb(Them, s) & pieces_pp(QUEEN, ROOK) & attacks_from_rook(s);

        if (!(pieces_c(Us) & bb))
          defendedSquares &= ei->attackedBy[Us][0];

        if (!(pieces_c(Them) & bb))
          unsafeSquares &= ei->attackedBy[Them][0] | pieces_c(Them);

        // If there aren't any enemy attacks, assign a big bonus. Otherwise
        // assign a smaller bonus if the block square isn't attacked.
        int k = !unsafeSquares ? 18 : !(unsafeSquares & sq_bb(blockSq)) ? 8 : 0;

        // If the path to the queen is fully defended, assign a big bonus.
        // Otherwise assign a smaller bonus if the block square is defended.
        if (defendedSquares == squaresToQueen)
          k += 6;

        else if (defendedSquares & blockSq)
          k += 4;

        mbonus += k * rr, ebonus += k * rr;
      }
      else if (pieces_c(Us) & sq_bb(blockSq))
        mbonus += rr + r * 2, ebonus += rr + r * 2;
    } // rr != 0

    score += make_score(mbonus, ebonus) + PassedFile[file_of(s)];
  }

//  if (DoTrace)
//    Trace::add(PASSED, Us, score);

  // Add the scores to the middlegame and endgame eval
  return score;
}


// evaluate_space() computes the space evaluation for a given side. The
// space evaluation is a simple bonus based on the number of safe squares
// available for minor pieces on the central four files on ranks 2--4. Safe
// squares one, two or three squares behind a friendly pawn are counted
// twice. Finally, the space bonus is multiplied by a weight. The aim is to
// improve play on game opening.
//  template<Color Us>
Score func(evaluate_space)(Pos *pos, EvalInfo *ei)
{
  const Bitboard SpaceMask =
    Us == WHITE ? (FileCBB | FileDBB | FileEBB | FileFBB) & (Rank2BB | Rank3BB | Rank4BB)
                : (FileCBB | FileDBB | FileEBB | FileFBB) & (Rank7BB | Rank6BB | Rank5BB);

  // Find the safe squares for our pieces inside the area defined by
  // SpaceMask. A square is unsafe if it is attacked by an enemy
  // pawn, or if it is undefended and attacked by an enemy piece.
  Bitboard safe =   SpaceMask
                 & ~pieces_cp(Us, PAWN)
                 & ~ei->attackedBy[Them][PAWN]
                 & (ei->attackedBy[Us][0] | ~ei->attackedBy[Them][0]);

  // Find all squares which are at most three squares behind some friendly pawn
  Bitboard behind = pieces_cp(Us, PAWN);
  behind |= (Us == WHITE ? behind >>  8 : behind <<  8);
  behind |= (Us == WHITE ? behind >> 16 : behind << 16);

  // Since SpaceMask[Us] is fully on our half of the board...
  assert((unsigned)(safe >> (Us == WHITE ? 32 : 0)) == 0);

  // ...count safe + (behind & safe) with a single popcount
  int bonus = popcount((Us == WHITE ? safe << 32 : safe >> 32) | (behind & safe));
  int weight = popcount(pieces_pp(KNIGHT, BISHOP));

  return make_score(bonus * weight * weight * 2 / 11, 0);
}

#undef Them
#undef func
#undef shift_bb_Up
#undef shift_bb_Down
#undef shift_bb_Left
#undef shift_bb_Right
#undef LOOP
#undef GENERAL
#undef MOBILITY
#undef MINOR

