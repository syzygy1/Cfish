#if NT == PV
#define name_NT(name,chk) name##_PV_##chk
#define BETA_ARG Value beta,
#if InCheck == false
#define name_NT_InCheck(name) name##_PV_false
#else
#define name_NT_InCheck(name) name##_PV_true
#endif
#else
#define name_NT(name,chk) name##_NonPV_##chk
#define BETA_ARG
#define beta (alpha+1)
#if InCheck == false
#define name_NT_InCheck(name) name##_NonPV_false
#else
#define name_NT_InCheck(name) name##_NonPV_true
#endif
#endif

#define PvNode (NT == PV)

Value name_NT_InCheck(qsearch)(Pos* pos, Stack* ss, Value alpha, BETA_ARG
                               Depth depth)
{
  assert(InCheck == !!pos_checkers());
  assert(alpha >= -VALUE_INFINITE && alpha < beta && beta <= VALUE_INFINITE);
  assert(PvNode || (alpha == beta - 1));
  assert(depth <= DEPTH_ZERO);

  Move pv[MAX_PLY+1];
  TTEntry *tte;
  Key posKey;
  Move ttMove, move, bestMove;
  Value bestValue, value, ttValue, futilityValue, futilityBase, oldAlpha;
  int ttHit, givesCheck, evasionPrunable;
  Depth ttDepth;
  int moveCount;

  if (PvNode) {
    oldAlpha = alpha; // To flag BOUND_EXACT when eval above alpha and no available moves
    (ss+1)->pv = pv;
    ss->pv[0] = 0;
  }

  bestMove = 0;
  moveCount = 0;

  // Check for an instant draw or if the maximum ply has been reached
  if (is_draw(pos) || ss->ply >= MAX_PLY)
    return ss->ply >= MAX_PLY && !InCheck ? evaluate(pos) : VALUE_DRAW;

  assert(0 <= ss->ply && ss->ply < MAX_PLY);

  // Decide whether or not to include checks: this fixes also the type of
  // TT entry depth that we are going to use. Note that in qsearch we use
  // only two types of depth in TT: DEPTH_QS_CHECKS or DEPTH_QS_NO_CHECKS.
  ttDepth = InCheck || depth >= DEPTH_QS_CHECKS ? DEPTH_QS_CHECKS
                                                : DEPTH_QS_NO_CHECKS;

  // Transposition table lookup
  posKey = pos_key();
  tte = tt_probe(posKey, &ttHit);
  ttMove = ttHit ? tte_move(tte) : 0;
  ttValue = ttHit ? value_from_tt(tte_value(tte), ss->ply) : VALUE_NONE;

  if (  !PvNode
      && ttHit
      && tte_depth(tte) >= ttDepth
      && ttValue != VALUE_NONE // Only in case of TT access race
      && (ttValue >= beta ? (tte_bound(tte) &  BOUND_LOWER)
                          : (tte_bound(tte) &  BOUND_UPPER)))
    return ttValue;

  // Evaluate the position statically
  if (InCheck) {
    ss->staticEval = VALUE_NONE;
    bestValue = futilityBase = -VALUE_INFINITE;
  } else {
    if (ttHit) {
      // Never assume anything on values stored in TT
      if ((ss->staticEval = bestValue = tte_eval(tte)) == VALUE_NONE)
         ss->staticEval = bestValue = evaluate(pos);

      // Can ttValue be used as a better position evaluation?
      if (ttValue != VALUE_NONE)
        if (tte_bound(tte) & (ttValue > bestValue ? BOUND_LOWER : BOUND_UPPER))
          bestValue = ttValue;
    } else
      ss->staticEval = bestValue =
      (ss-1)->currentMove != MOVE_NULL ? evaluate(pos)
                                       : -(ss-1)->staticEval + 2 * Tempo;

    // Stand pat. Return immediately if static value is at least beta
    if (bestValue >= beta) {
      if (!ttHit)
        tte_save(tte, posKey, value_to_tt(bestValue, ss->ply),
                 BOUND_LOWER, DEPTH_NONE, 0, ss->staticEval,
                 tt_generation());

      return bestValue;
    }

    if (PvNode && bestValue > alpha)
      alpha = bestValue;

    futilityBase = bestValue + 128;
  }

  ss->history = &(*pos->counterMoveHistory)[0][0];

  // Initialize move picker data for the current position, and prepare
  // to search the moves. Because the depth is <= 0 here, only captures,
  // queen promotions and checks (only if depth >= DEPTH_QS_CHECKS) will
  // be generated.
  mp_init_q(pos, ttMove, depth, to_sq((ss-1)->currentMove));

  // Loop through the moves until no moves remain or a beta cutoff occurs
  while ((move = next_move(pos, 0))) {
    assert(move_is_ok(move));

    givesCheck = gives_check(pos, ss, move);

    moveCount++;

    // Futility pruning
    if (   !InCheck
        && !givesCheck
        &&  futilityBase > -VALUE_KNOWN_WIN
        && !advanced_pawn_push(pos, move)) {
      assert(type_of_m(move) != ENPASSANT); // Due to !advanced_pawn_push

      futilityValue = futilityBase + PieceValue[EG][piece_on(to_sq(move))];

      if (futilityValue <= alpha) {
        bestValue = max(bestValue, futilityValue);
        continue;
      }

      if (futilityBase <= alpha && !see_test(pos, move, 1)) {
        bestValue = max(bestValue, futilityBase);
        continue;
      }
    }

    // Detect non-capture evasions that are candidates to be pruned
    evasionPrunable =    InCheck
                     && (depth != DEPTH_ZERO || moveCount > 2)
                     &&  bestValue > VALUE_MATED_IN_MAX_PLY
                     && !is_capture(pos, move);

    // Don't search moves with negative SEE values
    if (  (!InCheck || evasionPrunable)
        &&  !see_test(pos, move, 0))
      continue;

    // Speculative prefetch as early as possible
    prefetch(tt_first_entry(key_after(pos, move)));

    // Check for legality just before making the move
    if (!is_legal(pos, move)) {
      moveCount--;
      continue;
    }

    ss->currentMove = move;
    ss->history = &(*pos->counterMoveHistory)[moved_piece(move)][to_sq(move)];

    // Make and search the move
    do_move(pos, move, givesCheck);
#if PvNode
    value =  givesCheck
           ? -qsearch_PV_true(pos, ss+1, -beta, -alpha, depth - ONE_PLY)
           : -qsearch_PV_false(pos, ss+1, -beta, -alpha, depth - ONE_PLY);
#else
    value =  givesCheck
           ? -qsearch_NonPV_true(pos, ss+1, -beta, depth - ONE_PLY)
           : -qsearch_NonPV_false(pos, ss+1, -beta, depth - ONE_PLY);
#endif
    undo_move(pos, move);

    assert(value > -VALUE_INFINITE && value < VALUE_INFINITE);

    // Check for a new best move
    if (value > bestValue) {
      bestValue = value;

      if (value > alpha) {
        if (PvNode) // Update pv even in fail-high case
          update_pv(ss->pv, move, (ss+1)->pv);

        if (PvNode && value < beta) { // Update alpha here!
          alpha = value;
          bestMove = move;
        } else { // Fail high
          tte_save(tte, posKey, value_to_tt(value, ss->ply), BOUND_LOWER,
                   ttDepth, move, ss->staticEval, tt_generation());

          return value;
        }
      }
    }
  }

  // All legal moves have been searched. A special case: If we're in check
  // and no legal moves were found, it is checkmate.
  if (InCheck && bestValue == -VALUE_INFINITE)
    return mated_in(ss->ply); // Plies to mate from the root

  tte_save(tte, posKey, value_to_tt(bestValue, ss->ply),
           PvNode && bestValue > oldAlpha ? BOUND_EXACT : BOUND_UPPER,
           ttDepth, bestMove, ss->staticEval, tt_generation());

  assert(bestValue > -VALUE_INFINITE && bestValue < VALUE_INFINITE);

  return bestValue;
}

#undef PvNode
#undef name_NT_InCheck
#undef name_NT
#undef BETA_ARG
#ifdef beta
#undef beta
#endif

