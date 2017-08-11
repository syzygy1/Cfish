#if (NT == PV)
#define PvNode 1
#define cutNode 0
#define name_NT(name) name##_PV
#else
#define PvNode 0
#define name_NT(name) name##_NonPV
#define beta (alpha+1)
#endif

#if PvNode
Value search_PV(Pos *pos, Stack *ss, Value alpha, Value beta, Depth depth)
#else
Value search_NonPV(Pos *pos, Stack *ss, Value alpha, Depth depth, int cutNode)
#endif
{
  int rootNode = PvNode && (ss-1)->ply == 0;

  assert(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE);
  assert(PvNode || (alpha == beta - 1));
  assert(DEPTH_ZERO < depth && depth < DEPTH_MAX);
  assert(!(PvNode && cutNode));

  Move pv[MAX_PLY+1], quietsSearched[64];
  TTEntry *tte;
  Key posKey;
  Move ttMove, move, excludedMove, bestMove;
  Depth extension, newDepth;
  Value bestValue, value, ttValue, eval;
  int ttHit, inCheck, givesCheck, singularExtensionNode, improving;
  int captureOrPromotion, doFullDepthSearch, moveCountPruning, skipQuiets;
  int ttCapture;
  Piece movedPiece;
  int moveCount, quietCount;

  // Step 1. Initialize node
  inCheck = !!pos_checkers();
  moveCount = quietCount =  ss->moveCount = 0;
  ss->statScore = 0;
  bestValue = -VALUE_INFINITE;

  // Check for the available remaining time
  if (load_rlx(pos->resetCalls)) {
    store_rlx(pos->resetCalls, 0);
    pos->callsCnt = Limits.nodes ? min(4096, Limits.nodes / 1024)
                                 : 4096;
  }
  if (--pos->callsCnt <= 0) {
    for (int idx = 0; idx < Threads.num_threads; idx++)
      store_rlx(Threads.pos[idx]->resetCalls, 1);

    check_time();
  }

  // Used to send selDepth info to GUI
  if (PvNode && pos->selDepth < ss->ply)
    pos->selDepth = ss->ply;

  if (!rootNode) {
    // Step 2. Check for aborted search and immediate draw
    if (load_rlx(Signals.stop) || is_draw(pos) || ss->ply >= MAX_PLY)
      return ss->ply >= MAX_PLY && !inCheck ? evaluate(pos)
                                            : DrawValue[pos_stm()];

    // Step 3. Mate distance pruning. Even if we mate at the next move our
    // score would be at best mate_in(ss->ply+1), but if alpha is already
    // bigger because a shorter mate was found upward in the tree then
    // there is no need to search because we will never beat the current
    // alpha. Same logic but with reversed signs applies also in the
    // opposite condition of being mated instead of giving mate. In this
    // case return a fail-high score.
#if PvNode
    alpha = max(mated_in(ss->ply), alpha);
    beta = min(mate_in(ss->ply+1), beta);
    if (alpha >= beta)
      return alpha;
#else
    if (alpha < mated_in(ss->ply))
      return mated_in(ss->ply);
    if (alpha >= mate_in(ss->ply+1))
      return alpha;
#endif
  }

  assert(0 <= ss->ply && ss->ply < MAX_PLY);

  (ss+1)->excludedMove = bestMove = 0;
  ss->history = &(*pos->counterMoveHistory)[0][0];
  (ss+2)->killers[0] = (ss+2)->killers[1] = 0;
  Square prevSq = to_sq((ss-1)->currentMove);

  // Step 4. Transposition table lookup. We don't want the score of a
  // partial search to overwrite a previous full search TT value, so we
  // use a different position key in case of an excluded move.
  excludedMove = ss->excludedMove;
  posKey = pos_key() ^ (Key)excludedMove;
  tte = tt_probe(posKey, &ttHit);
  ttValue = ttHit ? value_from_tt(tte_value(tte), ss->ply) : VALUE_NONE;
  ttMove =  rootNode ? pos->rootMoves->move[pos->PVIdx].pv[0]
          : ttHit    ? tte_move(tte) : 0;

  // At non-PV nodes we check for an early TT cutoff.
  if (  !PvNode
      && ttHit
      && tte_depth(tte) >= depth
      && ttValue != VALUE_NONE // Possible in case of TT access race.
      && (ttValue >= beta ? (tte_bound(tte) & BOUND_LOWER)
                          : (tte_bound(tte) & BOUND_UPPER))) {
    // If ttMove is quiet, update move sorting heuristics on TT hit.
    if (ttMove) {
      if (ttValue >= beta) {
        if (!is_capture_or_promotion(pos, ttMove))
          update_stats(pos, ss, ttMove, NULL, 0, stat_bonus(depth));

        // Extra penalty for a quiet TT move in previous ply when it gets
        // refuted.
        if ((ss-1)->moveCount == 1 && !captured_piece())
          update_cm_stats(ss-1, piece_on(prevSq), prevSq,
                          -stat_bonus(depth + ONE_PLY));
      }
      // Penalty for a quiet ttMove that fails low
      else if (!is_capture_or_promotion(pos, ttMove)) {
        Value penalty = -stat_bonus(depth);
        history_update(*pos->history, pos_stm(), ttMove, penalty);
        update_cm_stats(ss, moved_piece(ttMove), to_sq(ttMove), penalty);
      }
    }
    return ttValue;
  }

  // Step 4a. Tablebase probe
  if (!rootNode && TB_Cardinality) {
    int piecesCnt = popcount(pieces());

    if (    piecesCnt <= TB_Cardinality
        && (piecesCnt <  TB_Cardinality || depth >= TB_ProbeDepth)
        &&  pos_rule50_count() == 0
        && !can_castle_any())
    {
      int found, v = TB_probe_wdl(pos, &found);

      if (found) {
        pos->tb_hits++;

        int drawScore = TB_UseRule50 ? 1 : 0;

        value =  v < -drawScore ? -VALUE_MATE + MAX_PLY + ss->ply
               : v >  drawScore ?  VALUE_MATE - MAX_PLY - ss->ply
                                :  VALUE_DRAW + 2 * v * drawScore;

        tte_save(tte, posKey, value_to_tt(value, ss->ply), BOUND_EXACT,
                 min(DEPTH_MAX - ONE_PLY, depth + 6 * ONE_PLY),
                 0, VALUE_NONE, tt_generation());

        return value;
      }
    }
  }

  // Step 5. Evaluate the position statically
  if (inCheck) {
    ss->staticEval = eval = VALUE_NONE;
    goto moves_loop;
  } else if (ttHit) {
    // Never assume anything on values stored in TT
    if ((ss->staticEval = eval = tte_eval(tte)) == VALUE_NONE)
      eval = ss->staticEval = evaluate(pos);

    // Can ttValue be used as a better position evaluation?
    if (ttValue != VALUE_NONE)
      if (tte_bound(tte) & (ttValue > eval ? BOUND_LOWER : BOUND_UPPER))
        eval = ttValue;
  } else {
    eval = ss->staticEval =
    (ss-1)->currentMove != MOVE_NULL ? evaluate(pos)
                                     : -(ss-1)->staticEval + 2 * Tempo;

    tte_save(tte, posKey, VALUE_NONE, BOUND_NONE, DEPTH_NONE, 0,
             ss->staticEval, tt_generation());
  }

  if (ss->skipEarlyPruning)
    goto moves_loop;

  // Step 6. Razoring (skipped when in check)
  if (   !PvNode
      &&  depth < 4 * ONE_PLY
      &&  eval + razor_margin[depth / ONE_PLY] <= alpha) {

    if (depth <= ONE_PLY)
      return qsearch_NonPV_false(pos, ss, alpha, DEPTH_ZERO);

    Value ralpha = alpha - razor_margin[depth / ONE_PLY];
    Value v = qsearch_NonPV_false(pos, ss, ralpha, DEPTH_ZERO);
    if (v <= ralpha)
      return v;
  }

  // Step 7. Futility pruning: child node (skipped when in check)
  if (   !rootNode
      &&  depth < 7 * ONE_PLY
      &&  eval - futility_margin(depth) >= beta
      &&  eval < VALUE_KNOWN_WIN  // Do not return unproven wins
      &&  pos_non_pawn_material(pos_stm()))
    return eval; // - futility_margin(depth); (do not do the right thing)

  // Step 8. Null move search with verification search (is omitted in PV nodes)
  if (   !PvNode
      &&  eval >= beta
      && (ss->staticEval >= beta - 35 * (depth / ONE_PLY - 6) || depth >= 13 * ONE_PLY)
      &&  pos_non_pawn_material(pos_stm())) {

    assert(eval - beta >= 0);

    // Null move dynamic reduction based on depth and value
    Depth R = ((823 + 67 * depth / ONE_PLY) / 256 + min((eval - beta) / PawnValueMg, 3)) * ONE_PLY;

    ss->currentMove = MOVE_NULL;
    ss->history = &(*pos->counterMoveHistory)[0][0];

    do_null_move(pos);
    ss->endMoves = (ss-1)->endMoves;
    (ss+1)->skipEarlyPruning = 1;
    Value nullValue =   depth-R < ONE_PLY
                     ? -qsearch_NonPV_false(pos, ss+1, -beta, DEPTH_ZERO)
                     : - search_NonPV(pos, ss+1, -beta, depth-R, !cutNode);
    (ss+1)->skipEarlyPruning = 0;
    undo_null_move(pos);

    if (nullValue >= beta) {
      // Do not return unproven mate scores
      if (nullValue >= VALUE_MATE_IN_MAX_PLY)
         nullValue = beta;

      if (depth < 12 * ONE_PLY && abs(beta) < VALUE_KNOWN_WIN)
         return nullValue;

      // Do verification search at high depths
      ss->skipEarlyPruning = 1;
      Value v = depth-R < ONE_PLY ? qsearch_NonPV_false(pos, ss, beta-1, DEPTH_ZERO)
                                  :  search_NonPV(pos, ss, beta-1, depth-R, 0);
      ss->skipEarlyPruning = 0;

      if (v >= beta)
        return nullValue;
    }
  }

  // Step 9. ProbCut (skipped when in check)
  // If we have a good enough capture and a reduced search returns a value
  // much above beta, we can (almost) safely prune the previous move.
  if (   !PvNode
      &&  depth >= 5 * ONE_PLY
      &&  abs(beta) < VALUE_MATE_IN_MAX_PLY) {

    Value rbeta = min(beta + 200, VALUE_INFINITE);
    Depth rdepth = depth - 4 * ONE_PLY;

    assert(rdepth >= ONE_PLY);
    assert(move_is_ok((ss-1)->currentMove));

    mp_init_pc(pos, ttMove, rbeta - ss->staticEval);

    while ((move = next_move(pos, 0)))
      if (is_legal(pos, move)) {
        ss->currentMove = move;
        ss->history = &(*pos->counterMoveHistory)[moved_piece(move)][to_sq(move)];
        do_move(pos, move, gives_check(pos, ss, move));
        value = -search_NonPV(pos, ss+1, -rbeta, rdepth, !cutNode);
        undo_move(pos, move);
        if (value >= rbeta)
          return value;
      }
  }

  // Step 10. Internal iterative deepening (skipped when in check)
  if (    depth >= 6 * ONE_PLY
      && !ttMove
      && (PvNode || ss->staticEval + 256 >= beta))
  {
    Depth d = (3 * depth / (4 * ONE_PLY) - 2) * ONE_PLY;
    ss->skipEarlyPruning = 1;
#if PvNode
    search_PV(pos, ss, alpha, beta, d);
#else
    search_NonPV(pos, ss, alpha, d, cutNode);
#endif
    ss->skipEarlyPruning = 0;

    tte = tt_probe(posKey, &ttHit);
    ttMove = ttHit ? tte_move(tte) : 0;
  }

moves_loop: // When in check search starts from here.
  ;  // Avoid a compiler warning. A label must be followed by a statement.
  PieceToHistory *cmh  = (ss-1)->history;
  PieceToHistory *fmh  = (ss-2)->history;
  PieceToHistory *fmh2 = (ss-4)->history;

  mp_init(pos, ttMove, depth);
  value = bestValue; // Workaround a bogus 'uninitialized' warning under gcc
  improving =   ss->staticEval >= (ss-2)->staticEval
          /* || ss->staticEval == VALUE_NONE Already implicit in the previous condition */
             ||(ss-2)->staticEval == VALUE_NONE;

  singularExtensionNode =   !rootNode
                         &&  depth >= 8 * ONE_PLY
                         &&  ttMove
                         && !excludedMove // Recursive singular search is not allowed
                         && (tte_bound(tte) & BOUND_LOWER)
                         &&  tte_depth(tte) >= depth - 3 * ONE_PLY;
  skipQuiets = 0;
  ttCapture = 0;

  // Step 11. Loop through moves
  // Loop through all pseudo-legal moves until no moves remain or a beta cutoff occurs
  while ((move = next_move(pos, skipQuiets))) {
    assert(move_is_ok(move));

    if (move == excludedMove)
      continue;

    // At root obey the "searchmoves" option and skip moves not listed
    // inRoot Move List. As a consequence any illegal move is also skipped.
    // In MultiPV mode we also skip PV moves which have been already
    // searched.
    if (rootNode) {
      int idx;
      for (idx = pos->PVIdx; idx < pos->PVLast; idx++)
        if (pos->rootMoves->move[idx].pv[0] == move)
          break;
      if (idx == pos->PVLast)
        continue;
    }

    ss->moveCount = ++moveCount;

    if (rootNode && pos->thread_idx == 0 && time_elapsed() > 3000) {
      char buf[16];
      IO_LOCK;
      printf("info depth %d currmove %s currmovenumber %d\n",
             depth / ONE_PLY,
             uci_move(buf, move, is_chess960()),
             moveCount + pos->PVIdx);
      fflush(stdout);
      IO_UNLOCK;
    }

    if (PvNode)
      (ss+1)->pv = NULL;

    extension = DEPTH_ZERO;
    captureOrPromotion = is_capture_or_promotion(pos, move);
    movedPiece = moved_piece(move);

    givesCheck = gives_check(pos, ss, move);

    moveCountPruning =   depth < 16 * ONE_PLY
                && moveCount >= FutilityMoveCounts[improving][depth / ONE_PLY];

    // Step 12. Singular and Gives Check Extensions

    // Singular extension search. If all moves but one fail low on a search
    // of (alpha-s, beta-s), and just one fails high on (alpha, beta), then
    // that move is singular and should be extended. To verify this we do a
    // reduced search on all the other moves but the ttMove and if the
    // result is lower than ttValue minus a margin then we extend the ttMove.
    if (    singularExtensionNode
        &&  move == ttMove
        && !extension
        &&  is_legal(pos, move))
    {
      Value rBeta = max(ttValue - 2 * depth / ONE_PLY, -VALUE_MATE);
      Depth d = (depth / (2 * ONE_PLY)) * ONE_PLY;
      ss->excludedMove = move;
      ss->skipEarlyPruning = 1;
      Move cm = ss->countermove;
      Move k1 = ss->mp_killers[0], k2 = ss->mp_killers[1];
      value = search_NonPV(pos, ss, rBeta - 1, d, cutNode);
      ss->skipEarlyPruning = 0;
      ss->excludedMove = 0;

      if (value < rBeta)
        extension = ONE_PLY;

      // The call to search_NonPV with the same value of ss messed up our
      // move picker data. So we fix it.
      mp_init(pos, ttMove, depth);
      ss->stage++;
      ss->countermove = cm; // pedantic
      ss->mp_killers[0] = k1; ss->mp_killers[1] = k2;
    }
    else if (    givesCheck
             && !moveCountPruning
             &&  see_test(pos, move, 0))
      extension = ONE_PLY;

    // Calculate new depth for this move
    newDepth = depth - ONE_PLY + extension;

    // Step 13. Pruning at shallow depth
    if (  !rootNode
        && pos_non_pawn_material(pos_stm())
        && bestValue > VALUE_MATED_IN_MAX_PLY)
    {
      if (   !captureOrPromotion
          && !givesCheck
          && (  !advanced_pawn_push(pos, move)
              || pos_non_pawn_material(WHITE) + pos_non_pawn_material(BLACK) >= 5000)
             )
      {
        // Move count based pruning
        if (moveCountPruning) {
          skipQuiets = 1;
          continue;
        }

        // Reduced depth of the next LMR search
        int lmrDepth = max(newDepth - reduction(improving, depth, moveCount, NT), DEPTH_ZERO) / ONE_PLY;

        // Countermoves based pruning
        if (   lmrDepth < 3
            && (*cmh )[movedPiece][to_sq(move)] < CounterMovePruneThreshold
            && (*fmh )[movedPiece][to_sq(move)] < CounterMovePruneThreshold)
          continue;

        // Futility pruning: parent node
        if (   lmrDepth < 7
            && !inCheck
            && ss->staticEval + 256 + 200 * lmrDepth <= alpha)
          continue;

        // Prune moves with negative SEE at low depths and below a decreasing
        // threshold at higher depths.
        if (   lmrDepth < 8
            && !extension
            && !see_test(pos, move, -35 * lmrDepth * lmrDepth))
          continue;
      }
//      else if (   depth < 7 * ONE_PLY && ss->stage != ST_GOOD_CAPTURES
//               && !see_test(pos, move, -35 * depth / ONE_PLY * depth / ONE_PLY))
      else if (    depth < 7 * ONE_PLY
               && !extension
               && !see_test(pos, move, -PawnValueEg * (depth / ONE_PLY)))
        continue;
    }

    // Speculative prefetch as early as possible
    prefetch(tt_first_entry(key_after(pos, move)));

    // Check for legality just before making the move
    if (!rootNode && !is_legal(pos, move)) {
      ss->moveCount = --moveCount;
      continue;
    }

    if (move == ttMove && captureOrPromotion)
      ttCapture = 1;

    // Update the current move (this must be done after singular extension
    // search)
    ss->currentMove = move;
    ss->history = &(*pos->counterMoveHistory)[movedPiece][to_sq(move)];

    // Step 14. Make the move.
    do_move(pos, move, givesCheck);
    // HACK: Fix bench after introduction of 2-fold MultiPV bug
    if (rootNode) pos->st[-1].key ^= pos->rootKeyFlip;

    // Step 15. Reduced depth search (LMR). If the move fails high it will be
    // re-searched at full depth.
    if (    depth >= 3 * ONE_PLY
        &&  moveCount > 1
        && (!captureOrPromotion || moveCountPruning))
    {
      Depth r = reduction(improving, depth, moveCount, NT);
      if (captureOrPromotion)
        r -= r ? ONE_PLY : DEPTH_ZERO;
      else {
        // Increase reduction if ttMove is a capture
        if (ttCapture)
          r += ONE_PLY;

        // Increase reduction for cut nodes
        if (cutNode)
          r += 2 * ONE_PLY;

        // Decrease reduction for moves that escape a capture. Filter out
        // castling moves, because they are coded as "king captures rook" and
        // hence break make_move(). Also use see() instead of see_sign(),
        // because the destination square is empty.
        else if (   type_of_m(move) == NORMAL
                 && !see_test(pos, make_move(to_sq(move), from_sq(move)), 0))
          r -= 2 * ONE_PLY;

        ss->statScore =  (*cmh )[movedPiece][to_sq(move)]
                       + (*fmh )[movedPiece][to_sq(move)]
                       + (*fmh2)[movedPiece][to_sq(move)]
                       + (*pos->history)[pos_stm() ^ 1][from_to(move)]
                       - 4000; // Correction factor.

        // Decrease/increase reduction by comparing with opponent's stat score.
        if (ss->statScore > 0 && (ss-1)->statScore < 0)
          r -= ONE_PLY;

        else if (ss->statScore < 0 && (ss-1)->statScore > 0)
          r += ONE_PLY;

        // Decrease/increase reduction for moves with a good/bad history.
        r = max(DEPTH_ZERO, (r / ONE_PLY - ss->statScore / 20000) * ONE_PLY);
      }

      Depth d = max(newDepth - r, ONE_PLY);

      value = -search_NonPV(pos, ss+1, -(alpha+1), d, 1);

      doFullDepthSearch = (value > alpha && d != newDepth);

    } else
      doFullDepthSearch = !PvNode || moveCount > 1;

    // Step 16. Full depth search when LMR is skipped or fails high.
    if (doFullDepthSearch)
        value = newDepth <   ONE_PLY ?
                          givesCheck ? -qsearch_NonPV_true(pos, ss+1, -(alpha+1), DEPTH_ZERO)
                                     : -qsearch_NonPV_false(pos, ss+1, -(alpha+1), DEPTH_ZERO)
                                     : - search_NonPV(pos, ss+1, -(alpha+1), newDepth, !cutNode);

    // For PV nodes only, do a full PV search on the first move or after a fail
    // high (in the latter case search only if value < beta), otherwise let the
    // parent node fail low with value <= alpha and try another move.
    if (PvNode && (moveCount == 1 || (value > alpha && (rootNode || value < beta)))) {
      (ss+1)->pv = pv;
      (ss+1)->pv[0] = 0;

      value = newDepth <   ONE_PLY ?
                        givesCheck ? -qsearch_PV_true(pos, ss+1, -beta, -alpha, DEPTH_ZERO)
                                   : -qsearch_PV_false(pos, ss+1, -beta, -alpha, DEPTH_ZERO)
                                   : - search_PV(pos, ss+1, -beta, -alpha, newDepth);
    }

    // Step 17. Undo move
    // HACK: Fix bench after introduction of 2-fold MultiPV bug
    if (rootNode) pos->st[-1].key ^= pos->rootKeyFlip;
    undo_move(pos, move);

    assert(value > -VALUE_INFINITE && value < VALUE_INFINITE);

    // Step 18. Check for a new best move
    // Finished searching the move. If a stop occurred, the return value of
    // the search cannot be trusted, and we return immediately without
    // updating best move, PV and TT.
    if (load_rlx(Signals.stop))
      return 0;

    if (rootNode) {
      RootMove *rm = NULL;
      for (int idx = 0; idx < pos->rootMoves->size; idx++)
        if (pos->rootMoves->move[idx].pv[0] == move) {
          rm = &pos->rootMoves->move[idx];
          break;
        }

      // PV move or new best move ?
      if (moveCount == 1 || value > alpha) {
        rm->score = value;
        rm->selDepth = pos->selDepth;
        rm->pv_size = 1;

        assert((ss+1)->pv);

        for (Move *m = (ss+1)->pv; *m; ++m)
          rm->pv[rm->pv_size++] = *m;

        // We record how often the best move has been changed in each
        // iteration. This information is used for time management: When
        // the best move changes frequently, we allocate some more time.
        if (moveCount > 1 && pos->thread_idx == 0)
          mainThread.bestMoveChanges++;
      } else
        // All other moves but the PV are set to the lowest value: this is
        // not a problem when sorting because the sort is stable and the
        // move position in the list is preserved - just the PV is pushed up.
        rm->score = -VALUE_INFINITE;
    }

    if (value > bestValue) {
      bestValue = value;

      if (value > alpha) {
        bestMove = move;

        if (PvNode && !rootNode) // Update pv even in fail-high case
          update_pv(ss->pv, move, (ss+1)->pv);

        if (PvNode && value < beta) // Update alpha! Always alpha < beta
          alpha = value;
        else {
          assert(value >= beta); // Fail high
          break;
        }
      }
    }

    if (!captureOrPromotion && move != bestMove && quietCount < 64)
      quietsSearched[quietCount++] = move;
  }

  // The following condition would detect a stop only after move loop has
  // been completed. But in this case bestValue is valid because we have
  // fully searched our subtree, and we can anyhow save the result in TT.
  /*
  if (Signals.stop)
    return VALUE_DRAW;
  */

  // Step 20. Check for mate and stalemate
  // All legal moves have been searched and if there are no legal moves,
  // it must be a mate or a stalemate. If we are in a singular extension
  // search then return a fail low score.
  if (!moveCount)
    bestValue = excludedMove ? alpha
               :     inCheck ? mated_in(ss->ply) : DrawValue[pos_stm()];
  else if (bestMove) {
    // Quiet best move: update move sorting heuristics.
    if (!is_capture_or_promotion(pos, bestMove))
      update_stats(pos, ss, bestMove, quietsSearched, quietCount,
                   stat_bonus(depth));

    // Extra penalty for a quiet TT move in previous ply when it gets refuted.
    if ((ss-1)->moveCount == 1 && !captured_piece())
      update_cm_stats(ss-1, piece_on(prevSq), prevSq,
                      -stat_bonus(depth + ONE_PLY));
  }
  // Bonus for prior countermove that caused the fail low.
  else if (    depth >= 3 * ONE_PLY
           && !captured_piece()
           && move_is_ok((ss-1)->currentMove))
    update_cm_stats(ss-1, piece_on(prevSq), prevSq, stat_bonus(depth));

  if (!excludedMove)
    tte_save(tte, posKey, value_to_tt(bestValue, ss->ply),
             bestValue >= beta ? BOUND_LOWER :
             PvNode && bestMove ? BOUND_EXACT : BOUND_UPPER,
             depth, bestMove, ss->staticEval, tt_generation());

  assert(bestValue > -VALUE_INFINITE && bestValue < VALUE_INFINITE);

  return bestValue;
}

#undef PvNode
#undef name_NT
#ifdef cutNode
#undef cutNode
#endif
#ifdef beta
#undef beta
#endif

