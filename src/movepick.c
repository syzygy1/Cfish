/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2008 Tord Romstad (Glaurung author)
  Copyright (C) 2008-2015 Marco Costalba, Joona Kiiski, Tord Romstad
  Copyright (C) 2015-2016 Marco Costalba, Joona Kiiski, Gary Linscott, Tord Romstad

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <assert.h>

#include "movepick.h"
#include "thread.h"

#define HistoryStats_Max ((Value)(1<<28))

#define ST_MAIN_SEARCH            0
#define ST_GOOD_CAPTURES          1
#define ST_KILLERS                2
#define ST_QUIET                  3
#define ST_BAD_CAPTURES           4
#define ST_EVASION                5
#define ST_ALL_EVASIONS           6
#define ST_QSEARCH_WITH_CHECKS    7
#define ST_QCAPTURES_1            8
#define ST_CHECKS                 9
#define ST_QSEARCH_WITHOUT_CHECKS 10
#define ST_QCAPTURES_2            11
#define ST_PROBCUT                12
#define ST_PROBCUT_CAPTURES       13
#define ST_RECAPTURE              14
#define ST_RECAPTURES             15
#define ST_STOP                   16

// Our insertion sort, which is guaranteed to be stable, as it should be.

static inline void insertion_sort(ExtMove *begin, ExtMove *end)
{
  ExtMove tmp, *p, *q;

  for (p = begin + 1; p < end; p++) {
    tmp = *p;
    for (q = p; q != begin && (q-1)->value < tmp.value; --q)
      *q = *(q-1);
    *q = tmp;
  }
}

// Our non-stable partition function, the one that Stockfish uses.

static inline ExtMove *partition(ExtMove *first, ExtMove *last)
{
  ExtMove tmp;

  while (1) {
    while (1)
      if (first == last)
        return first;
      else if (first->value > 0)
        first++;
      else
        break;
    last--;
    while (1)
      if (first == last)
        return first;
      else if (!(last->value > 0))
        last--;
      else
        break;
    tmp = *first;
    *first = *last;
    *last = tmp;
    first++;
  }
}

// pick_best() finds the best move in the range (begin, end) and moves
// it to the front. It's faster than sorting all the moves in advance
// when there are few moves, e.g., the possible captures.

Move pick_best(ExtMove *begin, ExtMove *end)
{
  ExtMove *p, *q, tmp;

  for (p = begin, q = begin + 1; q < end; q++)
    if (q->value > p->value)
      p = q;
  tmp = *begin;
  *begin = *p;
  *p = tmp;
  return begin->move;
}


// Constructors of the MovePicker class. As arguments we pass information
// to help it to return the (presumably) good moves first, to decide which
// moves to return (in the quiescence search, for instance, we only want
// to search captures, promotions, and some checks) and how important good
// move ordering is at the current node.

void mp_init(MovePicker *mp, Pos *pos, Move ttm, Depth depth, Stack *ss)
{
  assert(depth > DEPTH_ZERO);

  mp->pos = pos;
  mp->ss = ss;
  mp->depth = depth;

  Square prevSq = to_sq((ss-1)->currentMove);
  mp->countermove = (*pos->thisThread->counterMoves)[piece_on(prevSq)][prevSq];

  mp->stage = pos_checkers() ? ST_EVASION : ST_MAIN_SEARCH;
  mp->ttMove = ttm && is_pseudo_legal(pos, ttm) ? ttm : 0;
  mp->cur = mp->moves;
  mp->endMoves = mp->moves + (mp->ttMove != 0);
  mp->endBadCaptures = mp->moves + MAX_MOVES - 1;
}

void mp_init_q(MovePicker *mp, Pos *pos, Move ttm, Depth depth, Square s)
{
  assert (depth <= DEPTH_ZERO);

  mp->pos = pos;

  if (pos_checkers())
    mp->stage = ST_EVASION;
  else if (depth > DEPTH_QS_NO_CHECKS)
    mp->stage = ST_QSEARCH_WITH_CHECKS;
  else if (depth > DEPTH_QS_RECAPTURES)
    mp->stage = ST_QSEARCH_WITHOUT_CHECKS;
  else {
    mp->stage = ST_RECAPTURE;
    mp->recaptureSquare = s;
    ttm = 0;
  }

  mp->ttMove = ttm && is_pseudo_legal(pos, ttm) ? ttm : 0;
  mp->cur = mp->moves;
  mp->endMoves = mp->moves + (mp->ttMove != 0);
  mp->endBadCaptures = mp->moves + MAX_MOVES - 1;
}

void mp_init_pc(MovePicker *mp, Pos *pos, Move ttm, Value threshold)
{
  assert(!pos_checkers());

  mp->pos = pos;
  mp->threshold = threshold;

  mp->stage = ST_PROBCUT;

  // In ProbCut we generate captures with SEE higher than the given threshold
  mp->ttMove = ttm && is_pseudo_legal(pos, ttm) && is_capture(pos, ttm)
                   && see(pos, ttm) > threshold ? ttm : 0;

  mp->cur = mp->moves;
  mp->endMoves = mp->moves + (mp->ttMove != 0);
  mp->endBadCaptures = mp->moves + MAX_MOVES - 1;
}


// score() assigns a numerical value to each move in a move list. The moves with
// highest values will be picked first.

void score_captures(MovePicker *mp)
{
  Pos *pos = mp->pos;

  // Winning and equal captures in the main search are ordered by MVV,
  // preferring captures near our home rank. Surprisingly, this appears
  // to perform slightly better than SEE-based move ordering: exchanging
  // big pieces before capturing a hanging piece probably helps to reduce
  // the subtree size. In the main search we want to push captures with
  // negative SEE values to the badCaptures[] array, but instead of doing
  // it now we delay until the move has been picked up, saving some SEE
  // calls in case we get a cutoff.

  for (ExtMove *m = mp->moves; m < mp->endMoves; m++)
    m->value =  PieceValue[MG][piece_on(to_sq(m->move))]
              - (Value)(200 * relative_rank_s(pos_stm(), to_sq(m->move)));
}

void score_quiets(MovePicker *mp)
{
  Pos *pos = mp->pos;
  HistoryStats *history = pos->thisThread->history;
  FromToStats *fromTo = pos->thisThread->fromTo;

  CounterMoveStats *cm = (mp->ss-1)->counterMoves;
  CounterMoveStats *fm = (mp->ss-2)->counterMoves;
  CounterMoveStats *f2 = (mp->ss-4)->counterMoves;

  int c = pos_stm();

  for (ExtMove *m = mp->moves; m < mp->endMoves; m++)
    m->value =   (*history)[moved_piece(m->move)][to_sq(m->move)]
              + (cm ? (*cm)[moved_piece(m->move)][to_sq(m->move)] : 0)
              + (fm ? (*fm)[moved_piece(m->move)][to_sq(m->move)] : 0)
              + (f2 ? (*f2)[moved_piece(m->move)][to_sq(m->move)] : 0)
              + ft_get(*fromTo, c, m->move);
}

void score_evasions(MovePicker *mp)
{
  Pos *pos = mp->pos;

  // Try winning and equal captures ordered by MVV/LVA, then non-captures
  // ordered by history value, then bad captures and quiet moves with a
  // negative SEE ordered by SEE value.

  HistoryStats *history = pos->thisThread->history;
  FromToStats *fromTo = pos->thisThread->fromTo;
  int c = pos_stm();
  Value see;

  for (ExtMove *m = mp->moves; m < mp->endMoves; m++)
    if ((see = see_sign(pos, m->move)) < VALUE_ZERO)
      m->value = see - HistoryStats_Max; // At the bottom
    else if (is_capture(pos, m->move))
      m->value =  PieceValue[MG][piece_on(to_sq(m->move))]
                - (Value)type_of_p(moved_piece(m->move)) + HistoryStats_Max;
    else
      m->value =  (*history)[moved_piece(m->move)][to_sq(m->move)]
                + ft_get(*fromTo, c, m->move);
}


// generate_next_stage() generates, scores, and sorts the next bunch of
// moves when there are no more moves to try for the current stage.

void generate_next_stage(MovePicker *mp)
{
  assert(mp->stage != ST_STOP);

  mp->cur = mp->moves;

  switch (++(mp->stage)) {

  case ST_GOOD_CAPTURES: case ST_QCAPTURES_1: case ST_QCAPTURES_2:
  case ST_PROBCUT_CAPTURES: case ST_RECAPTURES:
    mp->endMoves = generate_captures(mp->pos, mp->moves);
    score_captures(mp);
    break;

  case ST_KILLERS:
    mp->killers[0].move = mp->ss->killers[0];
    mp->killers[1].move = mp->ss->killers[1];
    mp->killers[2].move = mp->countermove;
    mp->cur = mp->killers;
    mp->endMoves = mp->cur + 2 + (mp->countermove != mp->killers[0].move && mp->countermove != mp->killers[1].move);
    break;

  case ST_QUIET:
    mp->endMoves = generate_quiets(mp->pos, mp->moves);
    score_quiets(mp);
    if (mp->depth < 3 * ONE_PLY) {
      ExtMove *goodQuiet = partition(mp->cur, mp->endMoves);
      insertion_sort(mp->cur, goodQuiet);
    } else
      insertion_sort(mp->cur, mp->endMoves);
    break;

  case ST_BAD_CAPTURES:
    // Just pick them in reverse order to get correct ordering
    mp->cur = mp->moves + MAX_MOVES - 1;
    mp->endMoves = mp->endBadCaptures;
    break;

  case ST_ALL_EVASIONS:
    mp->endMoves = generate_evasions(mp->pos, mp->moves);
    if (mp->endMoves - mp->moves > 1)
      score_evasions(mp);
    break;

  case ST_CHECKS:
    mp->endMoves = generate_quiet_checks(mp->pos, mp->moves);
    break;

  case ST_EVASION: case ST_QSEARCH_WITH_CHECKS: case ST_QSEARCH_WITHOUT_CHECKS:
  case ST_PROBCUT: case ST_RECAPTURE: case ST_STOP:
    mp->stage = ST_STOP;
    break;

  default:
    assert(0);
  }
}


// next_move() is the most important method of the MovePicker class. It
// returns a new pseudo legal move every time it is called, until there
// are no more moves left. It picks the move with the biggest value from
// a list of generated moves taking care not to return the ttMove if it
// has already been searched.

Move next_move(MovePicker *mp)
{
  Move move;

  while (1) {
    while (mp->cur == mp->endMoves && mp->stage != ST_STOP)
      generate_next_stage(mp);

    switch (mp->stage) {

    case ST_MAIN_SEARCH: case ST_EVASION: case ST_QSEARCH_WITH_CHECKS:
    case ST_QSEARCH_WITHOUT_CHECKS: case ST_PROBCUT:
      mp->cur++;
      return mp->ttMove;

    case ST_GOOD_CAPTURES:
      move = pick_best(mp->cur++, mp->endMoves);
      if (move != mp->ttMove) {
        if (see_sign(mp->pos, move) >= 0)
          return move;

        // Losing capture, move it to the tail of the array
        (mp->endBadCaptures--)->move = move;
      }
      break;

    case ST_KILLERS:
      move = (mp->cur++)->move;
      if (   move != 0 && move != mp->ttMove && is_pseudo_legal(mp->pos, move)
          && !is_capture(mp->pos, move))
        return move;
      break;

    case ST_QUIET:
      move = (mp->cur++)->move;
      if (   move != mp->ttMove && move != mp->killers[0].move
          && move != mp->killers[1].move && move != mp->killers[2].move)
        return move;
      break;

    case ST_BAD_CAPTURES:
      return (mp->cur--)->move;

    case ST_ALL_EVASIONS: case ST_QCAPTURES_1: case ST_QCAPTURES_2:
      move = pick_best(mp->cur++, mp->endMoves);
      if (move != mp->ttMove)
        return move;
      break;

    case ST_PROBCUT_CAPTURES:
      move = pick_best(mp->cur++, mp->endMoves);
      if (move != mp->ttMove && see(mp->pos, move) > mp->threshold)
        return move;
      break;

    case ST_RECAPTURES:
      move = pick_best(mp->cur++, mp->endMoves);
      if (to_sq(move) == mp->recaptureSquare)
        return move;
      break;

    case ST_CHECKS:
      move = (mp->cur++)->move;
      if (move != mp->ttMove)
        return move;
      break;

    case ST_STOP:
      return 0;

    default:
      assert(0);
    }
  }
}

