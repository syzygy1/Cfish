#include <assert.h>
#include <ctype.h>
#include <inttypes.h>
#include <string.h>

#include "bitboard.h"
#include "material.h"
#include "misc.h"
#include "movegen.h"
#include "pawns.h"
#include "position.h"
#include "tbprobe.h"
#include "thread.h"
#include "tt.h"
#include "uci.h"

static void set_castling_right(Pos *pos, Color c, Square rfrom);
static void set_state(Pos *pos, Stack *st);

#ifndef NDEBUG
static int pos_is_ok(Pos *pos, int *failedStep);
static int check_pos(Pos *pos);
#else
#define check_pos(p) do {} while (0)
#endif

struct Zob zob;

Key matKey[16] = {
  0ULL,
  0x5ced000000000101ULL,
  0xe173000000001001ULL,
  0xd64d000000010001ULL,
  0xab88000000100001ULL,
  0x680b000001000001ULL,
  0x0000000000000001ULL,
  0ULL,
  0ULL,
  0xf219000010000001ULL,
  0xbb14000100000001ULL,
  0x58df001000000001ULL,
  0xa15f010000000001ULL,
  0x7c94100000000001ULL,
  0x0000000000000001ULL,
  0ULL
};

const char PieceToChar[] = " PNBRQK  pnbrqk";

int failed_step;

#ifdef PEDANTIC
INLINE void put_piece(Pos *pos, Color c, Piece piece, Square s)
{
  pos->board[s] = piece;
  pos->byTypeBB[0] |= sq_bb(s);
  pos->byTypeBB[type_of_p(piece)] |= sq_bb(s);
  pos->byColorBB[c] |= sq_bb(s);
  pos->index[s] = pos->pieceCount[piece]++;
  pos->pieceList[pos->index[s]] = s;
}

INLINE void remove_piece(Pos *pos, Color c, Piece piece, Square s)
{
  // WARNING: This is not a reversible operation.
  pos->byTypeBB[0] ^= sq_bb(s);
  pos->byTypeBB[type_of_p(piece)] ^= sq_bb(s);
  pos->byColorBB[c] ^= sq_bb(s);
  /* board[s] = 0;  Not needed, overwritten by the capturing one */
  Square lastSquare = pos->pieceList[--pos->pieceCount[piece]];
  pos->index[lastSquare] = pos->index[s];
  pos->pieceList[pos->index[lastSquare]] = lastSquare;
  pos->pieceList[pos->pieceCount[piece]] = SQ_NONE;
}

INLINE void move_piece(Pos *pos, Color c, Piece piece, Square from, Square to)
{
  // index[from] is not updated and becomes stale. This works as long as
  // index[] is accessed just by known occupied squares.
  Bitboard fromToBB = sq_bb(from) ^ sq_bb(to);
  pos->byTypeBB[0] ^= fromToBB;
  pos->byTypeBB[type_of_p(piece)] ^= fromToBB;
  pos->byColorBB[c] ^= fromToBB;
  pos->board[from] = 0;
  pos->board[to] = piece;
  pos->index[to] = pos->index[from];
  pos->pieceList[pos->index[to]] = to;
}
#endif


// Calculate CheckInfo data.

INLINE void set_check_info(Pos *pos)
{
  Stack *st = pos->st;

  st->blockersForKing[WHITE] = slider_blockers(pos, pieces_c(BLACK), square_of(WHITE, KING), &st->pinnersForKing[WHITE]);
  st->blockersForKing[BLACK] = slider_blockers(pos, pieces_c(WHITE), square_of(BLACK, KING), &st->pinnersForKing[BLACK]);

  Color them = pos_stm() ^ 1;
  st->ksq = square_of(them, KING);

  st->checkSquares[PAWN]   = attacks_from_pawn(st->ksq, them);
  st->checkSquares[KNIGHT] = attacks_from_knight(st->ksq);
  st->checkSquares[BISHOP] = attacks_from_bishop(st->ksq);
  st->checkSquares[ROOK]   = attacks_from_rook(st->ksq);
  st->checkSquares[QUEEN]  = st->checkSquares[BISHOP] | st->checkSquares[ROOK];
  st->checkSquares[KING]   = 0;
}


// print_pos() prints an ASCII representation of the position to stdout.

void print_pos(Pos *pos)
{
  char fen[128];
  pos_fen(pos, fen);

  flockfile(stdout);
  printf("\n +---+---+---+---+---+---+---+---+\n");

  for (int r = 7; r >= 0; r--) {
    for (int f = 0; f <= 7; f++)
      printf(" | %c", PieceToChar[pos->board[8 * r + f]]);

    printf(" |\n +---+---+---+---+---+---+---+---+\n");
  }

  printf("\nFen: %s\nKey: %16"PRIX64"\nCheckers: ", fen, pos_key());

  char buf[16];
  for (Bitboard b = pos_checkers(); b; )
    printf("%s ", uci_square(buf, pop_lsb(&b)));

  if (popcount(pieces()) <= TB_MaxCardinality && !can_castle_cr(ANY_CASTLING)) {
    int s1, s2;
    int wdl = TB_probe_wdl(pos, &s1);
    int dtz = TB_probe_dtz(pos, &s2);
    printf("\nTablebases WDL: %4d (%d)\nTablebases DTZ: %4d (%d)", wdl, s1, dtz, s2);
    if (s1 && wdl != 0) {
      Value dtm = TB_probe_dtm(pos, wdl, &s1);
      printf("\nTablebases DTM: %s (%d)", uci_value(buf, dtm), s1);
    }
  }
  printf("\n");
  fflush(stdout);
  funlockfile(stdout);
}

INLINE Key H1(Key h)
{
  return h & 0x1fff;
}

INLINE Key H2(Key h)
{
  return (h >> 16) & 0x1fff;
}

static Key cuckoo[8192];
static uint16_t cuckooMove[8192];

// zob_init() initializes at startup the various arrays used to compute
// hash keys.

void zob_init(void) {

  PRNG rng;
  prng_init(&rng, 1070372);

  for (int c = 0; c < 2; c++)
    for (int pt = PAWN; pt <= KING; pt++)
      for (Square s = 0; s < 64; s++)
        zob.psq[make_piece(c, pt)][s] = prng_rand(&rng);

  for (int f = 0; f < 8; f++)
    zob.enpassant[f] = prng_rand(&rng);

  for (int cr = 0; cr < 16; cr++) {
    zob.castling[cr] = 0;
    Bitboard b = (Bitboard)cr;
    while (b) {
      Key k = zob.castling[1ULL << pop_lsb(&b)];
      zob.castling[cr] ^= k ? k : prng_rand(&rng);
    }
  }

  zob.side = prng_rand(&rng);
  zob.noPawns = prng_rand(&rng);

  // Prepare the cuckoo tables
  int count = 0;
  for (int c = 0; c < 2; c++)
    for (int pt = PAWN; pt <= KING; pt++) {
      int pc = make_piece(c, pt);
      for (Square s1 = 0; s1 < 64; s1++)
        for (Square s2 = s1 + 1; s2 < 64; s2++)
          if (PseudoAttacks[pt][s1] & sq_bb(s2)) {
            Move move = between_bb(s1, s2) ? make_move(s1, s2)
                                           : make_move(SQ_C3, SQ_D5);
            Key key = zob.psq[pc][s1] ^ zob.psq[pc][s2] ^ zob.side;
            uint32_t i = H1(key);
            while (1) {
              Key tmpKey = cuckoo[i];
              cuckoo[i] = key;
              key = tmpKey;
              Move tmpMove = cuckooMove[i];
              cuckooMove[i] = move;
              move = tmpMove;
              if (!move) break;
              i = (i == H1(key)) ? H2(key) : H1(key);
            }
            count++;
          }
    }
  assert(count == 3668);
}


// pos_set() initializes the position object with the given FEN string.
// This function is not very robust - make sure that input FENs are correct,
// this is assumed to be the responsibility of the GUI.

void pos_set(Pos *pos, char *fen, int isChess960)
{
  unsigned char col, row, token;
  Square sq = SQ_A8;

  Stack *st = pos->st;
  memset(pos, 0, offsetof(Pos, moveList));
  pos->st = st;
  memset(st, 0, StateSize);
#ifdef PEDANTIC
  for (int i = 0; i < 256; i++)
    pos->pieceList[i] = SQ_NONE;
  for (int i = 0; i < 16; i++)
    pos->pieceCount[i] = 16 * i;
#else
  for (Square s = 0; s < 64; s++)
    CastlingRightsMask[s] = ANY_CASTLING;
#endif

  // Piece placement
  while ((token = *fen++) && token != ' ') {
    if (token >= '0' && token <= '9')
      sq += token - '0'; // Advance the given number of files
    else if (token == '/')
      sq -= 16;
    else {
      for (int piece = 0; piece < 16; piece++)
        if (PieceToChar[piece] == token) {
#ifdef PEDANTIC
          put_piece(pos, color_of(piece), piece, sq++);
#else
          pos->board[sq] = piece;
          pos->byTypeBB[0] |= sq_bb(sq);
          pos->byTypeBB[type_of_p(piece)] |= sq_bb(sq);
          pos->byColorBB[color_of(piece)] |= sq_bb(sq);
          sq++;
#endif
          break;
        }
    }
  }

  // Active color
  token = *fen++;
  pos->sideToMove = token == 'w' ? WHITE : BLACK;
  token = *fen++;

  // Castling availability. Compatible with 3 standards: Normal FEN
  // standard, Shredder-FEN that uses the letters of the columns on which
  // the rooks began the game instead of KQkq and also X-FEN standard
  // that, in case of Chess960, // if an inner rook is associated with
  // the castling right, the castling tag is replaced by the file letter
  // of the involved rook, as for the Shredder-FEN.
  while ((token = *fen++) && !isspace(token)) {
    Square rsq;
    int c = islower(token) ? BLACK : WHITE;
    Piece rook = make_piece(c, ROOK);

    token = toupper(token);

    if (token == 'K')
      for (rsq = relative_square(c, SQ_H1); piece_on(rsq) != rook; --rsq);
    else if (token == 'Q')
      for (rsq = relative_square(c, SQ_A1); piece_on(rsq) != rook; ++rsq);
    else if (token >= 'A' && token <= 'H')
      rsq = make_square(token - 'A', relative_rank(c, RANK_1));
    else
      continue;

    set_castling_right(pos, c, rsq);
  }

  // En passant square. Ignore if no pawn capture is possible.
  if (   ((col = *fen++) && (col >= 'a' && col <= 'h'))
      && ((row = *fen++) && (row == '3' || row == '6')))
  {
    st->epSquare = make_square(col - 'a', row - '1');

    if (!(attackers_to(st->epSquare) & pieces_cp(pos_stm(), PAWN)))
      st->epSquare = 0;
  }
  else
    st->epSquare = 0;

  // Halfmove clock and fullmove number
  st->rule50 = strtol(fen, &fen, 10);
  pos->gamePly = strtol(fen, NULL, 10);

  // Convert from fullmove starting from 1 to ply starting from 0,
  // handle also common incorrect FEN with fullmove = 0.
  pos->gamePly = max(2 * (pos->gamePly - 1), 0) + (pos_stm() == BLACK);

  pos->chess960 = isChess960;
  set_state(pos, st);

  assert(pos_is_ok(pos, &failed_step));
}


// set_castling_right() is a helper function used to set castling rights
// given the corresponding color and the rook starting square.

static void set_castling_right(Pos *pos, Color c, Square rfrom)
{
  Square kfrom = square_of(c, KING);
  int cs = kfrom < rfrom ? KING_SIDE : QUEEN_SIDE;
  int cr = (WHITE_OO << ((cs == QUEEN_SIDE) + 2 * c));

  Square kto = relative_square(c, cs == KING_SIDE ? SQ_G1 : SQ_C1);
  Square rto = relative_square(c, cs == KING_SIDE ? SQ_F1 : SQ_D1);

  pos->st->castlingRights |= cr;

#ifdef PEDANTIC
  pos->castlingRightsMask[kfrom] |= cr;
  pos->castlingRightsMask[rfrom] |= cr;
  pos->castlingRookSquare[cr] = rfrom;

  for (Square s = min(rfrom, rto); s <= max(rfrom, rto); s++)
    if (s != kfrom && s != rfrom)
      pos->castlingPath[cr] |= sq_bb(s);

  for (Square s = min(kfrom, kto); s <= max(kfrom, kto); s++)
    if (s != kfrom && s != rfrom)
      pos->castlingPath[cr] |= sq_bb(s);
#else
  CastlingRightsMask[kfrom] &= ~cr;
  CastlingRightsMask[rfrom] &= ~cr;
//  CastlingToSquare[rfrom & 0x0f] = kto;
  Piece rook = make_piece(c, ROOK);
  CastlingHash[kto & 0x0f] = zob.psq[rook][rto] ^ zob.psq[rook][rfrom];
  CastlingPSQ[kto & 0x0f] = psqt.psq[rook][rto] - psqt.psq[rook][rfrom];
  CastlingBits[kto & 0x0f] = sq_bb(rto) ^ sq_bb(rfrom);
  // need 2nd set of from/to, maybe... for undo
  CastlingRookFrom[kto & 0x0f] = rfrom != kto ? rfrom : rto;
  CastlingRookTo[kto & 0x0f] = rto;
  CastlingRookSquare[cr] = rfrom;

  for (Square s = min(rfrom, rto); s <= max(rfrom, rto); s++)
    if (s != kfrom && s != rfrom)
      CastlingPath[cr] |= sq_bb(s);

  for (Square s = min(kfrom, kto); s <= max(kfrom, kto); s++)
    if (s != kfrom && s != rfrom)
      CastlingPath[cr] |= sq_bb(s);
#endif
}


// set_state() computes the hash keys of the position, and other data
// that once computed is updated incrementally as moves are made. The
// function is only used when a new position is set up, and to verify
// the correctness of the Stack data when running in debug mode.

static void set_state(Pos *pos, Stack *st)
{
  st->key = st->materialKey = 0;
  st->pawnKey = zob.noPawns;
  st->nonPawn = 0;
  st->psq = 0;

  st->checkersBB = attackers_to(square_of(pos_stm(), KING)) & pieces_c(pos_stm() ^ 1);

  set_check_info(pos);

  for (Bitboard b = pieces(); b; ) {
    Square s = pop_lsb(&b);
    Piece pc = piece_on(s);
    st->key ^= zob.psq[pc][s];
    st->psq += psqt.psq[pc][s];
  }

  if (st->epSquare != 0)
      st->key ^= zob.enpassant[file_of(st->epSquare)];

  if (pos_stm() == BLACK)
      st->key ^= zob.side;

  st->key ^= zob.castling[st->castlingRights];

  for (Bitboard b = pieces_p(PAWN); b; ) {
    Square s = pop_lsb(&b);
    st->pawnKey ^= zob.psq[piece_on(s)][s];
  }

  for (Color c = 0; c < 2; c++)
    for (PieceType pt = PAWN; pt <= KING; pt++)
      st->materialKey += piece_count(c, pt) * matKey[8 * c + pt];

  for (Color c = 0; c < 2; c++)
    for (PieceType pt = KNIGHT; pt <= QUEEN; pt++)
      st->nonPawn += piece_count(c, pt) * NonPawnPieceValue[make_piece(c, pt)];
}


// pos_fen() returns a FEN representation of the position. In case of
// Chess960 the Shredder-FEN notation is used. This is used for copying
// the root position to search threads.

void pos_fen(const Pos *pos, char *str)
{
  int cnt;

  for (int r = 7; r >= 0; r--) {
    for (int f = 0; f < 8; f++) {
      for (cnt = 0; f < 8 && !piece_on(8 * r + f); f++)
        cnt++;
      if (cnt) *str++ = '0' + cnt;
      if (f < 8) *str++ = PieceToChar[piece_on(8 * r + f)];
    }
    if (r > 0) *str++ = '/';
  }

  *str++ = ' ';
  *str++ = pos_stm() == WHITE ? 'w' : 'b';
  *str++ = ' ';

  int cr = pos->st->castlingRights;

  if (!is_chess960()) {
    if (cr & WHITE_OO) *str++ = 'K';
    if (cr & WHITE_OOO) *str++ = 'Q';
    if (cr & BLACK_OO) *str++ = 'k';
    if (cr & BLACK_OOO) *str++ = 'q';
  } else {
    if (cr & WHITE_OO) *str++ = 'A' + file_of(castling_rook_square(make_castling_right(WHITE, KING_SIDE)));
    if (cr & WHITE_OOO) *str++ = 'A' + file_of(castling_rook_square(make_castling_right(WHITE, QUEEN_SIDE)));
    if (cr & BLACK_OO) *str++ = 'A' + file_of(castling_rook_square(make_castling_right(BLACK, KING_SIDE)));
    if (cr & BLACK_OOO) *str++ = 'A' + file_of(castling_rook_square(make_castling_right(BLACK, QUEEN_SIDE)));
  }
  if (!cr)
      *str++ = '-';

  *str++ = ' ';
  if (ep_square() != 0) {
    *str++ = 'a' + file_of(ep_square());
    *str++ = '1' + rank_of(ep_square());
  } else {
    *str++ = '-';
  }

  sprintf(str, " %d %d", pos_rule50_count(),
          1 + (pos_game_ply() - (pos_stm() == BLACK)) / 2);
}


// Turning slider_blockers() into an inline function was slower, even
// though it should only add a single slightly optimised copy to evaluate().
#if 1
// slider_blockers() returns a bitboard of all pieces that are blocking
// attacks on the square 's' from 'sliders'. A piece blocks a slider if
// removing that piece from the board would result in a position where
// square 's' is attacked. Both pinned pieces and discovered check
// candidates are slider blockers and are calculated by calling this
// function.

Bitboard slider_blockers(const Pos *pos, Bitboard sliders, Square s,
                         Bitboard *pinners)
{
  Bitboard result = 0, snipers;
  *pinners = 0;

  // Snipers are sliders that attack square 's'when a piece removed.
  snipers = (  (PseudoAttacks[ROOK  ][s] & pieces_pp(QUEEN, ROOK))
             | (PseudoAttacks[BISHOP][s] & pieces_pp(QUEEN, BISHOP))) & sliders;
  Bitboard occupancy = pieces() & ~snipers;

  while (snipers) {
    Square sniperSq = pop_lsb(&snipers);
    Bitboard b = between_bb(s, sniperSq) & occupancy;

    if (!more_than_one(b)) {
      result |= b;
      if (b & pieces_c(color_of(piece_on(s))))
        *pinners |= sq_bb(sniperSq);
    }
  }
  return result;
}
#endif


#if 0
// attackers_to() computes a bitboard of all pieces which attack a given
// square. Slider attacks use the occupied bitboard to indicate occupancy.

Bitboard pos_attackers_to_occ(const Pos *pos, Square s, Bitboard occupied)
{
  return  (attacks_from_pawn(s, BLACK)    & pieces_cp(WHITE, PAWN))
        | (attacks_from_pawn(s, WHITE)    & pieces_cp(BLACK, PAWN))
        | (attacks_from_knight(s)         & pieces_p(KNIGHT))
        | (attacks_bb_rook(s, occupied)   & pieces_pp(ROOK,   QUEEN))
        | (attacks_bb_bishop(s, occupied) & pieces_pp(BISHOP, QUEEN))
        | (attacks_from_king(s)           & pieces_p(KING));
}
#endif


// is_legal() tests whether a pseudo-legal move is legal

int is_legal(const Pos *pos, Move m)
{
  assert(move_is_ok(m));

  uint64_t us = pos_stm();
  Square from = from_sq(m);
  Square to = to_sq(m);

  assert(color_of(moved_piece(m)) == us);
  assert(piece_on(square_of(us, KING)) == make_piece(us, KING));

  // En passant captures are a tricky special case. Because they are rather
  // uncommon, we do it simply by testing whether the king is attacked after
  // the move is made.
  if (unlikely(type_of_m(m) == ENPASSANT)) {
    Square ksq = square_of(us, KING);
    Square capsq = to ^ 8;
    Bitboard occupied = pieces() ^ sq_bb(from) ^ sq_bb(capsq) ^ sq_bb(to);

    assert(to == ep_square());
    assert(moved_piece(m) == make_piece(us, PAWN));
    assert(piece_on(capsq) == make_piece(us ^ 1, PAWN));
    assert(piece_on(to) == 0);

    return   !(attacks_bb_rook  (ksq, occupied) & pieces_cpp(us ^ 1, QUEEN, ROOK))
          && !(attacks_bb_bishop(ksq, occupied) & pieces_cpp(us ^ 1, QUEEN, BISHOP));
  }

  // Check legality of castling moves.
  if (unlikely(type_of_m(m) == CASTLING)) {
    // to > from works both for standard chess and for Chess960.
    to = relative_square(us, to > from ? SQ_G1 : SQ_C1);
    int step = to > from ? WEST : EAST;

    for (Square s = to; s != from; s += step)
      if (attackers_to(s) & pieces_c(us ^ 1))
        return false;

    // For Chess960, verify that moving the castling rook does not discover
    // some hidden checker, e.g. on SQ_A1 when castling rook is on SQ_B1.
    return   !is_chess960()
          || !(attacks_bb_rook(to, pieces() ^ sq_bb(to_sq(m)))
               & pieces_cpp(us ^ 1, ROOK, QUEEN));
  }

  // If the moving piece is a king, check whether the destination
  // square is attacked by the opponent. Castling moves are checked
  // for legality during move generation.
  if (pieces_p(KING) & sq_bb(from))
    return !(attackers_to(to) & pieces_c(us ^ 1));

  // A non-king move is legal if and only if it is not pinned or it
  // is moving along the ray towards or away from the king.
  return   !(blockers_for_king(pos, us) & sq_bb(from))
        ||  aligned(m, square_of(us, KING));
}


// is_pseudo_legal() takes a random move and tests whether the move is
// pseudo legal. It is used to validate moves from TT that can be corrupted
// due to SMP concurrent access or hash position key aliasing.

#if 0
int is_pseudo_legal_old(Pos *pos, Move m)
{
  int us = pos_stm();
  Square from = from_sq(m);
  Square to = to_sq(m);
  Piece pc = moved_piece(m);

  // Use a slower but simpler function for uncommon cases
  if (type_of_m(m) != NORMAL) {
    ExtMove list[MAX_MOVES];
    ExtMove *last = generate_legal(pos, list);
    for (ExtMove *p = list; p < last; p++)
      if (p->move == m)
        return 1;
    return 0;
  }

  // Is not a promotion, so promotion piece must be empty
  if (promotion_type(m) - KNIGHT != 0)
    return 0;

  // If the 'from' square is not occupied by a piece belonging to the side to
  // move, the move is obviously not legal.
  if (pc == 0 || color_of(pc) != us)
    return 0;

  // The destination square cannot be occupied by a friendly piece
  if (pieces_c(us) & sq_bb(to))
    return 0;

  // Handle the special case of a pawn move
  if (type_of_p(pc) == PAWN) {
    // We have already handled promotion moves, so destination
    // cannot be on the 8th/1st rank.
    if (rank_of(to) == relative_rank(us, RANK_8))
      return 0;

    if (   !(attacks_from_pawn(from, us) & pieces_c(us ^ 1) & sq_bb(to)) // Not a capture
        && !((from + pawn_push(us) == to) && is_empty(to))       // Not a single push
        && !( (from + 2 * pawn_push(us) == to)              // Not a double push
           && (rank_of(from) == relative_rank(us, RANK_2))
           && is_empty(to)
           && is_empty(to - pawn_push(us))))
      return 0;
  }
  else if (!(attacks_from(pc, from) & sq_bb(to)))
    return 0;

  // Evasions generator already takes care to avoid some kind of illegal moves
  // and legal() relies on this. We therefore have to take care that the same
  // kind of moves are filtered out here.
  if (pos_checkers()) {
    if (type_of_p(pc) != KING) {
      // Double check? In this case a king move is required
      if (more_than_one(pos_checkers()))
        return 0;

      // Our move must be a blocking evasion or a capture of the checking piece
      if (!((between_bb(lsb(pos_checkers()), square_of(us, KING)) | pos_checkers()) & sq_bb(to)))
        return 0;
    }
    // In case of king moves under check we have to remove king so as to catch
    // invalid moves like b1a1 when opposite queen is on c1.
    else if (attackers_to_occ(to, pieces() ^ sq_bb(from)) & pieces_c(us ^ 1))
      return 0;
  }

  return 1;
}
#endif

int is_pseudo_legal(const Pos *pos, Move m)
{
  uint64_t us = pos_stm();
  Square from = from_sq(m);

  if (!(pieces_c(us) & sq_bb(from)))
    return 0;

  if (unlikely(type_of_m(m) == CASTLING)) {
    if (pos_checkers()) return 0;
    ExtMove list[MAX_MOVES];
    ExtMove *end = generate_quiets(pos, list);
    for (ExtMove *p = list; p < end; p++)
      if (p->move == m) return is_legal(pos, m);
    return 0;
  }

  Square to = to_sq(m);
  if (pieces_c(us) & sq_bb(to))
    return 0;

  PieceType pt = type_of_p(piece_on(from));
  if (pt != PAWN) {
    if (type_of_m(m) != NORMAL)
      return 0;
    switch (pt) {
    case KNIGHT:
      if (!(attacks_from_knight(from) & sq_bb(to)))
        return 0;
      break;
    case BISHOP:
      if (!(attacks_from_bishop(from) & sq_bb(to)))
        return 0;
      break;
    case ROOK:
      if (!(attacks_from_rook(from) & sq_bb(to)))
        return 0;
      break;
    case QUEEN:
      if (!(attacks_from_queen(from) & sq_bb(to)))
        return 0;
      break;
    case KING:
      if (!(attacks_from_king(from) & sq_bb(to)))
        return 0;
      // is_legal() does not remove the "from" square from the "occupied"
      // bitboard when checking that the king is not in check on the "to"
      // square. So we need to be careful here.
      if (   pos_checkers()
          && (attackers_to_occ(to, pieces() ^ sq_bb(from)) & pieces_c(us ^ 1)))
        return 0;
      return 1;
    default:
      assume(0);
      break;
    }
  } else {
    if (likely(type_of_m(m) == NORMAL)) {
      if (rank_of(to) == relative_rank(us, RANK_8))
        return 0;
      if (   !(attacks_from_pawn(from, us) & pieces_c(us ^ 1) & sq_bb(to))
          && !((from + pawn_push(us) == to) && is_empty(to))
          && !( (from + 2 * pawn_push(us) == to)
            && (rank_of(from) == relative_rank(us, RANK_2))
            && is_empty(to) && is_empty(to - pawn_push(us))))
        return 0;
    }
    else if (likely(type_of_m(m) == PROMOTION)) {
      // No need to test for pawn to 8th rank.
      if (   !(attacks_from_pawn(from, us) & pieces_c(us ^ 1) & sq_bb(to))
          && !((from + pawn_push(us) == to) && is_empty(to)))
        return 0;
    }
    else
      return to == ep_square() && (attacks_from_pawn(from, us) & sq_bb(to));
  }
  if (pos_checkers()) {
    // Again we need to be a bit careful.
    if (more_than_one(pos_checkers()))
      return 0;
    if (!((between_bb(lsb(pos_checkers()), square_of(us, KING))
                                      | pos_checkers()) & sq_bb(to)))
      return 0;
  }
  return 1;
}

#if 0
int is_pseudo_legal(Pos *pos, Move m)
{
  int r1 = is_pseudo_legal_old(pos, m);
  int r2 = is_pseudo_legal_new(pos, m);
  if (r1 != r2) {
    printf("old: %d, new: %d\n", r1, r2);
    printf("old: %d\n", is_pseudo_legal_old(pos, m));
    printf("new: %d\n", is_pseudo_legal_new(pos, m));
exit(1);
  }
  return r1;
}
#endif


// gives_check_special() is invoked by gives_check() if there are
// discovered check candidates or the move is of a special type

int gives_check_special(const Pos *pos, Stack *st, Move m)
{
  assert(move_is_ok(m));
  assert(color_of(moved_piece(m)) == pos_stm());

  Square from = from_sq(m);
  Square to = to_sq(m);

  if ((blockers_for_king(pos, pos_stm() ^ 1) & sq_bb(from)) && !aligned(m, st->ksq))
    return 1;

  switch (type_of_m(m)) {
  case NORMAL:
    return !!(st->checkSquares[type_of_p(piece_on(from))] & sq_bb(to));

  case PROMOTION:
    return !!(  attacks_bb(promotion_type(m), to, pieces() ^ sq_bb(from))
              & sq_bb(st->ksq));

  case ENPASSANT:
  {
    if (st->checkSquares[PAWN] & sq_bb(to))
      return 1;
    Square capsq = make_square(file_of(to), rank_of(from));
//    Bitboard b = pieces() ^ sq_bb(from) ^ sq_bb(capsq) ^ sq_bb(to);
    Bitboard b = inv_sq(inv_sq(inv_sq(pieces(), from), to), capsq);
    return  (attacks_bb_rook  (st->ksq, b) & pieces_cpp(pos_stm(), QUEEN, ROOK))
          ||(attacks_bb_bishop(st->ksq, b) & pieces_cpp(pos_stm(), QUEEN, BISHOP));
  }
  case CASTLING:
  {
#ifdef PEDANTIC
    // Castling is encoded as 'King captures the rook'
    Square rto = relative_square(pos_stm(), to > from ? SQ_F1 : SQ_D1);
#else
    Square rto = CastlingRookTo[to & 0x0f];
#endif
    return   (PseudoAttacks[ROOK][rto] & sq_bb(st->ksq))
          && (attacks_bb_rook(rto, pieces() ^ sq_bb(from)) & sq_bb(st->ksq));
  }
  default:
    assume(0);
    return 0;
  }
}


// do_move() makes a move. The move is assumed to be legal.
#ifndef PEDANTIC
void do_move(Pos *pos, Move m, int givesCheck)
{
  assert(move_is_ok(m));

  Stack *st = ++pos->st;
  st->pawnKey = (st-1)->pawnKey;
  st->materialKey = (st-1)->materialKey;
  st->psqnpm = (st-1)->psqnpm; // psq and nonPawnMaterial

  Square from = from_sq(m);
  Square to = to_sq(m);
  Key key = (st-1)->key ^ zob.side;

  // Update castling rights
  st->castlingRights =  (st-1)->castlingRights
                      & CastlingRightsMask[from]
                      & CastlingRightsMask[to];
  key ^= zob.castling[st->castlingRights ^ (st-1)->castlingRights];

  Piece captPiece = pos->board[to];
  Color us = pos->sideToMove;

  // Clear en passant
  st->epSquare = 0;
  if (unlikely((st-1)->epSquare)) {
    key ^= zob.enpassant[(st-1)->epSquare & 7];
    if (type_of_m(m) == ENPASSANT)
      captPiece = B_PAWN ^ (us << 3);
  }

  Piece piece = piece_on(from);
  Piece prom_piece;

  // Move the piece or carry out a promotion
  if (likely(type_of_m(m) != PROMOTION)) {
    // In Chess960, the king might seem to capture the friendly rook
    if (type_of_m(m) == CASTLING)
      captPiece = 0;
    pos->byTypeBB[type_of_p(piece)] ^= sq_bb(from) ^ sq_bb(to);
    st->psq += psqt.psq[piece][to] - psqt.psq[piece][from];
    key ^= zob.psq[piece][from] ^ zob.psq[piece][to];
    if (type_of_p(piece) == PAWN)
      st->pawnKey ^= zob.psq[piece][from] ^ zob.psq[piece][to];
    prom_piece = piece;
  } else {
    prom_piece = promotion_type(m);
    pos->byTypeBB[type_of_p(piece)] ^= sq_bb(from);
    pos->byTypeBB[prom_piece] ^= sq_bb(to);
    prom_piece |= piece & 8;
    st->psq += psqt.psq[prom_piece][to] - psqt.psq[piece][to];
    st->nonPawn += NonPawnPieceValue[prom_piece];
    st->materialKey += matKey[prom_piece] - matKey[piece];
    key ^= zob.psq[piece][from] ^ zob.psq[prom_piece][to];
    st->pawnKey ^= zob.psq[piece][from];
  }
  pos->byColorBB[us] ^= sq_bb(from) ^ sq_bb(to);
  pos->board[from] = 0;
  pos->board[to] = prom_piece;

  if (captPiece) {
    st->rule50 = 0;
    if ((captPiece & 7) == PAWN) {
      if (type_of_m(m) == ENPASSANT) {
        to += (us == WHITE ? -8 : 8);
        pos->board[to] = 0;
      }
      st->pawnKey ^= zob.psq[captPiece][to];
    }
    st->capturedPiece = captPiece;
    st->psq -= psqt.psq[captPiece][to];
    st->nonPawn -= NonPawnPieceValue[captPiece];
    st->materialKey -= matKey[captPiece];
    pos->byTypeBB[captPiece & 7] ^= sq_bb(to);
    pos->byColorBB[us ^ 1] ^= sq_bb(to);
    key ^= zob.psq[captPiece][to];
  } else { // Not a capture.
    st->capturedPiece = 0;
    st->rule50 = (st-1)->rule50 + 1;
    if ((piece & 7) == PAWN) {
      st->rule50 = 0;
      if ((from ^ to) == 16) {
        if (EPMask[to - SQ_A4] & pos->byTypeBB[PAWN] & pos->byColorBB[us^1]) {
          st->epSquare = to + (us == WHITE ? -8 : 8);
          key ^= zob.enpassant[to & 7];
        }
      }
    } else if (type_of_m(m) == CASTLING) {
      key ^= CastlingHash[to & 0x0f];
      pos->byTypeBB[ROOK] ^= CastlingBits[to & 0x0f];
      pos->byColorBB[us] ^= CastlingBits[to & 0x0f];
      pos->board[CastlingRookFrom[to & 0x0f]] = 0;
      pos->board[CastlingRookTo[to & 0x0f]] = ROOK | (to & 0x08);
      st->psq += CastlingPSQ[to & 0x0f];
    }
  }
  st->key = key;
  pos->byTypeBB[0] = pos->byColorBB[0] | pos->byColorBB[1];

  st->checkersBB =  givesCheck
                  ? attackers_to(square_of(us ^ 1, KING)) & pieces_c(us) : 0;

  st->pliesFromNull = (st-1)->pliesFromNull + 1;

  pos->sideToMove ^= 1;
  pos->nodes++;

  set_check_info(pos);

  check_pos(pos);
}

void undo_move(Pos *pos, Move m)
{
  Square from = from_sq(m);
  Square to = to_sq(m);
  Piece piece = pos->board[to];
  Stack *st = pos->st--;
  pos->sideToMove ^= 1;
  Color us = pos->sideToMove;

  if (likely(type_of_m(m) != PROMOTION)) {
    pos->byTypeBB[piece & 7] ^= sq_bb(from) ^ sq_bb(to);
  } else {
    pos->byTypeBB[piece & 7] ^= sq_bb(to);
    pos->byTypeBB[PAWN] ^= sq_bb(from);
    piece = PAWN | (piece & 8);
  }
  pos->byColorBB[us] ^= sq_bb(from) ^ sq_bb(to);
  pos->board[from] = piece;

  Piece captPiece = st->capturedPiece;
  pos->board[to] = captPiece;
  if (captPiece) {
    if (type_of_m(m) == ENPASSANT) {
      pos->board[to] = 0;
      to = (st-1)->epSquare + (us == WHITE ? -8 : 8);
      pos->board[to] = captPiece;
    }
    pos->byTypeBB[captPiece & 7] ^= sq_bb(to);
    pos->byColorBB[us ^ 1] ^= sq_bb(to);
  }
  else if (type_of_m(m) == CASTLING) {
    pos->byTypeBB[ROOK] ^= CastlingBits[to & 0x0f];
    pos->byColorBB[us] ^= CastlingBits[to & 0x0f];
    pos->board[CastlingRookTo[to & 0x0f]] = 0;
    pos->board[CastlingRookFrom[to & 0x0f]] = ROOK | (to & 0x08);
  }
  pos->byTypeBB[0] = pos->byColorBB[0] | pos->byColorBB[1];

  check_pos(pos);
}
#else
void do_move(Pos *pos, Move m, int givesCheck)
{
  assert(move_is_ok(m));

  Key key = pos_key() ^ zob.side;

  // Copy some fields of the old state to our new Stack object except the
  // ones which are going to be recalculated from scratch anyway and then
  // switch our state pointer to point to the new (ready to be updated)
  // state.
  Stack *st = ++pos->st;
  memcpy(st, st - 1, (StateCopySize + 7) & ~7);

  // Increment ply counters. Note that rule50 will be reset to zero later
  // on in case of a capture or a pawn move.
  st->plyCounters += 0x101; // Increment both rule50 and pliesFromNull

  Color us = pos_stm();
  Color them = us ^ 1;
  Square from = from_sq(m);
  Square to = to_sq(m);
  Piece piece = piece_on(from);
  Piece captured = type_of_m(m) == ENPASSANT
                   ? make_piece(them, PAWN) : piece_on(to);

  assert(color_of(piece) == us);
  assert(   is_empty(to)
         || color_of(piece_on(to)) == (type_of_m(m) != CASTLING ? them : us));
  assert(type_of_p(captured) != KING);

  if (unlikely(type_of_m(m) == CASTLING)) {
    assert(piece == make_piece(us, KING));
    assert(captured == make_piece(us, ROOK));

    Square rfrom, rto;

    int kingSide = to > from;
    rfrom = to; // Castling is encoded as "king captures friendly rook"
    rto = relative_square(us, kingSide ? SQ_F1 : SQ_D1);
    to = relative_square(us, kingSide ? SQ_G1 : SQ_C1);

    // Remove both pieces first since squares could overlap in Chess960
    remove_piece(pos, us, piece, from);
    remove_piece(pos, us, captured, rfrom);
    pos->board[from] = pos->board[rfrom] = 0;
    put_piece(pos, us, piece, to);
    put_piece(pos, us, captured, rto);

    st->psq += psqt.psq[captured][rto] - psqt.psq[captured][rfrom];
    key ^= zob.psq[captured][rfrom] ^ zob.psq[captured][rto];
    captured = 0;
  }

  else if (captured) {
    Square capsq = to;

    // If the captured piece is a pawn, update pawn hash key. Otherwise,
    // update non-pawn material.
    if (type_of_p(captured) == PAWN) {
      if (unlikely(type_of_m(m) == ENPASSANT)) {
        capsq ^= 8;

        assert(piece == make_piece(us, PAWN));
        assert(to == (st-1)->epSquare);
        assert(relative_rank_s(us, to) == RANK_6);
        assert(is_empty(to));
        assert(piece_on(capsq) == make_piece(them, PAWN));

        pos->board[capsq] = 0; // Not done by remove_piece()
      }

      st->pawnKey ^= zob.psq[captured][capsq];
    } else
      st->nonPawn -= NonPawnPieceValue[captured];

    // Update board and piece lists
    remove_piece(pos, them, captured, capsq);

    // Update material hash key and prefetch access to materialTable
    key ^= zob.psq[captured][capsq];
    st->materialKey -= matKey[captured];
    prefetch(&pos->materialTable[st->materialKey >> (64 - 13)]);

    // Update incremental scores
    st->psq -= psqt.psq[captured][capsq];

    // Reset ply counters
    st->plyCounters = 0;
  }

  // Set captured piece
  st->capturedPiece = captured;

  // Update hash key
  key ^= zob.psq[piece][from] ^ zob.psq[piece][to];

  // Reset en passant square
  if (unlikely((st-1)->epSquare != 0))
    key ^= zob.enpassant[file_of((st-1)->epSquare)];
  st->epSquare = 0;

  // Update castling rights if needed
  if (    st->castlingRights
      && (pos->castlingRightsMask[from] | pos->castlingRightsMask[to]))
  {
    uint32_t cr = pos->castlingRightsMask[from] | pos->castlingRightsMask[to];
    key ^= zob.castling[st->castlingRights & cr];
    st->castlingRights &= ~cr;
  }

  // Move the piece. The tricky Chess960 castling is handled earlier.
  if (likely(type_of_m(m) != CASTLING))
    move_piece(pos, us, piece, from, to);

  // If the moving piece is a pawn do some special extra work
  if (type_of_p(piece) == PAWN) {
    // Set en-passant square if the moved pawn can be captured
    if (   (to ^ from) == 16
        && (attacks_from_pawn(to ^ 8, us) & pieces_cp(them, PAWN)))
    {
      st->epSquare = to ^ 8;
      key ^= zob.enpassant[file_of(st->epSquare)];
    } else if (type_of_m(m) == PROMOTION) {
      PieceType promotion = promotion_type(m);

      assert(relative_rank_s(us, to) == RANK_8);
      assert(promotion >= KNIGHT && promotion <= QUEEN);

      promotion = make_piece(us, promotion);

      remove_piece(pos, us, piece, to);
      put_piece(pos, us, promotion, to);

      // Update hash keys
      key ^= zob.psq[piece][to] ^ zob.psq[promotion][to];
      st->pawnKey ^= zob.psq[piece][to];
      st->materialKey += matKey[promotion] - matKey[piece];

      // Update incremental score
      st->psq += psqt.psq[promotion][to] - psqt.psq[piece][to];

      // Update material
      st->nonPawn += NonPawnPieceValue[promotion];
    }

    // Update pawn hash key and prefetch access to pawnsTable
    st->pawnKey ^= zob.psq[piece][from] ^ zob.psq[piece][to];
    prefetch2(&pos->pawnTable[st->pawnKey & (PAWN_ENTRIES -1)]);

    // Reset ply counters.
    st->plyCounters = 0;
  }

  // Update incremental scores
  st->psq += psqt.psq[piece][to] - psqt.psq[piece][from];

  // Update the key with the final value
  st->key = key;

  // Calculate checkers bitboard (if move gives check)
#if 1
  st->checkersBB =  givesCheck
                  ? attackers_to(square_of(them, KING)) & pieces_c(us) : 0;
#else
  st->checkersBB = 0;
  if (givesCheck) {
    if (type_of_m(m) != NORMAL || ((st-1)->blockersForKing[them] & sq_bb(from)))
      st->checkersBB = attackers_to(square_of(them, KING)) & pieces_c(us);
    else
      st->checkersBB = (st-1)->checkSquares[piece & 7] & sq_bb(to);
  }
#endif

  pos->sideToMove ^= 1;
  pos->nodes++;

  set_check_info(pos);

  assert(pos_is_ok(pos, &failed_step));
}


// undo_move() unmakes a move. When it returns, the position should
// be restored to exactly the same state as before the move was made.

void undo_move(Pos *pos, Move m)
{
  assert(move_is_ok(m));

  pos->sideToMove ^= 1;

  Color us = pos_stm();
  Square from = from_sq(m);
  Square to = to_sq(m);
  Piece piece = piece_on(to);

  assert(is_empty(from) || type_of_m(m) == CASTLING);
  assert(type_of_p(pos->st->capturedPiece) != KING);

  if (unlikely(type_of_m(m) == PROMOTION)) {
    assert(relative_rank_s(us, to) == RANK_8);
    assert(type_of_p(piece) == promotion_type(m));
    assert(type_of_p(piece) >= KNIGHT && type_of_p(piece) <= QUEEN);

    remove_piece(pos, us, piece, to);
    piece = make_piece(us, PAWN);
    put_piece(pos, us, piece, to);
  }

  if (unlikely(type_of_m(m) == CASTLING)) {
    Square rfrom, rto;
    int kingSide = to > from;
    rfrom = to; // Castling is encoded as "king captures friendly rook"
    rto = relative_square(us, kingSide ? SQ_F1 : SQ_D1);
    to = relative_square(us, kingSide ? SQ_G1 : SQ_C1);

    // Remove both pieces first since squares could overlap in Chess960
    Piece king = make_piece(us, KING);
    Piece rook = make_piece(us, ROOK);
    remove_piece(pos, us, king, to);
    remove_piece(pos, us, rook, rto);
    pos->board[to] = pos->board[rto] = 0;
    put_piece(pos, us, king, from);
    put_piece(pos, us, rook, rfrom);
  } else {
    move_piece(pos, us, piece, to, from); // Put the piece back at the source square

    if (pos->st->capturedPiece) {
      Square capsq = to;

      if (unlikely(type_of_m(m) == ENPASSANT)) {
        capsq ^= 8;

        assert(type_of_p(piece) == PAWN);
        assert(to == (pos->st-1)->epSquare);
        assert(relative_rank_s(us, to) == RANK_6);
        assert(is_empty(capsq));
        assert(pos->st->capturedPiece == make_piece(us ^ 1, PAWN));
      }

      put_piece(pos, us ^ 1, pos->st->capturedPiece, capsq); // Restore the captured piece
    }
  }

  // Finally, point our state pointer back to the previous state
  pos->st--;

  assert(pos_is_ok(pos, &failed_step));
}
#endif


// do_null_move() is used to do a null move

void do_null_move(Pos *pos)
{
  assert(!pos_checkers());

  Stack *st = ++pos->st;
  memcpy(st, st - 1, (StateSize + 7) & ~7);

  if (unlikely(st->epSquare)) {
    st->key ^= zob.enpassant[file_of(st->epSquare)];
    st->epSquare = 0;
  }

  st->key ^= zob.side;
  prefetch(tt_first_entry(st->key));

  st->rule50++;
  st->pliesFromNull = 0;

  pos->sideToMove ^= 1;

  set_check_info(pos);

  assert(pos_is_ok(pos, &failed_step));
}

// See position.h for undo_null_move()


// key_after() computes the new hash key after the given move. Needed
// for speculative prefetch. It does not recognize special moves like
// castling, en-passant and promotions.

Key key_after(const Pos *pos, Move m)
{
  Square from = from_sq(m);
  Square to = to_sq(m);
  Piece pc = piece_on(from);
  Piece captured = piece_on(to);
  Key k = pos_key() ^ zob.side;

  if (captured)
    k ^= zob.psq[captured][to];

  return k ^ zob.psq[pc][to] ^ zob.psq[pc][from];
}


// Test whether SEE >= value.
int see_test(const Pos *pos, Move m, int value)
{
  if (unlikely(type_of_m(m) != NORMAL))
    return 0 >= value;

  Square from = from_sq(m), to = to_sq(m);
  Bitboard occ;

  int swap = PieceValue[MG][piece_on(to)] - value;
  if (swap < 0)
    return 0;

  swap = PieceValue[MG][piece_on(from)] - swap;
  if (swap <= 0)
    return 1;

  occ = pieces() ^ sq_bb(from) ^ sq_bb(to);
  Color stm = color_of(piece_on(from));
  Bitboard attackers = attackers_to_occ(to, occ), stmAttackers;
  int res = 1;

  while (1) {
    stm ^= 1;
    attackers &= occ;
    if (!(stmAttackers = attackers & pieces_c(stm))) break;
    if (    (stmAttackers & blockers_for_king(pos, stm))
        && (pos->st->pinnersForKing[stm] & occ))
      stmAttackers &= ~blockers_for_king(pos, stm);
    if (!stmAttackers) break;
    res ^= 1;
    Bitboard bb;
    if ((bb = stmAttackers & pieces_p(PAWN))) {
      if ((swap = PawnValueMg - swap) < res) break;
      occ ^= bb & -bb;
      attackers |= attacks_bb_bishop(to, occ) & pieces_pp(BISHOP, QUEEN);
    }
    else if ((bb = stmAttackers & pieces_p(KNIGHT))) {
      if ((swap = KnightValueMg - swap) < res) break;
      occ ^= bb & -bb;
    }
    else if ((bb = stmAttackers & pieces_p(BISHOP))) {
      if ((swap = BishopValueMg - swap) < res) break;
      occ ^= bb & -bb;
      attackers |= attacks_bb_bishop(to, occ) & pieces_pp(BISHOP, QUEEN);
    }
    else if ((bb = stmAttackers & pieces_p(ROOK))) {
      if ((swap = RookValueMg - swap) < res) break;
      occ ^= bb & -bb;
      attackers |= attacks_bb_rook(to, occ) & pieces_pp(ROOK, QUEEN);
    }
    else if ((bb = stmAttackers & pieces_p(QUEEN))) {
      if ((swap = QueenValueMg - swap) < res) break;
      occ ^= bb & -bb;
      attackers |=  (attacks_bb_bishop(to, occ) & pieces_pp(BISHOP, QUEEN))
                  | (attacks_bb_rook(to, occ) & pieces_pp(ROOK, QUEEN));
    }
    else // KING
      return (attackers & ~pieces_c(stm)) ? res ^ 1 : res;
  }

  return res;
}


// is_draw() tests whether the position is drawn by 50-move rule or by
// repetition. It does not detect stalemates.

__attribute__((optimize("Os")))
int is_draw(const Pos *pos)
{
  Stack *st = pos->st;

  if (unlikely(st->rule50 > 99)) {
    if (!pos_checkers())
      return 1;
    return generate_legal(pos, (st-1)->endMoves) != (st-1)->endMoves;
  }

  // st->pliesFromNull is reset both on null moves and on zeroing moves.
  int e = st->pliesFromNull - 4;
  if (e >= 0) {
    Stack *stp = st - 2;
    for (int i = 0; i <= e; i += 2) {
      stp -= 2;
      if (stp->key == st->key)
        return 1; // Draw at first repetition
    }
  }

  return 0;
}


// has_game_cycle() tests if the position has a move which draws by
// repetition or an earlier position has a move that directly reaches
// the current position.

bool has_game_cycle(const Pos *pos) {
  unsigned int j;

  int end = pos->st->pliesFromNull;

  Key originalKey = pos->st->key;
  Stack *stp = pos->st - 1;

  for (int i = 3; i <= end; i += 2) {
    stp -= 2;

    Key moveKey = originalKey ^ stp->key;
    if (   (j = H1(moveKey), cuckoo[j] == moveKey)
        || (j = H2(moveKey), cuckoo[j] == moveKey))
    {
      if (!(((Bitboard *)BetweenBB)[cuckooMove[j]] & pieces()))
        return true;
    }
  }
  return false;
}


void pos_set_check_info(Pos *pos)
{
  set_check_info(pos);
}

// pos_is_ok() performs some consistency checks for the position object.
// This is meant to be helpful when debugging.

#ifdef PEDANTIC
#ifndef NDEBUG
static int pos_is_ok(Pos *pos, int *failedStep)
{
  int Fast = 1; // Quick (default) or full check?

  enum { Default, King, Bitboards, StackOK, Lists, Castling };

  for (int step = Default; step <= (Fast ? Default : Castling); step++) {
    if (failedStep)
      *failedStep = step;

    if (step == Default)
      if (   (pos_stm() != WHITE && pos_stm() != BLACK)
          || piece_on(square_of(WHITE, KING)) != W_KING
          || piece_on(square_of(BLACK, KING)) != B_KING
          || ( ep_square() && relative_rank_s(pos_stm(), ep_square()) != RANK_6))
        return 0;

#if 0
    if (step == King)
      if (   std::count(board, board + SQUARE_NB, W_KING) != 1
          || std::count(board, board + SQUARE_NB, B_KING) != 1
          || attackers_to(square_of(pos_stm() ^ 1, KING)) & pieces_c(pos_stm()))
        return 0;
#endif

    if (step == Bitboards) {
      if (  (pieces_c(WHITE) & pieces_c(BLACK))
          ||(pieces_c(WHITE) | pieces_c(BLACK)) != pieces())
        return 0;

      for (int p1 = PAWN; p1 <= KING; p1++)
        for (int p2 = PAWN; p2 <= KING; p2++)
          if (p1 != p2 && (pieces_p(p1) & pieces_p(p2)))
            return 0;
    }

    if (step == StackOK) {
      Stack si = *(pos->st);
      set_state(pos, &si);
      if (memcmp(&si, pos->st, StateSize))
        return 0;
    }

    if (step == Lists)
      for (int c = 0; c < 2; c++)
        for (int pt = PAWN; pt <= KING; pt++) {
          if (piece_count(c, pt) != popcount(pieces_cp(c, pt)))
            return 0;

          for (int i = 0; i < piece_count(c, pt); i++)
            if (   piece_on(piece_list(c, pt)[i]) != make_piece(c, pt)
                || pos->index[piece_list(c, pt)[i]] != i)
              return 0;
        }

    if (step == Castling)
      for (int c = 0; c < 2; c++)
        for (int s = 0; s < 2; s++) {
          int cr = make_castling_right(c, s);
          if (!can_castle_cr(cr))
            continue;

          if (   piece_on(pos->castlingRookSquare[cr]) != make_piece(c, ROOK)
              || pos->castlingRightsMask[pos->castlingRookSquare[cr]] != cr
              || (pos->castlingRightsMask[square_of(c, KING)] & cr) != cr)
            return 0;
        }
  }

  return 1;
}
#endif
#else
static int pos_is_ok(Pos *pos, int *failedStep)
{
(void)pos;
(void)failedStep;
  return 1;
}

#ifndef NDEBUG
static int check_pos(Pos *pos)
{
  Bitboard colorBB[2];
  Bitboard pieceBB[8];

  colorBB[0] = colorBB[1] = 0;
  for (int i = 0; i < 8; i++)
    pieceBB[i] = 0;

  for (int sq = 0; sq < 64; sq++)
    if (pos->board[sq]) {
      colorBB[pos->board[sq] >> 3] |= sq_bb(sq);
      pieceBB[pos->board[sq] & 7] |= sq_bb(sq);
    }

  for (int i = PAWN; i <= KING; i++)
    assert(pos->byTypeBB[i] == pieceBB[i]);

  assert(pos->byColorBB[0] == colorBB[0]);
  assert(pos->byColorBB[1] == colorBB[1]);
  assert(pos->byTypeBB[0] == (colorBB[0] | colorBB[1]));

  Key key = 0, pawnKey = 0, matKey = 0;

  for (int c = 0; c < 2; c++)
    for (int i = PAWN; i <= KING; i++)
       matKey += matKey[8 * c + i] * piece_count(c, i);

  for (int sq = 0; sq < 64; sq++)
    if (pos->board[sq])
      key ^= zob.psq[pos->board[sq]][sq];
  if (pos->sideToMove == BLACK)
    key ^= zob.side;
  if (pos->st->epSquare)
    key ^= zob.enpassant[pos->st->epSquare & 7];
  key ^= zob.castling[pos->st->castlingRights];

  for (int sq = 0; sq < 64; sq++)
    if ((pos->board[sq] & 7) == PAWN)
      pawnKey ^= zob.psq[pos->board[sq]][sq];

  int npm_w = 0, npm_b = 0;
  for (int i = KNIGHT; i <= KING; i++) {
    npm_w += piece_count(WHITE, i) * PieceValue[MG][i];
    npm_b += piece_count(BLACK, i) * PieceValue[MG][i];
  }
  assert(npm_w == pos_non_pawn_material(WHITE));
  assert(npm_b == pos_non_pawn_material(BLACK));

  assert(key == pos->st->key);
  assert(pawnKey == pos->st->pawnKey);
  assert(matKey == pos->st->materialKey);

  return 1;
}
#endif
#endif
