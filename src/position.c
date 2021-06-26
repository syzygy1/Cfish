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

static void set_castling_right(Position *pos, Color c, Square rfrom);
static void set_state(Position *pos, Stack *st);

#ifndef NDEBUG
static int pos_is_ok(Position *pos, int *failedStep);
static int check_pos(Position *pos);
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

INLINE void put_piece(Position *pos, Color c, Piece piece, Square s)
{
  pos->board[s] = piece;
  pos->byTypeBB[0] |= sq_bb(s);
  pos->byTypeBB[type_of_p(piece)] |= sq_bb(s);
  pos->byColorBB[c] |= sq_bb(s);
}

INLINE void remove_piece(Position *pos, Color c, Piece piece, Square s)
{
  pos->byTypeBB[0] ^= sq_bb(s);
  pos->byTypeBB[type_of_p(piece)] ^= sq_bb(s);
  pos->byColorBB[c] ^= sq_bb(s);
  /* board[s] = 0;  Not needed, overwritten by the capturing one */
}

INLINE void move_piece(Position *pos, Color c, Piece piece, Square from,
    Square to)
{
  Bitboard fromToBB = sq_bb(from) ^ sq_bb(to);
  pos->byTypeBB[0] ^= fromToBB;
  pos->byTypeBB[type_of_p(piece)] ^= fromToBB;
  pos->byColorBB[c] ^= fromToBB;
  pos->board[from] = 0;
  pos->board[to] = piece;
}


// Calculate CheckInfo data.

INLINE void set_check_info(Position *pos)
{
  Stack *st = pos->st;

  st->blockersForKing[WHITE] = slider_blockers(pos, pieces_c(BLACK), square_of(WHITE, KING), &st->pinnersForKing[WHITE]);
  st->blockersForKing[BLACK] = slider_blockers(pos, pieces_c(WHITE), square_of(BLACK, KING), &st->pinnersForKing[BLACK]);

  Color them = !stm();
  st->ksq = square_of(them, KING);

  st->checkSquares[PAWN]   = attacks_from_pawn(st->ksq, them);
  st->checkSquares[KNIGHT] = attacks_from_knight(st->ksq);
  st->checkSquares[BISHOP] = attacks_from_bishop(st->ksq);
  st->checkSquares[ROOK]   = attacks_from_rook(st->ksq);
  st->checkSquares[QUEEN]  = st->checkSquares[BISHOP] | st->checkSquares[ROOK];
  st->checkSquares[KING]   = 0;
}


// print_pos() prints an ASCII representation of the position to stdout.

void print_pos(Position *pos)
{
  char fen[128];
  pos_fen(pos, fen);

  flockfile(stdout);
  printf("\n +---+---+---+---+---+---+---+---+\n");

  for (int r = 7; r >= 0; r--) {
    for (int f = 0; f <= 7; f++)
      printf(" | %c", PieceToChar[pos->board[8 * r + f]]);

    printf(" | %d\n +---+---+---+---+---+---+---+---+\n", r + 1);
  }

  printf("   a   b   c   d   e   f   g   h\n\nFen: %s\nKey: %16"PRIX64"\nCheckers: ", fen, key());

  char buf[16];
  for (Bitboard b = checkers(); b; )
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

  for (int cr = 0; cr < 16; cr++)
    zob.castling[cr] = prng_rand(&rng);

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
//            Move move = between_bb(s1, s2) ? make_move(s1, s2)
//                                           : make_move(SQ_C3, SQ_D5);
            Move move = make_move(s1, s2);
            Key key = zob.psq[pc][s1] ^ zob.psq[pc][s2] ^ zob.side;
            uint32_t i = H1(key);
            while (true) {
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

void pos_set(Position *pos, char *fen, int isChess960)
{
  unsigned char col, row, token;
  Square sq = SQ_A8;

  Stack *st = pos->st;
  memset(pos, 0, offsetof(Position, moveList));
  pos->st = st;
  memset(st, 0, StateSize);
  for (int i = 0; i < 16; i++)
    pos->pieceCount[i] = 0;

  // Piece placement
  while ((token = *fen++) && token != ' ') {
    if (token >= '0' && token <= '9')
      sq += token - '0'; // Advance the given number of files
    else if (token == '/')
      sq -= 16;
    else {
      for (int piece = 0; piece < 16; piece++)
        if (PieceToChar[piece] == token) {
          put_piece(pos, color_of(piece), piece, sq++);
          pos->pieceCount[piece]++;
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
      && ((row = *fen++) && (row == (stm() == WHITE ? '6' : '3'))))
  {
    st->epSquare = make_square(col - 'a', row - '1');

    // We assume a legal FEN, i.e. if epSquare is present, then the previous
    // move was a legal double pawn push.
    if (!(attackers_to(st->epSquare) & pieces_cp(stm(), PAWN)))
      st->epSquare = 0;
  }
  else
    st->epSquare = 0;

  // Halfmove clock and fullmove number
  st->rule50 = strtol(fen, &fen, 10);
  pos->gamePly = strtol(fen, NULL, 10);

  // Convert from fullmove starting from 1 to ply starting from 0,
  // handle also common incorrect FEN with fullmove = 0.
  pos->gamePly = max(2 * (pos->gamePly - 1), 0) + (stm() == BLACK);

  pos->chess960 = isChess960;
  set_state(pos, st);

  assert(pos_is_ok(pos, &failed_step));
}


// set_castling_right() is a helper function used to set castling rights
// given the corresponding color and the rook starting square.

static void set_castling_right(Position *pos, Color c, Square rfrom)
{
  Square kfrom = square_of(c, KING);
  int cs = kfrom < rfrom ? KING_SIDE : QUEEN_SIDE;
  int cr = (WHITE_OO << ((cs == QUEEN_SIDE) + 2 * c));

  Square kto = relative_square(c, cs == KING_SIDE ? SQ_G1 : SQ_C1);
  Square rto = relative_square(c, cs == KING_SIDE ? SQ_F1 : SQ_D1);

  pos->st->castlingRights |= cr;

  pos->castlingRightsMask[kfrom] |= cr;
  pos->castlingRightsMask[rfrom] |= cr;
  pos->castlingRookSquare[cr] = rfrom;

  for (Square s = min(rfrom, rto); s <= max(rfrom, rto); s++)
    if (s != kfrom && s != rfrom)
      pos->castlingPath[cr] |= sq_bb(s);

  for (Square s = min(kfrom, kto); s <= max(kfrom, kto); s++)
    if (s != kfrom && s != rfrom)
      pos->castlingPath[cr] |= sq_bb(s);
}


// set_state() computes the hash keys of the position, and other data
// that once computed is updated incrementally as moves are made. The
// function is only used when a new position is set up, and to verify
// the correctness of the Stack data when running in debug mode.

static void set_state(Position *pos, Stack *st)
{
  st->key = st->materialKey = 0;
#ifndef NNUE_PURE
  st->pawnKey = zob.noPawns;
  st->psq = 0;
#endif
  st->nonPawn = 0;

  st->checkersBB = attackers_to(square_of(stm(), KING)) & pieces_c(!stm());

  set_check_info(pos);

  for (Bitboard b = pieces(); b; ) {
    Square s = pop_lsb(&b);
    Piece pc = piece_on(s);
    st->key ^= zob.psq[pc][s];
#ifndef NNUE_PURE
    st->psq += psqt.psq[pc][s];
#endif
  }

// emulate a bug in Stockfish
//  if (st->epSquare != 0)
    st->key ^= zob.enpassant[file_of(st->epSquare)];

  if (stm() == BLACK)
    st->key ^= zob.side;

  st->key ^= zob.castling[st->castlingRights];

#ifndef NNUE_PURE
  for (Bitboard b = pieces_p(PAWN); b; ) {
    Square s = pop_lsb(&b);
    st->pawnKey ^= zob.psq[piece_on(s)][s];
  }
#endif

  for (PieceType pt = PAWN; pt <= KING; pt++) {
    st->materialKey += piece_count(WHITE, pt) * matKey[8 * WHITE + pt];
    st->materialKey += piece_count(BLACK, pt) * matKey[8 * BLACK + pt];
  }

  for (PieceType pt = KNIGHT; pt <= QUEEN; pt++)
    for (int c = 0; c < 2; c++)
      st->nonPawn += piece_count(c, pt) * NonPawnPieceValue[make_piece(c, pt)];
}


// pos_fen() returns a FEN representation of the position. In case of
// Chess960 the Shredder-FEN notation is used. This is used for copying
// the root position to search threads.

void pos_fen(const Position *pos, char *str)
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
  *str++ = stm() == WHITE ? 'w' : 'b';
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
    if (cr & BLACK_OO) *str++ = 'a' + file_of(castling_rook_square(make_castling_right(BLACK, KING_SIDE)));
    if (cr & BLACK_OOO) *str++ = 'a' + file_of(castling_rook_square(make_castling_right(BLACK, QUEEN_SIDE)));
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

  sprintf(str, " %d %d", rule50_count(), 1 + (game_ply()-(stm() == BLACK)) / 2);
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

Bitboard slider_blockers(const Position *pos, Bitboard sliders, Square s,
    Bitboard *pinners)
{
  Bitboard blockers = 0, snipers;
  *pinners = 0;

  // Snipers are sliders that attack square 's' when a piece and other
  // snipers are removed.
  snipers = (  (PseudoAttacks[ROOK  ][s] & pieces_pp(QUEEN, ROOK))
             | (PseudoAttacks[BISHOP][s] & pieces_pp(QUEEN, BISHOP))) & sliders;
  Bitboard occupancy = pieces() ^ snipers;

  while (snipers) {
    Square sniperSq = pop_lsb(&snipers);
    Bitboard b = between_bb(s, sniperSq) & occupancy;

    if (b && !more_than_one(b)) {
      blockers |= b;
      if (b & pieces_c(color_of(piece_on(s))))
        *pinners |= sq_bb(sniperSq);
    }
  }
  return blockers;
}
#endif


#if 0
// attackers_to() computes a bitboard of all pieces which attack a given
// square. Slider attacks use the occupied bitboard to indicate occupancy.

Bitboard attackers_to_occ(const Position *pos, Square s, Bitboard occupied)
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

bool is_legal(const Position *pos, Move m)
{
  assert(move_is_ok(m));

  Color us = stm();
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
    assert(piece_on(capsq) == make_piece(!us, PAWN));
    assert(piece_on(to) == 0);

    return   !(attacks_bb_rook  (ksq, occupied) & pieces_cpp(!us, QUEEN, ROOK))
          && !(attacks_bb_bishop(ksq, occupied) & pieces_cpp(!us, QUEEN, BISHOP));
  }

  // Check legality of castling moves.
  if (unlikely(type_of_m(m) == CASTLING)) {
    // to > from works both for standard chess and for Chess960.
    to = relative_square(us, to > from ? SQ_G1 : SQ_C1);
    int step = to > from ? WEST : EAST;

    for (Square s = to; s != from; s += step)
      if (attackers_to(s) & pieces_c(!us))
        return false;

    // For Chess960, verify that moving the castling rook does not discover
    // some hidden checker, e.g. on SQ_A1 when castling rook is on SQ_B1.
    return !is_chess960() || !(blockers_for_king(pos, us) & sq_bb(to_sq(m)));
  }

  // If the moving piece is a king, check whether the destination
  // square is attacked by the opponent. Castling moves are checked
  // for legality during move generation.
  if (pieces_p(KING) & sq_bb(from))
    return !(attackers_to_occ(pos, to, pieces() ^ sq_bb(from)) & pieces_c(!us));

  // A non-king move is legal if and only if it is not pinned or it
  // is moving along the ray towards or away from the king.
  return   !(blockers_for_king(pos, us) & sq_bb(from))
        ||  aligned(m, square_of(us, KING));
}


// is_pseudo_legal() takes a random move and tests whether the move is
// pseudo legal. It is used to validate moves from TT that can be corrupted
// due to SMP concurrent access or hash position key aliasing.

#if 0
bool is_pseudo_legal_old(Position *pos, Move m)
{
  Color us = stm();
  Square from = from_sq(m);
  Square to = to_sq(m);
  Piece pc = moved_piece(m);

  // Use a slower but simpler function for uncommon cases
  if (type_of_m(m) != NORMAL) {
    ExtMove list[MAX_MOVES];
    ExtMove *last = generate_legal(pos, list);
    for (ExtMove *p = list; p < last; p++)
      if (p->move == m)
        return true;
    return false;
  }

  // Is not a promotion, so promotion piece must be empty
  if (promotion_type(m) - KNIGHT != 0)
    return false;

  // If the 'from' square is not occupied by a piece belonging to the side to
  // move, the move is obviously not legal.
  if (pc == 0 || color_of(pc) != us)
    return false;

  // The destination square cannot be occupied by a friendly piece
  if (pieces_c(us) & sq_bb(to))
    return false;

  // Handle the special case of a pawn move
  if (type_of_p(pc) == PAWN) {
    // We have already handled promotion moves, so destination
    // cannot be on the 8th/1st rank.
    if (!((to + 0x08) & 0x30))
      return false;

    if (   !(attacks_from_pawn(from, us) & pieces_c(!us) & sq_bb(to)) // Not a capture
        && !((from + pawn_push(us) == to) && is_empty(to))       // Not a single push
        && !( (from + 2 * pawn_push(us) == to)              // Not a double push
           && (rank_of(from) == relative_rank(us, RANK_2))
           && is_empty(to)
           && is_empty(to - pawn_push(us))))
      return false;
  }
  else if (!(attacks_from(pc, from) & sq_bb(to)))
    return false;

  // Evasions generator already takes care to avoid some kind of illegal moves
  // and legal() relies on this. We therefore have to take care that the same
  // kind of moves are filtered out here.
  if (checkers()) {
    if (type_of_p(pc) != KING) {
      // Double check? In this case a king move is required
      if (more_than_one(checkers()))
        return false;

      // Our move must be a blocking evasion or a capture of the checking piece
      if (!((between_bb(lsb(checkers()), square_of(us, KING)) | checkers()) & sq_bb(to)))
        return false;
    }
    // In case of king moves under check we have to remove king so as to catch
    // invalid moves like b1a1 when opposite queen is on c1.
    else if (attackers_to_occ(pos, to, pieces() ^ sq_bb(from)) & pieces_c(!us))
      return false;
  }

  return true;
}
#endif

bool is_pseudo_legal(const Position *pos, Move m)
{
  Color us = stm();
  Square from = from_sq(m);

  if (!(pieces_c(us) & sq_bb(from)))
    return false;

  if (unlikely(type_of_m(m) == CASTLING)) {
    if (checkers()) return false;
    ExtMove list[MAX_MOVES];
    ExtMove *end = generate_quiets(pos, list);
    for (ExtMove *p = list; p < end; p++)
      if (p->move == m) return true;
    return false;
  }

  Square to = to_sq(m);
  if (pieces_c(us) & sq_bb(to))
    return false;

  PieceType pt = type_of_p(piece_on(from));
  if (pt != PAWN) {
    if (type_of_m(m) != NORMAL)
      return false;
    switch (pt) {
    case KNIGHT:
      if (!(attacks_from_knight(from) & sq_bb(to)))
        return false;
      break;
    case BISHOP:
      if (!(attacks_from_bishop(from) & sq_bb(to)))
        return false;
      break;
    case ROOK:
      if (!(attacks_from_rook(from) & sq_bb(to)))
        return false;
      break;
    case QUEEN:
      if (!(attacks_from_queen(from) & sq_bb(to)))
        return false;
      break;
    case KING:
      if (!(attacks_from_king(from) & sq_bb(to)))
        return false;
      // is_legal() does not remove the "from" square from the "occupied"
      // bitboard when checking that the king is not in check on the "to"
      // square. So we need to be careful here.
      if (   checkers()
          && (attackers_to_occ(pos, to, pieces() ^ sq_bb(from)) & pieces_c(!us)))
        return false;
      return true;
    default:
      assume(false);
      break;
    }
  } else {
    if (likely(type_of_m(m) == NORMAL)) {
      if (!((to + 0x08) & 0x30))
        return false;
      if (   !(attacks_from_pawn(from, us) & pieces_c(!us) & sq_bb(to))
          && !((from + pawn_push(us) == to) && is_empty(to))
          && !(   from + 2 * pawn_push(us) == to
               && rank_of(from) == relative_rank(us, RANK_2)
               && is_empty(to) && is_empty(to - pawn_push(us))))
        return false;
    }
    else if (likely(type_of_m(m) == PROMOTION)) {
      // No need to test for pawn to 8th rank.
      if (   !(attacks_from_pawn(from, us) & pieces_c(!us) & sq_bb(to))
          && !((from + pawn_push(us) == to) && is_empty(to)))
        return false;
    }
    else
      return to == ep_square() && (attacks_from_pawn(from, us) & sq_bb(to));
  }
  if (checkers()) {
    // Again we need to be a bit careful.
    if (more_than_one(checkers()))
      return false;
    if (!(between_bb(square_of(us, KING), lsb(checkers())) & sq_bb(to)))
      return false;
  }
  return true;
}

#if 0
int is_pseudo_legal(Position *pos, Move m)
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

bool gives_check_special(const Position *pos, Stack *st, Move m)
{
  assert(move_is_ok(m));
  assert(color_of(moved_piece(m)) == stm());

  Square from = from_sq(m);
  Square to = to_sq(m);

  if ((blockers_for_king(pos, !stm()) & sq_bb(from)) && !aligned(m, st->ksq))
    return true;

  switch (type_of_m(m)) {
  case NORMAL:
    return st->checkSquares[type_of_p(piece_on(from))] & sq_bb(to);

  case PROMOTION:
    return attacks_bb(promotion_type(m), to, pieces() ^ sq_bb(from)) & sq_bb(st->ksq);

  case ENPASSANT:
  {
    if (st->checkSquares[PAWN] & sq_bb(to))
      return true;
    Square capsq = make_square(file_of(to), rank_of(from));
//    Bitboard b = pieces() ^ sq_bb(from) ^ sq_bb(capsq) ^ sq_bb(to);
    Bitboard b = inv_sq(inv_sq(inv_sq(pieces(), from), to), capsq);
    return  (attacks_bb_rook  (st->ksq, b) & pieces_cpp(stm(), QUEEN, ROOK))
          ||(attacks_bb_bishop(st->ksq, b) & pieces_cpp(stm(), QUEEN, BISHOP));
  }
  case CASTLING:
  {
    // Castling is encoded as 'King captures the rook'
    Square rto = relative_square(stm(), to > from ? SQ_F1 : SQ_D1);
    return   (PseudoAttacks[ROOK][rto] & sq_bb(st->ksq))
          && (attacks_bb_rook(rto, pieces() ^ sq_bb(from)) & sq_bb(st->ksq));
  }
  default:
    assume(false);
    return false;
  }
}


// do_move() makes a move. The move is assumed to be legal.

void do_move(Position *pos, Move m, int givesCheck)
{
  assert(move_is_ok(m));

  Key key = pos->st->key ^ zob.side;

  // Copy some fields of the old state to our new Stack object except the
  // ones which are going to be recalculated from scratch anyway and then
  // switch our state pointer to point to the new (ready to be updated)
  // state.
  Stack *st = ++pos->st;
  memcpy(st, st - 1, (StateCopySize + 7) & ~7);

  // Increment ply counters. Note that rule50 will be reset to zero later
  // on in case of a capture or a pawn move.
  st->plyCounters += 0x101; // Increment both rule50 and pliesFromNull

#ifdef NNUE
  st->accumulator.state[WHITE] = ACC_EMPTY;
  st->accumulator.state[BLACK] = ACC_EMPTY;
  DirtyPiece *dp = &(st->dirtyPiece);
  dp->dirtyNum = 1;
#endif

  Color us = stm();
  Color them = !us;
  Square from = from_sq(m);
  Square to = to_sq(m);
  Piece piece = piece_on(from);
  Piece captured =  type_of_m(m) == ENPASSANT
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

#ifdef NNUE
    dp->dirtyNum = 2;
    dp->pc[1] = captured;
    dp->from[1] = rfrom;
    dp->to[1] = rto;
#endif

    // Remove both pieces first since squares could overlap in Chess960
    remove_piece(pos, us, piece, from);
    remove_piece(pos, us, captured, rfrom);
    pos->board[from] = pos->board[rfrom] = 0;
    put_piece(pos, us, piece, to);
    put_piece(pos, us, captured, rto);

#ifndef NNUE_PURE
    st->psq += psqt.psq[captured][rto] - psqt.psq[captured][rfrom];
#endif
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

#ifndef NNUE_PURE
      st->pawnKey ^= zob.psq[captured][capsq];
#endif
    } else
      st->nonPawn -= NonPawnPieceValue[captured];

#ifdef NNUE
    dp->dirtyNum = 2; // captured piece goes off the board
    dp->pc[1] = captured;
    dp->from[1] = capsq;
    dp->to[1] = SQ_NONE;
#endif

    // Update board
    remove_piece(pos, them, captured, capsq);
    pos->pieceCount[captured]--;

    // Update material hash key and prefetch access to materialTable
    key ^= zob.psq[captured][capsq];
    st->materialKey -= matKey[captured];
#ifndef NNUE_PURE
    prefetch(&pos->materialTable[st->materialKey >> (64 - 13)]);

    // Update incremental scores
    st->psq -= psqt.psq[captured][capsq];
#endif

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
//    uint32_t cr = pos->castlingRightsMask[from] | pos->castlingRightsMask[to];
//    key ^= zob.castling[st->castlingRights & cr];
//    st->castlingRights &= ~cr;
    key ^= zob.castling[st->castlingRights];
    st->castlingRights &= ~(pos->castlingRightsMask[from] | pos->castlingRightsMask[to]);
    key ^= zob.castling[st->castlingRights];
  }

#ifdef NNUE
    dp->pc[0] = piece;
    dp->from[0] = from;
    dp->to[0] = to;
#endif

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
    }
    else if (type_of_m(m) == PROMOTION) {
      Piece promotion = make_piece(us, promotion_type(m));

      assert(relative_rank_s(us, to) == RANK_8);
      assert(type_of_p(promotion) >= KNIGHT && type_of_p(promotion) <= QUEEN);

      remove_piece(pos, us, piece, to);
      pos->pieceCount[piece]--;
      put_piece(pos, us, promotion, to);
      pos->pieceCount[promotion]++;

#ifdef NNUE
      dp->to[0] = SQ_NONE;   // pawn to SQ_NONE, promoted piece from SQ_NONE
      dp->pc[dp->dirtyNum] = promotion;
      dp->from[dp->dirtyNum] = SQ_NONE;
      dp->to[dp->dirtyNum] = to;
      dp->dirtyNum++;
#endif

      // Update hash keys
      key ^= zob.psq[piece][to] ^ zob.psq[promotion][to];
#ifndef NNUE_PURE
      st->pawnKey ^= zob.psq[piece][to];
#endif
      st->materialKey += matKey[promotion] - matKey[piece];

#ifndef NNUE_PURE
      // Update incremental score
      st->psq += psqt.psq[promotion][to] - psqt.psq[piece][to];
#endif

      // Update material
      st->nonPawn += NonPawnPieceValue[promotion];
    }

#ifndef NNUE_PURE
    // Update pawn hash key and prefetch access to pawnsTable
    st->pawnKey ^= zob.psq[piece][from] ^ zob.psq[piece][to];
    prefetch2(&pos->pawnTable[st->pawnKey & (PAWN_ENTRIES -1)]);
#endif

    // Reset ply counters.
    st->plyCounters = 0;
  }

#ifndef NNUE_PURE
  // Update incremental scores
  st->psq += psqt.psq[piece][to] - psqt.psq[piece][from];
#endif

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

  pos->sideToMove = !pos->sideToMove;
  pos->nodes++;

  set_check_info(pos);

  assert(pos_is_ok(pos, &failed_step));
}


// undo_move() unmakes a move. When it returns, the position should
// be restored to exactly the same state as before the move was made.

void undo_move(Position *pos, Move m)
{
  assert(move_is_ok(m));

  pos->sideToMove = !pos->sideToMove;

  Color us = stm();
  Square from = from_sq(m);
  Square to = to_sq(m);
  Piece pc = piece_on(to);

  assert(is_empty(from) || type_of_m(m) == CASTLING);
  assert(type_of_p(pos->st->capturedPiece) != KING);

  if (unlikely(type_of_m(m) == PROMOTION)) {
    assert(relative_rank_s(us, to) == RANK_8);
    assert(type_of_p(pc) == promotion_type(m));
    assert(type_of_p(pc) >= KNIGHT && type_of_p(pc) <= QUEEN);

    remove_piece(pos, us, pc, to);
    pos->pieceCount[pc]--;
    pc = make_piece(us, PAWN);
    put_piece(pos, us, pc, to);
    pos->pieceCount[pc]++;
  }

  if (unlikely(type_of_m(m) == CASTLING)) {
    Square rfrom, rto;
    int kingSide = to > from;
    rfrom = to; // Castling is encoded as "king captures friendly rook"
    rto = relative_square(us, kingSide ? SQ_F1 : SQ_D1);
    to = relative_square(us, kingSide ? SQ_G1 : SQ_C1);
    Piece king = make_piece(us, KING);
    Piece rook = make_piece(us, ROOK);

    // Remove both pieces first since squares could overlap in Chess960
    remove_piece(pos, us, king, to);
    remove_piece(pos, us, rook, rto);
    pos->board[to] = pos->board[rto] = 0;
    put_piece(pos, us, king, from);
    put_piece(pos, us, rook, rfrom);
  } else {
    move_piece(pos, us, pc, to, from); // Put the piece back at the source square

    if (pos->st->capturedPiece) {
      Square capsq = to;

      if (unlikely(type_of_m(m) == ENPASSANT)) {
        capsq ^= 8;

        assert(type_of_p(pc) == PAWN);
        assert(to == (pos->st-1)->epSquare);
        assert(relative_rank_s(us, to) == RANK_6);
        assert(is_empty(capsq));
        assert(pos->st->capturedPiece == make_piece(!us, PAWN));
      }

      put_piece(pos, !us, pos->st->capturedPiece, capsq); // Restore the captured piece
      pos->pieceCount[pos->st->capturedPiece]++;
    }
  }

  // Finally, point our state pointer back to the previous state
  pos->st--;

  assert(pos_is_ok(pos, &failed_step));
}


// do_null_move() is used to do a null move

void do_null_move(Position *pos)
{
  assert(!checkers());

  Stack *st = ++pos->st;
  memcpy(st, st - 1, (StateSize + 7) & ~7);
#ifdef NNUE
  st->accumulator.state[WHITE] = ACC_EMPTY;
  st->accumulator.state[BLACK] = ACC_EMPTY;
  st->dirtyPiece.dirtyNum = 0;
  st->dirtyPiece.pc[0] = 0;
#endif

  if (unlikely(st->epSquare)) {
    st->key ^= zob.enpassant[file_of(st->epSquare)];
    st->epSquare = 0;
  }

  st->key ^= zob.side;
  prefetch(tt_first_entry(st->key));

  st->rule50++;
  st->pliesFromNull = 0;

  pos->sideToMove = !pos->sideToMove;

  set_check_info(pos);

  assert(pos_is_ok(pos, &failed_step));
}

// See position.h for undo_null_move()


// key_after() computes the new hash key after the given move. Needed
// for speculative prefetch. It does not recognize special moves like
// castling, en-passant and promotions.

Key key_after(const Position *pos, Move m)
{
  Square from = from_sq(m);
  Square to = to_sq(m);
  Piece pc = piece_on(from);
  Piece captured = piece_on(to);
  Key k = pos->st->key ^ zob.side;

  if (captured)
    k ^= zob.psq[captured][to];

  return k ^ zob.psq[pc][to] ^ zob.psq[pc][from];
}


// Test whether SEE >= value.
bool see_test(const Position *pos, Move m, int value)
{
  if (unlikely(type_of_m(m) != NORMAL))
    return 0 >= value;

  Square from = from_sq(m), to = to_sq(m);
  Bitboard occ;

  int swap = PieceValue[MG][piece_on(to)] - value;
  if (swap < 0)
    return false;

  swap = PieceValue[MG][piece_on(from)] - swap;
  if (swap <= 0)
    return true;

  occ = pieces() ^ sq_bb(from) ^ sq_bb(to);
  Color stm = color_of(piece_on(from));
  Bitboard attackers = attackers_to_occ(pos, to, occ), stmAttackers;
  bool res = true;

  while (true) {
    stm = !stm;
    attackers &= occ;
    if (!(stmAttackers = attackers & pieces_c(stm))) break;
    if (    (stmAttackers & blockers_for_king(pos, stm))
        && (pos->st->pinnersForKing[stm] & occ))
      stmAttackers &= ~blockers_for_king(pos, stm);
    if (!stmAttackers) break;
    res = !res;
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
      return (attackers & ~pieces_c(stm)) ? !res : res;
  }

  return res;
}


// is_draw() tests whether the position is drawn by 50-move rule or by
// repetition. It does not detect stalemates.

SMALL
bool is_draw(const Position *pos)
{
  Stack *st = pos->st;

  if (unlikely(st->rule50 > 99)) {
    if (!checkers())
      return true;
    return generate_legal(pos, (st-1)->endMoves) != (st-1)->endMoves;
  }

  // st->pliesFromNull is reset both on null moves and on zeroing moves.
  int e = st->pliesFromNull - 4;
  if (e >= 0) {
    Stack *stp = st - 2;
    for (int i = 0; i <= e; i += 2) {
      stp -= 2;
      if (stp->key == st->key)
        return true; // Draw at first repetition
    }
  }

  return false;
}


// has_game_cycle() tests if the position has a move which draws by
// repetition or an earlier position has a move that directly reaches
// the current position.

bool has_game_cycle(const Position *pos, int ply)
{
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
      Move m = cuckooMove[j];
      if (!((((Bitboard *)BetweenBB)[m] ^ sq_bb(to_sq(m))) & pieces())) {
        if (   ply > i
            || color_of(piece_on(is_empty(from_sq(m)) ? to_sq(m) : from_sq(m))) == stm())
          return true;
      }
    }
  }
  return false;
}


void pos_set_check_info(Position *pos)
{
  set_check_info(pos);
}

// pos_is_ok() performs some consistency checks for the position object.
// This is meant to be helpful when debugging.

#ifndef NDEBUG
static int pos_is_ok(Position *pos, int *failedStep)
{
  int Fast = 1; // Quick (default) or full check?

  enum { Default, King, Bitboards, StackOK, Lists, Castling };

  for (int step = Default; step <= (Fast ? Default : Castling); step++) {
    if (failedStep)
      *failedStep = step;

    if (step == Default)
      if (   (stm() != WHITE && stm() != BLACK)
          || piece_on(square_of(WHITE, KING)) != W_KING
          || piece_on(square_of(BLACK, KING)) != B_KING
          || ( ep_square() && relative_rank_s(stm(), ep_square()) != RANK_6))
        return 0;

#if 0
    if (step == King)
      if (   std::count(board, board + SQUARE_NB, W_KING) != 1
          || std::count(board, board + SQUARE_NB, B_KING) != 1
          || attackers_to(square_of(!stm(), KING)) & pieces_c(stm()))
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
        for (int pt = PAWN; pt <= KING; pt++)
          if (piece_count(c, pt) != popcount(pieces_cp(c, pt)))
            return 0;

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
