#include <assert.h>
#include <string.h>
#include <ctype.h>

#include "bitboard.h"
#include "misc.h"
#include "movegen.h"
#include "position.h"
#include "thread.h"
#include "tt.h"
#include "uci.h"
#include "tbprobe.h"

static void set_castling_right(Pos *pos, int c, Square rfrom);
static void set_state(Pos *pos, Stack *st);

#ifndef NDEBUG
int check_pos(Pos *pos);
#else
#define check_pos(p) do {} while (0)
#endif

#define min(a,b) ((a) < (b) ? (a) : (b))
#define max(a,b) ((a) > (b) ? (a) : (b))

struct Zob zob;

#ifndef PEDANTIC
Key mat_key[16] = {
    0ULL,
    0x5ced000000000101ULL,
    0xe173000000001001ULL,
    0xd64d000000010001ULL,
    0xab88000000100001ULL,
    0x680b000001000001ULL,
    0x0000000000000001ULL,
    0ULL,
    0ULL,
    0xf209000010000001ULL,
    0xbb14000100000001ULL,
    0x58df001000000001ULL,
    0xa15f010000000001ULL,
    0x7c94100000000001ULL,
    0x0000000000000001ULL,
    0ULL
};
#endif

const char *PieceToChar = " PNBRQK  pnbrqk";

int failed_step;

#ifdef PEDANTIC
static inline void put_piece(Pos *pos, int c, int pt, Square s)
{
  pos->board[s] = make_piece(c, pt);
  pos->byTypeBB[0] |= sq_bb(s);
  pos->byTypeBB[pt] |= sq_bb(s);
  pos->byColorBB[c] |= sq_bb(s);
  pos->index[s] = pos->pieceCount[c][pt]++;
  pos->pieceList[c][pt][pos->index[s]] = s;
}

static inline void remove_piece(Pos *pos, int c, int pt, Square s)
{
  // WARNING: This is not a reversible operation.
  pos->byTypeBB[0] ^= sq_bb(s);
  pos->byTypeBB[pt] ^= sq_bb(s);
  pos->byColorBB[c] ^= sq_bb(s);
  /* board[s] = 0;  Not needed, overwritten by the capturing one */
  Square lastSquare = pos->pieceList[c][pt][--pos->pieceCount[c][pt]];
  pos->index[lastSquare] = pos->index[s];
  pos->pieceList[c][pt][pos->index[lastSquare]] = lastSquare;
  pos->pieceList[c][pt][pos->pieceCount[c][pt]] = SQ_NONE;
}

static inline void move_piece(Pos *pos, int c, int pt, Square from, Square to)
{
  // index[from] is not updated and becomes stale. This works as long as
  // index[] is accessed just by known occupied squares.
  Bitboard from_to_bb = sq_bb(from) ^ sq_bb(to);
  pos->byTypeBB[0] ^= from_to_bb;
  pos->byTypeBB[pt] ^= from_to_bb;
  pos->byColorBB[c] ^= from_to_bb;
  pos->board[from] = 0;
  pos->board[to] = make_piece(c, pt);
  pos->index[to] = pos->index[from];
  pos->pieceList[c][pt][pos->index[to]] = to;
}
#endif


// CheckInfo initialisation.

void checkinfo_init(CheckInfo *ci, Pos *pos)
{
  int them = pos_stm() ^ 1;
  ci->ksq = square_of(them, KING);

  ci->pinned = pinned_pieces(pos, pos_stm());
  ci->dcCandidates = discovered_check_candidates(pos);

  ci->checkSquares[PAWN]   = attacks_from_pawn(ci->ksq, them);
  ci->checkSquares[KNIGHT] = attacks_from_knight(ci->ksq);
  ci->checkSquares[BISHOP] = attacks_from_bishop(ci->ksq);
  ci->checkSquares[ROOK]   = attacks_from_rook(ci->ksq);
  ci->checkSquares[QUEEN]  = ci->checkSquares[BISHOP] | ci->checkSquares[ROOK];
  ci->checkSquares[KING]   = 0;
}


// print_pos() prints an ASCII representation of the position to stdout.

void print_pos(Pos *pos)
{
  char fen[128];
  pos_fen(pos, fen);

  printf("\n +---+---+---+---+---+---+---+---+\n");

  for (int r = 7; r >= 0; r--) {
    for (int f = 0; f <= 7; f++)
      printf(" | %c", PieceToChar[pos->board[8 * r + f]]);

    printf(" |\n +---+---+---+---+---+---+---+---+\n");
  }

  printf("\nFen: %s\nKey: %16llX\nCheckers: ",
         fen, (unsigned long long)pos_key());

  char buf[8]; 
  for (Bitboard b = pos_checkers(); b; )
    printf("%s ", uci_square(buf, pop_lsb(&b)));

  if (popcount(pieces()) <= TB_MaxCardinality && !can_castle_cr(ANY_CASTLING)) {
    int s1, s2;
    int wdl = TB_probe_wdl(pos, &s1);
    int dtz = TB_probe_dtz(pos, &s2);
    printf("\nTablebases WDL: %4d (%d)\nTablebases DTZ: %4d (%d)",
           wdl, s1, dtz, s2);
  }
  printf("\n");
}


// zob_init() initializes at startup the various arrays used to compute
// hash keys.

void zob_init(void) {

  PRNG rng;
  prng_init(&rng, 1070372);

  for (int c = 0; c < 2; c++)
    for (int pt = PAWN; pt <= KING; pt++)
      for (Square s = 0; s < 64; s++)
        zob.psq[c][pt][s] = prng_rand(&rng);

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
  zob.exclusion = prng_rand(&rng);
}


// pos_set() initializes the position object with the given FEN string.
// This function is not very robust - make sure that input FENs are correct,
// this is assumed to be the responsibility of the GUI.

void pos_set(Pos *pos, char *fen, int isChess960)
{
  unsigned char col, row, token;
  Square sq = SQ_A8;

  memset(pos, 0, offsetof(Pos, st));
  Stack *st = pos->st = pos->stack;
  memset(st, 0, StateSize);
#ifdef PEDANTIC
  for (int c = 0; c < 2; c++)
    for (int pt = 0; pt < 8; pt++)
      for (int i = 0; i < 16; i++)
        pos->pieceList[c][pt][i] = SQ_NONE;
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
      for (int pt = 0; pt < 16; pt++)
        if (PieceToChar[pt] == token) {
#ifdef PEDANTIC
          put_piece(pos, color_of(pt), type_of_p(pt), sq++);
#else
          pos->board[sq] = pt;
          pos->byTypeBB[0] |= sq_bb(sq);
          pos->byTypeBB[pt & 7] |= sq_bb(sq);
          pos->byColorBB[pt >> 3] |= sq_bb(sq);
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
      && ((row = *fen++) && (row == '3' || row == '6'))) {

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

static void set_castling_right(Pos *pos, int c, Square rfrom)
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
  CastlingHash[kto & 0x0f] = zob.psq[c][ROOK][rto] ^ zob.psq[c][ROOK][rfrom];
  CastlingPSQ[kto & 0x0f] = psqt.psq[c][ROOK][rto] - psqt.psq[c][ROOK][rfrom];
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
  st->key = st->pawnKey = st->materialKey = 0;
  st->nonPawnMaterial[WHITE] = st->nonPawnMaterial[BLACK] = 0;
  st->psq = 0;

  st->checkersBB = attackers_to(square_of(pos_stm(), KING)) & pieces_c(pos_stm() ^ 1);

  for (Bitboard b = pieces(); b; ) {
    Square s = pop_lsb(&b);
    Piece pc = piece_on(s);
    st->key ^= zob.psq[color_of(pc)][type_of_p(pc)][s];
    st->psq += psqt.psq[color_of(pc)][type_of_p(pc)][s];
  }

  if (st->epSquare != 0)
      st->key ^= zob.enpassant[file_of(st->epSquare)];

  if (pos_stm() == BLACK)
      st->key ^= zob.side;

  st->key ^= zob.castling[st->castlingRights];

  for (Bitboard b = pieces_p(PAWN); b; ) {
    Square s = pop_lsb(&b);
    st->pawnKey ^= zob.psq[color_of(piece_on(s))][PAWN][s];
  }

#ifdef PEDANTIC
  for (int c = 0; c < 2; c++)
    for (int pt = PAWN; pt <= KING; pt++)
      for (int cnt = 0; cnt < piece_count(c, pt); cnt++)
        st->materialKey ^= zob.psq[c][pt][cnt];
#else
  for (int c = 0; c < 2; c++)
    for (int pt = PAWN; pt <= KING; pt++)
      st->materialKey += piece_count(c, pt) * mat_key[8 * c + pt];
#endif

  for (int c = 0; c < 2; c++)
    for (int pt = KNIGHT; pt <= QUEEN; pt++)
       st->nonPawnMaterial[c] += piece_count(c, pt) * PieceValue[MG][pt];
}


// pos_fen() returns a FEN representation of the position. In case of
// Chess960 the Shredder-FEN notation is used. This is used for copying
// the root position to search threads.

void pos_fen(Pos *pos, char *str)
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
//  int ch960 = is_chess960();

// FIXME: Chess960
  if (cr & WHITE_OO) *str++ = 'K';
  if (cr & WHITE_OOO) *str++ = 'Q';
  if (cr & BLACK_OO) *str++ = 'k';
  if (cr & BLACK_OOO) *str++ = 'q';
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


// game_phase() calculates the game phase interpolating total non-pawn
// material between endgame and midgame limits.

int game_phase(Pos *pos)
{
  Value npm = pos_non_pawn_material(WHITE) + pos_non_pawn_material(BLACK);

  if (npm > MidgameLimit)
      npm = MidgameLimit;

  if (npm < EndgameLimit)
      npm = EndgameLimit;

  return ((npm - EndgameLimit) * PHASE_MIDGAME) / (MidgameLimit - EndgameLimit);
}


// slider_blockers() returns a bitboard of all the pieces in 'target' that
// are blocking attacks on the square 's' from 'sliders'. A piece blocks a
// slider if removing that piece from the board would result in a position
// where square 's' is attacked. For example, a king attack blocking piece
// can be either a pinned or a discovered check piece, according if its
// color is the opposite or the same of the color of the slider.

Bitboard slider_blockers(Pos *pos, Bitboard target, Bitboard sliders, Square s)
{
  Bitboard b, pinners, result = 0;

  // Pinners are sliders that attack 's' when a pinned piece is removed
  pinners = (  (PseudoAttacks[ROOK  ][s] & pieces_pp(QUEEN, ROOK))
             | (PseudoAttacks[BISHOP][s] & pieces_pp(QUEEN, BISHOP))) & sliders;

  while (pinners) {
    b = between_bb(s, pop_lsb(&pinners)) & pieces();

    if (!more_than_one(b))
      result |= b & target;
  }
  return result;
}


// attackers_to() computes a bitboard of all pieces which attack a given
// square. Slider attacks use the occupied bitboard to indicate occupancy.

Bitboard pos_attackers_to_occ(Pos *pos, Square s, Bitboard occupied)
{
  return  (attacks_from_pawn(s, BLACK)    & pieces_cp(WHITE, PAWN))
        | (attacks_from_pawn(s, WHITE)    & pieces_cp(BLACK, PAWN))
        | (attacks_from_knight(s)         & pieces_p(KNIGHT))
        | (attacks_bb_rook(s, occupied)   & pieces_pp(ROOK,   QUEEN))
        | (attacks_bb_bishop(s, occupied) & pieces_pp(BISHOP, QUEEN))
        | (attacks_from_king(s)           & pieces_p(KING));
}


// is_legal() tests whether a pseudo-legal move is legal

int is_legal(Pos *pos, Move m, Bitboard pinned)
{
  assert(move_is_ok(m));
  assert(pinned == pinned_pieces(pos, pos_stm()));

  int us = pos_stm();
  Square from = from_sq(m);

  assert(color_of(moved_piece(m)) == us);
  assert(piece_on(square_of(us, KING)) == make_piece(us, KING));

  // En passant captures are a tricky special case. Because they are rather
  // uncommon, we do it simply by testing whether the king is attacked after
  // the move is made.
  if (type_of_m(m) == ENPASSANT) {
    Square ksq = square_of(us, KING);
    Square to = to_sq(m);
    Square capsq = to - pawn_push(us);
    Bitboard occupied = (pieces() ^ sq_bb(from) ^ sq_bb(capsq)) | sq_bb(to);

    assert(to == ep_square());
    assert(moved_piece(m) == make_piece(us, PAWN));
    assert(piece_on(capsq) == make_piece(us ^ 1, PAWN));
    assert(piece_on(to) == 0);

    return   !(attacks_bb_rook  (ksq, occupied) & pieces_cpp(us ^ 1, QUEEN, ROOK))
          && !(attacks_bb_bishop(ksq, occupied) & pieces_cpp(us ^ 1, QUEEN, BISHOP));
  }

  // If the moving piece is a king, check whether the destination
  // square is attacked by the opponent. Castling moves are checked
  // for legality during move generation.
  if (type_of_p(piece_on(from)) == KING)
    return   type_of_m(m) == CASTLING
          || !(attackers_to(to_sq(m)) & pieces_c(us ^ 1));

  // A non-king move is legal if and only if it is not pinned or it
  // is moving along the ray towards or away from the king.
  return   !(pinned & sq_bb(from))
        ||  aligned(from, to_sq(m), square_of(us, KING));
}


// is_pseudo_legal() takes a random move and tests whether the move is
// pseudo legal. It is used to validate moves from TT that can be corrupted
// due to SMP concurrent access or hash position key aliasing.

int is_pseudo_legal(Pos *pos, Move m)
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
  if (type_of_p(pc) == PAWN)
  {
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


// gives_check() tests whether a pseudo-legal move gives a check.

int gives_check(Pos *pos, Move m, const CheckInfo *ci)
{
  assert(move_is_ok(m));
  assert(ci->dcCandidates == discovered_check_candidates(pos));
  assert(color_of(moved_piece(m)) == pos_stm());

  Square from = from_sq(m);
  Square to = to_sq(m);

  // Is there a direct check?
  if (ci->checkSquares[type_of_p(piece_on(from))] & sq_bb(to))
    return 1;

  // Is there a discovered check?
  if ((ci->dcCandidates & sq_bb(from)) && !aligned(from, to, ci->ksq))
    return 1;

  switch (type_of_m(m)) {
  case NORMAL:
    return 0;

  case PROMOTION:
    return (  attacks_bb(promotion_type(m), to, pieces() ^ sq_bb(from))
            & sq_bb(ci->ksq)) != 0;

  // En passant capture with check? We have already handled the case
  // of direct checks and ordinary discovered check, so the only case we
  // need to handle is the unusual case of a discovered check through
  // the captured pawn.
  case ENPASSANT:
  {
    Square capsq = make_square(file_of(to), rank_of(from));
    Bitboard b = (pieces() ^ sq_bb(from) ^ sq_bb(capsq)) | sq_bb(to);

    return  (attacks_bb_rook  (ci->ksq, b) & pieces_cpp(pos_stm(), QUEEN, ROOK))
          || (attacks_bb_bishop(ci->ksq, b) & pieces_cpp(pos_stm(), QUEEN, BISHOP));
  }
  case CASTLING:
  {
    Square kfrom = from;
    Square rfrom = to; // Castling is encoded as 'King captures the rook'
    Square kto = relative_square(pos_stm(), rfrom > kfrom ? SQ_G1 : SQ_C1);
    Square rto = relative_square(pos_stm(), rfrom > kfrom ? SQ_F1 : SQ_D1);

    return   (PseudoAttacks[ROOK][rto] & sq_bb(ci->ksq))
          && (attacks_bb_rook(rto, (pieces() ^ sq_bb(kfrom) ^ sq_bb(rfrom))
                                    | sq_bb(rto) | sq_bb(kto))
                  & sq_bb(ci->ksq));
  }
  default:
    assert(0);
    return 0;
  }
}


// do_move() makes a move. The move is assumed to be legal.
#ifndef PEDANTIC
void do_move(Pos *pos, Move m, int givesCheck)
{
  assert(move_is_ok(m));

  Stack *st = ++pos->st;
  st->previous = st - 1;
  st->nonPawnMaterial[0] = (st-1)->nonPawnMaterial[0];
  st->nonPawnMaterial[1] = (st-1)->nonPawnMaterial[1];
  st->psq = (st-1)->psq;
  st->materialKey = (st-1)->materialKey;

  Square from = from_sq(m);
  Square to = to_sq(m);
  Key key = (st-1)->key ^ zob.side;
  Key pawnKey = (st-1)->pawnKey;

  // Update castling rights.
  st->castlingRights =  (st-1)->castlingRights
                      & CastlingRightsMask[from]
                      & CastlingRightsMask[to];
  key ^= zob.castling[st->castlingRights ^ (st-1)->castlingRights];

  int capt_piece = pos->board[to];
  int us = pos->sideToMove;

  // Clear en passant.
  st->epSquare = 0;
  if ((st-1)->epSquare) {
    key ^= zob.enpassant[(st-1)->epSquare & 7];
    if (type_of_m(m) == ENPASSANT)
      capt_piece = B_PAWN ^ (us << 3);
  }

  int piece = piece_on(from);
  int prom_piece;

  // Move the piece or carry out a promotion.
  if (type_of_m(m) != PROMOTION) {
    // In Chess960, the king might seem to capture the friendly rook.
    if (type_of_m(m) == CASTLING)
      capt_piece = 0;
    pos->byTypeBB[piece & 7] ^= sq_bb(from) ^ sq_bb(to);
    st->psq += psqt.psq2d[piece][to] - psqt.psq2d[piece][from];
    key ^= zob.psq2d[piece][from] ^ zob.psq2d[piece][to];
    if (type_of_p(piece) == PAWN)
      pawnKey ^= zob.psq2d[piece][from] ^ zob.psq2d[piece][to];
    prom_piece = piece;
  } else {
    prom_piece = promotion_type(m);
    pos->byTypeBB[piece & 7] ^= sq_bb(from);
    pos->byTypeBB[prom_piece] ^= sq_bb(to);
    prom_piece |= piece & 8;
    st->psq += psqt.psq2d[prom_piece][to] - psqt.psq2d[piece][to];
    st->nonPawnMaterial[us] += NonPawnPieceValue[prom_piece];
    st->materialKey += mat_key[prom_piece] - mat_key[piece];
    key ^= zob.psq2d[piece][from] ^ zob.psq2d[prom_piece][to];
    pawnKey ^= zob.psq2d[piece][from];
  }
  pos->byColorBB[us] ^= sq_bb(from) ^ sq_bb(to);
  pos->board[from] = 0;
  pos->board[to] = prom_piece;

  if (capt_piece) {
    st->rule50 = 0;
    if (type_of_m(m) == ENPASSANT) {
      to += (us == WHITE ? -8 : 8);
      pos->board[to] = 0;
    }
    st->capturedType = capt_piece;
    st->psq -= psqt.psq2d[capt_piece][to];
    st->nonPawnMaterial[us ^ 1] -= NonPawnPieceValue[capt_piece];
    st->materialKey -= mat_key[capt_piece];
    pos->byTypeBB[capt_piece & 7] ^= sq_bb(to);
    pos->byColorBB[us ^ 1] ^= sq_bb(to);
    key ^= zob.psq2d[capt_piece][to];
    if ((st->capturedType & 7) == PAWN)
      pawnKey ^= zob.psq2d[capt_piece][to];
  } else { // Not a capture.
    st->capturedType = 0;
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
  st->pawnKey = pawnKey;
  pos->byTypeBB[0] = pos->byColorBB[0] | pos->byColorBB[1];

  st->checkersBB =  givesCheck
                  ? attackers_to(square_of(us ^ 1, KING)) & pieces_c(us) : 0;

  st->pliesFromNull = (st-1)->pliesFromNull + 1;

  pos->sideToMove ^= 1;
  pos->nodes++;

  check_pos(pos);
}

void undo_move(Pos *pos, Move m)
{
  int from = from_sq(m);
  int to = to_sq(m);
  int piece = pos->board[to];
  Stack *st = pos->st--;
  pos->sideToMove ^= 1;
  int us = pos->sideToMove;
 
  if (type_of_m(m) != PROMOTION) {
    pos->byTypeBB[piece & 7] ^= sq_bb(from) ^ sq_bb(to);
  } else {
    pos->byTypeBB[piece & 7] ^= sq_bb(to);
    pos->byTypeBB[PAWN] ^= sq_bb(from);
    piece = PAWN | (piece & 8);
  }
  pos->byColorBB[us] ^= sq_bb(from) ^ sq_bb(to);
  pos->board[from] = piece;

  int capt_piece = st->capturedType;
  pos->board[to] = capt_piece;
  if (capt_piece) {
    if (type_of_m(m) == ENPASSANT) {
      pos->board[to] = 0;
      to = (st-1)->epSquare + (us == WHITE ? -8 : 8);
      pos->board[to] = capt_piece;
    }
    pos->byTypeBB[capt_piece & 7] ^= sq_bb(to);
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
//  memcpy(st, st - 1, StateCopySize);
  st->previous = st - 1;

  st->pawnKey = (st-1)->pawnKey;
  // The following fields should probably be moved to Pos.
  st->materialKey = (st-1)->materialKey;
  st->nonPawnMaterial[0] = (st-1)->nonPawnMaterial[0];
  st->nonPawnMaterial[1] = (st-1)->nonPawnMaterial[1];
  st->psq = (st-1)->psq;

// castlingRights
  st->castlingRights = (st-1)->castlingRights;

  // Increment ply counters. In particular, rule50 will be reset to zero
  // later on in case of a capture or a pawn move.
  st->rule50 = (st-1)->rule50 + 1;

  int us = pos_stm();
  int them = us ^ 1;
  Square from = from_sq(m);
  Square to = to_sq(m);
  int pt = type_of_p(piece_on(from));
  int captured = type_of_m(m) == ENPASSANT ? PAWN : type_of_p(piece_on(to));

  assert(color_of(piece_on(from)) == us);
  assert(   is_empty(to)
         || color_of(piece_on(to)) == (type_of_m(m) != CASTLING ? them : us));
  assert(captured != KING);

  if (type_of_m(m) == CASTLING) {
    assert(pt == KING);

    Square rfrom, rto;

    int kingSide = to > from;
    rfrom = to; // Castling is encoded as "king captures friendly rook"
    rto = relative_square(us, kingSide ? SQ_F1 : SQ_D1);
    to = relative_square(us, kingSide ? SQ_G1 : SQ_C1);

    // Remove both pieces first since squares could overlap in Chess960
    remove_piece(pos, us, KING, from);
    remove_piece(pos, us, ROOK, rfrom);
    pos->board[from] = pos->board[rfrom] = 0;
    put_piece(pos, us, KING, to);
    put_piece(pos, us, ROOK, rto);

    captured = 0;
    st->psq += psqt.psq[us][ROOK][rto] - psqt.psq[us][ROOK][rfrom];
    key ^= zob.psq[us][ROOK][rfrom] ^ zob.psq[us][ROOK][rto];
  }

  if (captured) {
    Square capsq = to;

    // If the captured piece is a pawn, update pawn hash key, otherwise
    // update non-pawn material.
    if (captured == PAWN) {
      if (type_of_m(m) == ENPASSANT) {
        capsq -= pawn_push(us);

        assert(pt == PAWN);
        assert(to == st->epSquare);
        assert(relative_rank_s(us, to) == RANK_6);
        assert(is_empty(to));
        assert(piece_on(capsq) == make_piece(them, PAWN));

        pos->board[capsq] = 0; // Not done by remove_piece()
      }

      st->pawnKey ^= zob.psq[them][PAWN][capsq];
    } else
      st->nonPawnMaterial[them] -= PieceValue[MG][captured];

    // Update board and piece lists
    remove_piece(pos, them, captured, capsq);

    // Update material hash key and prefetch access to materialTable
    key ^= zob.psq[them][captured][capsq];
    st->materialKey ^= zob.psq[them][captured][piece_count(them, captured)];
    prefetch(&pos->materialTable[st->materialKey & 8191]);

    // Update incremental scores
    st->psq -= psqt.psq[them][captured][capsq];

    // Reset rule 50 counter
    st->rule50 = 0;
  }

  // Update hash key
  key ^= zob.psq[us][pt][from] ^ zob.psq[us][pt][to];

  // Reset en passant square
  if ((st-1)->epSquare != 0)
    key ^= zob.enpassant[file_of((st-1)->epSquare)];
  st->epSquare = 0;

  // Update castling rights if needed
  if (    st->castlingRights
      && (pos->castlingRightsMask[from] | pos->castlingRightsMask[to])) {
    int cr = pos->castlingRightsMask[from] | pos->castlingRightsMask[to];
    key ^= zob.castling[st->castlingRights & cr];
    st->castlingRights &= ~cr;
  }

  // Move the piece. The tricky Chess960 castling is handled earlier
  if (type_of_m(m) != CASTLING)
    move_piece(pos, us, pt, from, to);

  // If the moving piece is a pawn do some special extra work
  if (pt == PAWN) {
    // Set en-passant square if the moved pawn can be captured
    if ( (to ^ from) == 16
        && (attacks_from_pawn(to - pawn_push(us), us) & pieces_cp(them, PAWN))) {
      st->epSquare = (from + to) / 2;
      key ^= zob.enpassant[file_of(st->epSquare)];
    } else if (type_of_m(m) == PROMOTION) {
      int promotion = promotion_type(m);

      assert(relative_rank_s(us, to) == RANK_8);
      assert(promotion >= KNIGHT && promotion <= QUEEN);

      remove_piece(pos, us, PAWN, to);
      put_piece(pos, us, promotion, to);

      // Update hash keys
      key ^= zob.psq[us][PAWN][to] ^ zob.psq[us][promotion][to];
      st->pawnKey ^= zob.psq[us][PAWN][to];
      st->materialKey ^=  zob.psq[us][promotion][piece_count(us, promotion)-1]
                        ^ zob.psq[us][PAWN][piece_count(us, PAWN)];

      // Update incremental score
      st->psq += psqt.psq[us][promotion][to] - psqt.psq[us][PAWN][to];

      // Update material
      st->nonPawnMaterial[us] += PieceValue[MG][promotion];
    }

    // Update pawn hash key and prefetch access to pawnsTable
    st->pawnKey ^= zob.psq[us][PAWN][from] ^ zob.psq[us][PAWN][to];
    prefetch(&pos->pawnTable[st->pawnKey & 16383]);

    // Reset rule 50 draw counter
    st->rule50 = 0;
  }

  // Update incremental scores
  st->psq += psqt.psq[us][pt][to] - psqt.psq[us][pt][from];

  // Set capture piece
  st->capturedType = captured;

  // Update the key with the final value
  st->key = key;

  // Calculate checkers bitboard (if move gives check)
  st->checkersBB =  givesCheck
                  ? attackers_to(square_of(them, KING)) & pieces_c(us) : 0;

  st->pliesFromNull = (st-1)->pliesFromNull + 1;

  pos->sideToMove ^= 1;
  pos->nodes++;

  assert(pos_is_ok(pos, &failed_step));
}


// undo_move() unmakes a move. When it returns, the position should
// be restored to exactly the same state as before the move was made.

void undo_move(Pos *pos, Move m)
{
  assert(move_is_ok(m));

  pos->sideToMove ^= 1;

  int us = pos_stm();
  Square from = from_sq(m);
  Square to = to_sq(m);
  int pt = type_of_p(piece_on(to));

  assert(is_empty(from) || type_of_m(m) == CASTLING);
  assert(pos->st->capturedType != KING);

  if (type_of_m(m) == PROMOTION) {
    assert(relative_rank_s(us, to) == RANK_8);
    assert(pt == promotion_type(m));
    assert(pt >= KNIGHT && pt <= QUEEN);

    remove_piece(pos, us, pt, to);
    put_piece(pos, us, PAWN, to);
    pt = PAWN;
  }

  if (type_of_m(m) == CASTLING) {
    Square rfrom, rto;
    int kingSide = to > from;
    rfrom = to; // Castling is encoded as "king captures friendly rook"
    rto = relative_square(us, kingSide ? SQ_F1 : SQ_D1);
    to = relative_square(us, kingSide ? SQ_G1 : SQ_C1);

    // Remove both pieces first since squares could overlap in Chess960
    remove_piece(pos, us, KING, to);
    remove_piece(pos, us, ROOK, rto);
    pos->board[to] = pos->board[rto] = 0;
    put_piece(pos, us, KING, from);
    put_piece(pos, us, ROOK, rfrom);
  } else {
    move_piece(pos, us, pt, to, from); // Put the piece back at the source square

    if (pos->st->capturedType) {
      Square capsq = to;

      if (type_of_m(m) == ENPASSANT) {
        capsq -= pawn_push(us);

        assert(pt == PAWN);
        assert(to == pos->st->previous->epSquare);
        assert(relative_rank_s(us, to) == RANK_6);
        assert(is_empty(capsq));
        assert(pos->st->capturedType == PAWN);
      }

      put_piece(pos, us ^ 1, pos->st->capturedType, capsq); // Restore the captured piece
    }
  }

  // Finally point our state pointer back to the previous state
  pos->st--;

  assert(pos_is_ok(pos, &failed_step));
}
#endif


// do_null_move() is used to do a "null move".

void do_null_move(Pos *pos)
{
  assert(!pos_checkers());

  Stack *st = ++pos->st;
  memcpy(st, st - 1, StateSize);
  st->previous = st - 1;

  if (st->epSquare) {
    st->key ^= zob.enpassant[file_of(st->epSquare)];
    st->epSquare = 0;
  }

  st->key ^= zob.side;
  prefetch(tt_first_entry(st->key));

  st->rule50++;
  st->pliesFromNull = 0;

  pos->sideToMove ^= 1;

  assert(pos_is_ok(pos, &failed_step));
}

// undo_null_move() is used to undo a "null move".

void undo_null_move(Pos *pos)
{
  assert(!pos_checkers());

  pos->st = pos->st->previous;
  pos->sideToMove ^= 1;
}


// key_after() computes the new hash key after the given move. Needed
// for speculative prefetch. It does not recognize special moves like
// castling, en-passant and promotions.

Key key_after(Pos *pos, Move m)
{
  int us = pos_stm();
  Square from = from_sq(m);
  Square to = to_sq(m);
  int pt = type_of_p(piece_on(from));
  int captured = type_of_p(piece_on(to));
  Key k = pos_key() ^ zob.side;

  if (captured)
    k ^= zob.psq[us ^ 1][captured][to];

  return k ^ zob.psq[us][pt][to] ^ zob.psq[us][pt][from];
}

// see() is a static exchange evaluator: It tries to estimate the
// material gain or loss resulting from a move.

Value see_sign(Pos *pos, Move m) 
{
  assert(move_is_ok(m));

  // Early return if SEE cannot be negative because captured piece value
  // is not less then capturing one. Note that king moves always return
  // here because king midgame value is set to 0.
  if (PieceValue[MG][moved_piece(m)] <= PieceValue[MG][piece_on(to_sq(m))])
    return VALUE_KNOWN_WIN;

  return see(pos, m);
}

Value see(Pos *pos, Move m)
{
  Square from, to;
  Bitboard occ, attackers, stmAttackers;
  Value swapList[32];
  int slIndex = 1;
  int captured;
  int stm;

  assert(move_is_ok(m));

  from = from_sq(m);
  to = to_sq(m);
  swapList[0] = PieceValue[MG][piece_on(to)];
  stm = color_of(piece_on(from));
  occ = pieces() ^ sq_bb(from);

  // Castling moves are implemented as king capturing the rook so cannot
  // be handled correctly. Simply return VALUE_ZERO that is always correct
  // unless in the rare case the rook ends up under attack.
  if (type_of_m(m) == CASTLING)
    return 0;

  if (type_of_m(m) == ENPASSANT) {
    occ ^= sq_bb(to - pawn_push(stm)); // Remove the captured pawn
    swapList[0] = PieceValue[MG][PAWN];
  }

  // Find all attackers to the destination square, with the moving piece
  // removed, but possibly an X-ray attacker added behind it.
  attackers = attackers_to_occ(to, occ) & occ;

  // If the opponent has no attackers we are finished
  stm ^= 1;
  stmAttackers = attackers & pieces_c(stm);
  if (!stmAttackers)
    return swapList[0];

  // The destination square is defended. We proceed by building up a
  // "swap list" containing the material gain or loss at each stop in a
  // sequence of captures to the destination square, where the sides
  // alternately capture, and always capture with the least valuable
  // piece. After each capture, we look for new X-ray attacks from
  // behind the capturing piece.
  captured = type_of_p(piece_on(from));

  do {
    assert(slIndex < 32);

    // Add the new entry to the swap list
    swapList[slIndex] = -swapList[slIndex - 1] + PieceValue[MG][captured];

    // Locate and remove the next least valuable attacker
    Bitboard bb;
    for (captured = PAWN; captured <= KING; captured++)
      if ((bb = stmAttackers & pieces_p(captured)))
        break;
    occ ^= (bb & -bb);
    if (captured & 1) // PAWN, BISHOP, QUEEN
      attackers |= attacks_bb_bishop(to, occ) & pieces_pp(BISHOP, QUEEN);
    if ((captured & 4) && captured != KING) // ROOK, QUEEN
      attackers |= attacks_bb_rook(to, occ) & pieces_pp(ROOK, QUEEN);
    attackers &= occ;
    stm ^= 1;
    stmAttackers = attackers & pieces_c(stm);
    slIndex++;

  } while (stmAttackers && (captured != KING || (--slIndex, 0))); // Stop before a king capture

  // Having built the swap list, we negamax through it to find the best
  // achievable score from the point of view of the side to move.
  while (--slIndex)
    swapList[slIndex - 1] = min(-swapList[slIndex], swapList[slIndex - 1]);

  return swapList[0];
}


// is_draw() tests whether the position is drawn by 50-move rule or by
// repetition. It does not detect stalemates.

int is_draw(Pos *pos)
{
  Stack *st = pos->st;

  if (st->rule50 > 99) {
    if (!pos_checkers())
      return 1;
    ExtMove list[MAX_MOVES];
    return generate_legal(pos, list) != list;
  }

  Stack *stp = st;
  for (int i = 2, e = min(st->rule50, st->pliesFromNull); i <= e; i += 2)
  {
      stp = stp->previous->previous;

      if (stp->key == st->key)
          return 1; // Draw at first repetition
  }

  return 0;
}


void pos_copy(Pos *dest, Pos *src)
{
  memcpy(dest, src, offsetof(Pos, st));
  dest->st = dest->stack;
  memcpy(dest->st, src->st, StateSize);
}


// pos_is_ok() performs some consistency checks for the position object.
// This is meant to be helpful when debugging.

#ifdef PEDANTIC
int pos_is_ok(Pos *pos, int *failedStep)
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
#else
int pos_is_ok(Pos *pos, int *failedStep)
{
(void)pos;
(void)failedStep;
  return 1;
}

#ifndef NDEBUG
int check_pos(Pos *pos)
{
  Bitboard color_bb[2];
  Bitboard piece_bb[8];

  color_bb[0] = color_bb[1] = 0;
  for (int i = 0; i < 8; i++)
    piece_bb[i] = 0;

  for (int sq = 0; sq < 64; sq++)
    if (pos->board[sq]) {
      color_bb[pos->board[sq] >> 3] |= sq_bb(sq);
      piece_bb[pos->board[sq] & 7] |= sq_bb(sq);
    }

  for (int i = PAWN; i <= KING; i++)
    assert(pos->byTypeBB[i] == piece_bb[i]);

  assert(pos->byColorBB[0] == color_bb[0]);
  assert(pos->byColorBB[1] == color_bb[1]);
  assert(pos->byTypeBB[0] == (color_bb[0] | color_bb[1]));

  Key key = 0, pawnKey = 0, matKey = 0;

  for (int c = 0; c < 2; c++)
    for (int i = PAWN; i <= KING; i++)
       matKey += mat_key[8 * c + i] * piece_count(c, i);

  for (int sq = 0; sq < 64; sq++)
    if (pos->board[sq])
      key ^= zob.psq2d[pos->board[sq]][sq];
  if (pos->sideToMove == BLACK)
    key ^= zob.side;
  if (pos->st->epSquare)
    key ^= zob.enpassant[pos->st->epSquare & 7];
  key ^= zob.castling[pos->st->castlingRights];

  for (int sq = 0; sq < 64; sq++)
    if ((pos->board[sq] & 7) == PAWN)
      pawnKey ^= zob.psq2d[pos->board[sq]][sq];

  assert(key == pos->st->key);
  assert(pawnKey == pos->st->pawnKey);
  assert(matKey == pos->st->materialKey);

  return 1;
}
#endif
#endif

