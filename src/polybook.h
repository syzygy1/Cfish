/* polybook.h from BrainFish, Copyright (C) 2016-2017 Thomas Zipproth */

#ifndef POLYBOOK_H_INCLUDED
#define POLYBOOK_H_INCLUDED

#include "bitboard.h"
#include "misc.h"
#include "position.h"

struct PolyBook {
  ssize_t keycount;
  const struct PolyHash *polyhash;

  map_t mapping;

//  int use_best_book_move;
//  int max_book_depth;
  int book_depth_count;

  ssize_t index_first;
  int index_count;
  ssize_t index_best;
  ssize_t index_rand;
  int index_weight_count;

//  PRNG sr;

  Bitboard last_position;
  Bitboard akt_position;
  int last_anz_pieces;
  int akt_anz_pieces;
  int search_counter;

  bool enabled, do_search;
};

typedef struct PolyBook PolyBook;

extern PolyBook polybook, polybook2;

void pb_init(PolyBook *pb, const char *bookfile);
void pb_free(void);
void pb_set_best_book_move(bool best_book_move);
void pb_set_book_depth(int book_depth);
Move pb_probe(PolyBook *pb, Position *pos);

#endif
