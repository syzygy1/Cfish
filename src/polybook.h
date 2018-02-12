/* polybook.h from BrainFish, Copyright (C) 2016-2017 Thomas Zipproth */

#ifndef POLYBOOK_H_INCLUDED
#define POLYBOOK_H_INCLUDED

#include "bitboard.h"
#include "position.h"

void pb_init(const char *bookfile);
void pb_free(void);
void pb_set_best_book_move(int best_book_move);
void pb_set_book_depth(int book_depth);
Move pb_probe(Pos *pos);

#endif
