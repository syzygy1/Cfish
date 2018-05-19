/*
  Copyright (c) 2011-2018 Ronald de Man
*/

#ifndef TBCORE_H
#define TBCORE_H

#include <stdatomic.h>

#ifndef _WIN32
#include <pthread.h>
#define SEP_CHAR ':'
#else
#include <windows.h>
#define SEP_CHAR ';'
#endif

#define TB_PIECES 7

#define WDL_SUFFIX ".rtbw"
#define DTZ_SUFFIX ".rtbz"
#define DTM_SUFFIX ".rtbm"

const uint32_t WDL_MAGIC = 0x5d23e871;
const uint32_t DTZ_MAGIC = 0xa50c66d7;
const uint32_t DTM_MAGIC = 0x88ac504b;


#define TBHASHBITS 10

struct TBHashEntry;

typedef uint64_t base_t;

#ifdef _WIN32
typedef HANDLE map_t;
#else
typedef size_t map_t;
#endif

enum { WDL, DTM, DTZ };

enum { PIECE_ENC, FILE_ENC, RANK_ENC };

struct PairsData {
  uint8_t *indextable;
  uint16_t *sizetable;
  uint8_t *data;
  uint16_t *offset;
  uint8_t *symlen;
  uint8_t *sympat;
  uint32_t blocksize;
  uint32_t idxbits;
  uint8_t min_len;
  uint8_t const_val[2];
  base_t base[]; // must be base[1] in C++
};

struct EncInfo {
  struct PairsData *precomp;
  uint32_t factor[TB_PIECES];
  uint8_t pieces[TB_PIECES];
  uint8_t norm[TB_PIECES];
};

struct BaseEntry {
  Key key;
  uint8_t *data[3];
  map_t mapping[3];
  atomic_bool ready[3];
  uint8_t num;
  bool symmetric, has_pawns, has_dtm, has_dtz;
  union {
    bool kk_enc;
    uint8_t pawns[2];
  };
  bool dtm_loss_only;
};

struct PieceEntry {
  struct BaseEntry be;
  struct EncInfo ei[5]; // 2 + 2 + 1
  uint16_t *dtm_map;
  uint16_t dtm_map_idx[2][2];
  void *dtz_map;
  uint16_t dtz_map_idx[4];
  uint8_t dtz_flags;
};

struct PawnEntry {
  struct BaseEntry be;
  struct EncInfo ei[24]; // 4 * 2 + 6 * 2 + 4
  uint16_t *dtm_map;
  uint16_t dtm_map_idx[6][2][2];
  void *dtz_map;
  uint16_t dtz_map_idx[4][4];
  uint8_t dtz_flags[4];
  bool dtm_switched;
};

struct TBHashEntry {
  Key key;
  struct BaseEntry *ptr;
};

#endif
