/*
  Copyright (c) 2011-2018 Ronald de Man
  This file may be redistributed and/or modified without restrictions.

  tbcore.c contains engine-independent routines of the tablebase probing code.
  This file should not need too much adaptation to add tablebase probing to
  a particular engine, provided the engine is written in C or C++.
*/

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tbcore.h"
#include "thread.h"

#define TBMAX_PIECE 650
#define TBMAX_PAWN 861
#define HSHMAX 10

#define Swap(a,b) {int tmp=a;a=b;b=tmp;}

static LOCK_T TB_mutex;

static int initialized = 0;
static int num_paths = 0;
static char *path_string = NULL;
static char **paths = NULL;

static int TBnum_piece, TBnum_pawn;
static int num_wdl, num_dtm, num_dtz;
static struct PieceEntry pieceEntry[TBMAX_PIECE];
static struct PawnEntry pawnEntry[TBMAX_PAWN];

static struct TBHashEntry TB_hash[1 << TBHASHBITS][HSHMAX];

static void init_indices(void);
static Key calc_key_from_pcs(int *pcs, int flip);
static Key calc_key_from_pieces(uint8_t *pieces, int num);
static void free_tb_entry(struct BaseEntry *ptr);

static const char *tbSuffix[] = { WDL_SUFFIX, DTM_SUFFIX, DTZ_SUFFIX };
static uint32_t tbMagic[] = { WDL_MAGIC, DTM_MAGIC, DTZ_MAGIC };

static FD open_tb(const char *str, const char *suffix)
{
  char name[256];

  for (int i = 0; i < num_paths; i++) {
    strcpy(name, paths[i]);
    strcat(name, "/");
    strcat(name, str);
    strcat(name, suffix);
    FD fd = open_file(name);
    if (fd != FD_ERR) return fd;
  }
  return FD_ERR;
}

static bool test_tb(const char *str, const char *suffix)
{
  FD fd = open_tb(str, suffix);
  if (fd != FD_ERR)
    close (fd);
  return fd != FD_ERR;
}

static void *map_tb(const char *name, const char *suffix, map_t *mapping)
{
  FD fd = open_tb(name, suffix);
  if (fd == FD_ERR)
    return NULL;

  void *data = map_file(fd, mapping);
  if (data == NULL) {
    fprintf(stderr, "Could not map %s%s into memory.\n", name, suffix);
    exit(EXIT_FAILURE);
  }

  close_file(fd);

  return data;
}

static void add_to_hash(void *ptr, Key key)
{
  int i, hshidx;

  hshidx = key >> (64 - TBHASHBITS);
  i = 0;
  while (i < HSHMAX && TB_hash[hshidx][i].ptr)
    i++;
  if (i == HSHMAX) {
    fprintf(stderr, "HSHMAX too low!\n");
    exit(EXIT_FAILURE);
  } else {
    TB_hash[hshidx][i].key = key;
    TB_hash[hshidx][i].ptr = ptr;
  }
}

static char pchr[] = {'K', 'Q', 'R', 'B', 'N', 'P'};

static void init_tb(char *str)
{
  if (!test_tb(str, tbSuffix[WDL]))
    return;

  int pcs[16];
  for (int i = 0; i < 16; i++)
    pcs[i] = 0;
  int color = 0;
  for (char *s = str; *s; s++)
    switch (*s) {
    case 'P':
      pcs[PAWN | color]++;
      break;
    case 'N':
      pcs[KNIGHT | color]++;
      break;
    case 'B':
      pcs[BISHOP | color]++;
      break;
    case 'R':
      pcs[ROOK | color]++;
      break;
    case 'Q':
      pcs[QUEEN | color]++;
      break;
    case 'K':
      pcs[KING | color]++;
      break;
    case 'v':
      color = 0x08;
      break;
    }

  bool has_pawns = pcs[W_PAWN] || pcs[B_PAWN];

  struct BaseEntry *be;
  if (!has_pawns) {
    if (TBnum_piece == TBMAX_PIECE) {
      fprintf(stderr, "TBMAX_PIECE limit too low!\n");
      exit(EXIT_FAILURE);
    }
    be = &pieceEntry[TBnum_piece++].be;
  } else {
    if (TBnum_pawn == TBMAX_PAWN) {
      fprintf(stderr, "TBMAX_PAWN limit too low!\n");
      exit(EXIT_FAILURE);
    }
    be = &pawnEntry[TBnum_pawn++].be;
  }

  be->has_pawns = has_pawns;

  be->has_dtm = test_tb(str, tbSuffix[DTM]);
  be->has_dtz = test_tb(str, tbSuffix[DTZ]);

  Key key = calc_key_from_pcs(pcs, 0);
  Key key2 = calc_key_from_pcs(pcs, 1);

  be->key = key;

  be->num = 0;
  for (int i = 0; i < 16; i++)
    be->num += pcs[i];

  TB_MaxCardinality = max(TB_MaxCardinality, be->num);
  if (be->has_dtm)
    TB_MaxCardinalityDTM = max(TB_MaxCardinalityDTM, be->num);

  be->symmetric = key == key2;

  num_wdl++;
  num_dtm += be->has_dtm;
  num_dtz += be->has_dtz;

  atomic_init(&be->ready[WDL], false);
  atomic_init(&be->ready[DTM], false);
  atomic_init(&be->ready[DTZ], false);

  if (!be->has_pawns) {
    int j = 0;
    for (int i = 0; i < 16; i++)
      if (pcs[i] == 1) j++;
    be->kk_enc = j == 2;
  } else {
    if (!pcs[B_PAWN] || (pcs[W_PAWN] && pcs[W_PAWN] < pcs[B_PAWN])) {
      be->pawns[0] = pcs[W_PAWN];
      be->pawns[1] = pcs[B_PAWN];
    } else {
      be->pawns[0] = pcs[B_PAWN];
      be->pawns[1] = pcs[W_PAWN];
    }
  }

  add_to_hash(be, key);
  if (key != key2)
    add_to_hash(be, key2);
}

void TB_free(void)
{
  TB_init("");
}

void TB_init(char *path)
{
  if (!initialized) {
    init_indices();
    initialized = 1;
  }

  // if path_string is set, we need to clean up first.
  if (path_string) {
    free(path_string);
    free(paths);

    for (int i = 0; i < TBnum_piece; i++)
      free_tb_entry((struct BaseEntry *)&pieceEntry[i]);
    for (int i = 0; i < TBnum_pawn; i++)
      free_tb_entry((struct BaseEntry *)&pawnEntry[i]);

    LOCK_DESTROY(TB_mutex);

    path_string = NULL;
    num_wdl = num_dtm = num_dtz = 0;
  }

  // if path is an empty string or equals "<empty>", we are done.
  const char *p = path;
  if (strlen(p) == 0 || !strcmp(p, "<empty>")) return;

  path_string = malloc(strlen(p) + 1);
  strcpy(path_string, p);
  num_paths = 0;
  for (int i = 0;; i++) {
    if (path_string[i] != SEP_CHAR)
      num_paths++;
    while (path_string[i] && path_string[i] != SEP_CHAR)
      i++;
    if (!path_string[i]) break;
    path_string[i] = 0;
  }
  paths = malloc(num_paths * sizeof(*paths));
  for (int i = 0, j = 0; i < num_paths; i++) {
    while (!path_string[j]) j++;
    paths[i] = &path_string[j];
    while (path_string[j]) j++;
  }

  LOCK_INIT(TB_mutex);

  TBnum_piece = TBnum_pawn = 0;
  TB_MaxCardinality = TB_MaxCardinalityDTM = 0;

  for (int i = 0; i < (1 << TBHASHBITS); i++)
    for (int j = 0; j < HSHMAX; j++) {
      TB_hash[i][j].key = 0;
      TB_hash[i][j].ptr = NULL;
    }

  char str[16];
  int i, j, k, l, m;

  for (i = 1; i < 6; i++) {
    sprintf(str, "K%cvK", pchr[i]);
    init_tb(str);
  }

  for (i = 1; i < 6; i++)
    for (j = i; j < 6; j++) {
      sprintf(str, "K%cvK%c", pchr[i], pchr[j]);
      init_tb(str);
    }

  for (i = 1; i < 6; i++)
    for (j = i; j < 6; j++) {
      sprintf(str, "K%c%cvK", pchr[i], pchr[j]);
      init_tb(str);
    }

  for (i = 1; i < 6; i++)
    for (j = i; j < 6; j++)
      for (k = 1; k < 6; k++) {
        sprintf(str, "K%c%cvK%c", pchr[i], pchr[j], pchr[k]);
        init_tb(str);
      }

  for (i = 1; i < 6; i++)
    for (j = i; j < 6; j++)
      for (k = j; k < 6; k++) {
        sprintf(str, "K%c%c%cvK", pchr[i], pchr[j], pchr[k]);
        init_tb(str);
      }

  // 6- and 7-piece TBs make sense only with a 64-bit address space
  if (sizeof(size_t) < 8)
    goto finished;

  for (i = 1; i < 6; i++)
    for (j = i; j < 6; j++)
      for (k = i; k < 6; k++)
        for (l = (i == k) ? j : k; l < 6; l++) {
          sprintf(str, "K%c%cvK%c%c", pchr[i], pchr[j], pchr[k], pchr[l]);
          init_tb(str);
        }

  for (i = 1; i < 6; i++)
    for (j = i; j < 6; j++)
      for (k = j; k < 6; k++)
        for (l = 1; l < 6; l++) {
          sprintf(str, "K%c%c%cvK%c", pchr[i], pchr[j], pchr[k], pchr[l]);
          init_tb(str);
        }

  for (i = 1; i < 6; i++)
    for (j = i; j < 6; j++)
      for (k = j; k < 6; k++)
        for (l = k; l < 6; l++) {
          sprintf(str, "K%c%c%c%cvK", pchr[i], pchr[j], pchr[k], pchr[l]);
          init_tb(str);
        }

  for (i = 1; i < 6; i++)
    for (j = i; j < 6; j++)
      for (k = j; k < 6; k++)
        for (l = k; l < 6; l++)
          for (m = l; m < 6; m++) {
            sprintf(str, "K%c%c%c%c%cvK", pchr[i], pchr[j], pchr[k], pchr[l], pchr[m]);
            init_tb(str);
          }

  for (i = 1; i < 6; i++)
    for (j = i; j < 6; j++)
      for (k = j; k < 6; k++)
        for (l = k; l < 6; l++)
          for (m = 1; m < 6; m++) {
            sprintf(str, "K%c%c%c%cvK%c", pchr[i], pchr[j], pchr[k], pchr[l], pchr[m]);
            init_tb(str);
          }

  for (i = 1; i < 6; i++)
    for (j = i; j < 6; j++)
      for (k = j; k < 6; k++)
        for (l = 1; l < 6; l++)
          for (m = l; m < 6; m++) {
            sprintf(str, "K%c%c%cvK%c%c", pchr[i], pchr[j], pchr[k], pchr[l], pchr[m]);
            init_tb(str);
          }

finished:
  printf("info string Found %d WDL, %d DTM and %d DTZ tablebase files.\n",
      num_wdl, num_dtm, num_dtz);
  fflush(stdout);
}

static const int8_t offdiag[] = {
  0,-1,-1,-1,-1,-1,-1,-1,
  1, 0,-1,-1,-1,-1,-1,-1,
  1, 1, 0,-1,-1,-1,-1,-1,
  1, 1, 1, 0,-1,-1,-1,-1,
  1, 1, 1, 1, 0,-1,-1,-1,
  1, 1, 1, 1, 1, 0,-1,-1,
  1, 1, 1, 1, 1, 1, 0,-1,
  1, 1, 1, 1, 1, 1, 1, 0
};

static const uint8_t triangle[] = {
  6, 0, 1, 2, 2, 1, 0, 6,
  0, 7, 3, 4, 4, 3, 7, 0,
  1, 3, 8, 5, 5, 8, 3, 1,
  2, 4, 5, 9, 9, 5, 4, 2,
  2, 4, 5, 9, 9, 5, 4, 2,
  1, 3, 8, 5, 5, 8, 3, 1,
  0, 7, 3, 4, 4, 3, 7, 0,
  6, 0, 1, 2, 2, 1, 0, 6
};

static const uint8_t flipdiag[] = {
   0,  8, 16, 24, 32, 40, 48, 56,
   1,  9, 17, 25, 33, 41, 49, 57,
   2, 10, 18, 26, 34, 42, 50, 58,
   3, 11, 19, 27, 35, 43, 51, 59,
   4, 12, 20, 28, 36, 44, 52, 60,
   5, 13, 21, 29, 37, 45, 53, 61,
   6, 14, 22, 30, 38, 46, 54, 62,
   7, 15, 23, 31, 39, 47, 55, 63
};

static const uint8_t lower[] = {
  28,  0,  1,  2,  3,  4,  5,  6,
   0, 29,  7,  8,  9, 10, 11, 12,
   1,  7, 30, 13, 14, 15, 16, 17,
   2,  8, 13, 31, 18, 19, 20, 21,
   3,  9, 14, 18, 32, 22, 23, 24,
   4, 10, 15, 19, 22, 33, 25, 26,
   5, 11, 16, 20, 23, 25, 34, 27,
   6, 12, 17, 21, 24, 26, 27, 35
};

static const uint8_t diag[] = {
   0,  0,  0,  0,  0,  0,  0,  8,
   0,  1,  0,  0,  0,  0,  9,  0,
   0,  0,  2,  0,  0, 10,  0,  0,
   0,  0,  0,  3, 11,  0,  0,  0,
   0,  0,  0, 12,  4,  0,  0,  0,
   0,  0, 13,  0,  0,  5,  0,  0,
   0, 14,  0,  0,  0,  0,  6,  0,
  15,  0,  0,  0,  0,  0,  0,  7
};

static const uint8_t flap[2][64] = {
  {  0,  0,  0,  0,  0,  0,  0,  0,
     0,  6, 12, 18, 18, 12,  6,  0,
     1,  7, 13, 19, 19, 13,  7,  1,
     2,  8, 14, 20, 20, 14,  8,  2,
     3,  9, 15, 21, 21, 15,  9,  3,
     4, 10, 16, 22, 22, 16, 10,  4,
     5, 11, 17, 23, 23, 17, 11,  5,
     0,  0,  0,  0,  0,  0,  0,  0  },
  {  0,  0,  0,  0,  0,  0,  0,  0,
     0,  1,  2,  3,  3,  2,  1,  0,
     4,  5,  6,  7,  7,  6,  5,  4,
     8,  9, 10, 11, 11, 10,  9,  8,
    12, 13, 14, 15, 15, 14, 13, 12,
    16, 17, 18, 19, 19, 18, 17, 16,
    20, 21, 22, 23, 23, 22, 21, 20,
     0,  0,  0,  0,  0,  0,  0,  0  }
};

static const uint8_t ptwist[2][64] = {
  {  0,  0,  0,  0,  0,  0,  0,  0,
    47, 35, 23, 11, 10, 22, 34, 46,
    45, 33, 21,  9,  8, 20, 32, 44,
    43, 31, 19,  7,  6, 18, 30, 42,
    41, 29, 17,  5,  4, 16, 28, 40,
    39, 27, 15,  3,  2, 14, 26, 38,
    37, 25, 13,  1,  0, 12, 24, 36,
     0,  0,  0,  0,  0,  0,  0,  0 },
  {  0,  0,  0,  0,  0,  0,  0,  0,
    47, 45, 43, 41, 40, 42, 44, 46,
    39, 37, 35, 33, 32, 34, 36, 38,
    31, 29, 27, 25, 24, 26, 28, 30,
    23, 21, 19, 17, 16, 18, 20, 22,
    15, 13, 11,  9,  8, 10, 12, 14,
     7,  5,  3,  1,  0,  2,  4,  6,
     0,  0,  0,  0,  0,  0,  0,  0 }
};

static const uint8_t invflap[] = {
  8, 16, 24, 32, 40, 48,
  9, 17, 25, 33, 41, 49,
  10, 18, 26, 34, 42, 50,
  11, 19, 27, 35, 43, 51
};

static const uint8_t invflap2[] = {
  8, 9, 10, 11,
  16, 17, 18, 19,
  24, 25, 26, 27,
  32, 33, 34, 35,
  40, 41, 42, 43,
  48, 49, 50, 51
};

static const uint8_t file_to_file[] = {
  0, 1, 2, 3, 3, 2, 1, 0
};

static const int16_t KK_idx[10][64] = {
  { -1, -1, -1,  0,  1,  2,  3,  4,
    -1, -1, -1,  5,  6,  7,  8,  9,
    10, 11, 12, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33,
    34, 35, 36, 37, 38, 39, 40, 41,
    42, 43, 44, 45, 46, 47, 48, 49,
    50, 51, 52, 53, 54, 55, 56, 57 },
  { 58, -1, -1, -1, 59, 60, 61, 62,
    63, -1, -1, -1, 64, 65, 66, 67,
    68, 69, 70, 71, 72, 73, 74, 75,
    76, 77, 78, 79, 80, 81, 82, 83,
    84, 85, 86, 87, 88, 89, 90, 91,
    92, 93, 94, 95, 96, 97, 98, 99,
   100,101,102,103,104,105,106,107,
   108,109,110,111,112,113,114,115},
  {116,117, -1, -1, -1,118,119,120,
   121,122, -1, -1, -1,123,124,125,
   126,127,128,129,130,131,132,133,
   134,135,136,137,138,139,140,141,
   142,143,144,145,146,147,148,149,
   150,151,152,153,154,155,156,157,
   158,159,160,161,162,163,164,165,
   166,167,168,169,170,171,172,173 },
  {174, -1, -1, -1,175,176,177,178,
   179, -1, -1, -1,180,181,182,183,
   184, -1, -1, -1,185,186,187,188,
   189,190,191,192,193,194,195,196,
   197,198,199,200,201,202,203,204,
   205,206,207,208,209,210,211,212,
   213,214,215,216,217,218,219,220,
   221,222,223,224,225,226,227,228 },
  {229,230, -1, -1, -1,231,232,233,
   234,235, -1, -1, -1,236,237,238,
   239,240, -1, -1, -1,241,242,243,
   244,245,246,247,248,249,250,251,
   252,253,254,255,256,257,258,259,
   260,261,262,263,264,265,266,267,
   268,269,270,271,272,273,274,275,
   276,277,278,279,280,281,282,283 },
  {284,285,286,287,288,289,290,291,
   292,293, -1, -1, -1,294,295,296,
   297,298, -1, -1, -1,299,300,301,
   302,303, -1, -1, -1,304,305,306,
   307,308,309,310,311,312,313,314,
   315,316,317,318,319,320,321,322,
   323,324,325,326,327,328,329,330,
   331,332,333,334,335,336,337,338 },
  { -1, -1,339,340,341,342,343,344,
    -1, -1,345,346,347,348,349,350,
    -1, -1,441,351,352,353,354,355,
    -1, -1, -1,442,356,357,358,359,
    -1, -1, -1, -1,443,360,361,362,
    -1, -1, -1, -1, -1,444,363,364,
    -1, -1, -1, -1, -1, -1,445,365,
    -1, -1, -1, -1, -1, -1, -1,446 },
  { -1, -1, -1,366,367,368,369,370,
    -1, -1, -1,371,372,373,374,375,
    -1, -1, -1,376,377,378,379,380,
    -1, -1, -1,447,381,382,383,384,
    -1, -1, -1, -1,448,385,386,387,
    -1, -1, -1, -1, -1,449,388,389,
    -1, -1, -1, -1, -1, -1,450,390,
    -1, -1, -1, -1, -1, -1, -1,451 },
  {452,391,392,393,394,395,396,397,
    -1, -1, -1, -1,398,399,400,401,
    -1, -1, -1, -1,402,403,404,405,
    -1, -1, -1, -1,406,407,408,409,
    -1, -1, -1, -1,453,410,411,412,
    -1, -1, -1, -1, -1,454,413,414,
    -1, -1, -1, -1, -1, -1,455,415,
    -1, -1, -1, -1, -1, -1, -1,456 },
  {457,416,417,418,419,420,421,422,
    -1,458,423,424,425,426,427,428,
    -1, -1, -1, -1, -1,429,430,431,
    -1, -1, -1, -1, -1,432,433,434,
    -1, -1, -1, -1, -1,435,436,437,
    -1, -1, -1, -1, -1,459,438,439,
    -1, -1, -1, -1, -1, -1,460,440,
    -1, -1, -1, -1, -1, -1, -1,461 }
};

static size_t binomial[7][64];
static size_t pawnidx[2][6][24];
static size_t pfactor_file[6][4];
static size_t pfactor_rank[6][6];

static void init_indices(void)
{
  int i, j, k;

// binomial[k][n] = Bin(n, k)
  for (i = 0; i < 7; i++)
    for (j = 0; j < 64; j++) {
      size_t f = 1;
      size_t l = 1;
      for (k = 0; k < i; k++) {
        f *= (j - k);
        l *= (k + 1);
      }
      binomial[i][j] = f / l;
    }

  for (i = 0; i < 6; i++) {
    size_t s = 0;
    for (j = 0; j < 24; j++) {
      pawnidx[0][i][j] = s;
      s += binomial[i][ptwist[0][(1 + (j % 6)) * 8 + (j  /6)]];
      if ((j + 1) % 6 == 0) {
        pfactor_file[i][j / 6] = s;
        s = 0;
      }
    }
  }

  for (i = 0; i < 6; i++) {
    size_t s = 0;
    for (j = 0; j < 24; j++) {
      pawnidx[1][i][j] = s;
      s += binomial[i][ptwist[1][(1 + (j / 4)) * 8 + (j % 4)]];
      if ((j + 1) % 4 == 0) {
        pfactor_rank[i][j / 4] = s;
        s = 0;
      }
    }
  }
}

static int leading_pawn_file(Bitboard pawns)
{
  return  (pawns & (FileABB | FileBBB | FileGBB | FileHBB))
        ? (pawns & (FileABB | FileHBB)) ? FILE_A : FILE_B
        : (pawns & (FileCBB | FileFBB)) ? FILE_C : FILE_D;
}

static int leading_pawn_rank(Bitboard pawns, bool flip)
{
  Bitboard b = flip ? BSWAP64(pawns) : pawns;
  return (lsb(b) >> 3) - 1;
}

INLINE size_t encode(int *p, struct EncInfo *ei, struct BaseEntry *be,
    const int enc)
{
  int n = be->num;
  size_t idx;
  int k;

  if (p[0] & 0x04)
    for (int i = 0; i < n; i++)
      p[i] ^= 0x07;

  if (enc == PIECE_ENC) {
    if (p[0] & 0x20)
      for (int i = 0; i < n; i++)
        p[i] ^= 0x38;

    for (int i = 0; i < n; i++)
      if (offdiag[p[i]]) {
        if (offdiag[p[i]] > 0 && i < (be->kk_enc ? 2 : 3))
          for (int j = 0; j < n; j++)
            p[j] = flipdiag[p[j]];
        break;
      }

    if (be->kk_enc) {
      idx = KK_idx[triangle[p[0]]][p[1]];
      k = 2;
    } else {
      int s1 = (p[1] > p[0]);
      int s2 = (p[2] > p[0]) + (p[2] > p[1]);

      if (offdiag[p[0]])
        idx = triangle[p[0]] * 63*62 + (p[1] - s1) * 62 + (p[2] - s2);
      else if (offdiag[p[1]])
        idx = 6*63*62 + diag[p[0]] * 28*62 + lower[p[1]] * 62 + p[2] - s2;
      else if (offdiag[p[2]])
        idx = 6*63*62 + 4*28*62 + diag[p[0]] * 7*28 + (diag[p[1]] - s1) * 28 + lower[p[2]];
      else
        idx = 6*63*62 + 4*28*62 + 4*7*28 + diag[p[0]] * 7*6 + (diag[p[1]] - s1) * 6 + (diag[p[2]] - s2);
      k = 3;
    }
    idx *= ei->factor[0];
  } else {
    for (int i = 0; i < be->pawns[0]; i++)
      for (int j = i + 1; j < be->pawns[0]; j++)
        if (ptwist[enc-1][p[i]] < ptwist[enc-1][p[j]])
          Swap(p[i], p[j]);

    k = be->pawns[0];
    idx = pawnidx[enc-1][k-1][flap[enc-1][p[0]]];
    for (int i = 1; i < k; i++)
      idx += binomial[k-i][ptwist[enc-1][p[i]]];
    idx *= ei->factor[0];

    // Remaining pawns
    if (be->pawns[1]) {
      int t = k + be->pawns[1];
      for (int i = k; i < t; i++)
        for (int j = i + 1; j < t; j++)
          if (p[i] > p[j]) Swap(p[i], p[j]);
      size_t s = 0;
      for (int i = k; i < t; i++) {
        int sq = p[i];
        int skips = 0;
        for (int j = 0; j < k; j++)
          skips += (sq > p[j]);
        s += binomial[i - k + 1][sq - skips - 8];
      }
      idx += s * ei->factor[k];
      k = t;
    }
  }

  for (; k < n;) {
    int t = k + ei->norm[k];
    for (int i = k; i < t; i++)
      for (int j = i + 1; j < t; j++)
        if (p[i] > p[j]) Swap(p[i], p[j]);
    size_t s = 0;
    for (int i = k; i < t; i++) {
      int sq = p[i];
      int skips = 0;
      for (int j = 0; j < k; j++)
        skips += (sq > p[j]);
      s += binomial[i - k + 1][sq - skips];
    }
    idx += s * ei->factor[k];
    k = t;
  }

  return idx;
}

static size_t encode_piece(int *p, struct EncInfo *ei, struct BaseEntry *be)
{
  return encode(p, ei, be, PIECE_ENC);
}

static size_t encode_pawn(int *p, struct EncInfo *ei, struct BaseEntry *be)
{
  return encode(p, ei, be, FILE_ENC);
}

static size_t encode_pawn2(int *p, struct EncInfo *ei, struct BaseEntry *be)
{
  return encode(p, ei, be, RANK_ENC);
}

// place k like pieces on n squares
static size_t subfactor(size_t k, size_t n)
{
  size_t f = n;
  size_t l = 1;
  for (size_t i = 1; i < k; i++) {
    f *= n - i;
    l *= i + 1;
  }

  return f / l;
}

static size_t calc_factors(struct EncInfo *ei, struct BaseEntry *be,
    uint8_t order, uint8_t order2, int t, const int enc)
{
  int i = ei->norm[0];
  if (order2 < 0x0f) {
    i += ei->norm[i];
  }

  int n = 64 - i;
  size_t f = 1;

  for (int k = 0; i < be->num || k == order || k == order2; k++) {
    if (k == order) {
      ei->factor[0] = f;
      f *=  enc == FILE_ENC ? pfactor_file[ei->norm[0] - 1][t]
          : enc == RANK_ENC ? pfactor_rank[ei->norm[0] - 1][t]
          : be->kk_enc ? 462 : 31332;
    } else if (k == order2) {
      ei->factor[ei->norm[0]] = f;
      f *= subfactor(ei->norm[ei->norm[0]], 48 - ei->norm[0]);
    } else {
      ei->factor[i] = f;
      f *= subfactor(ei->norm[i], n);
      n -= ei->norm[i];
      i += ei->norm[i];
    }
  }

  return f;
}

static void set_norm(struct EncInfo *ei, struct BaseEntry *be, const int enc)
{
  for (int i = 0; i < be->num; i++)
    ei->norm[i] = 0;

  int i = ei->norm[0] =  enc != PIECE_ENC ? be->pawns[0]
                       : be->kk_enc ? 2 : 3;

  if (enc != PIECE_ENC && be->pawns[1]) {
    ei->norm[i] = be->pawns[1];
    i += ei->norm[i];
  }

  for (; i < be->num; i += ei->norm[i])
    for (int j = i; j < be->num && ei->pieces[j] == ei->pieces[i]; j++)
      ei->norm[i]++;
}

static size_t setup_pieces(struct EncInfo *ei, struct BaseEntry *be,
    uint8_t *tb, int shift, int t, const int enc)
{
  int j = 1 + (enc != PIECE_ENC && be->pawns[1] > 0);

  for (int i = 0; i < be->num; i++)
    ei->pieces[i] = (tb[i + j] >> shift) & 0x0f;
  int order = (tb[0] >> shift) & 0x0f;
  int order2 =  enc != PIECE_ENC && be->pawns[1] > 0
              ? (tb[1] >> shift) & 0x0f : 0x0f;

  set_norm(ei, be, enc);
  return calc_factors(ei, be, order, order2, t, enc);
}

static void calc_symlen(struct PairsData *d, uint32_t s, char *tmp)
{
  uint8_t *w = d->sympat + 3 * s;
  uint32_t s2 = (w[2] << 4) | (w[1] >> 4);
  if (s2 == 0x0fff)
    d->symlen[s] = 0;
  else {
    uint32_t s1 = ((w[1] & 0xf) << 8) | w[0];
    if (!tmp[s1]) calc_symlen(d, s1, tmp);
    if (!tmp[s2]) calc_symlen(d, s2, tmp);
    d->symlen[s] = d->symlen[s1] + d->symlen[s2] + 1;
  }
  tmp[s] = 1;
}

static struct PairsData *setup_pairs(uint8_t **ptr, size_t tb_size,
    size_t *size, uint8_t *flags, int type)
{
  struct PairsData *d;
  uint8_t *data = *ptr;

  *flags = data[0];
  if (data[0] & 0x80) {
    d = malloc(sizeof(*d));
    d->idxbits = 0;
    d->const_val[0] = type == WDL ? data[1] : 0;
    d->const_val[1] = 0;
    *ptr = data + 2;
    size[0] = size[1] = size[2] = 0;
    return d;
  }

  uint32_t blocksize = data[1];
  uint32_t idxbits = data[2];
  uint32_t real_num_blocks = read_le_u32(&data[4]);
  uint32_t num_blocks = real_num_blocks + data[3];
  int max_len = data[8];
  int min_len = data[9];
  int h = max_len - min_len + 1;
  uint32_t num_syms = read_le_u16(&data[10 + 2 * h]);
  d = malloc(sizeof(*d) + h * sizeof(base_t) + num_syms);
  d->blocksize = blocksize;
  d->idxbits = idxbits;
  d->offset = (uint16_t *)&data[10];
  d->symlen = (uint8_t *)d + sizeof(*d) + h * sizeof(base_t);
  d->sympat = &data[12 + 2 * h];
  d->min_len = min_len;
  *ptr = &data[12 + 2 * h + 3 * num_syms + (num_syms & 1)];

  size_t num_indices = (tb_size + (1ULL << idxbits) - 1) >> idxbits;
  size[0] = 6ULL * num_indices;
  size[1] = 2ULL * num_blocks;
  size[2] = (size_t)real_num_blocks << blocksize;

  char tmp[num_syms];
  memset(tmp, 0, num_syms);
  for (uint32_t s = 0; s < num_syms; s++)
    if (!tmp[s])
      calc_symlen(d, s, tmp);

  d->base[h - 1] = 0;
  for (int i = h - 2; i >= 0; i--)
    d->base[i] = (d->base[i + 1] + read_le_u16((uint8_t *)(d->offset + i)) - read_le_u16((uint8_t *)(d->offset + i + 1))) / 2;
  for (int i = 0; i < h; i++)
    d->base[i] <<= 64 - (min_len + i);

  d->offset -= d->min_len;

  return d;
}

#define PIECE(x) ((struct PieceEntry *)(x))
#define PAWN(x) ((struct PawnEntry *)(x))

INLINE int num_tables(struct BaseEntry *be, const int type)
{
  return be->has_pawns ? type == DTM ? 6 : 4 : 1;
}

INLINE struct EncInfo *first_ei(struct BaseEntry *be, const int type)
{
  return  !be->has_pawns
        ? &PIECE(be)->ei[type == WDL ? 0 : type == DTM ? 2 : 4]
        : &PAWN(be)->ei[type == WDL ? 0 : type == DTM ? 8 : 20];
}

static bool init_table(struct BaseEntry *be, const char *str, int type)
{
  uint8_t *data = map_tb(str, tbSuffix[type], &be->mapping[type]);
  if (!data) return false;

  if (read_le_u32(data) != tbMagic[type]) {
    fprintf(stderr, "Corrupted table.\n");
    unmap_file(data, be->mapping[type]);
    return false;
  }

  be->data[type] = data;

  bool split = type != DTZ && (data[4] & 0x01);
  if (type == DTM)
    be->dtm_loss_only = data[4] & 0x04;

  data += 5;

  size_t tb_size[6][2];
  int num = num_tables(be, type);
  struct EncInfo *ei = first_ei(be, type);
  int enc = !be->has_pawns ? PIECE_ENC : type != DTM ? FILE_ENC : RANK_ENC;

  for (int t = 0; t < num; t++) {
    tb_size[t][0] = setup_pieces(&ei[t], be, data, 0, t, enc);
    if (split)
      tb_size[t][1] = setup_pieces(&ei[num + t], be, data, 4, t, enc);
    data += be->num + 1 + (be->has_pawns && be->pawns[1]);
  }
  data += (uintptr_t)data & 1;

  size_t size[6][2][3];
  for (int t = 0; t < num; t++) {
    uint8_t flags;
    ei[t].precomp = setup_pairs(&data, tb_size[t][0], size[t][0], &flags, type);
    if (type == DTZ) {
      if (!be->has_pawns)
        PIECE(be)->dtz_flags = flags;
      else
        PAWN(be)->dtz_flags[t] = flags;
    }
    if (split)
      ei[num + t].precomp = setup_pairs(&data, tb_size[t][1], size[t][1], &flags, type);
    else if (type != DTZ)
      ei[num + t].precomp = NULL;
  }

  if (type == DTM && !be->dtm_loss_only) {
    uint16_t *map = (uint16_t *)data;
    if (!be->has_pawns)
      PIECE(be)->dtm_map = map;
    else
      PAWN(be)->dtm_map = map;
    uint16_t (*map_idx)[2][2] = !be->has_pawns ? &PIECE(be)->dtm_map_idx
                                               : &PAWN(be)->dtm_map_idx[0];
    for (int t = 0; t < num; t++) {
      for (int i = 0; i < 2; i++) {
        map_idx[t][0][i] = (uint16_t *)data + 1 - map;
        data += 2 + 2 * read_le_u16(data);
      }
      if (split) {
        for (int i = 0; i < 2; i++) {
          map_idx[t][1][i] = (uint16_t *)data + 1 - map;
          data += 2 + 2 * read_le_u16(data);
        }
      }
    }
  }

  if (type == DTZ) {
    void *map = data;
    if (!be->has_pawns)
      PIECE(be)->dtz_map = map;
    else
      PAWN(be)->dtz_map = map;
    uint16_t (*map_idx)[4] = !be->has_pawns ? &PIECE(be)->dtz_map_idx
                                            : &PAWN(be)->dtz_map_idx[0];
    uint8_t *flags = !be->has_pawns ? &PIECE(be)->dtz_flags
                                    : &PAWN(be)->dtz_flags[0];
    for (int t = 0; t < num; t++) {
      if (flags[t] & 2) {
        if (!(flags[t] & 16)) {
          for (int i = 0; i < 4; i++) {
            map_idx[t][i] = data + 1 - (uint8_t *)map;
            data += 1 + data[0];
          }
        } else {
          data += (uintptr_t)data & 0x01;
          for (int i = 0; i < 4; i++) {
            map_idx[t][i] = (uint16_t *)data + 1 - (uint16_t *)map;
            data += 2 + 2 * read_le_u16(data);
          }
        }
      }
    }
    data += (uintptr_t)data & 0x01;
  }

  for (int t = 0; t < num; t++) {
    ei[t].precomp->indextable = data;
    data += size[t][0][0];
    if (split) {
      ei[num + t].precomp->indextable = data;
      data += size[t][1][0];
    }
  }

  for (int t = 0; t < num; t++) {
    ei[t].precomp->sizetable = (uint16_t *)data;
    data += size[t][0][1];
    if (split) {
      ei[num + t].precomp->sizetable = (uint16_t *)data;
      data += size[t][1][1];
    }
  }

  for (int t = 0; t < num; t++) {
    data = (uint8_t *)(((uintptr_t)data + 0x3f) & ~0x3f);
    ei[t].precomp->data = data;
    data += size[t][0][2];
    if (split) {
      data = (uint8_t *)(((uintptr_t)data + 0x3f) & ~0x3f);
      ei[num + t].precomp->data = data;
      data += size[t][1][2];
    }
  }

  if (type == DTM && be->has_pawns)
    PAWN(be)->dtm_switched =
      calc_key_from_pieces(PAWN(be)->ei[0].pieces, be->num) != be->key;

  return true;
}

static uint8_t *decompress_pairs(struct PairsData *d, size_t idx)
{
  if (!d->idxbits)
    return d->const_val;

  uint32_t mainidx = idx >> d->idxbits;
  int litidx = (idx & (((size_t)1 << d->idxbits) - 1)) - ((size_t)1 << (d->idxbits - 1));
  uint32_t block;
  memcpy(&block, d->indextable + 6 * mainidx, sizeof(block));
  block = from_le_u32(block);

  uint16_t idxOffset = *(uint16_t *)(d->indextable + 6 * mainidx + 4);
  idxOffset = from_le_u16(idxOffset);
  litidx += idxOffset;

  if (litidx < 0) {
    do {
      litidx += d->sizetable[--block] + 1;
    } while (litidx < 0);
  } else {
    while (litidx > d->sizetable[block])
      litidx -= d->sizetable[block++] + 1;
  }

  uint32_t *ptr = (uint32_t *)(d->data + ((size_t)block << d->blocksize));

  int m = d->min_len;
  uint16_t *offset = d->offset;
  base_t *base = d->base - m;
  uint8_t *symlen = d->symlen;
  int sym, bitcnt;

  uint64_t code = from_be_u64(*(uint64_t *)ptr);

  ptr += 2;
  bitcnt = 0; // number of "empty bits" in code
  for (;;) {
    int l = m;
    while (code < base[l]) l++;
    sym = from_le_u16(offset[l]);
    sym += (code - base[l]) >> (64 - l);
    if (litidx < (int)symlen[sym] + 1) break;
    litidx -= (int)symlen[sym] + 1;
    code <<= l;
    bitcnt += l;
    if (bitcnt >= 32) {
      bitcnt -= 32;
      uint32_t tmp = from_be_u32(*ptr++);
      code |= (uint64_t)tmp << bitcnt;
     }
   }

  uint8_t *sympat = d->sympat;
  while (symlen[sym] != 0) {
    uint8_t *w = sympat + (3 * sym);
    int s1 = ((w[1] & 0xf) << 8) | w[0];
    if (litidx < (int)symlen[s1] + 1)
      sym = s1;
    else {
      litidx -= (int)symlen[s1] + 1;
      sym = (w[2] << 4) | (w[1] >> 4);
    }
  }

  return &sympat[3 * sym];
}

static void free_tb_entry(struct BaseEntry *be)
{
  for (int type = 0; type < 3; type++) {
    if (atomic_load_explicit(&be->ready[type], memory_order_relaxed)) {
      unmap_file(be->data[type], be->mapping[type]);
      int num = num_tables(be, type);
      struct EncInfo *ei = first_ei(be, type);
      for (int t = 0; t < num; t++) {
        free(ei[t].precomp);
        if (type != DTZ)
          free(ei[num + t].precomp);
      }
      atomic_store_explicit(&be->ready[type], false, memory_order_relaxed);
    }
  }
}

static int wdl_to_map[5] = { 1, 3, 0, 2, 0 };
static uint8_t pa_flags[5] = { 8, 0, 0, 0, 4 };
