Bitboard  RookMasks  [64];
Bitboard  RookMagics [64];
Bitboard *RookAttacks[64];

Bitboard  BishopMasks  [64];
Bitboard  BishopMagics [64];
Bitboard *BishopAttacks[64];

static Bitboard AttacksTable[87988];

// Black magics found by Volker Annuss and Niklas Fiekas
// http://talkchess.com/forum/viewtopic.php?t=64790

struct MagicInit {
  Bitboard magic;
  int index;
};

static struct MagicInit bishop_init[64] = {
  { 0xa7020080601803d8ull, 60984 },
  { 0x13802040400801f1ull, 66046 },
  { 0x0a0080181001f60cull, 32910 },
  { 0x1840802004238008ull, 16369 },
  { 0xc03fe00100000000ull, 42115 },
  { 0x24c00bffff400000ull,   835 },
  { 0x0808101f40007f04ull, 18910 },
  { 0x100808201ec00080ull, 25911 },
  { 0xffa2feffbfefb7ffull, 63301 },
  { 0x083e3ee040080801ull, 16063 },
  { 0xc0800080181001f8ull, 17481 },
  { 0x0440007fe0031000ull, 59361 },
  { 0x2010007ffc000000ull, 18735 },
  { 0x1079ffe000ff8000ull, 61249 },
  { 0x3c0708101f400080ull, 68938 },
  { 0x080614080fa00040ull, 61791 },
  { 0x7ffe7fff817fcff9ull, 21893 },
  { 0x7ffebfffa01027fdull, 62068 },
  { 0x53018080c00f4001ull, 19829 },
  { 0x407e0001000ffb8aull, 26091 },
  { 0x201fe000fff80010ull, 15815 },
  { 0xffdfefffde39ffefull, 16419 },
  { 0xcc8808000fbf8002ull, 59777 },
  { 0x7ff7fbfff8203fffull, 16288 },
  { 0x8800013e8300c030ull, 33235 },
  { 0x0420009701806018ull, 15459 },
  { 0x7ffeff7f7f01f7fdull, 15863 },
  { 0x8700303010c0c006ull, 75555 },
  { 0xc800181810606000ull, 79445 },
  { 0x20002038001c8010ull, 15917 },
  { 0x087ff038000fc001ull,  8512 },
  { 0x00080c0c00083007ull, 73069 },
  { 0x00000080fc82c040ull, 16078 },
  { 0x000000407e416020ull, 19168 },
  { 0x00600203f8008020ull, 11056 },
  { 0xd003fefe04404080ull, 62544 },
  { 0xa00020c018003088ull, 80477 },
  { 0x7fbffe700bffe800ull, 75049 },
  { 0x107ff00fe4000f90ull, 32947 },
  { 0x7f8fffcff1d007f8ull, 59172 },
  { 0x0000004100f88080ull, 55845 },
  { 0x00000020807c4040ull, 61806 },
  { 0x00000041018700c0ull, 73601 },
  { 0x0010000080fc4080ull, 15546 },
  { 0x1000003c80180030ull, 45243 },
  { 0xc10000df80280050ull, 20333 },
  { 0xffffffbfeff80fdcull, 33402 },
  { 0x000000101003f812ull, 25917 },
  { 0x0800001f40808200ull, 32875 },
  { 0x084000101f3fd208ull,  4639 },
  { 0x080000000f808081ull, 17077 },
  { 0x0004000008003f80ull, 62324 },
  { 0x08000001001fe040ull, 18159 },
  { 0x72dd000040900a00ull, 61436 },
  { 0xfffffeffbfeff81dull, 57073 },
  { 0xcd8000200febf209ull, 61025 },
  { 0x100000101ec10082ull, 81259 },
  { 0x7fbaffffefe0c02full, 64083 },
  { 0x7f83fffffff07f7full, 56114 },
  { 0xfff1fffffff7ffc1ull, 57058 },
  { 0x0878040000ffe01full, 58912 },
  { 0x945e388000801012ull, 22194 },
  { 0x0840800080200fdaull, 70880 },
  { 0x100000c05f582008ull, 11140 }
};

static struct MagicInit rook_init[64] = {
  { 0x80280013ff84ffffull, 10890 },
  { 0x5ffbfefdfef67fffull, 50579 },
  { 0xffeffaffeffdffffull, 62020 },
  { 0x003000900300008aull, 67322 },
  { 0x0050028010500023ull, 80251 },
  { 0x0020012120a00020ull, 58503 },
  { 0x0030006000c00030ull, 51175 },
  { 0x0058005806b00002ull, 83130 },
  { 0x7fbff7fbfbeafffcull, 50430 },
  { 0x0000140081050002ull, 21613 },
  { 0x0000180043800048ull, 72625 },
  { 0x7fffe800021fffb8ull, 80755 },
  { 0xffffcffe7fcfffafull, 69753 },
  { 0x00001800c0180060ull, 26973 },
  { 0x4f8018005fd00018ull, 84972 },
  { 0x0000180030620018ull, 31958 },
  { 0x00300018010c0003ull, 69272 },
  { 0x0003000c0085ffffull, 48372 },
  { 0xfffdfff7fbfefff7ull, 65477 },
  { 0x7fc1ffdffc001fffull, 43972 },
  { 0xfffeffdffdffdfffull, 57154 },
  { 0x7c108007befff81full, 53521 },
  { 0x20408007bfe00810ull, 30534 },
  { 0x0400800558604100ull, 16548 },
  { 0x0040200010080008ull, 46407 },
  { 0x0010020008040004ull, 11841 },
  { 0xfffdfefff7fbfff7ull, 21112 },
  { 0xfebf7dfff8fefff9ull, 44214 },
  { 0xc00000ffe001ffe0ull, 57925 },
  { 0x4af01f00078007c3ull, 29574 },
  { 0xbffbfafffb683f7full, 17309 },
  { 0x0807f67ffa102040ull, 40143 },
  { 0x200008e800300030ull, 64659 },
  { 0x0000008780180018ull, 70469 },
  { 0x0000010300180018ull, 62917 },
  { 0x4000008180180018ull, 60997 },
  { 0x008080310005fffaull, 18554 },
  { 0x4000188100060006ull, 14385 },
  { 0xffffff7fffbfbfffull,     0 },
  { 0x0000802000200040ull, 38091 },
  { 0x20000202ec002800ull, 25122 },
  { 0xfffff9ff7cfff3ffull, 60083 },
  { 0x000000404b801800ull, 72209 },
  { 0x2000002fe03fd000ull, 67875 },
  { 0xffffff6ffe7fcffdull, 56290 },
  { 0xbff7efffbfc00fffull, 43807 },
  { 0x000000100800a804ull, 73365 },
  { 0x6054000a58005805ull, 76398 },
  { 0x0829000101150028ull, 20024 },
  { 0x00000085008a0014ull,  9513 },
  { 0x8000002b00408028ull, 24324 },
  { 0x4000002040790028ull, 22996 },
  { 0x7800002010288028ull, 23213 },
  { 0x0000001800e08018ull, 56002 },
  { 0xa3a80003f3a40048ull, 22809 },
  { 0x2003d80000500028ull, 44545 },
  { 0xfffff37eefefdfbeull, 36072 },
  { 0x40000280090013c1ull,  4750 },
  { 0xbf7ffeffbffaf71full,  6014 },
  { 0xfffdffff777b7d6eull, 36054 },
  { 0x48300007e8080c02ull, 78538 },
  { 0xafe0000fff780402ull, 28745 },
  { 0xee73fffbffbb77feull,  8555 },
  { 0x0002000308482882ull,  1009 }
};

typedef unsigned (Fn)(Square, Bitboard);

static void init_magics(struct MagicInit *magic_init, Bitboard *attacks[],
                        Bitboard magics[], Bitboard masks[], int deltas[],
                        Fn index)
{
  Bitboard edges, b, m;

  for (int s = 0; s < 64; s++) {
    magics[s] = magic_init[s].magic;
    attacks[s] = &AttacksTable[magic_init[s].index];

    // Board edges are not considered in the relevant occupancies
    edges = ((Rank1BB | Rank8BB) & ~rank_bb_s(s)) | ((FileABB | FileHBB) & ~file_bb_s(s));

    masks[s] = ~(m = sliding_attack(deltas, s, 0) & ~edges);

    // Use Carry-Rippler trick to enumerate all subsets of m and
    // fill the attacks table.
    b = 0;
    do {
      attacks[s][index(s, b)] = sliding_attack(deltas, s, b);
      b = (b - m) & m;
    } while (b);
  }
}

static void init_sliding_attacks(void)
{
  init_magics(rook_init, RookAttacks, RookMagics, RookMasks,
              RookDirs, magic_index_rook);
  init_magics(bishop_init, BishopAttacks, BishopMagics, BishopMasks,
              BishopDirs, magic_index_bishop);
}

