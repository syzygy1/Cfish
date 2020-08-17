#ifndef CONFIG_H
#define CONFIG_H

//#define LONG_MATES
#define PER_THREAD_CMH

#ifdef USE_PEXT
//#define BMI2_PLAIN
#define BMI2_FANCY
#else
//#define MAGIC_BLACK
#define MAGIC_PLAIN
//#define MAGIC_FANCY
//#define AVX2_BITBOARD
#endif

#endif
