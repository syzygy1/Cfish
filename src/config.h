#ifndef CONFIG_H
#define CONFIG_H

#define PEDANTIC

#ifdef USE_PEXT
#define BMI2_PLAIN
//#define BMI2_FANCY
#else
#define MAGIC_PLAIN
//#define MAGIC_FANCY
#endif

#endif

