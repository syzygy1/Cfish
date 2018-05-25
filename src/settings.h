#ifndef SETTINGS_H
#define SETTINGS_H

#include "numa.h"

struct settings {
  NodeMask mask;
  size_t ttSize;
  size_t numThreads;
  bool numaEnabled;
  bool largePages;
  bool clear;
};

extern struct settings settings, delayedSettings;

void process_delayed_settings(void);

#endif

