#include "numa.h"
#include "search.h"
#include "settings.h"
#include "thread.h"
#include "tt.h"
#include "types.h"

struct settings settings, delayedSettings;

// Process Hash, Threads, NUMA and LargePages settings.

void process_delayed_settings(void)
{
  bool ttChange = delayedSettings.ttSize != settings.ttSize;
  bool lpChange = delayedSettings.largePages != settings.largePages;
  bool numaChange =   (settings.numaEnabled != delayedSettings.numaEnabled)
                   || (   settings.numaEnabled
                       && !masks_equal(settings.mask, delayedSettings.mask));

#ifdef NUMA
  if (numaChange) {
    threads_set_number(0);
    settings.numThreads = 0;
#ifndef _WIN32
    if ((settings.numaEnabled = delayedSettings.numaEnabled))
      copy_bitmask_to_bitmask(delayedSettings.mask, settings.mask);
#endif
    settings.numaEnabled = delayedSettings.numaEnabled;
  }
#endif

  if (settings.numThreads != delayedSettings.numThreads) {
    settings.numThreads = delayedSettings.numThreads;
    threads_set_number(settings.numThreads);
  }

  if (numaChange || ttChange || lpChange) {
    tt_free();
    settings.largePages = delayedSettings.largePages;
    settings.ttSize = delayedSettings.ttSize;
    tt_allocate(settings.ttSize);
  }

  if (delayedSettings.clear) {
    delayedSettings.clear = false;
    search_clear();
  }
}
