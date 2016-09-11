#include "numa.h"
#include "settings.h"
#include "thread.h"
#include "tt.h"
#include "types.h"

struct settings settings, delayed_settings;

// Process Hash, Threads, NUMA and LargePages settings.

void process_delayed_settings(void)
{
  int tt_change = delayed_settings.tt_size != settings.tt_size;
  int lp_change = delayed_settings.large_pages != settings.large_pages;
  int numa_change =   (settings.numa_enabled != delayed_settings.numa_enabled)
                   || (   settings.numa_enabled
                       && !masks_equal(settings.mask, delayed_settings.mask));

#ifdef NUMA
  if (numa_change) {
    threads_set_number(0);
    settings.num_threads = 0;
#ifndef __WIN32__
    if ((settings.numa_enabled = delayed_settings.numa_enabled))
      copy_bitmask_to_bitmask(delayed_settings.mask, settings.mask);
#endif
    settings.numa_enabled = delayed_settings.numa_enabled;
  }
#endif

  if (settings.num_threads != delayed_settings.num_threads) {
    settings.num_threads = delayed_settings.num_threads;
    threads_set_number(settings.num_threads);
  }

  if (numa_change || tt_change || lp_change) {
    tt_free();
    settings.large_pages = delayed_settings.large_pages;
    settings.tt_size = delayed_settings.tt_size;
    tt_allocate(settings.tt_size);
  }
}

