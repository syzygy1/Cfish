#ifndef NUMA_H
#define NUMA_H
#include "types.h"

#ifdef NUMA
#ifndef _WIN32
#include <numa.h>
#else
#include <windows.h>
#endif

extern bool numaAvail;
void numa_init(void);
void numa_exit(void);
void read_numa_nodes(char *str);
struct bitmask *numa_thread_to_node(int idx);
int bind_thread_to_numa_node(int idx);

#ifndef _WIN32
typedef struct bitmask *NodeMask;
#define masks_equal numa_bitmask_equal
#else
typedef int NodeMask;
#define masks_equal(a,b) 1
void *numa_alloc(size_t size);
void numa_free(void *ptr, size_t size);
void numa_interleave_memory(void *ptr, size_t size, ULONGLONG mask);
#endif

#else

typedef int NodeMask;
#define masks_equal(a,b) 1
#define numa_alloc(size) calloc(size, 1)
#define numa_interleave_memory(a, b, c) do {} while (0)
#define numa_free(ptr, size) free(ptr)
#define bind_thread_to_numa_node(a) 0

#endif

#endif
