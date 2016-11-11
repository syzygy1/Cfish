#ifdef NUMA

#define _GNU_SOURCE

#ifndef __WIN32__
#include <numa.h>
#else
#define _WIN32_WINNT 0x0600
#include <windows.h>
#endif
#include <stdio.h>

#include "settings.h"
#include "types.h"

static int num_nodes;
static int *num_physical_cores;
int numa_avail;

#ifndef __WIN32__
static int *num_logical_cores;
static struct bitmask **nodemask;

// Override libnuma's numa_warn().
void numa_warn(int num, char *fmt, ...)
{
  (void)num, (void)fmt;
}

void numa_init(void)
{
  FILE *F;

  if (numa_available() == -1 || numa_max_node() == 0) {
    numa_avail = 0;
    settings.numa_enabled = delayed_settings.numa_enabled = 0;
    return;
  }

  numa_avail = 1;
  num_nodes = numa_max_node() + 1;
#if 0
  printf("numa nodes = %d\n", num_nodes);
  for (int node = 0; node < num_nodes; node++)
    if (numa_bitmask_isbitset(numa_all_nodes_ptr, node))
      printf("node %d is present.\n", node);
    else
      printf("node %d is absent.\n", node);
#endif

  // Determine number of logical and physical cores per node.
  num_physical_cores = malloc(num_nodes * sizeof(int));
  num_logical_cores = malloc(num_nodes * sizeof(int));
  nodemask = malloc(num_nodes * sizeof(struct bitmask *));
  struct bitmask *cpu_mask = numa_allocate_cpumask();
  int num_cpus = numa_num_configured_cpus();
  char name[96];
  char *line = NULL;
  size_t len = 0;
  for (int node = 0; node < num_nodes; node++) {
    nodemask[node] = numa_allocate_nodemask();
    numa_bitmask_setbit(nodemask[node], node);
    num_physical_cores[node] = 0;
    num_logical_cores[node] = 0;
    numa_node_to_cpus(node, cpu_mask);
    for (int cpu = 0; cpu < num_cpus; cpu++)
      if (numa_bitmask_isbitset(cpu_mask, cpu)) {
        num_logical_cores[node]++;
        // Find out about the thread_siblings of this cpu.
        sprintf(name,
                "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list",
                cpu);
        F = fopen(name, "r");
        if (F && getline(&line, &len, F) > 0)
          if (atoi(line) == cpu)
            num_physical_cores[node]++;
        if (F) fclose(F);
      }
  }
  numa_bitmask_free(cpu_mask);
  if (line) free(line);
#if 0
  for (int node = 0; node < num_nodes; node++)
    printf("node %d has %d logical and %d physical cpu cores.\n",
           node, num_logical_cores[node], num_physical_cores[node]);
#endif

  delayed_settings.numa_enabled = 1;
  settings.numa_enabled = 0;
  delayed_settings.mask = numa_allocate_nodemask();
  copy_bitmask_to_bitmask(numa_all_nodes_ptr, delayed_settings.mask);
  settings.mask = numa_allocate_nodemask();
}

void numa_exit(void)
{
  if (!numa_avail)
    return;

  for (int node = 0; node < num_nodes; node++)
    free(nodemask[node]);
  free(nodemask);
  free(num_physical_cores);
  free(num_logical_cores);
  numa_bitmask_free(delayed_settings.mask);
  numa_bitmask_free(settings.mask);
}

void read_numa_nodes(char *str)
{
  struct bitmask *mask = NULL;

  if (!numa_avail) {
    printf("info string NUMA not supported by OS.\n");
  }
  else if (strcmp(str, "off") == 0) {
    delayed_settings.numa_enabled = 0;
    printf("info string NUMA disabled.\n");
  }
  else if (strcmp(str, "on") == 0) {
    delayed_settings.numa_enabled = 1;
    printf("info string NUMA enabled.\n");
  }
  else if (!(mask = numa_parse_nodestring(str))) {
    printf("info string Invalid specification of NUMA nodes.\n");
  }
  else if (numa_bitmask_equal(mask, numa_no_nodes_ptr)) {
    printf("info string NUMA disabled.\n");
    delayed_settings.numa_enabled = 0;
  }
  else {
    printf("info string NUMA enabled.\n");
    delayed_settings.numa_enabled = 1;
    copy_bitmask_to_bitmask(mask, delayed_settings.mask);
  }
  fflush(stdout);

  if (mask)
    numa_bitmask_free(mask);
}

int bind_thread_to_numa_node(int thread_idx)
{
  int idx = thread_idx;
  int node, k;

  // First assign threads to all physical cores of the first node, then
  // to all physical cores of the second node, etc.
  for (node = 0, k= 0; node < num_nodes; node++)
    if (numa_bitmask_isbitset(settings.mask, node)) {
      if (idx < num_physical_cores[node])
        break;
      idx -= num_physical_cores[node];
      k++;
    }

  // Then assign threads round-robin.
  if (node == num_nodes) {
    idx %= k;
    for (node = 0; node < num_nodes; node++)
      if (numa_bitmask_isbitset(settings.mask, node)) {
        if (idx == 0)
          break;
        idx--;
      }
  }

  printf("info string Binding thread %d to node %d.\n", thread_idx, node);
  fflush(stdout);
  numa_bind(nodemask[node]);

  return node;
}

#else /* NUMA on Windows */

typedef BOOL (WINAPI *GLPIEX)(LOGICAL_PROCESSOR_RELATIONSHIP, PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, PDWORD);
typedef BOOL (WINAPI *STGA)(HANDLE, const GROUP_AFFINITY *, PGROUP_AFFINITY);
typedef LPVOID (WINAPI *VAEN)(HANDLE, LPVOID, SIZE_T, DWORD, DWORD, DWORD);

static GLPIEX imp_GetLogicalProcessorInformationEx;
static STGA   imp_SetThreadGroupAffinity;
static VAEN   imp_VirtualAllocExNuma;

static int num_nodes;
static int *node_number;
static GROUP_AFFINITY *node_group_mask;
static ULONGLONG *node_mask = NULL;
static int *num_physical_cores;
//static int *num_logical_cores;

void numa_init(void)
{
  numa_avail = 1;
  num_nodes = 0;
  int max_nodes = 16;
  node_number = malloc(max_nodes * sizeof(int));
  
  DWORD len = 0;
  DWORD offset = 0;

  imp_GetLogicalProcessorInformationEx =
                (GLPIEX)GetProcAddress(GetModuleHandle("kernel32.dll"),
                                       "GetLogicalProcessorInformationEx");
  imp_SetThreadGroupAffinity =
                (STGA)GetProcAddress(GetModuleHandle("kernel32.dll"),
                                     "SetThreadGroupAffinity");

  imp_VirtualAllocExNuma =
                (VAEN)GetProcAddress(GetModuleHandle("kernel32.dll"),
                                     "VirtualAllocExNuma");

  if (imp_GetLogicalProcessorInformationEx && imp_SetThreadGroupAffinity) {
    // use windows processor groups

    // get array of node and core data
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *buffer = NULL;
    while (1) {
      if (imp_GetLogicalProcessorInformationEx(RelationAll, buffer, &len))
        break;
      if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
        buffer = realloc(buffer, len);
        if (!buffer) {
          fprintf(stderr, "GetLogicalProcessorInformationEx malloc failed.\n");
          exit(EXIT_FAILURE);
        }
      } else {
        free(buffer);
        fprintf(stderr, "GetLogicalProcessorInformationEx failed.\n");
        exit(EXIT_FAILURE);
      }
    }

    // First get nodes.
    node_group_mask = malloc(max_nodes * sizeof(GROUP_AFFINITY));
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *ptr = buffer;
    while (ptr->Size > 0 && offset + ptr->Size <= len) {
      if (ptr->Relationship == RelationNumaNode) {
        if (num_nodes == max_nodes) {
          max_nodes += 16;
          node_number = realloc(node_number, max_nodes * sizeof(int));
          node_group_mask = realloc(node_group_mask,
                                    max_nodes * sizeof(GROUP_AFFINITY));
        }
        node_number[num_nodes] = ptr->NumaNode.NodeNumber;
        node_group_mask[num_nodes] = ptr->NumaNode.GroupMask;
        num_nodes++;
      }
      offset += ptr->Size;
      ptr = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)(((char *)ptr) + ptr->Size);        
    }

    // Then count cores in each node.
    num_physical_cores = calloc(num_nodes, sizeof(int));
//    num_logical_cores = calloc(num_nodes, sizeof(int));
    ptr = buffer;
    offset = 0;
    while (ptr->Size > 0 && offset + ptr->Size <= len) {
      if (ptr->Relationship == RelationProcessorCore) {
        // Loop through nodes to find one with matching group number
        // and intersecting mask.
        for (int i = 0; i < num_nodes; i++)
          if (   node_group_mask[i].Group == ptr->Processor.GroupMask[0].Group
              && (node_group_mask[i].Mask & ptr->Processor.GroupMask[0].Mask))
            num_physical_cores[i]++;
      }
      offset += ptr->Size;
      ptr = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)(((char*)ptr) + ptr->Size);        
    }
    free(buffer);

  } else {
    // Use windows but not its processor groups.

    // Get array of node and core data.
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION *buffer = NULL;
    while (1) {
      if (GetLogicalProcessorInformation(buffer, &len))
        break;
      if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
        buffer = realloc(buffer, len);
        if (!buffer) {
          fprintf(stderr, "GetLogicalProcessorInformation malloc failed.\n");
          exit(EXIT_FAILURE);
        }
      } else {
        free(buffer);
        fprintf(stderr, "GetLogicalProcessorInformation failed.\n");
        exit(EXIT_FAILURE);
      }
    }

    // First get nodes.
    node_mask = malloc(max_nodes * sizeof(ULONGLONG));
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION *ptr = buffer;
    while (offset + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION) <= len) {
      if (ptr->Relationship == RelationNumaNode) {
        if (num_nodes == max_nodes) {
          max_nodes += 16;
          node_number = realloc(node_number, max_nodes * sizeof(int));
          node_mask = realloc(node_mask, max_nodes * sizeof(ULONGLONG));
        }
        node_number[num_nodes] = ptr->NumaNode.NodeNumber;
        node_mask[num_nodes] = ptr->ProcessorMask;
        num_nodes++;
      }
      offset += sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
      ptr++;
    }

    // Then count cores in each node.
    ptr = buffer;
    offset = 0;
    while (offset + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION) <= len) {
      if (ptr->Relationship == RelationProcessorCore) {
        // Loop through nodes to find one with intersecting mask.
        for (int i = 0; i < num_nodes; i++)
          if (node_mask[i] & ptr->ProcessorMask)
            num_physical_cores[i]++;
      }
      offset += sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
      ptr++;
    }
    free(buffer);
  }

  if (num_nodes <= 1) {
    numa_avail = 0;
    settings.numa_enabled = delayed_settings.numa_enabled = 0;
  }
}

void numa_exit(void)
{
  if (!numa_avail)
    return;

  free(node_number);
  if (!node_mask)
    free(node_group_mask);
  else
    free(node_mask);
  free(num_physical_cores);
//  free(num_logical_cores);
}

void read_numa_nodes(char *str)
{
  if (!numa_avail) {
    printf("info string NUMA not supported by OS.\n");
  }
  else if (strcmp(str, "off") == 0) {
    delayed_settings.numa_enabled = 0;
    printf("info string NUMA disabled.\n");
  }
  else if (strcmp(str, "on") == 0 || strcmp(str, "all") == 0) {
    delayed_settings.numa_enabled = 1;
    printf("info string NUMA enabled.\n");
  }
#if 0
  else if (!(mask = numa_parse_nodestring(str))) {
    printf("info string Invalid specification of NUMA nodes.\n");
  }
  else if (numa_bitmask_equal(mask, numa_no_nodes_ptr)) {
    printf("info string NUMA disabled.\n");
    delayed_settings.numa_enabled = 0;
  }
  else {
    printf("info string NUMA enabled.\n");
    delayed_settings.numa_enabled = 1;
    copy_bitmask_to_bitmask(mask, delayed_settings.mask);
  }
#else
  else
    printf("info string Invalid argument.\n");
#endif
  fflush(stdout);
}

int bind_thread_to_numa_node(int thread_idx)
{
  int idx = thread_idx;
  int node;

  // First assign threads to all physical cores of the first node, then
  // to all physical cores of the second node, etc.
  for (node = 0; node < num_nodes; node++) {
    if (idx < num_physical_cores[node])
      break;
    idx -= num_physical_cores[node];
  }

  // Then assign threads round-robin.
  if (node == num_nodes)
    node = idx % num_nodes;

  if (!node_mask) {
    GROUP_AFFINITY aff;
    memset(&aff, 0, sizeof(aff));
    aff.Group = node_group_mask[node].Group;
    aff.Mask = node_group_mask[node].Mask;
    printf("info string Binding thread %d to node %d in group %d.\n",
           thread_idx, node_number[node], aff.Group);
    if (!imp_SetThreadGroupAffinity(GetCurrentThread(), &aff, NULL))
      printf("info string error code = %d\n", (int)GetLastError());
  } else {
    printf("info string Binding thread %d to node %d.\n", thread_idx, node);
    if (!SetThreadAffinityMask(GetCurrentThread(), node_mask[node]))
      printf("info string error code = %d\n", (int)GetLastError());
  }
  fflush(stdout);

  return node;
}

void *numa_alloc(size_t size)
{
  if (imp_VirtualAllocExNuma) {
    unsigned char num_node;
    GetNumaProcessorNode(GetCurrentProcessorNumber(), &num_node);
    return VirtualAllocExNuma(GetCurrentProcess(), NULL, size,
                              MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE,
                              num_node);
  }
  return VirtualAllocEx(GetCurrentProcess(), NULL, size,
                        MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
}

void numa_free(void *ptr, size_t size)
{
  (void)size;
  VirtualFree(ptr, 0, MEM_RELEASE);
}

void numa_interleave_memory(void *ptr, size_t size, ULONGLONG mask)
{
  (void)ptr;
  (void)size;
  (void)mask;
}

#endif

#else

typedef int make_iso_compilers_happy;

#endif

