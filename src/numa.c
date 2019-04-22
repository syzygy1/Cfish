#ifdef NUMA

#ifndef _WIN32
#include <numa.h>
#else
#define _WIN32_WINNT 0x0600
#include <windows.h>
#endif
#include <stdio.h>

#include "misc.h"
#include "settings.h"
#include "types.h"

static int numNodes;
static int *numPhysicalCores;
bool numaAvail;

#ifndef _WIN32
static struct bitmask **nodeMask;

// Override libnuma's numa_warn()
void numa_warn(int num, char *fmt, ...)
{
  (void)num, (void)fmt;
}

void numa_init(void)
{
  FILE *F;

  if (numa_available() == -1 || numa_max_node() == 0) {
    numaAvail = false;
    settings.numaEnabled = delayedSettings.numaEnabled = false;
    return;
  }

  numaAvail = true;
  numNodes = numa_max_node() + 1;
  delayedSettings.mask = numa_allocate_nodemask();

  // Determine number of logical and physical cores per node
  numPhysicalCores = malloc(numNodes * sizeof(int));
  nodeMask = malloc(numNodes * sizeof(struct bitmask *));
  struct bitmask *cpuMask = numa_allocate_cpumask();
  int numCpus = numa_num_configured_cpus();
  char name[96];
  char *line = NULL;
  size_t len = 0;
  for (int node = 0; node < numNodes; node++) {
    nodeMask[node] = numa_allocate_nodemask();
    numa_bitmask_setbit(nodeMask[node], node);
    numa_bitmask_setbit(delayedSettings.mask, node);
    numPhysicalCores[node] = 0;
    numa_node_to_cpus(node, cpuMask);
    for (int cpu = 0; cpu < numCpus; cpu++)
      if (numa_bitmask_isbitset(cpuMask, cpu)) {
        // Find out about the thread_siblings of this cpu
        sprintf(name,
                "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list",
                cpu);
        F = fopen(name, "r");
        if (F && getline(&line, &len, F) > 0)
          if (atoi(line) == cpu)
            numPhysicalCores[node]++;
        if (F) fclose(F);
      }
  }
  numa_bitmask_free(cpuMask);
  if (line) free(line);

  delayedSettings.numaEnabled = true;
  settings.numaEnabled = false;
  settings.mask = numa_allocate_nodemask();
}

void numa_exit(void)
{
  if (!numaAvail)
    return;

  for (int node = 0; node < numNodes; node++)
    free(nodeMask[node]);
  free(nodeMask);
  free(numPhysicalCores);
  numa_bitmask_free(delayedSettings.mask);
  numa_bitmask_free(settings.mask);
}

void read_numa_nodes(char *str)
{
  struct bitmask *mask = NULL;

  if (!numaAvail) {
    printf("info string NUMA not supported by OS.\n");
  }
  else if (strcmp(str, "off") == 0) {
    delayedSettings.numaEnabled = false;
    printf("info string NUMA disabled.\n");
  }
  else if (strcmp(str, "on") == 0) {
    delayedSettings.numaEnabled = true;
    printf("info string NUMA enabled.\n");
  }
  else if (!(mask = numa_parse_nodestring(str))) {
    printf("info string Invalid specification of NUMA nodes.\n");
  }
  else if (numa_bitmask_equal(mask, numa_no_nodes_ptr)) {
    printf("info string NUMA disabled.\n");
    delayedSettings.numaEnabled = false;
  }
  else {
    printf("info string NUMA enabled.\n");
    delayedSettings.numaEnabled = true;
    copy_bitmask_to_bitmask(mask, delayedSettings.mask);
  }
  fflush(stdout);

  if (mask)
    numa_bitmask_free(mask);
}

int bind_thread_to_numa_node(int threadIdx)
{
  int idx = threadIdx;
  int node, k;

  // First assign threads to all physical cores of the first node, then
  // to all physical cores of the second node, etc.
  for (node = 0, k= 0; node < numNodes; node++)
    if (numa_bitmask_isbitset(settings.mask, node)) {
      if (idx < numPhysicalCores[node])
        break;
      idx -= numPhysicalCores[node];
      k++;
    }

  // Then assign threads round-robin
  if (node == numNodes) {
    idx %= k;
    for (node = 0; node < numNodes; node++)
      if (numa_bitmask_isbitset(settings.mask, node)) {
        if (idx == 0)
          break;
        idx--;
      }
  }

  printf("info string Binding thread %d to node %d.\n", threadIdx, node);
  fflush(stdout);
  numa_bind(nodeMask[node]);

  return node;
}

#else /* NUMA on Windows */

typedef BOOL (WINAPI *GLPIEX)(LOGICAL_PROCESSOR_RELATIONSHIP,
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, PDWORD);
typedef BOOL (WINAPI *STGA)(HANDLE, const GROUP_AFFINITY *, PGROUP_AFFINITY);
typedef LPVOID (WINAPI *VAEN)(HANDLE, LPVOID, SIZE_T, DWORD, DWORD, DWORD);

static GLPIEX impGetLogicalProcessorInformationEx;
static STGA   impSetThreadGroupAffinity;
static VAEN   impVirtualAllocExNuma;

static int numNodes;
static int *nodeNumber;
static GROUP_AFFINITY *nodeGroupMask;
static ULONGLONG *nodeMask = NULL;
static int *numPhysicalCores;

void numa_init(void)
{
  numaAvail = true;
  numNodes = 0;
  int maxNodes = 16;
  nodeNumber = malloc(maxNodes * sizeof(int));
 
  DWORD len = 0;
  DWORD offset = 0;

  impGetLogicalProcessorInformationEx =
    (GLPIEX)(void (*)(void))GetProcAddress(GetModuleHandle("kernel32.dll"),
        "GetLogicalProcessorInformationEx");
  impSetThreadGroupAffinity =
    (STGA)(void (*)(void))GetProcAddress(GetModuleHandle("kernel32.dll"),
        "SetThreadGroupAffinity");
  impVirtualAllocExNuma =
    (VAEN)(void (*)(void))GetProcAddress(GetModuleHandle("kernel32.dll"),
        "VirtualAllocExNuma");

  if (impGetLogicalProcessorInformationEx && impSetThreadGroupAffinity) {
    // Use windows processor groups

    // Get array of node and core data
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *buffer = NULL;
    while (1) {
      if (impGetLogicalProcessorInformationEx(RelationAll, buffer, &len))
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

    // First get nodes
    nodeGroupMask = malloc(maxNodes * sizeof(GROUP_AFFINITY));
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *ptr = buffer;
    while (offset < len && offset + ptr->Size <= len) {
      if (ptr->Relationship == RelationNumaNode) {
        if (numNodes == maxNodes) {
          maxNodes += 16;
          nodeNumber = realloc(nodeNumber, maxNodes * sizeof(int));
          nodeGroupMask = realloc(nodeGroupMask,
                                    maxNodes * sizeof(GROUP_AFFINITY));
        }
        nodeNumber[numNodes] = ptr->NumaNode.NodeNumber;
        nodeGroupMask[numNodes] = ptr->NumaNode.GroupMask;
        numNodes++;
      }
      offset += ptr->Size;
      ptr = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)(((char *)ptr) + ptr->Size);
    }

    // Then count cores in each node
    numPhysicalCores = calloc(numNodes, sizeof(int));
    ptr = buffer;
    offset = 0;
    while (offset < len && offset + ptr->Size <= len) {
      if (ptr->Relationship == RelationProcessorCore) {
        // Loop through nodes to find one with matching group number
        // and intersecting mask
        for (int i = 0; i < numNodes; i++)
          if (   nodeGroupMask[i].Group == ptr->Processor.GroupMask[0].Group
              && (nodeGroupMask[i].Mask & ptr->Processor.GroupMask[0].Mask))
            numPhysicalCores[i]++;
      }
      offset += ptr->Size;
      ptr = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)(((char*)ptr) + ptr->Size);
    }
    free(buffer);

  } else {
    // Use windows but not its processor groups

    // Get array of node and core data
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

    // First get nodes
    nodeMask = malloc(maxNodes * sizeof(ULONGLONG));
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION *ptr = buffer;
    while (offset + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION) <= len) {
      if (ptr->Relationship == RelationNumaNode) {
        if (numNodes == maxNodes) {
          maxNodes += 16;
          nodeNumber = realloc(nodeNumber, maxNodes * sizeof(int));
          nodeMask = realloc(nodeMask, maxNodes * sizeof(ULONGLONG));
        }
        nodeNumber[numNodes] = ptr->NumaNode.NodeNumber;
        nodeMask[numNodes] = ptr->ProcessorMask;
        numNodes++;
      }
      offset += sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
      ptr++;
    }

    // Then count cores in each node
    ptr = buffer;
    offset = 0;
    while (offset + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION) <= len) {
      if (ptr->Relationship == RelationProcessorCore) {
        // Loop through nodes to find one with intersecting mask
        for (int i = 0; i < numNodes; i++)
          if (nodeMask[i] & ptr->ProcessorMask)
            numPhysicalCores[i]++;
      }
      offset += sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
      ptr++;
    }
    free(buffer);
  }

  if (numNodes <= 1) {
    numaAvail = false;
    settings.numaEnabled = delayedSettings.numaEnabled = false;
  }
}

void numa_exit(void)
{
  if (!numaAvail)
    return;

  free(nodeNumber);
  if (!nodeMask)
    free(nodeGroupMask);
  else
    free(nodeMask);
  free(numPhysicalCores);
}

void read_numa_nodes(char *str)
{
  if (!numaAvail) {
    printf("info string NUMA not supported by OS.\n");
  }
  else if (strcmp(str, "off") == 0) {
    delayedSettings.numaEnabled = false;
    printf("info string NUMA disabled.\n");
  }
  else if (strcmp(str, "on") == 0 || strcmp(str, "all") == 0) {
    delayedSettings.numaEnabled = true;
    printf("info string NUMA enabled.\n");
  }
#if 0
  else if (!(mask = numa_parse_nodestring(str))) {
    printf("info string Invalid specification of NUMA nodes.\n");
  }
  else if (numa_bitmask_equal(mask, numa_no_nodes_ptr)) {
    printf("info string NUMA disabled.\n");
    delayedSettings.numaEnabled = false;
  }
  else {
    printf("info string NUMA enabled.\n");
    delayedSettings.numaEnabled = true;
    copy_bitmask_to_bitmask(mask, delayedSettings.mask);
  }
#else
  else
    printf("info string Invalid argument.\n");
#endif
  fflush(stdout);
}

int bind_thread_to_numa_node(int threadIdx)
{
  int idx = threadIdx;
  int node;

  // First assign threads to all physical cores of the first node, then
  // to all physical cores of the second node, etc.
  for (node = 0; node < numNodes; node++) {
    if (idx < numPhysicalCores[node])
      break;
    idx -= numPhysicalCores[node];
  }

  // Then assign threads round-robin
  if (node == numNodes)
    node = idx % numNodes;

  if (!nodeMask) {
    GROUP_AFFINITY aff;
    memset(&aff, 0, sizeof(aff));
    aff.Group = nodeGroupMask[node].Group;
    aff.Mask = nodeGroupMask[node].Mask;
    printf("info string Binding thread %d to node %d in group %d.\n",
           threadIdx, nodeNumber[node], aff.Group);
    if (!impSetThreadGroupAffinity(GetCurrentThread(), &aff, NULL))
      printf("info string error code = %d\n", (int)GetLastError());
  } else {
    printf("info string Binding thread %d to node %d.\n", threadIdx, node);
    if (!SetThreadAffinityMask(GetCurrentThread(), nodeMask[node]))
      printf("info string error code = %d\n", (int)GetLastError());
  }
  fflush(stdout);

  return node;
}

void *numa_alloc(size_t size)
{
  if (impVirtualAllocExNuma) {
    unsigned char numNode;
    GetNumaProcessorNode(GetCurrentProcessorNumber(), &numNode);
    return impVirtualAllocExNuma(GetCurrentProcess(), NULL, size,
                                  MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE,
                                  numNode);
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
