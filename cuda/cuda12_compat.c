// CUDA 12.6 Compatibility Wrapper
//
// This file provides wrapper functions for CUDA Driver API functions that are
// referenced with _v2 suffixes in the CUDA headers but exported without the
// suffix in the actual library.
//
// We don't include cuda.h here to avoid macro conflicts.

// CUDA types (minimal declarations needed)
typedef struct CUstream_st *CUstream;
typedef struct CUgraphNode_st *CUgraphNode;
typedef struct CUgraph_st *CUgraph;
typedef unsigned long long CUdeviceptr;
typedef unsigned int cuuint32_t;
typedef int CUresult;

// Forward declare structures we need
struct CUDA_KERNEL_NODE_PARAMS;
typedef struct CUDA_KERNEL_NODE_PARAMS CUDA_KERNEL_NODE_PARAMS;

// Declare the actual functions exported by the CUDA library (without _v2 suffix)
extern CUresult cuStreamWaitValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags);
extern CUresult cuStreamWriteValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags);
extern CUresult cuGraphAddKernelNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, unsigned long numDependencies, const CUDA_KERNEL_NODE_PARAMS *nodeParams);

// Provide the missing _v2 symbols by wrapping the non-versioned ones
CUresult cuStreamWaitValue32_v2(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags) {
    return cuStreamWaitValue32(stream, addr, value, flags);
}

CUresult cuStreamWriteValue32_v2(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags) {
    return cuStreamWriteValue32(stream, addr, value, flags);
}

CUresult cuGraphAddKernelNode_v2(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, unsigned long numDependencies, const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
    return cuGraphAddKernelNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
}
