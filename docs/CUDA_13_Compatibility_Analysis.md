# CUDA 13 Compatibility Analysis - Phase 1 Findings

**Date:** December 15, 2025
**Analysis:** Gorgonia CUDA Upgrade Compatibility

## Executive Summary

**Critical Discovery:** Your system has CUDA 13.1 installed (not CUDA 11.4 as initially assumed). The current gorgonia/cu package (all released versions v0.8.0 through v0.10.0-working) **does not support CUDA 13** or **cuDNN 9**, which introduces breaking API changes.

## System Configuration

### Actual Environment
```
CUDA Toolkit:    13.1.80 (nvcc version)
CUDA Runtime:    13.0.2
cuDNN:           9.16.0
Driver:          580.95.05 (from version.json)
GPU:             NVIDIA GeForce (consumer GPU)
Install Path:    /usr/local/cuda-13.0/
```

### nvidia-smi Output (Misleading)
```
Driver Version:  470.256.02
CUDA Version:    11.4
```
**Note:** This shows backward compatibility version, not actual toolkit version.

## Compatibility Testing Results

### cu Package Version Testing

| Version | CUDA Target | Test Result | Issues |
|---------|-------------|-------------|---------|
| v0.9.4 | CUDA 11 | ❌ FAILED | Missing CU_TARGET_COMPUTE_20, cuDNN header issues |
| v0.9.5 | CUDA 12 (initial) | ❌ FAILED | API signature mismatches with CUDA 13 |
| v0.9.6 | CUDA 12 (Windows) | ❌ FAILED | API signature mismatches with CUDA 13 |
| v0.10.0-working | Development | ❌ FAILED | cuDNN 9 incompatibility, missing COMPUTE_20 |

### Specific Incompatibilities

#### 1. cuDNN API Breaking Changes
**cuDNN 9 removed deprecated APIs** that cu package depends on:
```c
// REMOVED in cuDNN 9:
cudnnAlgorithmDescriptor_t
cudnnCreateAlgorithmDescriptor()
cudnnDestroyAlgorithmDescriptor()
cudnnSetAlgorithmDescriptor()
cudnnAlgorithm_t
```

**Error Example:**
```
error: unknown type name 'cudnnAlgorithm_t'
```

#### 2. CUDA Driver API Changes
**CUDA 13 changed function signatures:**

```go
// OLD (CUDA 11/12):
cuCtxCreate(ctx *CUcontext, flags uint, device CUdevice)

// NEW (CUDA 13):
cuCtxCreate(ctx *CUcontext, params *CUctxCreateParams_st, flags uint, device CUdevice)
```

```go
// OLD (CUDA 11/12):
cuMemPrefetchAsync(ptr CUdeviceptr, size size_t, device CUdevice, stream CUstream)

// NEW (CUDA 13):
cuMemPrefetchAsync(ptr CUdeviceptr, size size_t, location CUmemLocation_st, flags uint, stream *CUstream_st)
```

#### 3. Removed Compute Capabilities
**CUDA 13 removed support for:**
- `CU_TARGET_COMPUTE_20` (Fermi)
- `CU_TARGET_COMPUTE_21` (Fermi)

These are still referenced in cu package code.

## Root Cause Analysis

### Why Incompatibility Exists

1. **CUDA 13 Released Recently:** NVIDIA CUDA 13.1 was released in November 2025 - just weeks ago
2. **Major Breaking Changes:** CUDA 13 is described as "the largest and most comprehensive update to the CUDA platform since it was invented"
3. **cu Package Lag:** gorgonia/cu development hasn't caught up to CUDA 13 yet
4. **cuDNN Major Version:** cuDNN 9 introduced breaking API changes from cuDNN 8

### Timeline
- **CUDA 12:** Supported by cu v0.9.5-v0.9.6 (2023)
- **CUDA 13.1:** Released November 2025
- **cu Package:** Last release v0.9.6 (June 2023), development v0.10.0-working status unknown

## Options & Recommendations

### Option 1: Downgrade CUDA Toolkit ⭐ RECOMMENDED
**Install CUDA 11.x or 12.x alongside CUDA 13**

**Pros:**
- cu v0.9.4 (CUDA 11) or v0.9.6 (CUDA 12) will work
- Existing Gorgonia code will work without modifications
- Proven stable combination
- Multiple CUDA versions can coexist

**Cons:**
- Requires installing additional CUDA toolkit (~3-4 GB)
- Need to manage multiple CUDA versions
- Won't use latest CUDA 13 features

**Implementation Steps:**
1. Download CUDA 11.8 or CUDA 12.x from NVIDIA
2. Install to `/usr/local/cuda-11.8/` or `/usr/local/cuda-12.x/`
3. Install cuDNN 8.x for that CUDA version
4. Set environment variables to point to older CUDA:
   ```bash
   export CUDA_HOME=/usr/local/cuda-11.8
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
   export CGO_CFLAGS="-I$CUDA_HOME/include"
   export CGO_LDFLAGS="-L$CUDA_HOME/lib64"
   ```
5. Use cu v0.9.4 (CUDA 11) or v0.9.6 (CUDA 12)
6. Proceed with original upgrade plan

**Estimated Time:** 2-4 hours (download + install + test)

---

### Option 2: Patch/Fork cu Package
**Update cu package for CUDA 13/cuDNN 9 compatibility**

**Pros:**
- Use latest CUDA 13 features
- Long-term solution for community
- Could contribute back to upstream

**Cons:**
- Significant development effort (2-4 weeks)
- Requires deep CUDA/cuDNN expertise
- Need to maintain fork until upstream catches up
- Risk of introducing bugs

**Required Changes:**
1. Remove all cuDNN algorithm descriptor APIs
2. Update to cuDNN 9 alternative APIs
3. Update CUDA Driver API calls (cuCtxCreate, cuMemPrefetchAsync, etc.)
4. Remove references to COMPUTE_20, COMPUTE_21
5. Update compute capability list to CUDA 13 supported range (SM35+)
6. Test thoroughly with CUDA 13 + cuDNN 9

**Estimated Time:** 2-4 weeks for experienced CUDA developer

---

### Option 3: Wait for Official Support
**Monitor gorgonia/cu repository for CUDA 13 support**

**Pros:**
- No immediate effort required
- Official support will be tested and stable
- Community benefits

**Cons:**
- Unknown timeline (could be weeks to months)
- Blocks Gorgonia CUDA usage in meantime
- CUDA 13 is very new (Nov 2025)

**Actions:**
1. File GitHub issue at https://github.com/gorgonia/cu/issues
2. Mention CUDA 13.1 + cuDNN 9 incompatibility
3. Offer to test once support is added
4. Monitor for updates

**Estimated Time:** Unknown (weeks to months)

---

### Option 4: CPU-Only Mode (Temporary)
**Use Gorgonia without CUDA while awaiting support**

**Pros:**
- Works immediately
- No CUDA compatibility issues
- Good for development/testing

**Cons:**
- Much slower for large models (10-100x slower)
- Can't utilize GPU hardware
- Not viable for production training

**Implementation:**
- Build without `-tags=cuda`
- Use native Go operations
- Consider OpenBLAS for CPU optimization

---

## Revised Implementation Plan

### If Option 1 (Downgrade CUDA) - RECOMMENDED

**Phase 1A: Install CUDA 11.8 or 12.x (Week 1)**
1. Download CUDA 11.8.0 from NVIDIA archive
2. Install to `/usr/local/cuda-11.8/`
3. Download and install cuDNN 8.9.x for CUDA 11.8
4. Verify installation with `nvcc --version`
5. Set environment variables in `.bashrc` or project-specific

**Phase 1B: Update cu Package**
- For CUDA 11.8: Use cu v0.9.4
- For CUDA 12.x: Use cu v0.9.6

**Phase 2-6: Continue with Original Plan**
- Proceed with kernel compilation updates (compute capabilities 35-87)
- Add CUDA version checks to engine
- Run tests
- Create documentation

### If Option 2 (Patch cu Package)

**Phase 1A: Fork and Setup**
1. Fork https://github.com/gorgonia/cu
2. Set up development environment with CUDA 13 + cuDNN 9
3. Create feature branch `cuda-13-support`

**Phase 1B: API Updates**
1. Replace cuDNN algorithm descriptor usage
2. Update CUDA Driver API calls
3. Remove deprecated compute capabilities
4. Update tests

**Phase 1C: Testing**
1. Unit tests with CUDA 13
2. Integration tests with Gorgonia
3. Performance benchmarks

**Phase 2-6: Continue with Gorgonia Updates**
- Update go.mod to use forked cu package
- Proceed with kernel compilation and testing

## Decision Matrix

| Criteria | Option 1 (Downgrade) | Option 2 (Patch) | Option 3 (Wait) | Option 4 (CPU) |
|----------|---------------------|------------------|-----------------|----------------|
| Time to working state | 4 hours | 2-4 weeks | Unknown | Immediate |
| Development effort | Low | High | None | None |
| Long-term viability | Medium | High | High | Low |
| Uses CUDA 13 features | No | Yes | Yes | No |
| Risk level | Low | Medium | Low | None |
| GPU performance | Full | Full | N/A | N/A |
| **Recommendation** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐ |

## Recommended Next Steps

1. **Immediate (Today):**
   - Decide on Option 1 (downgrade) vs Option 2 (patch)
   - If Option 1: Download CUDA 11.8 or 12.x toolkit

2. **Short-term (This Week):**
   - Install chosen CUDA version + cuDNN 8.x
   - Update environment variables
   - Test basic CUDA build
   - Update go.mod with appropriate cu version

3. **Medium-term (Next 2 Weeks):**
   - Execute Phases 2-6 of original plan
   - Run comprehensive tests
   - Document configuration

4. **Long-term (Future):**
   - Monitor gorgonia/cu for CUDA 13 support
   - Plan migration to CUDA 13 when supported
   - Consider contributing patches back to cu package

## References

- [Gorgonia GitHub](https://github.com/gorgonia/gorgonia)
- [cu Package GitHub](https://github.com/gorgonia/cu)
- [NVIDIA CUDA 13.1 Blog Post](https://developer.nvidia.com/blog/nvidia-cuda-13-1-powers-next-gen-gpu-programming-with-nvidia-cuda-tile-and-performance-gains/)
- [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
- [cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive)

## Contact & Support

- File issues: https://github.com/gorgonia/cu/issues
- Gorgonia Slack: (check README for invite link)
- CUDA Developer Forums: https://forums.developer.nvidia.com/c/accelerated-computing/cuda/
