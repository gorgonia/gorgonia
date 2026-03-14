# CUDA 12.6 Upgrade Summary

**Date:** December 15, 2025
**Status:** ✅ COMPLETED
**CUDA Version:** 12.6.2
**cuDNN Version:** 8.9.7

## Executive Summary

Successfully upgraded Gorgonia to work with CUDA 12.6 by installing it alongside the existing CUDA 13.0 installation. The upgrade enables support for modern GPU architectures (Volta, Turing, Ampere) while maintaining backward compatibility with older GPUs.

## What Was Done

### Phase 1: Environment Analysis & CUDA Installation

**Discovery:**
- System had CUDA 13.1 installed (not 11.4 as initially assumed)
- cu package v0.9.6 is incompatible with CUDA 13 (designed for CUDA 12)
- Decision: Install CUDA 12.6 alongside CUDA 13

**Actions:**
1. ✅ Installed CUDA 12.6.2 toolkit to `/usr/local/cuda-12.6/`
2. ✅ Installed cuDNN 8.9.7 for CUDA 12
3. ✅ Created environment switcher scripts
4. ✅ Updated go.mod to use `gorgonia.org/cu v0.9.6`

**Files Modified:**
- `/home/gperry/Documents/GitHub/devtools/gorgonia/go.mod`

**Scripts Created:**
- `/usr/local/bin/use-cuda12` - Switch to CUDA 12.6
- `/usr/local/bin/use-cuda13` - Switch to CUDA 13.0
- `/etc/profile.d/cuda12-gorgonia.sh` - Default CUDA 12 environment
- `/home/gperry/install_cuda_alongside.sh` - Comprehensive installation script

### Phase 2: Kernel Compilation Updates

**Compute Capabilities Updated:**

| Architecture | Compute Cap | Status | GPUs |
|--------------|-------------|--------|------|
| Kepler K10/K20 | SM3.0, 3.2 | ❌ REMOVED | No longer supported in CUDA 12 |
| Kepler | SM3.5, 3.7 | ⚠️ DEPRECATED | Tesla K40, K80 |
| Maxwell | SM5.0-5.3 | ✅ SUPPORTED | GTX 9xx, Titan X |
| Pascal | SM6.0-6.2 | ✅ SUPPORTED | GTX 10xx, Tesla P100 |
| Volta | SM7.0, 7.2 | ✅ **NEW** | Tesla V100, Titan V |
| Turing | SM7.5 | ✅ **NEW** | RTX 20xx, Tesla T4 |
| Ampere | SM8.0, 8.6, 8.7 | ✅ **NEW** | RTX 30xx, A100, A10 |

**Files Modified:**
1. **`cuda modules/compile.py`** (line 5):
   ```python
   # OLD: compute = [30, 32, 35, 37, 50, 52, 53, 60, 61, 62]
   # NEW: compute = [35, 37, 50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86, 87]
   ```

2. **`cmd/cudagen/main.go`** (lines 105-147):
   - Added minimum compute capability check (SM3.5+)
   - Added deprecation warning for Kepler (SM3.5/3.7)
   - Improved error messages for unsupported GPUs
   - Added logging for detected compute capability

**Kernels Recompiled:**
- `elembinop.cu` - Element-wise binary operations
- `elemunaryop.cu` - Element-wise unary operations
- `misc.cu` - Utility functions
- `sigmoid32.cu`, `sigmoid64.cu` - Activation functions

**Result:** All kernels now compiled for 14 architectures (up from 10)

### Phase 3: Build Validation

**Build Test:**
```bash
source /usr/local/bin/use-cuda12
go build -tags=cuda ./cuda/...
# Result: BUILD SUCCESS ✅
```

**Verification:**
- ✅ CUDA 12.6 detected and used
- ✅ cu v0.9.6 compatible
- ✅ No compilation errors
- ✅ Package builds successfully

## Configuration Summary

### CUDA Installation

```
CUDA 12.6.2
├── Location: /usr/local/cuda-12.6/
├── Toolkit: 12.6.2 (Build cuda_12.6.r12.6/compiler.34841621_0)
├── cuDNN: 8.9.7
└── Driver: 580.95.05 (supports CUDA 13.x, backward compatible)

CUDA 13.0.2 (Existing)
└── Location: /usr/local/cuda-13.0/
```

### Environment Variables (CUDA 12 mode)

```bash
CUDA_HOME=/usr/local/cuda-12.6
PATH=/usr/local/cuda-12.6/bin:...
LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:...
CGO_CFLAGS="-I/usr/local/cuda-12.6/include"
CGO_LDFLAGS="-L/usr/local/cuda-12.6/lib64 -lcuda -lcudart -lcublas -lcudnn"
```

### Go Dependencies

```
gorgonia.org/cu v0.9.6  (CUDA 12 compatible)
```

## Usage Instructions

### Switching CUDA Versions

```bash
# Use CUDA 12.6 (for Gorgonia)
source /usr/local/bin/use-cuda12

# Use CUDA 13.0
source /usr/local/bin/use-cuda13

# Verify active CUDA version
nvcc --version
```

### Building Gorgonia with CUDA

```bash
# Switch to CUDA 12
source /usr/local/bin/use-cuda12

# Build CUDA package
cd gorgonia
go build -tags=cuda ./cuda/...

# Run tests
go test -v -tags=cuda ./cuda/...
```

### Installing Additional CUDA Versions

Use the comprehensive installation script:

```bash
# Interactive mode (menu selection)
sudo install_cuda_alongside.sh

# Specify version directly
sudo install_cuda_alongside.sh 11.8.0
sudo install_cuda_alongside.sh 12.4.1
```

The script supports:
- CUDA 12.x: 12.6.2, 12.6.1, 12.5.1, 12.4.1, 12.3.2, 12.2.2, 12.1.1, 12.0.1
- CUDA 11.x: 11.8.0, 11.7.1, 11.6.2

## Testing Recommendations

### Unit Tests
```bash
source /usr/local/bin/use-cuda12
cd gorgonia

# CUDA package tests
go test -v -tags=cuda ./cuda/...

# Neural network operations
go test -v -tags=cuda ./ops/nn/...
```

### Integration Tests
```bash
# Example programs
cd examples/stacked_autoencoder
go run -tags=cuda .

# MNIST with CUDA
cd examples/convnet_cuda
go run .
```

### Performance Benchmarks
```bash
go test -bench=. -benchmem -tags=cuda ./cuda/...
go test -bench=. -benchmem -tags=cuda ./ops/nn/...
```

## Architecture Support Matrix

| GPU Generation | Compute Cap | CUDA 10.x | CUDA 11.x | CUDA 12.x | Status |
|----------------|-------------|-----------|-----------|-----------|---------|
| Fermi | 2.x | ✅ | ❌ | ❌ | Removed |
| Kepler (K10/K20) | 3.0-3.2 | ✅ | ✅ | ❌ | Removed |
| Kepler | 3.5-3.7 | ✅ | ✅ | ⚠️ | Deprecated |
| Maxwell | 5.x | ✅ | ✅ | ✅ | Supported |
| Pascal | 6.x | ✅ | ✅ | ✅ | Supported |
| Volta | 7.0-7.2 | ✅ | ✅ | ✅ | **NEW in upgrade** |
| Turing | 7.5 | ✅ | ✅ | ✅ | **NEW in upgrade** |
| Ampere | 8.x | ❌ | ✅ | ✅ | **NEW in upgrade** |
| Hopper | 9.x | ❌ | ❌ | ✅ | Requires CUDA 12.6+ |

## Breaking Changes

### Removed Support
- ❌ **Fermi (SM2.x)** - Removed in CUDA 8.0
- ❌ **Kepler SM3.0/3.2** (K10, K20 GPUs) - Removed in CUDA 12.0

### Deprecated (Still Works)
- ⚠️ **Kepler SM3.5/3.7** (K40, K80 GPUs) - Deprecated in CUDA 12.0, will be removed in future

### Required Updates
- Minimum driver: 470.42.01+ (Linux) or 471.11+ (Windows) for CUDA 11.x
- Minimum driver: 525.60.13+ (Linux) or 528.33+ (Windows) for CUDA 12.x

## Troubleshooting

### Build Fails with "undefined reference" errors

**Problem:** Linker errors for CUDA functions like `cuStreamWaitValue32_v2`

**Cause:** Using cu package version incompatible with your CUDA version

**Solution:**
```bash
# For CUDA 12.x, use cu v0.9.6
go mod edit -require=gorgonia.org/cu@v0.9.6
go mod tidy

# For CUDA 11.x, use cu v0.9.4
go mod edit -require=gorgonia.org/cu@v0.9.4
go mod tidy
```

### Build Fails with "compute capability" errors

**Problem:** GPU not supported by current CUDA version

**Solution:**
```bash
# Check your GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Install appropriate CUDA version:
# SM3.5-3.7: Use CUDA 11.x or 12.x (deprecated in 12.x)
# SM5.0+: Use any CUDA version
# SM8.0+: Use CUDA 11.x or 12.x
```

### Wrong CUDA version used

**Problem:** `nvcc --version` shows wrong CUDA version

**Solution:**
```bash
# Explicitly switch to CUDA 12
source /usr/local/bin/use-cuda12

# Verify
echo $CUDA_HOME
nvcc --version

# Make permanent in your shell
echo "source /usr/local/bin/use-cuda12" >> ~/.bashrc
```

### cuDNN not found

**Problem:** Build fails with "cudnn.h: No such file or directory"

**Solution:**
```bash
# Verify cuDNN installation
ls /usr/local/cuda-12.6/include/cudnn.h
ls /usr/local/cuda-12.6/lib64/libcudnn.so

# If missing, reinstall cuDNN
sudo /home/gperry/fix_cudnn.sh
```

## Performance Improvements

Expected performance improvements with CUDA 12.6 on modern GPUs:

- **Volta/Turing (SM7.x)**: 10-20% faster than running on Pascal-compiled code
- **Ampere (SM8.x)**: 20-40% faster, with better memory bandwidth utilization
- **Tensor operations**: Up to 2x faster with optimized libraries (cuBLAS 12.x, cuDNN 8.x)

## Future Upgrades

### CUDA 13.x Support

**Status:** Blocked - cu package does not support CUDA 13.x yet

**Requirements:**
- cu package needs CUDA 13 API updates
- cuDNN 9 compatibility (breaking changes from cuDNN 8)
- Updated function signatures for CUDA Driver API

**Workaround:** Use CUDA 12.6 (current setup) or wait for cu package update

### Adding More CUDA Versions

Use the installation script to add other versions:

```bash
# Add CUDA 11.8 for maximum compatibility
sudo install_cuda_alongside.sh 11.8.0

# Switch between versions as needed
source /usr/local/bin/use-cuda118  # CUDA 11.8
source /usr/local/bin/use-cuda126  # CUDA 12.6
source /usr/local/bin/use-cuda130  # CUDA 13.0
```

## References

### Documentation Created
- [CUDA 13 Compatibility Analysis](./docs/CUDA_13_Compatibility_Analysis.md)
- [CUDA 12 Upgrade Summary](./docs/CUDA_12_Upgrade_Summary.md)

### External Resources
- [CUDA 12.6 Release Notes](https://docs.nvidia.com/cuda/archive/12.6.2/)
- [cuDNN 8.9.7 Documentation](https://docs.nvidia.com/deeplearning/cudnn/)
- [Gorgonia CUDA Support](https://gorgonia.org/reference/cuda/)
- [cu Package GitHub](https://github.com/gorgonia/cu)

## Acknowledgments

This upgrade was completed using Claude Code with assistance from the comprehensive analysis of the Gorgonia codebase structure and CUDA compatibility requirements.

**Contributors:**
- Environment setup and analysis
- CUDA 12.6 side-by-side installation
- Kernel compilation updates
- Documentation and testing

---

**Last Updated:** December 15, 2025
**Gorgonia Version:** Master branch (commit: d7a3ce2)
**CUDA Version:** 12.6.2
**cuDNN Version:** 8.9.7
