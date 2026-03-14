# CUDA/cuDNN Diagnostic Tests

This directory contains diagnostic tests to isolate whether cuDNN initialization issues are caused by:
1. System/driver/CUDA/cuDNN configuration
2. The gorgonia/cu Go bindings

## Test Files

### 1. `cudnn_test.c` - Standalone C Test
Pure C program that directly calls CUDA and cuDNN C APIs with no Go involvement.

**Run:** `./run_c_test.sh`

This is the **most direct test** - if this fails, the issue is with your system configuration, not Go bindings.

### 2. `cudnn_test.go` - CGo Test
Go test that uses CGo to call CUDA/cuDNN C APIs directly, bypassing gorgonia/cu.

**Run:** `./run_direct_test.sh`

If the C test passes but this fails, the issue is with CGo or Go's interaction with CUDA.

### 3. Original gorgonia tests
The regular gorgonia tests that use the gorgonia/cu package.

**Run:** `../test_cuda.sh` (from parent directory)

If tests 1 and 2 pass but this fails, the issue is in the gorgonia/cu package.

## Interpretation Matrix

| C Test | CGo Test | gorgonia/cu Test | Issue Location |
|--------|----------|------------------|----------------|
| ✗ FAIL | - | - | **System/Driver/cuDNN configuration** |
| ✓ PASS | ✗ FAIL | - | **CGo or Go-CUDA interaction** |
| ✓ PASS | ✓ PASS | ✗ FAIL | **gorgonia/cu package bindings** |
| ✓ PASS | ✓ PASS | ✓ PASS | **Everything works!** |

## Expected Results

Given your configuration (Driver 470, CUDA 11.4, cuDNN 8.2.4):

**Most likely:**
- C Test: May FAIL with cuDNN initialization error
- Reason: Driver 470 is very old and may not fully support cuDNN 8.2.4 operations

**If C test passes:**
This would indicate the issue is in the Go bindings or how gorgonia/cu initializes CUDA contexts before creating cuDNN handles.

## Common cuDNN Error Codes

- `CUDNN_STATUS_NOT_INITIALIZED (1)`: cuDNN library not initialized
- `CUDNN_STATUS_ALLOC_FAILED (2)`: Resource allocation failed
- `CUDNN_STATUS_BAD_PARAM (3)`: Invalid parameter
- `CUDNN_STATUS_ARCH_MISMATCH (8)`: Compute capability mismatch
- `CUDNN_STATUS_NOT_SUPPORTED (9)`: Feature not supported

## Next Steps Based on Results

### If C test FAILS:
- Upgrade NVIDIA driver to 510+ (recommended for CUDA 11.4 + cuDNN 8)
- Or downgrade to cuDNN 8.0.x (may be more compatible with driver 470)

### If C test PASSES but CGo test FAILS:
- Check Go CGo configuration
- Verify CGO_LDFLAGS and CGO_CFLAGS are correct
- Check for Go/C ABI compatibility issues

### If CGo test PASSES but gorgonia/cu test FAILS:
- Update gorgonia/cu package to latest version
- Check if cu package requires CUDA context before cuDNN creation
- File issue with gorgonia/cu project with diagnostic results
