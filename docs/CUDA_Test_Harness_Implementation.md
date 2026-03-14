# CUDA Test Harness Implementation Plan

This document outlines the plan to create a custom test harness for verifying the memory allocation logic within `cuda/external.go` in the `gorgonia` project. This approach addresses the challenges of testing against an external `gorgonia.org/cu` dependency and its global functions.

## Goal

To effectively test the `doInit` memory allocation logic in `cuda/external.go` by simulating the behavior of the `gorgonia.org/cu` package, allowing for controlled testing of various memory scenarios without a physical CUDA device.

---

## Step 1: Refactor `cuda/external.go` for Testability (Minimal Intrusion)

The primary challenge is to mock global functions like `cu.MemInfo` and `cu.MemAllocManaged`. We will introduce package-level function variables in `cuda/external.go` that initially point to the real `cu` functions. These variables can then be overwritten by test code to inject mock implementations.

### Action: Modify `cuda/external.go`

1.  **Introduce Package-Level Variables:**
    Add the following global variables near the top of `cuda/external.go`, for example, after the `const` block or after imports:

    ```go
    // In cuda/external.go
    var (
        getCuMemInfo      = cu.MemInfo
        callCuMemAllocManaged = cu.MemAllocManaged
        // Add other cu functions here if they become relevant for future testing
    )
    ```

2.  **Replace Direct `cu` Calls with Variables:**
    Update the `doInit` function to use these new package-level variables instead of direct calls to `cu.MemInfo` and `cu.MemAllocManaged`:

    ```go
    // ... in doInit function ...
    func (e *Engine) doInit(size int64) (err error) {
        // ...
        // Change: if e.freeMem, e.totalMem, err = cu.MemInfo(); err != nil {
        if e.freeMem, e.totalMem, err = getCuMemInfo(); err != nil {
            // ...
        }

        // ...
        // Change: ptr, err := cu.MemAllocManaged(allocsize, cu.AttachGlobal)
        ptr, err := callCuMemAllocManaged(allocsize, cu.AttachGlobal)
        // ...
    }
    ```

    *This change is minimal and ensures that production builds continue to use the real `cu` functions, while tests gain the ability to inject mock behavior.*

---

## Step 2: Create `cuda/external_test.go` (The Test Harness and Tests)

This file will contain the mock implementations for the `cu` functions and the actual test cases to verify the memory allocation logic.

### Action: Create `cuda/external_test.go`

1.  **File Structure and Imports:**
    Create a new file `cuda/external_test.go` with the following basic structure:

    ```go
    // In cuda/external_test.go
    package cuda

    import (
        "os"
        "testing"
        "gorgonia.org/cu" // Import the real cu package for types and constants
    )

    // --- Mock Implementations and Global State ---

    // Define mock implementations for cu.Device and cu.CUContext if needed by Engine.Init
    // For this specific test, minimal implementations are likely sufficient.
    type mockDevice struct{}
    func (m *mockDevice) MakeContext(flags cu.ContextFlags) (cu.CUContext, error) { return &mockCUContext{}, nil }
    func (m *mockDevice) Attributes(attrs ...cu.DeviceAttribute) ([]int, error) {
        // Return dummy attributes. Adjust if specific attributes affect doInit logic.
        return make([]int, len(attrs)), nil
    }
    // Implement other cu.Device methods as required by Engine.Init/doInit

    type mockCUContext struct{}
    func (m *mockCUContext) Destroy() error { return nil }
    func (m *mockCUContext) CUDAContext() uintptr { return 0 } // Dummy value
    // Implement other cu.CUContext methods as required by Engine.Init/doInit

    // Global variables to store mock behavior and captured values for MemInfo and MemAllocManaged
    var (
        mockMemInfoFunc       func() (uint64, uint64, error)
        mockAllocManagedFunc  func(size int64, flags cu.MemAttachFlags) (cu.DevicePtr, error)
        capturedAllocSize     int64 // To capture the size argument passed to MemAllocManaged
    )

    // --- TestMain Function for Mock Setup/Teardown ---
    // This function runs once for all tests in the package.
    func TestMain(m *testing.M) {
        // Save original functions to restore them after all tests run
        originalGetCuMemInfo := getCuMemInfo
        originalCallCuMemAllocManaged := callCuMemAllocManaged

        // Replace global function variables with mock implementations
        getCuMemInfo = func() (uint64, uint64, error) {
            if mockMemInfoFunc != nil {
                return mockMemInfoFunc()
            }
            // Default mock behavior if mockMemInfoFunc is not set by a specific test
            return 0, 0, nil
        }
        callCuMemAllocManaged = func(size int64, flags cu.MemAttachFlags) (cu.DevicePtr, error) {
            capturedAllocSize = size // IMPORTANT: Capture the size argument
            if mockAllocManagedFunc != nil {
                return mockAllocManagedFunc(size, flags)
            }
            // Default mock behavior
            return 0, nil
        }

        // Run all tests
        code := m.Run()

        // Restore original functions to ensure no side effects on other packages/tests
        getCuMemInfo = originalGetCuMemInfo
        callCuMemAllocManaged = originalCallCuMemAllocManaged

        os.Exit(code)
    }

    // --- Test Functions to Verify Memory Allocation Logic ---

    // Example test suite
    func TestEngineInitMemoryAllocationLogic(t *testing.T) {
        // Assuming minAllocSize is a package constant or can be derived.
        // If minAllocSize is unexported, it would need to be exposed for testing or its value hardcoded.
        // For demonstration, let's assume it's directly accessible or a known value.
        const assumedMinAllocSize int64 = 32 // Example value, adjust based on actual minAllocSize

        tests := []struct {
            name               string
            inputSize          int64  // The 'size' parameter passed to Engine.Init
            mockFreeMem        uint64 // The free memory value reported by mockMemInfoFunc
            expectedAllocSize  int64  // The expected size passed to mockAllocManagedFunc
            expectInitError    bool   // Whether Engine.Init is expected to return an error
        }{
            {
                name:              "Standard allocation (1.5x requested size)",
                inputSize:         1000,
                mockFreeMem:       10000, // Plenty of free memory
                expectedAllocSize: (1000 + (1000 / 2)) + assumedMinAllocSize, // 1.5 * 1000 + minAllocSize
                expectInitError:   false,
            },
            {
                name:              "Allocation capped by free memory",
                inputSize:         1000,
                mockFreeMem:       uint64(500 + assumedMinAllocSize), // Less than 1.5x requested size, exactly 500 bytes + minAllocSize
                expectedAllocSize: 500 + assumedMinAllocSize, // Should allocate all available free memory
                expectInitError:   false,
            },
            {
                name:              "Minimum allocation when requested size is very small",
                inputSize:         1,
                mockFreeMem:       1000,
                expectedAllocSize: (1 + (1 / 2)) + assumedMinAllocSize, // 1.5 * 1 + minAllocSize (integer division)
                expectInitError:   false,
            },
            {
                name:              "Allocation with zero input size",
                inputSize:         0,
                mockFreeMem:       1000,
                expectedAllocSize: assumedMinAllocSize, // Just minAllocSize
                expectInitError:   false,
            },
            {
                name:              "Free memory extremely low, should allocate available",
                inputSize:         1000,
                mockFreeMem:       uint64(10), // Very little free memory
                expectedAllocSize: 10,
                expectInitError:   false,
            },
        }

        for _, tt := range tests {
            t.Run(tt.name, func(t *testing.T) {
                // Reset captured size for each test run
                capturedAllocSize = 0

                // Configure mock MemInfo for this test case
                mockMemInfoFunc = func() (uint64, uint64, error) {
                    return tt.mockFreeMem, tt.mockFreeMem + 5000, nil // totalMem can be arbitrary
                }

                // Configure mock AllocManaged (no specific error for these tests)
                mockAllocManagedFunc = func(size int64, flags cu.MemAttachFlags) (cu.DevicePtr, error) {
                    return cu.DevicePtr(0x1000), nil // Return a dummy pointer
                }

                e := &Engine{} // Create a new Engine instance for each test
                err := e.Init(&mockDevice{}, tt.inputSize) // Pass our mock device

                // Assert error conditions
                if (err != nil) != tt.expectInitError {
                    t.Errorf("Init() error mismatch for test '%s'. Expected error: %v, Got: %v (Error: %v)", tt.name, tt.expectInitError, (err != nil), err)
                    return
                }

                // Assert the size passed to MemAllocManaged if no error was expected
                if !tt.expectInitError && capturedAllocSize != tt.expectedAllocSize {
                    t.Errorf("MemAllocManaged called with unexpected size for test '%s'. Expected: %d, Got: %d", tt.name, tt.expectedAllocSize, capturedAllocSize)
                }
            })
        }
    }
    ```

### Step 3: Addressing `minAllocSize` and other constants

The constant `minAllocSize` is used in the allocation logic. If it is an unexported constant in `cuda/external.go`, it might need to be temporarily exposed (e.g., via a test-specific getter) or its value hardcoded in the test for precise comparisons. Assuming it's `32` for the example above. We can infer `minAllocSize` from the `cuda/external.go` file.

**This plan sets up a robust and isolated testing environment for the memory allocation logic, allowing precise control over simulated CUDA conditions.**
