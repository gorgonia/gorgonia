//go:build cuda && !darwin && !arm64

package cuda

import (
	"errors" // Import errors for error handling in mock functions
	"os"
	"testing"

	"gorgonia.org/cu"
)

// Global variables to store mock behavior and captured values for MemInfo and MemAllocManaged
var (
	mockMemInfoFunc      func() (int64, int64, error) // Changed from uint64 to int64
	mockAllocManagedFunc func(size int64, flags cu.MemAttachFlags) (cu.DevicePtr, error)
	capturedAllocSize    int64 // To capture the size argument passed to MemAllocManaged
)

// --- TestMain Function for Mock Setup/Teardown ---
// This function runs once for all tests in the package.
func TestMain(m *testing.M) {
	// Save original functions to restore them after all tests run
	originalGetCuMemInfo := getCuMemInfo
	originalCallCuMemAllocManaged := callCuMemAllocManaged

	// Replace global function variables with mock implementations
	getCuMemInfo = func() (int64, int64, error) { // Changed from uint64 to int64
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

func TestEngineAllocateMemory(t *testing.T) {
	// minAllocSize = 1 << 8 = 256 (from cuda/bfc.go)
	const assumedMinAllocSize int64 = 256

	tests := []struct {
		name              string
		inputSize         int64  // The 'size' parameter passed to allocateMemory
		mockFreeMem       uint64 // The free memory value reported by mockMemInfoFunc (still uint64 in test struct for input)
		expectedAllocSize int64  // The expected size passed to mockAllocManagedFunc
		expectError       bool   // Whether allocateMemory is expected to return an error
	}{
		{
			name:              "Standard allocation (1.5x requested size)",
			inputSize:         1000,
			mockFreeMem:       10000,                                     // Plenty of free memory
			expectedAllocSize: (1000 + (1000 / 2)) + assumedMinAllocSize, // 1.5 * 1000 + minAllocSize
			expectError:       false,
		},
		{
			name:              "Allocation capped by free memory",
			inputSize:         1000,
			mockFreeMem:       uint64(500 + assumedMinAllocSize - 1), // Less than 1.5x requested size
			expectedAllocSize: 500 + assumedMinAllocSize - 1,         // Should allocate all available free memory
			expectError:       false,
		},
		{
			name:              "Minimum allocation when requested size is very small",
			inputSize:         1,
			mockFreeMem:       1000,
			expectedAllocSize: (1 + (1 / 2)) + assumedMinAllocSize, // 1.5 * 1 + minAllocSize (integer division)
			expectError:       false,
		},
		{
			name:              "Allocation with zero input size",
			inputSize:         0,
			mockFreeMem:       1000,
			expectedAllocSize: assumedMinAllocSize, // Just minAllocSize
			expectError:       false,
		},
		{
			name:              "Free memory extremely low, should allocate available",
			inputSize:         1000,
			mockFreeMem:       10, // Very little free memory
			expectedAllocSize: 10,
			expectError:       false,
		},
		{
			name:              "MemInfo returns error",
			inputSize:         1000,
			mockFreeMem:       0, // Not used if error
			expectedAllocSize: 0, // Not called if MemInfo errors
			expectError:       true,
		},
		{
			name:              "MemAllocManaged returns error",
			inputSize:         1000,
			mockFreeMem:       10000,
			expectedAllocSize: (1000 + (1000 / 2)) + assumedMinAllocSize,
			expectError:       true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Reset captured size for each test run
			capturedAllocSize = 0

			// Configure mock MemInfo for this test case
			if tt.name == "MemInfo returns error" {
				mockMemInfoFunc = func() (int64, int64, error) {
					return 0, 0, errors.New("mock MemInfo error")
				}
			} else {
				mockMemInfoFunc = func() (int64, int64, error) { // Cast tt.mockFreeMem to int64
					return int64(tt.mockFreeMem), int64(tt.mockFreeMem + 5000), nil // totalMem can be arbitrary
				}
			}

			// Configure mock AllocManaged (no specific error for these tests)
			if tt.name == "MemAllocManaged returns error" {
				mockAllocManagedFunc = func(size int64, flags cu.MemAttachFlags) (cu.DevicePtr, error) {
					return 0, errors.New("mock MemAllocManaged error")
				}
			} else {
				mockAllocManagedFunc = func(size int64, flags cu.MemAttachFlags) (cu.DevicePtr, error) {
					return cu.DevicePtr(0x1000), nil // Return a dummy pointer
				}
			}

			e := &Engine{} // Create a new Engine instance for each test
			e.a = makeBFC(memalign)
			err := e.allocateMemory(tt.inputSize)

			// Assert error conditions
			if (err != nil) != tt.expectError {
				t.Errorf("allocateMemory() error mismatch for test '%s'. Expected error: %v, Got: %v (Error: %v)", tt.name, tt.expectError, (err != nil), err)
				return
			}

			// Assert the size passed to MemAllocManaged if no error was expected
			if !tt.expectError && capturedAllocSize != tt.expectedAllocSize {
				t.Errorf("MemAllocManaged called with unexpected size for test '%s'. Expected: %d, Got: %d", tt.name, tt.expectedAllocSize, capturedAllocSize)
			}
		})
	}
}
