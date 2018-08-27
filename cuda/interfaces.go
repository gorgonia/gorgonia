package cuda

import (
	"gorgonia.org/cu"
	"gorgonia.org/cu/blas"
	"gorgonia.org/cu/dnn"
	"gorgonia.org/tensor"
)

// Arena is a representation of a memory arena which is managed.
type Arena interface {
	// Get returns a NoOpError when it cannot get a memory. Please allocate
	Get(size int64) (tensor.Memory, error)

	// Puts the memory back into arena
	Put(mem tensor.Memory, size int64)

	// ResetAllocator resets the allocator statisttics, but doesn't actually deallocate  real memory
	ResetAllocator()
}

// External is a representation of an external device, conceptually modelled as a machine
type External interface {
	// Arena implies that the machine has to be able to manage its own memory
	Arena

	// Engine implies that the machine is able to allocate and free memory
	tensor.Engine

	// HasFunc checks if a function exists within this machine
	HasFunc(string) bool

	// Sync returns a channel of sync signals
	Sync() chan struct{}

	// Signal signals the machine to do work
	Signal()

	// Context returns the Context (the default implementation returns a *cu.BatchedContext)
	Context() *cu.BatchedContext

	// CUDNNContext returns the cuDNN context
	CUDNNContext() *cudnn.Context

	// BLASContext returns the cuBLAS context
	BLASContext() *cublas.Standard

	// Modules returns the loaded modules. It is indexed by name
	Modules() map[string]cu.Module

	// Functions returns the loaded functions. It is indexed by name
	Functions() map[string]cu.Function

	// ElemGridSize calculates the grid sizes for elementwise operations
	ElemGridSize(n int) (gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ int)

	// Init initializes the machine
	Init(device cu.Device, size int64) error

	// Close cleans up the machine, and closes all available resources
	Close() error

	// DoWork sends a signal to the batched CUDA Context to actually do work
	DoWork() error
}
