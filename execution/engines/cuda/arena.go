package cuda

import (
	"gorgonia.org/cu"
	"gorgonia.org/tensor"
)

// this file implements an Arena.

// Get allocates memory of certain size and returns a pointer to it
func (e *Engine[DT, T]) Get(size int64) (tensor.Memory, error) {
	ptr, err := e.a.Alloc(size)
	return cu.DevicePtr(ptr), err
}

// Put releases a chunk of memory of certain size
func (e *Engine[DT, T]) Put(mem tensor.Memory, size int64) { e.a.Free(mem.Uintptr()) }

// ResetAllocator releases used memory of the engine
func (e *Engine[DT, T]) ResetAllocator() { e.a.Reset() }
