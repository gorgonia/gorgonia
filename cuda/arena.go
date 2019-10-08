package cuda

import (
	"gorgonia.org/cu"
	"gorgonia.org/tensor"
)

// this file implements the arena

var _ Arena = &Engine{}

// Get allocates memory of certain size and returns a pointer to it
func (e *Engine) Get(size int64) (tensor.Memory, error) {
	ptr, err := e.a.alloc(size)
	return cu.DevicePtr(ptr), err
}

// Put releases a chunk of memory of certain size
func (e *Engine) Put(mem tensor.Memory, size int64) {
	addr := uintptr(mem.Uintptr())
	e.a.free(addr)
}

// ResetAllocator releases used memory of the engine
func (e *Engine) ResetAllocator() {
	used := make([]uintptr, 0, len(e.a.used))
	for k := range e.a.used {
		used = append(used, k)
	}

	for _, ptr := range used {
		e.a.free(ptr + e.a.start)
	}
	e.a.coalesce()
	e.a.reset() // reset statistics
}
