package cuda

import (
	"gorgonia.org/cu"
	"gorgonia.org/tensor"
)

// this file implements the arena

var _ Arena = &Engine{}

func (e *Engine) Get(size int64) (tensor.Memory, error) {
	ptr, err := e.a.alloc(size)
	return cu.DevicePtr(ptr), err
}

func (e *Engine) Put(mem tensor.Memory, size int64) {
	addr := uintptr(mem.Uintptr())
	e.a.free(addr)
}
