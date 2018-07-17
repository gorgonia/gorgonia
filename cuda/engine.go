package cuda

import (
	"sync"

	"gorgonia.org/cu"
	"gorgonia.org/cu/blas"
	"gorgonia.org/cu/dnn"
	"gorgonia.org/tensor"
)

type Engine struct {
	tensor.Engine
	sync.Mutex

	a bfc
	b cublas.Standard
	c cu.BatchedContext
	d cu.Device
	f map[string]cu.Function
	m map[string]cu.Module
	n cudnn.Context

	warp int
	mtpb int
	mgdx int
	mgdy int
	mgdz int
	mbdx int
	mbdy int
	mbdz int

	freeMem  int64
	totalMem int64

	syncChan      chan struct{}
	workAvailable chan bool
	initialized   bool
}

func (e *Engine) AllocAccessible() bool { return true }

func (e *Engine) Alloc(size int64) (tensor.Memory, error) {
	// return e.c.MemAllocManaged(size, cu.AttachGlobal)
	return e.Get(size)
}

func (e *Engine) AllocFlags() (tensor.MemoryFlag, tensor.DataOrder) {
	return tensor.MakeMemoryFlag(tensor.ManuallyManaged), tensor.ColMajor
}

func (e *Engine) Free(mem tensor.Memory, size int64) error {
	// e.c.MemFree(mem.(cu.DevicePtr))
	// return e.c.Error()
	e.Put(mem, size)
	return nil
}

func (e *Engine) Memset(mem tensor.Memory, val interface{}) error {
	panic("not implemented")
}

func (e *Engine) Memclr(mem tensor.Memory) {
	panic("not implemented")
}

func (e *Engine) Memcpy(dst tensor.Memory, src tensor.Memory) error {
	panic("not implemented")
}

func (e *Engine) Accessible(mem tensor.Memory) (tensor.Memory, error) {
	panic("not implemented")

}

func (e *Engine) WorksWith(order tensor.DataOrder) bool { return true }

func (e *Engine) NonStdAlloc() {}

func (e *Engine) ContextErr() error { return e.c.Error() }
