package cuda

import "C"

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/pkg/errors"
	"gorgonia.org/cu"
	"gorgonia.org/cu/blas"
	"gorgonia.org/cu/dnn"
	"gorgonia.org/tensor"
)

var (
	_ tensor.Adder = &Engine{}
	_ tensor.Suber = &Engine{}
	_ tensor.Muler = &Engine{}
	_ tensor.Diver = &Engine{}
	_ tensor.Power = &Engine{}
	_ tensor.Moder = &Engine{}
	// _ tensor.FMAer       = &Engine{}
	_ tensor.MatMuler    = &Engine{}
	_ tensor.MatVecMuler = &Engine{}
	_ tensor.OuterProder = &Engine{}
	// _ tensor.Dotter      = &Engine{}
	// _ tensor.SVDer       = &Engine{}
	_ tensor.Lter   = &Engine{}
	_ tensor.Lteer  = &Engine{}
	_ tensor.Gter   = &Engine{}
	_ tensor.Gteer  = &Engine{}
	_ tensor.ElEqer = &Engine{}
)

// Engine is a CUDA engine
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
	finishChan    chan struct{}
	finishChan2   chan struct{}
	workAvailable chan bool
	err           error
	initialized   bool
	running       bool
}

// AllocAccessible returns true because the engine return Go-accessible memory pointers
func (e *Engine) AllocAccessible() bool { return true }

// Alloc allocates a chunk of certain size from engine memory
func (e *Engine) Alloc(size int64) (tensor.Memory, error) {
	// return e.c.MemAllocManaged(size, cu.AttachGlobal)
	return e.Get(size)
}

// AllocFlags returns allocation flags
func (e *Engine) AllocFlags() (tensor.MemoryFlag, tensor.DataOrder) {
	return tensor.MakeMemoryFlag(tensor.ManuallyManaged), tensor.ColMajor
}

// Free rees memory
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

// Memcpy is part of the implementation of tensor.Engine. It is eager, and will signal the context to actually do work.
// The memory that will be copied is up to the smallest of sizes between dst and src.
// i.e. if dst is 8 bytes and src is 16 bytes, only the first 8 bytes of src will be copied.
// Likewise, if dst is 20 bytes and src is 3 bytes, only 3 bytes will be copied.
func (e *Engine) Memcpy(dst tensor.Memory, src tensor.Memory) error {
	sSize := src.MemSize()
	dSize := dst.MemSize()

	var size int64
	switch {
	case dSize < sSize:
		size = int64(dSize)
	case sSize < dSize:
		size = int64(sSize)
	default:
		size = int64(dSize)
	}
	d := cu.DevicePtr(dst.Uintptr())
	s := cu.DevicePtr(src.Uintptr())
	e.c.Memcpy(d, s, size)
	e.Signal()
	<-e.syncChan
	return e.c.Error()
}

func (e *Engine) memcpy(dst cu.DevicePtr, src cu.DevicePtr, size int64) {
	e.c.Memcpy(dst, src, size)
}

func (e *Engine) Accessible(mem tensor.Memory) (tensor.Memory, error) {
	panic("not implemented")
}

// WorksWith returns true because the data order can be directly worked with
func (e *Engine) WorksWith(order tensor.DataOrder) bool { return true }

// NonStdAlloc nothing instead of running the default built in allocator
func (e *Engine) NonStdAlloc() {}

// Errors returns an error message
func (e *Engine) Errors() error { return e.c.Errors() }

// NaNChecker checks that the tensor contains a NaN
func (e *Engine) HasNaN(a tensor.Tensor) (bool, error) {
	dt := a.Dtype()
	name := fmt.Sprintf("misc.hasNaN_f%v", int(dt.Size()*8))

	if !e.HasFunc(name) {
		return false, errors.Errorf("Unable to perform HasNaN(). The tensor engine does not have the function %q", name)
	}

	mem := cu.DevicePtr(a.Uintptr())
	size := int64(logicalSize(a.Shape()))
	fn := e.f[name]

	var retVal C.int
	gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ := e.ElemGridSize(int(size))
	args := []unsafe.Pointer{
		unsafe.Pointer(&mem),
		unsafe.Pointer(&size),
		unsafe.Pointer(&retVal),
	}
	e.c.LaunchAndSync(fn, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 0, cu.NoStream, args)
	e.DoWork()
	return int(retVal) > 0, e.c.Error()
}

// InfChecker checks that the tensor contains a Inf
func (e *Engine) HasInf(a tensor.Tensor) (bool, error) {
	dt := a.Dtype()
	name := fmt.Sprintf("misc.hasInf_f%v", int(dt.Size()*8))

	if !e.HasFunc(name) {
		return false, errors.Errorf("Unable to perform HasInf(). The tensor engine does not have the function %q", name)
	}

	mem := cu.DevicePtr(a.Uintptr())
	size := int64(logicalSize(a.Shape()))
	fn := e.f[name]

	var retVal C.int
	gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ := e.ElemGridSize(int(size))
	args := []unsafe.Pointer{
		unsafe.Pointer(&mem),
		unsafe.Pointer(&size),
		unsafe.Pointer(&retVal),
	}
	e.c.LaunchAndSync(fn, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 0, cu.NoStream, args)
	e.DoWork()
	return int(retVal) > 0, e.c.Error()
}
