package cuda

import "C"
import (
	"fmt"
	"reflect"

	"github.com/pkg/errors"
	"gorgonia.org/cu"
	cublas "gorgonia.org/cu/blas"
	cudnn "gorgonia.org/cu/dnn"
	"gorgonia.org/gorgonia/internal/allocator"
	"gorgonia.org/tensor"

	"sync"
	"unsafe"
)

var (
	_ tensor.StandardEngine2 = &Engine{}
)

// Engine is a CUDA engine
type Engine struct {
	tensor.Engine
	sync.Mutex

	a allocator.BFC
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

	syncChan    chan struct{}
	finishChan  chan struct{}
	wg          sync.WaitGroup // wait for Run() to finish.
	once        sync.Once
	err         error
	initialized bool
	running     bool
}

// New creates and initializes the engine. This is to be used when then engine is standalone.
// It will reserve 80% of your GPU memory if no hints are provided.
// This function will panic if there are any CUDA errors.
//
// You will need to manually call `.Run()`
func New(hint int64) *Engine {
	dev, err := cu.GetDevice(0)
	if err != nil {
		panic(fmt.Sprintf("Failed to get CUDA device 0. Error %v", err))
	}
	logf("hint %v", hint)
	e := &Engine{totalMem: hint, d: dev}

	return e
}

// IsInitialized returns true when the engine has been initialized
func (e *Engine) IsInitialized() bool {
	e.Lock()
	initialized := e.initialized
	e.Unlock()
	return initialized
}

// AllocAccessible returns true because the engine return Go-accessible memory pointers
func (e *Engine) AllocAccessible() bool { return true }

// Alloc allocates a chunk of certain size from engine memory
func (e *Engine) Alloc(size int64) (tensor.Memory, error) {
	// return e.c.MemAllocManaged(size, cu.AttachGlobal)

	// loop here to wait for initialization
	for initialized := e.IsInitialized(); !initialized; initialized = e.IsInitialized() {
	}
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
	size := mem.MemSize()
	bs := make([]byte, int(size))
	e.c.Context.MemcpyDtoH(unsafe.Pointer(&bs[0]), cu.DevicePtr(mem.Uintptr()), int64(size))
	if err := e.c.Context.Error(); err != nil {
		return nil, err
	}
	switch t := mem.(type) {
	case *tensor.Dense:
		dt := t.Dtype()
		l := int(size / dt.Size())
		backingHdr := &reflect.SliceHeader{
			Data: uintptr(unsafe.Pointer(&bs[0])),
			Len:  l,
			Cap:  l,
		}
		switch dt {
		case tensor.Float64:
			backing := *(*[]float64)(unsafe.Pointer(backingHdr))
			retVal := tensor.New(tensor.WithShape(t.Shape().Clone()...), tensor.WithBacking(backing))
			return retVal, e.c.Error()
		case tensor.Float32:
			backing := *(*[]float32)(unsafe.Pointer(backingHdr))
			retVal := tensor.New(tensor.WithShape(t.Shape().Clone()...), tensor.WithBacking(backing))
			return retVal, e.c.Error()
		}
	default:
		return nil, errors.Errorf("mem of type %T unsupported by Accessible", mem)
	}
	panic("Unreachable")
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
	e.Signal()
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
	e.Signal()
	return int(retVal) > 0, e.c.Error()
}
