package cuda

import "C"
import (
	"fmt"
	"reflect"

	"gorgonia.org/cu"
	cublas "gorgonia.org/cu/blas"
	cudnn "gorgonia.org/cu/dnn"
	"gorgonia.org/gorgonia/internal/allocator"
	"gorgonia.org/gorgonia/internal/errors"
	"gorgonia.org/internal/debug"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/dense"

	"sync"
	"unsafe"
)

var (
// _ tensor.StandardEngine2 = &Engine{}
)

// EngineState represents the internal, type-free state. In typical runs, you'd have
// potentially multiple engines and single engine state.
type EngineState struct {
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

// NewState  will reserve 80% of your GPU memory if `hint` <= 0 are provided.
// This function will panic if there are any CUDA errors
func NewState(hint int64) *EngineState {
	dev, err := cu.GetDevice(0)
	if err != nil {
		panic(fmt.Sprintf("Failed to get CUDA device 0. Error %v", err))
	}
	return &EngineState{
		totalMem: hint,
		d:        dev,
	}
}

// Engine is a CUDA engine
type Engine[DT any, T tensor.Basic[DT]] struct {
	// tensor.Engine // is this needed?

	*EngineState
}

// New creates and initializes the engine. This is to be used when then engine is standalone.
// You will need to manually call `.Run()`
func New[DT any, T tensor.Basic[DT]](state *EngineState) *Engine[DT, T] {
	e := &Engine[DT, T]{EngineState: state}

	return e
}

func (e *Engine[DT, T]) Workhorse() tensor.Engine { return e }

func (e *Engine[DT, T]) BasicEng() tensor.Engine {
	return &Engine[DT, tensor.Basic[DT]]{EngineState: e.EngineState}
}

// IsInitialized returns true when the engine has been initialized
func (e *Engine[DT, T]) IsInitialized() bool {
	e.Lock()
	initialized := e.initialized
	e.Unlock()
	return initialized
}

// AllocAccessible returns true because the engine return Go-accessible memory pointers
func (e *Engine[DT, T]) AllocAccessible() bool { return true }

// Alloc allocates a chunk of certain size from engine memory
func (e *Engine[DT, T]) Alloc(size int64) (tensor.Memory, error) {
	e.waitForInit()
	return e.Get(size)
}

// AllocFlags returns allocation flags
func (e *Engine[DT, T]) AllocFlags() (tensor.MemoryFlag, tensor.DataOrder) {
	return tensor.MakeMemoryFlag(tensor.ManuallyManaged), tensor.ColMajor // TODO: NativelyInaccessible?
}

// Free rees memory
func (e *Engine[DT, T]) Free(mem tensor.Memory, size int64) error {
	e.waitForInit()
	e.Put(mem, size)
	return nil
}

func (e *Engine[DT, T]) Memset(mem tensor.Memory, val interface{}) error {
	panic("not implemented")
}

func (e *Engine[DT, T]) Memclr(mem tensor.Memory) {
	panic("not implemented")
}

// Memcpy is part of the implementation of tensor.Engine. It is eager, and will signal the context to actually do work.
// The memory that will be copied is up to the smallest of sizes between dst and src.
// i.e. if dst is 8 bytes and src is 16 bytes, only the first 8 bytes of src will be copied.
// Likewise, if dst is 20 bytes and src is 3 bytes, only 3 bytes will be copied.
func (e *Engine[DT, T]) Memcpy(dst tensor.Memory, src tensor.Memory) error {
	debug.Logf("Memcpy src %p %T %v, dst %p %T %v, called by %v", src, src, src.Uintptr(), dst, dst, dst.Uintptr(), errors.ThisFn(1))

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
	e.memcpy(dst, src, size)
	e.Signal()
	//<-e.syncChan

	return e.c.Error()
}

func (e *Engine[DT, T]) Accessible(mem tensor.Memory) (tensor.Memory, error) {
	size := mem.MemSize()
	bs := make([]byte, int(size))
	e.c.Context.MemcpyDtoH(unsafe.Pointer(&bs[0]), cu.DevicePtr(mem.Uintptr()), int64(size))
	if err := e.c.Context.Error(); err != nil {
		return nil, err
	}
	switch t := mem.(type) {
	case *dense.Dense[DT]:
		dt := t.Dtype()
		l := int(size / dt.Size())
		backingHdr := &reflect.SliceHeader{
			Data: uintptr(unsafe.Pointer(&bs[0])),
			Len:  l,
			Cap:  l,
		}
		backing := *(*[]DT)(unsafe.Pointer(backingHdr))
		retVal := dense.New[DT](tensor.WithShape(t.Shape().Clone()...), tensor.WithBacking(backing))
		return retVal, e.c.Error()
	default:
		return nil, errors.Errorf("mem of type %T unsupported by Accessible", mem)
	}
	panic("Unreachable")
}

// WorksWith returns true because the data order can be directly worked with
func (e *Engine[DT, T]) WorksWith(flags tensor.MemoryFlag, order tensor.DataOrder) bool { return true }

// NonStdAlloc nothing instead of running the default built in allocator
func (e *Engine[DT, T]) NonStdAlloc() {}

// Errors returns an error message
func (e *Engine[DT, T]) Errors() error { return e.c.Errors() }

// waitForInit is a loop that blocks and waits till the engine is initialized.
func (e *Engine[DT, T]) waitForInit() {
	for ok := e.IsInitialized(); !ok; ok = e.IsInitialized() {
	}
}
