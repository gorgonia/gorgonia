package cuda

import (
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
)

// internal_op_prep.go provides the syntactic abstractions for code that is relevant to extracting a CUDA memory from a tensor.

// Engine cannot handle incr or safe. See opMem()

func (e *Engine[DT, T]) HandleFuncOptsSpecialized(a T, expShape shapes.Shape, opts ...tensor.FuncOpt) (retVal T, fo tensor.Option, err error) {
	panic("NYI")
}

func (e *Engine[DT, T]) HandleFuncOpts(a tensor.Basic[DT], expShape shapes.Shape, opts ...tensor.FuncOpt) (retVal tensor.Basic[DT], fo tensor.Option, err error) {
	panic("NYI")
}

func (e *Engine[DT, T]) HandleFuncOptsDesc(a tensor.Basic[DT], expShape shapes.Shape, opts ...tensor.FuncOpt) (retVal tensor.DescWithStorage, fo tensor.Option, err error) {
	panic("NYI")
}

/*
func (e *Engine[DT, T]) opMem(a tensor.Tensor, opts ...tensor.FuncOpt) (mem cu.DevicePtr, size int64, retVal tensor.Tensor, err error) {
	var reuse tensor.DenseTensor
	var safe, toReuse bool
	var ctx context.Context
	if ctx, reuse, safe, toReuse, _, _, err = gtu.HandleFuncOpts(a.Shape(), a.Dtype(), a.DataOrder(), true, opts...); err != nil {
		return mem, 0, nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	if err := gctx.Handle(ctx); err != nil {
		return mem, 0, nil, err
	}

	switch {
	case toReuse:
		mem = cu.DevicePtr(reuse.Uintptr())
		memA := cu.DevicePtr(a.Uintptr())
		memSize := int64(a.MemSize())
		e.memcpy(mem, memA, memSize)

		size = int64(logicalSize(reuse.Shape()))
		retVal = reuse
	case !safe:
		mem = cu.DevicePtr(a.Uintptr())
		retVal = a
		size = int64(logicalSize(a.Shape()))
	default:
		err = errors.New("Impossible state: A reuse tensor must be passed in, or the operation must be unsafe. Incr and safe operations are not supported")
	}
	return mem, size, retVal, err
}
*/
