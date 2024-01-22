package cuda

import (
	"log"
	"unsafe"

	"gorgonia.org/cu"
	"gorgonia.org/dtype"
	"gorgonia.org/gorgonia/internal/errors"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
)

// internal_op_prep.go provides the syntactic abstractions for code that is relevant to extracting a CUDA memory from a tensor.

// Engine cannot handle incr or safe. See opMem()

// some error messages
const (
	cannotUseReuse  = "Cannot use `reuse`."
	cannotUseUnsafe = "Cannot use unsafe as an option."

	cannotDoSafeOps = "CUDA Engines do not support Safe operations"
)

func checkArrayShape(a tensor.DataSizer, expShape shapes.Shape) error {
	dataSize := a.DataSize()
	expSize := expShape.TotalSize()
	if dataSize < expSize && !expShape.IsScalar() {
		return errors.Errorf(errors.ArrayMismatch, dataSize, expSize)
	}
	return nil
}

func checkDtype(expDtype dtype.Dtype, got dtype.Dtype) error {
	if expDtype == nil {
		return nil
	}
	if !got.Eq(expDtype) {
		return errors.Errorf(errors.DtypeError, expDtype, got)
	}
	return nil
}

func (e *Engine[DT, T]) HandleFuncOptsSpecialized(a T, expShape shapes.Shape, opts ...tensor.FuncOpt) (retVal T, fo tensor.Option, err error) {
	fo = tensor.ParseFuncOpts(opts...)

	reuseAny := fo.Reuse
	toReuse := reuseAny != nil
	safe := fo.Safe()

	var ok bool
	var empty T

	if fo.AsType != nil {
		err = errors.Errorf("Cannot use HandleFuncOptsSpecialized with a return tensor of a different dtype %v.")
		return
	}
	if fo.Incr {
		err = errors.New("CUDA engines do not support increment")
		return
	}

	switch {
	case toReuse:
		if retVal, ok = reuseAny.(T); !ok {
			err = errors.Wrap(errors.Errorf(errors.TypeMismatch, empty, reuseAny), cannotUseReuse)
			return
		}

		// restore the state to the original
		retVal.Restore()

		if err = checkArrayShape(retVal, expShape); err != nil {
			err = errors.Wrap(err, cannotUseReuse)
			return
		}

		if !retVal.Shape().Eq(expShape) {
			if err = retVal.Reshape(expShape...); err != nil {
				return
			}
		}

		if !fo.Incr {
			retVal.SetDataOrder(a.DataOrder())
		}
	case !safe:
		// if the reuse tensor has a smaller array size than the expected array size, we cannot use unsafe.
		if err = checkArrayShape(a, expShape); err != nil {
			err = errors.Wrap(err, cannotUseUnsafe)
			return
		}

		// we reshape the value to the expected shape
		if !a.Shape().Eq(expShape) {
			if err = a.Reshape(expShape...); err != nil {
				err = errors.Wrap(err, cannotUseReuse)
				return
			}
		}
		retVal = a
	default:
		// safe
		err = errors.Errorf(cannotDoSafeOps)
		return
	}
	return
}

func (e *Engine[DT, T]) HandleFuncOpts(a tensor.Basic[DT], expShape shapes.Shape, opts ...tensor.FuncOpt) (retVal tensor.Basic[DT], fo tensor.Option, err error) {
	fo = tensor.ParseFuncOpts(opts...)

	reuseAny := fo.Reuse
	toReuse := reuseAny != nil
	safe := fo.Safe()

	var ok bool
	var empty tensor.Basic[DT]

	if fo.AsType != nil {
		err = errors.Errorf("Cannot use HandleFuncOptsSpecialized with a return tensor of a different dtype %v.")
		return
	}

	if fo.Incr {
		err = errors.New("CUDA engines do not support increment")
		return
	}

	switch {
	case toReuse:
		if retVal, ok = reuseAny.(tensor.Basic[DT]); !ok {
			err = errors.Wrap(errors.Errorf(errors.TypeMismatch, empty, reuseAny), cannotUseReuse)
			return
		}

		// restore the state to the original
		retVal.Restore()

		if err = checkArrayShape(retVal, expShape); err != nil {
			err = errors.Wrap(err, cannotUseReuse)
			return
		}

		if !retVal.Shape().Eq(expShape) {
			if err = retVal.Reshape(expShape...); err != nil {
				return
			}
		}

		if !fo.Incr {
			retVal.SetDataOrder(a.DataOrder())
		}
	case !safe:
		// if the reuse tensor has a smaller array size than the expected array size, we cannot use unsafe.
		if err = checkArrayShape(a, expShape); err != nil {
			err = errors.Wrap(err, cannotUseUnsafe)
			return
		}

		// we reshape the value to the expected shape
		if !a.Shape().Eq(expShape) {
			if err = a.Reshape(expShape...); err != nil {
				err = errors.Wrap(err, cannotUseReuse)
				return
			}
		}
		retVal = a
	default:
		// safe
		err = errors.Errorf(cannotDoSafeOps)
		return
	}
	return
}

func (e *Engine[DT, T]) HandleFuncOptsDesc(a tensor.Basic[DT], expShape shapes.Shape, opts ...tensor.FuncOpt) (retVal tensor.DescWithStorage, fo tensor.Option, err error) {
	fo = tensor.ParseFuncOpts(opts...)

	reuseAny := fo.Reuse
	toReuse := reuseAny != nil
	safe := fo.Safe()

	asType := fo.AsType

	var ok bool
	switch {
	case toReuse:
		if retVal, ok = reuseAny.(tensor.DescWithStorage); !ok {
			err = errors.Wrap(errors.Errorf(errors.TypeMismatch, retVal, reuseAny), cannotUseReuse)
			return
		}
		// if asType is not nil, then we would expect that the reuse should be of the same dtype
		if err = checkDtype(asType, retVal.Dtype()); err != nil {
			err = errors.Wrap(err, cannotUseReuse)
			return
		}

		// restore the state to the original
		retVal.Restore()

		// if the reuse tensor has a smaller array size than the expected array size, then we cannot proceed
		if err = checkArrayShape(retVal, expShape); err != nil {
			err = errors.Wrap(err, cannotUseReuse)
			return
		}

		// naughty...
		if !retVal.Shape().Eq(expShape) {
			if err = retVal.Reshape(expShape...); err != nil {
				err = errors.Wrap(err, cannotUseReuse)
				return
			}
		}

		// if the reuse is the target of an increment operation, then we don't set the data order.
		if !fo.Incr {
			retVal.SetDataOrder(a.DataOrder())
		}
	case !safe:
		// if asType is not nil, then we would expect that the reuse should be of the same dtype
		if err = checkDtype(asType, retVal.Dtype()); err != nil {
			err = errors.Wrap(err, cannotUseReuse)
			return
		}

		// if the reuse tensor has a smaller array size than the expected array size, we cannot use unsafe.
		if err = checkArrayShape(a, expShape); err != nil {
			err = errors.Wrap(err, cannotUseUnsafe)
			return
		}

		// we reshape the value to the expected shape
		if !a.Shape().Eq(expShape) {
			if err = a.Reshape(expShape...); err != nil {
				err = errors.Wrap(err, cannotUseReuse)
				return
			}
		}

		retVal = a
	default:
		// safe
		if asType != nil {
			retVal = a.AlikeAsType(asType, tensor.WithShape(expShape...))
			return
		}

		retVal = a.AlikeAsDescWithStorage(tensor.WithShape(expShape...))
	}
	return
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
func (e *Engine[DT, T]) opMem(operands ...T) (mem, memB cu.DevicePtr, size int64) {
	var a, b, retVal T
	switch len(operands) {
	case 2:
		a = operands[0]
		retVal = operands[1]
	case 3:
		a = operands[0]
		b = operands[1]
		retVal = operands[2]
		memB = cu.DevicePtr(b.Uintptr())
	default:
		panic("opMem only supports either two operands or three operands")
	}
	mem = cu.DevicePtr(retVal.Uintptr())
	memA := cu.DevicePtr(a.Uintptr())
	if memA != mem {
		log.Printf("opmem")
		memSize := int64(a.MemSize())
		e.memcpy(mem, memA, memSize)
	}

	size = int64(logicalSize(retVal.Shape()))
	return
}
func (e *Engine[DT, T]) opMemBC(a, b, retVal T) (memA, memB, memC cu.DevicePtr, size int64) {
	memA = cu.DevicePtr(a.Uintptr())
	memB = cu.DevicePtr(b.Uintptr())
	memC = cu.DevicePtr(retVal.Uintptr())
	size = int64(logicalSize(retVal.Shape()))
	return
}

func (e *Engine[DT, T]) prepShapes(expAPA, expAPB *tensor.AP, retVal T) (sp []cu.DevicePtr, totalAlloc uintptr, err error) {
	expShapeA := expAPA.Shape()
	expShapeB := expAPB.Shape()
	retShape := retVal.Shape()
	expStridesA := expAPA.Strides()
	expStridesB := expAPB.Strides()
	totalAlloc = sliceMemSize(expShapeA)
	// we call Get here instead of e.c.MemAlloc because this engine is manually managed, baby!
	memShape, err := e.Get(int64(totalAlloc * 5)) // ×5 is all that is needed because it's assumed that all the shapes and strides have the same dims
	if err != nil {
		return nil, 0, err
	}
	ss := [][]int{
		[]int(expShapeA),
		[]int(expShapeB),
		[]int(retShape),
		expStridesA,
		expStridesB,
	}
	sp = make([]cu.DevicePtr, len(ss))
	var offset uintptr
	for i, s := range ss {
		offset += uintptr(i) * totalAlloc
		ptr := cu.DevicePtr(memShape.Uintptr() + offset)
		sp[i] = ptr
		e.c.MemcpyHtoD(ptr, unsafe.Pointer(&s[0]), int64(totalAlloc))
	}
	return sp, totalAlloc, nil
}
