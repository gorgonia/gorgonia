package cuda

import (
	"context"
	"unsafe"

	"gorgonia.org/gorgonia/internal/errors"
	"gorgonia.org/internal/debug"
	"gorgonia.org/tensor"
)

// Code generated by gencudaengine, which is a API generation tool for Gorgonia. DO NOT EDIT.

// Lt implements tensor.Lter. It does not support safe or increment operation options and will return an error if those options are passed in.
func (e *Engine[DT, T]) Lt(ctx context.Context, a, b, retVal T, toIncr bool) (err error) {
	name, _, _ := constructBinName2(a, b, "lt", false)

	if err = binaryCheck[DT](a, b); err != nil {
		return errors.Wrap(err, "Basic checks failed for Lt")
	}
	mem, memB, size := e.opMem(a, b, retVal)

	debug.Logf("CUDADO %q, Mem: %v MemB: %v size %v", name, mem, memB, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&memB), unsafe.Pointer(&size)); err != nil {
		err = errors.Wrap(err, "Unable to perform engine.Lt - CUDA LaunchAndSync failed.")
	}
	return
}

// LtScalar implements tensor.Lter. It does not support safe or increment operation options and will return an error if those options are passed in.
func (e *Engine[DT, T]) LtScalar(ctx context.Context, a T, b DT, retVal T, leftTensor, toIncr bool) (err error) {
	return errors.NYI()
	/*
		name := constructBinName1(a, leftTensor, "lt")

		var bMem tensor.Memory
		var ok bool
		if bMem, ok = b.(tensor.Memory); !ok {
			return errors.Errorf("b has to be a tensor.Memory. Got %T instead", b)
		}

		if err = unaryCheck[DT](a); err != nil {
			return errors.Wrap(err, "Basic checks failed for LtScalar")
		}


		var mem, memB cu.DevicePtr
		var size int64
		if mem, size, retVal, err = e.opMem(a, opts...); err != nil{
			return errors.Wrap(err, "Unable to perform Lt")
		}
		memB = cu.DevicePtr(bMem.Uintptr())
		if !leftTensor {
			mem, memB = memB, mem
		}

		debug.Logf("CUDADO %q, Mem: %v size %v, args %v", name, mem, size)
		debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
		if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&memB), unsafe.Pointer(&size)); err != nil{
			err = errors.Wrap(err, "Unable to perform engine.Lt - CUDA LaunchAndSync failed.")
		}
		return
	*/
}

func (e *Engine[DT, T]) LtBroadcastable(ctx context.Context, a, b, retVal T, expAPA, expAPB *tensor.AP, toIncr bool) (err error) {
	// check if it's a scalar in a or b
	name, scalarOnLeft, scalarOnRight := constructBinName2(a, b, "lt", true)
	isScalar := scalarOnLeft || scalarOnRight
	// scalar
	if isScalar {
		var t T
		if scalarOnLeft {
			t = b
		} else {
			t = a
		}
		if err = unaryCheck[DT](t); err != nil {
			return errors.Wrap(err, "Basic checks failed for LtBroadcastable")
		}
		mem, memB, size := e.opMem(a, b, retVal)
		if scalarOnLeft {
			mem, memB = memB, mem
		}

		debug.Logf("CUDADO %q, Mem: %v MemB: %v size %v", name, mem, memB, size)
		debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
		if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&memB), unsafe.Pointer(&size)); err != nil {
			err = errors.Wrap(err, "Unable to perform engine.Add - CUDA LaunchAndSync failed.")
		}
		return
	}

	sp, totalAlloc, err := e.prepShapes(expAPA, expAPB, retVal)
	if err != nil {
		return errors.Wrap(err, "Failed to prep shapes")
	}
	_ = totalAlloc
	// TODO: sp is a slice of CUDA memory. They need to be freed. Add to this once the hook architecture is finished in package cu.

	mem, memB, memRetVal, size := e.opMemBC(a, b, retVal)
	debug.Logf("CUDADO %q, Mem: %v MemB: %v size %v", name, mem, memB, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&memB), unsafe.Pointer(&memRetVal),
		unsafe.Pointer(&sp[0]), unsafe.Pointer(&sp[1]), unsafe.Pointer(&sp[2]),
		unsafe.Pointer(&sp[3]), unsafe.Pointer(&sp[4]), unsafe.Pointer(&size)); err != nil {
		err = errors.Wrap(err, "Unable to perform engine.AddBC - CUDA LaunchAndSync failed.")
	}

	return

}

// Lte implements tensor.Lteer. It does not support safe or increment operation options and will return an error if those options are passed in.
func (e *Engine[DT, T]) Lte(ctx context.Context, a, b, retVal T, toIncr bool) (err error) {
	name, _, _ := constructBinName2(a, b, "lte", false)

	if err = binaryCheck[DT](a, b); err != nil {
		return errors.Wrap(err, "Basic checks failed for Lte")
	}
	mem, memB, size := e.opMem(a, b, retVal)

	debug.Logf("CUDADO %q, Mem: %v MemB: %v size %v", name, mem, memB, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&memB), unsafe.Pointer(&size)); err != nil {
		err = errors.Wrap(err, "Unable to perform engine.Lte - CUDA LaunchAndSync failed.")
	}
	return
}

// LteScalar implements tensor.Lteer. It does not support safe or increment operation options and will return an error if those options are passed in.
func (e *Engine[DT, T]) LteScalar(ctx context.Context, a T, b DT, retVal T, leftTensor, toIncr bool) (err error) {
	return errors.NYI()
	/*
		name := constructBinName1(a, leftTensor, "lte")

		var bMem tensor.Memory
		var ok bool
		if bMem, ok = b.(tensor.Memory); !ok {
			return errors.Errorf("b has to be a tensor.Memory. Got %T instead", b)
		}

		if err = unaryCheck[DT](a); err != nil {
			return errors.Wrap(err, "Basic checks failed for LteScalar")
		}


		var mem, memB cu.DevicePtr
		var size int64
		if mem, size, retVal, err = e.opMem(a, opts...); err != nil{
			return errors.Wrap(err, "Unable to perform Lte")
		}
		memB = cu.DevicePtr(bMem.Uintptr())
		if !leftTensor {
			mem, memB = memB, mem
		}

		debug.Logf("CUDADO %q, Mem: %v size %v, args %v", name, mem, size)
		debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
		if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&memB), unsafe.Pointer(&size)); err != nil{
			err = errors.Wrap(err, "Unable to perform engine.Lte - CUDA LaunchAndSync failed.")
		}
		return
	*/
}

func (e *Engine[DT, T]) LteBroadcastable(ctx context.Context, a, b, retVal T, expAPA, expAPB *tensor.AP, toIncr bool) (err error) {
	// check if it's a scalar in a or b
	name, scalarOnLeft, scalarOnRight := constructBinName2(a, b, "lte", true)
	isScalar := scalarOnLeft || scalarOnRight
	// scalar
	if isScalar {
		var t T
		if scalarOnLeft {
			t = b
		} else {
			t = a
		}
		if err = unaryCheck[DT](t); err != nil {
			return errors.Wrap(err, "Basic checks failed for LteBroadcastable")
		}
		mem, memB, size := e.opMem(a, b, retVal)
		if scalarOnLeft {
			mem, memB = memB, mem
		}

		debug.Logf("CUDADO %q, Mem: %v MemB: %v size %v", name, mem, memB, size)
		debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
		if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&memB), unsafe.Pointer(&size)); err != nil {
			err = errors.Wrap(err, "Unable to perform engine.Add - CUDA LaunchAndSync failed.")
		}
		return
	}

	sp, totalAlloc, err := e.prepShapes(expAPA, expAPB, retVal)
	if err != nil {
		return errors.Wrap(err, "Failed to prep shapes")
	}
	_ = totalAlloc
	// TODO: sp is a slice of CUDA memory. They need to be freed. Add to this once the hook architecture is finished in package cu.

	mem, memB, memRetVal, size := e.opMemBC(a, b, retVal)
	debug.Logf("CUDADO %q, Mem: %v MemB: %v size %v", name, mem, memB, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&memB), unsafe.Pointer(&memRetVal),
		unsafe.Pointer(&sp[0]), unsafe.Pointer(&sp[1]), unsafe.Pointer(&sp[2]),
		unsafe.Pointer(&sp[3]), unsafe.Pointer(&sp[4]), unsafe.Pointer(&size)); err != nil {
		err = errors.Wrap(err, "Unable to perform engine.AddBC - CUDA LaunchAndSync failed.")
	}

	return

}

// Gt implements tensor.Gter. It does not support safe or increment operation options and will return an error if those options are passed in.
func (e *Engine[DT, T]) Gt(ctx context.Context, a, b, retVal T, toIncr bool) (err error) {
	name, _, _ := constructBinName2(a, b, "gt", false)

	if err = binaryCheck[DT](a, b); err != nil {
		return errors.Wrap(err, "Basic checks failed for Gt")
	}
	mem, memB, size := e.opMem(a, b, retVal)

	debug.Logf("CUDADO %q, Mem: %v MemB: %v size %v", name, mem, memB, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&memB), unsafe.Pointer(&size)); err != nil {
		err = errors.Wrap(err, "Unable to perform engine.Gt - CUDA LaunchAndSync failed.")
	}
	return
}

// GtScalar implements tensor.Gter. It does not support safe or increment operation options and will return an error if those options are passed in.
func (e *Engine[DT, T]) GtScalar(ctx context.Context, a T, b DT, retVal T, leftTensor, toIncr bool) (err error) {
	return errors.NYI()
	/*
		name := constructBinName1(a, leftTensor, "gt")

		var bMem tensor.Memory
		var ok bool
		if bMem, ok = b.(tensor.Memory); !ok {
			return errors.Errorf("b has to be a tensor.Memory. Got %T instead", b)
		}

		if err = unaryCheck[DT](a); err != nil {
			return errors.Wrap(err, "Basic checks failed for GtScalar")
		}


		var mem, memB cu.DevicePtr
		var size int64
		if mem, size, retVal, err = e.opMem(a, opts...); err != nil{
			return errors.Wrap(err, "Unable to perform Gt")
		}
		memB = cu.DevicePtr(bMem.Uintptr())
		if !leftTensor {
			mem, memB = memB, mem
		}

		debug.Logf("CUDADO %q, Mem: %v size %v, args %v", name, mem, size)
		debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
		if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&memB), unsafe.Pointer(&size)); err != nil{
			err = errors.Wrap(err, "Unable to perform engine.Gt - CUDA LaunchAndSync failed.")
		}
		return
	*/
}

func (e *Engine[DT, T]) GtBroadcastable(ctx context.Context, a, b, retVal T, expAPA, expAPB *tensor.AP, toIncr bool) (err error) {
	// check if it's a scalar in a or b
	name, scalarOnLeft, scalarOnRight := constructBinName2(a, b, "gt", true)
	isScalar := scalarOnLeft || scalarOnRight
	// scalar
	if isScalar {
		var t T
		if scalarOnLeft {
			t = b
		} else {
			t = a
		}
		if err = unaryCheck[DT](t); err != nil {
			return errors.Wrap(err, "Basic checks failed for GtBroadcastable")
		}
		mem, memB, size := e.opMem(a, b, retVal)
		if scalarOnLeft {
			mem, memB = memB, mem
		}

		debug.Logf("CUDADO %q, Mem: %v MemB: %v size %v", name, mem, memB, size)
		debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
		if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&memB), unsafe.Pointer(&size)); err != nil {
			err = errors.Wrap(err, "Unable to perform engine.Add - CUDA LaunchAndSync failed.")
		}
		return
	}

	sp, totalAlloc, err := e.prepShapes(expAPA, expAPB, retVal)
	if err != nil {
		return errors.Wrap(err, "Failed to prep shapes")
	}
	_ = totalAlloc
	// TODO: sp is a slice of CUDA memory. They need to be freed. Add to this once the hook architecture is finished in package cu.

	mem, memB, memRetVal, size := e.opMemBC(a, b, retVal)
	debug.Logf("CUDADO %q, Mem: %v MemB: %v size %v", name, mem, memB, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&memB), unsafe.Pointer(&memRetVal),
		unsafe.Pointer(&sp[0]), unsafe.Pointer(&sp[1]), unsafe.Pointer(&sp[2]),
		unsafe.Pointer(&sp[3]), unsafe.Pointer(&sp[4]), unsafe.Pointer(&size)); err != nil {
		err = errors.Wrap(err, "Unable to perform engine.AddBC - CUDA LaunchAndSync failed.")
	}

	return

}

// Gte implements tensor.Gteer. It does not support safe or increment operation options and will return an error if those options are passed in.
func (e *Engine[DT, T]) Gte(ctx context.Context, a, b, retVal T, toIncr bool) (err error) {
	name, _, _ := constructBinName2(a, b, "gte", false)

	if err = binaryCheck[DT](a, b); err != nil {
		return errors.Wrap(err, "Basic checks failed for Gte")
	}
	mem, memB, size := e.opMem(a, b, retVal)

	debug.Logf("CUDADO %q, Mem: %v MemB: %v size %v", name, mem, memB, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&memB), unsafe.Pointer(&size)); err != nil {
		err = errors.Wrap(err, "Unable to perform engine.Gte - CUDA LaunchAndSync failed.")
	}
	return
}

// GteScalar implements tensor.Gteer. It does not support safe or increment operation options and will return an error if those options are passed in.
func (e *Engine[DT, T]) GteScalar(ctx context.Context, a T, b DT, retVal T, leftTensor, toIncr bool) (err error) {
	return errors.NYI()
	/*
		name := constructBinName1(a, leftTensor, "gte")

		var bMem tensor.Memory
		var ok bool
		if bMem, ok = b.(tensor.Memory); !ok {
			return errors.Errorf("b has to be a tensor.Memory. Got %T instead", b)
		}

		if err = unaryCheck[DT](a); err != nil {
			return errors.Wrap(err, "Basic checks failed for GteScalar")
		}


		var mem, memB cu.DevicePtr
		var size int64
		if mem, size, retVal, err = e.opMem(a, opts...); err != nil{
			return errors.Wrap(err, "Unable to perform Gte")
		}
		memB = cu.DevicePtr(bMem.Uintptr())
		if !leftTensor {
			mem, memB = memB, mem
		}

		debug.Logf("CUDADO %q, Mem: %v size %v, args %v", name, mem, size)
		debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
		if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&memB), unsafe.Pointer(&size)); err != nil{
			err = errors.Wrap(err, "Unable to perform engine.Gte - CUDA LaunchAndSync failed.")
		}
		return
	*/
}

func (e *Engine[DT, T]) GteBroadcastable(ctx context.Context, a, b, retVal T, expAPA, expAPB *tensor.AP, toIncr bool) (err error) {
	// check if it's a scalar in a or b
	name, scalarOnLeft, scalarOnRight := constructBinName2(a, b, "gte", true)
	isScalar := scalarOnLeft || scalarOnRight
	// scalar
	if isScalar {
		var t T
		if scalarOnLeft {
			t = b
		} else {
			t = a
		}
		if err = unaryCheck[DT](t); err != nil {
			return errors.Wrap(err, "Basic checks failed for GteBroadcastable")
		}
		mem, memB, size := e.opMem(a, b, retVal)
		if scalarOnLeft {
			mem, memB = memB, mem
		}

		debug.Logf("CUDADO %q, Mem: %v MemB: %v size %v", name, mem, memB, size)
		debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
		if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&memB), unsafe.Pointer(&size)); err != nil {
			err = errors.Wrap(err, "Unable to perform engine.Add - CUDA LaunchAndSync failed.")
		}
		return
	}

	sp, totalAlloc, err := e.prepShapes(expAPA, expAPB, retVal)
	if err != nil {
		return errors.Wrap(err, "Failed to prep shapes")
	}
	_ = totalAlloc
	// TODO: sp is a slice of CUDA memory. They need to be freed. Add to this once the hook architecture is finished in package cu.

	mem, memB, memRetVal, size := e.opMemBC(a, b, retVal)
	debug.Logf("CUDADO %q, Mem: %v MemB: %v size %v", name, mem, memB, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&memB), unsafe.Pointer(&memRetVal),
		unsafe.Pointer(&sp[0]), unsafe.Pointer(&sp[1]), unsafe.Pointer(&sp[2]),
		unsafe.Pointer(&sp[3]), unsafe.Pointer(&sp[4]), unsafe.Pointer(&size)); err != nil {
		err = errors.Wrap(err, "Unable to perform engine.AddBC - CUDA LaunchAndSync failed.")
	}

	return

}

// ElEq implements tensor.ElEqer. It does not support safe or increment operation options and will return an error if those options are passed in.
func (e *Engine[DT, T]) ElEq(ctx context.Context, a, b, retVal T, toIncr bool) (err error) {
	name, _, _ := constructBinName2(a, b, "eq", false)

	if err = binaryCheck[DT](a, b); err != nil {
		return errors.Wrap(err, "Basic checks failed for ElEq")
	}
	mem, memB, size := e.opMem(a, b, retVal)

	debug.Logf("CUDADO %q, Mem: %v MemB: %v size %v", name, mem, memB, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&memB), unsafe.Pointer(&size)); err != nil {
		err = errors.Wrap(err, "Unable to perform engine.ElEq - CUDA LaunchAndSync failed.")
	}
	return
}

// EqScalar implements tensor.ElEqer. It does not support safe or increment operation options and will return an error if those options are passed in.
func (e *Engine[DT, T]) EqScalar(ctx context.Context, a T, b DT, retVal T, leftTensor, toIncr bool) (err error) {
	return errors.NYI()
	/*
		name := constructBinName1(a, leftTensor, "eq")

		var bMem tensor.Memory
		var ok bool
		if bMem, ok = b.(tensor.Memory); !ok {
			return errors.Errorf("b has to be a tensor.Memory. Got %T instead", b)
		}

		if err = unaryCheck[DT](a); err != nil {
			return errors.Wrap(err, "Basic checks failed for EqScalar")
		}


		var mem, memB cu.DevicePtr
		var size int64
		if mem, size, retVal, err = e.opMem(a, opts...); err != nil{
			return errors.Wrap(err, "Unable to perform ElEq")
		}
		memB = cu.DevicePtr(bMem.Uintptr())
		if !leftTensor {
			mem, memB = memB, mem
		}

		debug.Logf("CUDADO %q, Mem: %v size %v, args %v", name, mem, size)
		debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
		if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&memB), unsafe.Pointer(&size)); err != nil{
			err = errors.Wrap(err, "Unable to perform engine.Eq - CUDA LaunchAndSync failed.")
		}
		return
	*/
}

func (e *Engine[DT, T]) EqBroadcastable(ctx context.Context, a, b, retVal T, expAPA, expAPB *tensor.AP, toIncr bool) (err error) {
	// check if it's a scalar in a or b
	name, scalarOnLeft, scalarOnRight := constructBinName2(a, b, "eq", true)
	isScalar := scalarOnLeft || scalarOnRight
	// scalar
	if isScalar {
		var t T
		if scalarOnLeft {
			t = b
		} else {
			t = a
		}
		if err = unaryCheck[DT](t); err != nil {
			return errors.Wrap(err, "Basic checks failed for EqBroadcastable")
		}
		mem, memB, size := e.opMem(a, b, retVal)
		if scalarOnLeft {
			mem, memB = memB, mem
		}

		debug.Logf("CUDADO %q, Mem: %v MemB: %v size %v", name, mem, memB, size)
		debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
		if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&memB), unsafe.Pointer(&size)); err != nil {
			err = errors.Wrap(err, "Unable to perform engine.Add - CUDA LaunchAndSync failed.")
		}
		return
	}

	sp, totalAlloc, err := e.prepShapes(expAPA, expAPB, retVal)
	if err != nil {
		return errors.Wrap(err, "Failed to prep shapes")
	}
	_ = totalAlloc
	// TODO: sp is a slice of CUDA memory. They need to be freed. Add to this once the hook architecture is finished in package cu.

	mem, memB, memRetVal, size := e.opMemBC(a, b, retVal)
	debug.Logf("CUDADO %q, Mem: %v MemB: %v size %v", name, mem, memB, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&memB), unsafe.Pointer(&memRetVal),
		unsafe.Pointer(&sp[0]), unsafe.Pointer(&sp[1]), unsafe.Pointer(&sp[2]),
		unsafe.Pointer(&sp[3]), unsafe.Pointer(&sp[4]), unsafe.Pointer(&size)); err != nil {
		err = errors.Wrap(err, "Unable to perform engine.AddBC - CUDA LaunchAndSync failed.")
	}

	return

}

// ElNe implements tensor.ElNeer. It does not support safe or increment operation options and will return an error if those options are passed in.
func (e *Engine[DT, T]) ElNe(ctx context.Context, a, b, retVal T, toIncr bool) (err error) {
	name, _, _ := constructBinName2(a, b, "ne", false)

	if err = binaryCheck[DT](a, b); err != nil {
		return errors.Wrap(err, "Basic checks failed for ElNe")
	}
	mem, memB, size := e.opMem(a, b, retVal)

	debug.Logf("CUDADO %q, Mem: %v MemB: %v size %v", name, mem, memB, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&memB), unsafe.Pointer(&size)); err != nil {
		err = errors.Wrap(err, "Unable to perform engine.ElNe - CUDA LaunchAndSync failed.")
	}
	return
}

// NeScalar implements tensor.ElNeer. It does not support safe or increment operation options and will return an error if those options are passed in.
func (e *Engine[DT, T]) NeScalar(ctx context.Context, a T, b DT, retVal T, leftTensor, toIncr bool) (err error) {
	return errors.NYI()
	/*
		name := constructBinName1(a, leftTensor, "ne")

		var bMem tensor.Memory
		var ok bool
		if bMem, ok = b.(tensor.Memory); !ok {
			return errors.Errorf("b has to be a tensor.Memory. Got %T instead", b)
		}

		if err = unaryCheck[DT](a); err != nil {
			return errors.Wrap(err, "Basic checks failed for NeScalar")
		}


		var mem, memB cu.DevicePtr
		var size int64
		if mem, size, retVal, err = e.opMem(a, opts...); err != nil{
			return errors.Wrap(err, "Unable to perform ElNe")
		}
		memB = cu.DevicePtr(bMem.Uintptr())
		if !leftTensor {
			mem, memB = memB, mem
		}

		debug.Logf("CUDADO %q, Mem: %v size %v, args %v", name, mem, size)
		debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
		if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&memB), unsafe.Pointer(&size)); err != nil{
			err = errors.Wrap(err, "Unable to perform engine.Ne - CUDA LaunchAndSync failed.")
		}
		return
	*/
}

func (e *Engine[DT, T]) NeBroadcastable(ctx context.Context, a, b, retVal T, expAPA, expAPB *tensor.AP, toIncr bool) (err error) {
	// check if it's a scalar in a or b
	name, scalarOnLeft, scalarOnRight := constructBinName2(a, b, "ne", true)
	isScalar := scalarOnLeft || scalarOnRight
	// scalar
	if isScalar {
		var t T
		if scalarOnLeft {
			t = b
		} else {
			t = a
		}
		if err = unaryCheck[DT](t); err != nil {
			return errors.Wrap(err, "Basic checks failed for NeBroadcastable")
		}
		mem, memB, size := e.opMem(a, b, retVal)
		if scalarOnLeft {
			mem, memB = memB, mem
		}

		debug.Logf("CUDADO %q, Mem: %v MemB: %v size %v", name, mem, memB, size)
		debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
		if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&memB), unsafe.Pointer(&size)); err != nil {
			err = errors.Wrap(err, "Unable to perform engine.Add - CUDA LaunchAndSync failed.")
		}
		return
	}

	sp, totalAlloc, err := e.prepShapes(expAPA, expAPB, retVal)
	if err != nil {
		return errors.Wrap(err, "Failed to prep shapes")
	}
	_ = totalAlloc
	// TODO: sp is a slice of CUDA memory. They need to be freed. Add to this once the hook architecture is finished in package cu.

	mem, memB, memRetVal, size := e.opMemBC(a, b, retVal)
	debug.Logf("CUDADO %q, Mem: %v MemB: %v size %v", name, mem, memB, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&memB), unsafe.Pointer(&memRetVal),
		unsafe.Pointer(&sp[0]), unsafe.Pointer(&sp[1]), unsafe.Pointer(&sp[2]),
		unsafe.Pointer(&sp[3]), unsafe.Pointer(&sp[4]), unsafe.Pointer(&size)); err != nil {
		err = errors.Wrap(err, "Unable to perform engine.AddBC - CUDA LaunchAndSync failed.")
	}

	return

}
