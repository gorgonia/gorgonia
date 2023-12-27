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
func (e *Engine[DT, T]) Lt(ctx context.Context, a, b, retVal T, opts ...tensor.FuncOpt) (err error) {
	name := constructBinName2(a, b, "lt")

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

// Lte implements tensor.Lteer. It does not support safe or increment operation options and will return an error if those options are passed in.
func (e *Engine[DT, T]) Lte(ctx context.Context, a, b, retVal T, opts ...tensor.FuncOpt) (err error) {
	name := constructBinName2(a, b, "lte")

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

// Gt implements tensor.Gter. It does not support safe or increment operation options and will return an error if those options are passed in.
func (e *Engine[DT, T]) Gt(ctx context.Context, a, b, retVal T, opts ...tensor.FuncOpt) (err error) {
	name := constructBinName2(a, b, "gt")

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

// Gte implements tensor.Gteer. It does not support safe or increment operation options and will return an error if those options are passed in.
func (e *Engine[DT, T]) Gte(ctx context.Context, a, b, retVal T, opts ...tensor.FuncOpt) (err error) {
	name := constructBinName2(a, b, "gte")

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

// ElEq implements tensor.ElEqer. It does not support safe or increment operation options and will return an error if those options are passed in.
func (e *Engine[DT, T]) ElEq(ctx context.Context, a, b, retVal T, opts ...tensor.FuncOpt) (err error) {
	name := constructBinName2(a, b, "eq")

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

// ElNe implements tensor.ElNeer. It does not support safe or increment operation options and will return an error if those options are passed in.
func (e *Engine[DT, T]) ElNe(ctx context.Context, a, b, retVal T, opts ...tensor.FuncOpt) (err error) {
	name := constructBinName2(a, b, "ne")

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
