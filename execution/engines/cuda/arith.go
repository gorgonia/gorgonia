package cuda

import (
	"context"
	"unsafe"

	"gorgonia.org/gorgonia/internal/errors"
	"gorgonia.org/internal/debug"
)

// Code generated by gencudaengine, which is a API generation tool for Gorgonia. DO NOT EDIT.

// Add implements tensor.Adder. It does not support safe or increment operation options and will return an error if those options are passed in.
func (e *Engine[DT, T]) Add(ctx context.Context, a, b, retVal T, toIncr bool) (err error) {
	name := constructBinName2(a, b, "add")

	if err = binaryCheck[DT](a, b); err != nil {
		return errors.Wrap(err, "Basic checks failed for Add")
	}
	mem, memB, size := e.opMem(a, b, retVal)

	debug.Logf("CUDADO %q, Mem: %v MemB: %v size %v", name, mem, memB, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&memB), unsafe.Pointer(&size)); err != nil {
		err = errors.Wrap(err, "Unable to perform engine.Add - CUDA LaunchAndSync failed.")
	}
	return
}

// AddScalar implements tensor.Adder. It does not support safe or increment operation options and will return an error if those options are passed in.
func (e *Engine[DT, T]) AddScalar(ctx context.Context, a T, b DT, retVal T, leftTensor, toIncr bool) (err error) {
	return errors.NYI()
	/*
		name := constructBinName1(a, leftTensor, "add")

		var bMem tensor.Memory
		var ok bool
		if bMem, ok = b.(tensor.Memory); !ok {
			return errors.Errorf("b has to be a tensor.Memory. Got %T instead", b)
		}

		if err = unaryCheck[DT](a); err != nil {
			return errors.Wrap(err, "Basic checks failed for AddScalar")
		}


		var mem, memB cu.DevicePtr
		var size int64
		if mem, size, retVal, err = e.opMem(a, opts...); err != nil{
			return errors.Wrap(err, "Unable to perform Add")
		}
		memB = cu.DevicePtr(bMem.Uintptr())
		if !leftTensor {
			mem, memB = memB, mem
		}

		debug.Logf("CUDADO %q, Mem: %v size %v, args %v", name, mem, size)
		debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
		if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&memB), unsafe.Pointer(&size)); err != nil{
			err = errors.Wrap(err, "Unable to perform engine.Add - CUDA LaunchAndSync failed.")
		}
		return
	*/
}

// Sub implements tensor.Suber. It does not support safe or increment operation options and will return an error if those options are passed in.
func (e *Engine[DT, T]) Sub(ctx context.Context, a, b, retVal T, toIncr bool) (err error) {
	name := constructBinName2(a, b, "sub")

	if err = binaryCheck[DT](a, b); err != nil {
		return errors.Wrap(err, "Basic checks failed for Sub")
	}
	mem, memB, size := e.opMem(a, b, retVal)

	debug.Logf("CUDADO %q, Mem: %v MemB: %v size %v", name, mem, memB, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&memB), unsafe.Pointer(&size)); err != nil {
		err = errors.Wrap(err, "Unable to perform engine.Sub - CUDA LaunchAndSync failed.")
	}
	return
}

// SubScalar implements tensor.Suber. It does not support safe or increment operation options and will return an error if those options are passed in.
func (e *Engine[DT, T]) SubScalar(ctx context.Context, a T, b DT, retVal T, leftTensor, toIncr bool) (err error) {
	return errors.NYI()
	/*
		name := constructBinName1(a, leftTensor, "sub")

		var bMem tensor.Memory
		var ok bool
		if bMem, ok = b.(tensor.Memory); !ok {
			return errors.Errorf("b has to be a tensor.Memory. Got %T instead", b)
		}

		if err = unaryCheck[DT](a); err != nil {
			return errors.Wrap(err, "Basic checks failed for SubScalar")
		}


		var mem, memB cu.DevicePtr
		var size int64
		if mem, size, retVal, err = e.opMem(a, opts...); err != nil{
			return errors.Wrap(err, "Unable to perform Sub")
		}
		memB = cu.DevicePtr(bMem.Uintptr())
		if !leftTensor {
			mem, memB = memB, mem
		}

		debug.Logf("CUDADO %q, Mem: %v size %v, args %v", name, mem, size)
		debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
		if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&memB), unsafe.Pointer(&size)); err != nil{
			err = errors.Wrap(err, "Unable to perform engine.Sub - CUDA LaunchAndSync failed.")
		}
		return
	*/
}

// Mul implements tensor.Muler. It does not support safe or increment operation options and will return an error if those options are passed in.
func (e *Engine[DT, T]) Mul(ctx context.Context, a, b, retVal T, toIncr bool) (err error) {
	name := constructBinName2(a, b, "mul")

	if err = binaryCheck[DT](a, b); err != nil {
		return errors.Wrap(err, "Basic checks failed for Mul")
	}
	mem, memB, size := e.opMem(a, b, retVal)

	debug.Logf("CUDADO %q, Mem: %v MemB: %v size %v", name, mem, memB, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&memB), unsafe.Pointer(&size)); err != nil {
		err = errors.Wrap(err, "Unable to perform engine.Mul - CUDA LaunchAndSync failed.")
	}
	return
}

// MulScalar implements tensor.Muler. It does not support safe or increment operation options and will return an error if those options are passed in.
func (e *Engine[DT, T]) MulScalar(ctx context.Context, a T, b DT, retVal T, leftTensor, toIncr bool) (err error) {
	return errors.NYI()
	/*
		name := constructBinName1(a, leftTensor, "mul")

		var bMem tensor.Memory
		var ok bool
		if bMem, ok = b.(tensor.Memory); !ok {
			return errors.Errorf("b has to be a tensor.Memory. Got %T instead", b)
		}

		if err = unaryCheck[DT](a); err != nil {
			return errors.Wrap(err, "Basic checks failed for MulScalar")
		}


		var mem, memB cu.DevicePtr
		var size int64
		if mem, size, retVal, err = e.opMem(a, opts...); err != nil{
			return errors.Wrap(err, "Unable to perform Mul")
		}
		memB = cu.DevicePtr(bMem.Uintptr())
		if !leftTensor {
			mem, memB = memB, mem
		}

		debug.Logf("CUDADO %q, Mem: %v size %v, args %v", name, mem, size)
		debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
		if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&memB), unsafe.Pointer(&size)); err != nil{
			err = errors.Wrap(err, "Unable to perform engine.Mul - CUDA LaunchAndSync failed.")
		}
		return
	*/
}

// Div implements tensor.Diver. It does not support safe or increment operation options and will return an error if those options are passed in.
func (e *Engine[DT, T]) Div(ctx context.Context, a, b, retVal T, toIncr bool) (err error) {
	name := constructBinName2(a, b, "div")

	if err = binaryCheck[DT](a, b); err != nil {
		return errors.Wrap(err, "Basic checks failed for Div")
	}
	mem, memB, size := e.opMem(a, b, retVal)

	debug.Logf("CUDADO %q, Mem: %v MemB: %v size %v", name, mem, memB, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&memB), unsafe.Pointer(&size)); err != nil {
		err = errors.Wrap(err, "Unable to perform engine.Div - CUDA LaunchAndSync failed.")
	}
	return
}

// DivScalar implements tensor.Diver. It does not support safe or increment operation options and will return an error if those options are passed in.
func (e *Engine[DT, T]) DivScalar(ctx context.Context, a T, b DT, retVal T, leftTensor, toIncr bool) (err error) {
	return errors.NYI()
	/*
		name := constructBinName1(a, leftTensor, "div")

		var bMem tensor.Memory
		var ok bool
		if bMem, ok = b.(tensor.Memory); !ok {
			return errors.Errorf("b has to be a tensor.Memory. Got %T instead", b)
		}

		if err = unaryCheck[DT](a); err != nil {
			return errors.Wrap(err, "Basic checks failed for DivScalar")
		}


		var mem, memB cu.DevicePtr
		var size int64
		if mem, size, retVal, err = e.opMem(a, opts...); err != nil{
			return errors.Wrap(err, "Unable to perform Div")
		}
		memB = cu.DevicePtr(bMem.Uintptr())
		if !leftTensor {
			mem, memB = memB, mem
		}

		debug.Logf("CUDADO %q, Mem: %v size %v, args %v", name, mem, size)
		debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
		if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&memB), unsafe.Pointer(&size)); err != nil{
			err = errors.Wrap(err, "Unable to perform engine.Div - CUDA LaunchAndSync failed.")
		}
		return
	*/
}

// Pow implements tensor.Power. It does not support safe or increment operation options and will return an error if those options are passed in.
func (e *Engine[DT, T]) Pow(ctx context.Context, a, b, retVal T, toIncr bool) (err error) {
	name := constructBinName2(a, b, "pow")

	if err = binaryCheck[DT](a, b); err != nil {
		return errors.Wrap(err, "Basic checks failed for Pow")
	}
	mem, memB, size := e.opMem(a, b, retVal)

	debug.Logf("CUDADO %q, Mem: %v MemB: %v size %v", name, mem, memB, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&memB), unsafe.Pointer(&size)); err != nil {
		err = errors.Wrap(err, "Unable to perform engine.Pow - CUDA LaunchAndSync failed.")
	}
	return
}

// PowScalar implements tensor.Power. It does not support safe or increment operation options and will return an error if those options are passed in.
func (e *Engine[DT, T]) PowScalar(ctx context.Context, a T, b DT, retVal T, leftTensor, toIncr bool) (err error) {
	return errors.NYI()
	/*
		name := constructBinName1(a, leftTensor, "pow")

		var bMem tensor.Memory
		var ok bool
		if bMem, ok = b.(tensor.Memory); !ok {
			return errors.Errorf("b has to be a tensor.Memory. Got %T instead", b)
		}

		if err = unaryCheck[DT](a); err != nil {
			return errors.Wrap(err, "Basic checks failed for PowScalar")
		}


		var mem, memB cu.DevicePtr
		var size int64
		if mem, size, retVal, err = e.opMem(a, opts...); err != nil{
			return errors.Wrap(err, "Unable to perform Pow")
		}
		memB = cu.DevicePtr(bMem.Uintptr())
		if !leftTensor {
			mem, memB = memB, mem
		}

		debug.Logf("CUDADO %q, Mem: %v size %v, args %v", name, mem, size)
		debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
		if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&memB), unsafe.Pointer(&size)); err != nil{
			err = errors.Wrap(err, "Unable to perform engine.Pow - CUDA LaunchAndSync failed.")
		}
		return
	*/
}

// Mod implements tensor.Moder. It does not support safe or increment operation options and will return an error if those options are passed in.
func (e *Engine[DT, T]) Mod(ctx context.Context, a, b, retVal T, toIncr bool) (err error) {
	name := constructBinName2(a, b, "mod")

	if err = binaryCheck[DT](a, b); err != nil {
		return errors.Wrap(err, "Basic checks failed for Mod")
	}
	mem, memB, size := e.opMem(a, b, retVal)

	debug.Logf("CUDADO %q, Mem: %v MemB: %v size %v", name, mem, memB, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&memB), unsafe.Pointer(&size)); err != nil {
		err = errors.Wrap(err, "Unable to perform engine.Mod - CUDA LaunchAndSync failed.")
	}
	return
}

// ModScalar implements tensor.Moder. It does not support safe or increment operation options and will return an error if those options are passed in.
func (e *Engine[DT, T]) ModScalar(ctx context.Context, a T, b DT, retVal T, leftTensor, toIncr bool) (err error) {
	return errors.NYI()
	/*
		name := constructBinName1(a, leftTensor, "mod")

		var bMem tensor.Memory
		var ok bool
		if bMem, ok = b.(tensor.Memory); !ok {
			return errors.Errorf("b has to be a tensor.Memory. Got %T instead", b)
		}

		if err = unaryCheck[DT](a); err != nil {
			return errors.Wrap(err, "Basic checks failed for ModScalar")
		}


		var mem, memB cu.DevicePtr
		var size int64
		if mem, size, retVal, err = e.opMem(a, opts...); err != nil{
			return errors.Wrap(err, "Unable to perform Mod")
		}
		memB = cu.DevicePtr(bMem.Uintptr())
		if !leftTensor {
			mem, memB = memB, mem
		}

		debug.Logf("CUDADO %q, Mem: %v size %v, args %v", name, mem, size)
		debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
		if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&memB), unsafe.Pointer(&size)); err != nil{
			err = errors.Wrap(err, "Unable to perform engine.Mod - CUDA LaunchAndSync failed.")
		}
		return
	*/
}
