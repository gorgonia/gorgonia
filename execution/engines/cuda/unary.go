package cuda

import (
	"context"
	"unsafe"

	"gorgonia.org/gorgonia/internal/errors"
	"gorgonia.org/internal/debug"
)

// Code generated by gencudaengine, which is a API generation tool for Gorgonia. DO NOT EDIT.

// Neg implements tensor.Neger. It does not support safe or increment options and will return an error if those options are passed in.
func (e *Engine[DT, T]) Neg(ctx context.Context, a T, retVal T) (err error) {
	name := constructUnOpName(a, "neg")
	mem, _, size := e.opMem(a, retVal)

	debug.Logf("CUDADO %q, Mem: %v size %v, args %v", name, mem, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&size)); err != nil {
		err = errors.Wrap(err, "Unable to perform engine.Neg - CUDA LaunchAndSync failed")
	}
	return
}

// Inv implements tensor.Inver. It does not support safe or increment options and will return an error if those options are passed in.
func (e *Engine[DT, T]) Inv(ctx context.Context, a T, retVal T) (err error) {
	name := constructUnOpName(a, "inverse")
	mem, _, size := e.opMem(a, retVal)

	debug.Logf("CUDADO %q, Mem: %v size %v, args %v", name, mem, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&size)); err != nil {
		err = errors.Wrap(err, "Unable to perform engine.Neg - CUDA LaunchAndSync failed")
	}
	return
}

// Square implements tensor.Squareer. It does not support safe or increment options and will return an error if those options are passed in.
func (e *Engine[DT, T]) Square(ctx context.Context, a T, retVal T) (err error) {
	name := constructUnOpName(a, "square")
	mem, _, size := e.opMem(a, retVal)

	debug.Logf("CUDADO %q, Mem: %v size %v, args %v", name, mem, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&size)); err != nil {
		err = errors.Wrap(err, "Unable to perform engine.Neg - CUDA LaunchAndSync failed")
	}
	return
}

// Cube implements tensor.Cubeer. It does not support safe or increment options and will return an error if those options are passed in.
func (e *Engine[DT, T]) Cube(ctx context.Context, a T, retVal T) (err error) {
	name := constructUnOpName(a, "cube")
	mem, _, size := e.opMem(a, retVal)

	debug.Logf("CUDADO %q, Mem: %v size %v, args %v", name, mem, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&size)); err != nil {
		err = errors.Wrap(err, "Unable to perform engine.Neg - CUDA LaunchAndSync failed")
	}
	return
}

// Exp implements tensor.Exper. It does not support safe or increment options and will return an error if those options are passed in.
func (e *Engine[DT, T]) Exp(ctx context.Context, a T, retVal T) (err error) {
	name := constructUnOpName(a, "exp")
	mem, _, size := e.opMem(a, retVal)

	debug.Logf("CUDADO %q, Mem: %v size %v, args %v", name, mem, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&size)); err != nil {
		err = errors.Wrap(err, "Unable to perform engine.Neg - CUDA LaunchAndSync failed")
	}
	return
}

// Tanh implements tensor.Tanher. It does not support safe or increment options and will return an error if those options are passed in.
func (e *Engine[DT, T]) Tanh(ctx context.Context, a T, retVal T) (err error) {
	name := constructUnOpName(a, "tanh")
	mem, _, size := e.opMem(a, retVal)

	debug.Logf("CUDADO %q, Mem: %v size %v, args %v", name, mem, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&size)); err != nil {
		err = errors.Wrap(err, "Unable to perform engine.Neg - CUDA LaunchAndSync failed")
	}
	return
}

// Log implements tensor.Loger. It does not support safe or increment options and will return an error if those options are passed in.
func (e *Engine[DT, T]) Log(ctx context.Context, a T, retVal T) (err error) {
	name := constructUnOpName(a, "ln")
	mem, _, size := e.opMem(a, retVal)

	debug.Logf("CUDADO %q, Mem: %v size %v, args %v", name, mem, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&size)); err != nil {
		err = errors.Wrap(err, "Unable to perform engine.Neg - CUDA LaunchAndSync failed")
	}
	return
}

// Log2 implements tensor.Log2er. It does not support safe or increment options and will return an error if those options are passed in.
func (e *Engine[DT, T]) Log2(ctx context.Context, a T, retVal T) (err error) {
	name := constructUnOpName(a, "log2")
	mem, _, size := e.opMem(a, retVal)

	debug.Logf("CUDADO %q, Mem: %v size %v, args %v", name, mem, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&size)); err != nil {
		err = errors.Wrap(err, "Unable to perform engine.Neg - CUDA LaunchAndSync failed")
	}
	return
}

// Log10 implements tensor.Log10er. It does not support safe or increment options and will return an error if those options are passed in.
func (e *Engine[DT, T]) Log10(ctx context.Context, a T, retVal T) (err error) {
	name := constructUnOpName(a, "log10")
	mem, _, size := e.opMem(a, retVal)

	debug.Logf("CUDADO %q, Mem: %v size %v, args %v", name, mem, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&size)); err != nil {
		err = errors.Wrap(err, "Unable to perform engine.Neg - CUDA LaunchAndSync failed")
	}
	return
}

// Sqrt implements tensor.Sqrter. It does not support safe or increment options and will return an error if those options are passed in.
func (e *Engine[DT, T]) Sqrt(ctx context.Context, a T, retVal T) (err error) {
	name := constructUnOpName(a, "sqrt")
	mem, _, size := e.opMem(a, retVal)

	debug.Logf("CUDADO %q, Mem: %v size %v, args %v", name, mem, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&size)); err != nil {
		err = errors.Wrap(err, "Unable to perform engine.Neg - CUDA LaunchAndSync failed")
	}
	return
}

// Cbrt implements tensor.Cbrter. It does not support safe or increment options and will return an error if those options are passed in.
func (e *Engine[DT, T]) Cbrt(ctx context.Context, a T, retVal T) (err error) {
	name := constructUnOpName(a, "cbrt")
	mem, _, size := e.opMem(a, retVal)

	debug.Logf("CUDADO %q, Mem: %v size %v, args %v", name, mem, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&size)); err != nil {
		err = errors.Wrap(err, "Unable to perform engine.Neg - CUDA LaunchAndSync failed")
	}
	return
}

// InvSqrt implements tensor.InvSqrter. It does not support safe or increment options and will return an error if those options are passed in.
func (e *Engine[DT, T]) InvSqrt(ctx context.Context, a T, retVal T) (err error) {
	name := constructUnOpName(a, "invsqrt")
	mem, _, size := e.opMem(a, retVal)

	debug.Logf("CUDADO %q, Mem: %v size %v, args %v", name, mem, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&size)); err != nil {
		err = errors.Wrap(err, "Unable to perform engine.Neg - CUDA LaunchAndSync failed")
	}
	return
}

// Sign implements tensor.Signer. It does not support safe or increment options and will return an error if those options are passed in.
func (e *Engine[DT, T]) Sign(ctx context.Context, a T, retVal T) (err error) {
	name := constructUnOpName(a, "sign")
	mem, _, size := e.opMem(a, retVal)

	debug.Logf("CUDADO %q, Mem: %v size %v, args %v", name, mem, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&size)); err != nil {
		err = errors.Wrap(err, "Unable to perform engine.Neg - CUDA LaunchAndSync failed")
	}
	return
}

// Log1p implements tensor.Log1per. It does not support safe or increment options and will return an error if those options are passed in.
func (e *Engine[DT, T]) Log1p(ctx context.Context, a T, retVal T) (err error) {
	name := constructUnOpName(a, "log1p")
	mem, _, size := e.opMem(a, retVal)

	debug.Logf("CUDADO %q, Mem: %v size %v, args %v", name, mem, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&size)); err != nil {
		err = errors.Wrap(err, "Unable to perform engine.Neg - CUDA LaunchAndSync failed")
	}
	return
}

// Expm1 implements tensor.Expm1er. It does not support safe or increment options and will return an error if those options are passed in.
func (e *Engine[DT, T]) Expm1(ctx context.Context, a T, retVal T) (err error) {
	name := constructUnOpName(a, "expm1")
	mem, _, size := e.opMem(a, retVal)

	debug.Logf("CUDADO %q, Mem: %v size %v, args %v", name, mem, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&size)); err != nil {
		err = errors.Wrap(err, "Unable to perform engine.Neg - CUDA LaunchAndSync failed")
	}
	return
}

// Cos implements tensor.Coser. It does not support safe or increment options and will return an error if those options are passed in.
func (e *Engine[DT, T]) Cos(ctx context.Context, a T, retVal T) (err error) {
	name := constructUnOpName(a, "cos")
	mem, _, size := e.opMem(a, retVal)

	debug.Logf("CUDADO %q, Mem: %v size %v, args %v", name, mem, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&size)); err != nil {
		err = errors.Wrap(err, "Unable to perform engine.Neg - CUDA LaunchAndSync failed")
	}
	return
}

// Sin implements tensor.Siner. It does not support safe or increment options and will return an error if those options are passed in.
func (e *Engine[DT, T]) Sin(ctx context.Context, a T, retVal T) (err error) {
	name := constructUnOpName(a, "sin")
	mem, _, size := e.opMem(a, retVal)

	debug.Logf("CUDADO %q, Mem: %v size %v, args %v", name, mem, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&size)); err != nil {
		err = errors.Wrap(err, "Unable to perform engine.Neg - CUDA LaunchAndSync failed")
	}
	return
}

// Abs implements tensor.Abser. It does not support safe or increment options and will return an error if those options are passed in.
func (e *Engine[DT, T]) Abs(ctx context.Context, a T, retVal T) (err error) {
	name := constructUnOpName(a, "abs")
	mem, _, size := e.opMem(a, retVal)

	debug.Logf("CUDADO %q, Mem: %v size %v, args %v", name, mem, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&size)); err != nil {
		err = errors.Wrap(err, "Unable to perform engine.Neg - CUDA LaunchAndSync failed")
	}
	return
}

// Ceil implements tensor.Ceiler. It does not support safe or increment options and will return an error if those options are passed in.
func (e *Engine[DT, T]) Ceil(ctx context.Context, a T, retVal T) (err error) {
	name := constructUnOpName(a, "ceil")
	mem, _, size := e.opMem(a, retVal)

	debug.Logf("CUDADO %q, Mem: %v size %v, args %v", name, mem, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&size)); err != nil {
		err = errors.Wrap(err, "Unable to perform engine.Neg - CUDA LaunchAndSync failed")
	}
	return
}

// Floor implements tensor.Floorer. It does not support safe or increment options and will return an error if those options are passed in.
func (e *Engine[DT, T]) Floor(ctx context.Context, a T, retVal T) (err error) {
	name := constructUnOpName(a, "floor")
	mem, _, size := e.opMem(a, retVal)

	debug.Logf("CUDADO %q, Mem: %v size %v, args %v", name, mem, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&size)); err != nil {
		err = errors.Wrap(err, "Unable to perform engine.Neg - CUDA LaunchAndSync failed")
	}
	return
}

// Softplus implements tensor.Softpluser. It does not support safe or increment options and will return an error if those options are passed in.
func (e *Engine[DT, T]) Softplus(ctx context.Context, a T, retVal T) (err error) {
	name := constructUnOpName(a, "softplus")
	mem, _, size := e.opMem(a, retVal)

	debug.Logf("CUDADO %q, Mem: %v size %v, args %v", name, mem, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&size)); err != nil {
		err = errors.Wrap(err, "Unable to perform engine.Neg - CUDA LaunchAndSync failed")
	}
	return
}

// Sigmoid implements tensor.Sigmoider. It does not support safe or increment options and will return an error if those options are passed in.
func (e *Engine[DT, T]) Sigmoid(ctx context.Context, a T, retVal T) (err error) {
	name := constructUnOpName(a, "sigmoid")
	mem, _, size := e.opMem(a, retVal)

	debug.Logf("CUDADO %q, Mem: %v size %v, args %v", name, mem, size)
	debug.Logf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	if err = e.Call(name, int(size), unsafe.Pointer(&mem), unsafe.Pointer(&size)); err != nil {
		err = errors.Wrap(err, "Unable to perform engine.Neg - CUDA LaunchAndSync failed")
	}
	return
}