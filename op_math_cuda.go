// +build cuda

package gorgonia

import (
	"fmt"
	"unsafe"

	"github.com/chewxy/cu"
	"github.com/chewxy/gorgonia/tensor"
	"github.com/chewxy/hm"
	"github.com/pkg/errors"
)

func (op elemUnaryOp) CallsExtern() bool { return true }

func (op elemUnaryOp) CUDADo(extern External, dev Device, inputTypes hm.Types, prealloc Memory, inputs ...Memory) (retVal Memory, err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}
	cudaLogf("CUDADoint %v", op)
	enterLoggingContext()
	defer leaveLoggingContext()

	// check
	cudaLogf("checking if input is scalar")
	a := inputs[0]
	if v, ok := a.(Value); ok && v.Shape().IsScalar() {
		return op.do(v)
	}

	var dt tensor.Dtype
	if dt, err = dtypeOf(inputTypes[0]); err != nil {
		return
	}

	name := fmt.Sprintf("%v%d", op.CUDAFuncName(), int(dt.Size())*8)
	if !extern.HasFunc(name) {
		cudaLogf("extern does not have func %q", name)
		if v, ok := a.(Value); ok {
			return op.do(v)
		}
		err = errors.Errorf("Cannot do %v on a non-Value mem %v", op, a)
		return
	}

	machine := extern.(CUDAMachine)
	fn := machine.Functions()[name][int(dev)]
	ctx := machine.Contexts()[int(dev)]

	// allocate if necessary
	cudaLogf("allocating if necessary")
	var mem cu.DevicePtr
	switch pre := prealloc.(type) {
	case Value:
		memsize := int64(pre.MemSize())
		if mem, err = cu.MemAlloc(memsize); err != nil {
			err = errors.Wrapf(err, "Failed to allocate %v bytes", memsize)
			return
		}

		// if the prealloc is a Value we want to copy the value back and then free
		defer func(ctx *cu.BatchedContext, val Value, mem cu.DevicePtr) {
			err = devPtrToValue(ctx, val, mem)
			ctx.MemFree(mem)
		}(ctx, pre, mem)

	case cu.DevicePtr:
		mem = pre
	}

	// copy
	cudaLogf("copying")
	var size int64
	switch at := a.(type) {
	case Value:
		size = int64(at.MemSize())
		ctx.MemcpyHtoD(mem, at.Pointer(), size)
	case cu.DevicePtr:
		size = int64(at.MemSize())
		ctx.Memcpy(mem, at, size)
	}

	gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ := machine.ElemGridSize(int(size), int(dev))
	args := []unsafe.Pointer{
		unsafe.Pointer(&mem),
		unsafe.Pointer(&size),
	}

	cudaLogf("CUDADO %q, size %v, args %v", name, size, args)
	ctx.LaunchAndSync(fn, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 0, cu.Stream(0), args)
	return mem, nil
}

func (op elemUnaryOp) CUDAFuncName() string {
	return op.String()
}

func (op elemBinOp) CallsExtern() bool { return true }

// func (op elemBinOp) CUDADo(extern External, fromDevs []Device, toDev Device, prealloc Memory, inputs ...Memory) (retVal Value, err error) {
// 	if err = checkArity(op, len(inputs)); err != nil {
// 		return
// 	}

// 	if !op.isArith() {
// 		if prealloc != nil && !prealloc.Shape().IsScalar() {
// 			return op.UsePreallocDo(prealloc, inputs...)
// 		}
// 		return op.Do(inputs...)
// 	}

// 	a := inputs[0]
// 	b := inputs[1]
// 	if a.Shape().IsScalar() || b.Shape().IsScalar() || prealloc == nil || prealloc.Shape().IsScalar() {
// 		return op.Do(inputs...)
// 	}

// 	name := fmt.Sprintf("%v%d", op.CUDAFuncName(), int(a.Dtype().Size())*8)
// 	if !extern.HasFunc(name) {
// 		cudaLogf("extern %T has no function %q", extern, name)
// 		if prealloc != nil {
// 			return op.UsePreallocDo(prealloc, inputs...)
// 		}
// 		return op.Do(inputs...)
// 	}

// 	// TODO: maybe check arity of fromDevs?
// 	dev := toDev

// 	machine := extern.(CUDAMachine)
// 	fns := machine.Functions()

// 	if len(machine.Contexts()) == 0 {
// 		if prealloc != nil {
// 			return op.UsePreallocDo(prealloc, inputs...)
// 		}
// 		return op.Do(inputs...)
// 	}

// 	if retVal, err = Copy(prealloc, a); err != nil {
// 		// TODO: maybe warn?
// 		if prealloc != nil {
// 			return op.UsePreallocDo(prealloc, inputs...)
// 		}
// 		return op.Do(inputs...)
// 	}

// 	fn := fns[name][int(dev)]
// 	// TODO: optimize kernel to maximize parallelization
// 	// var maxThreads int
// 	// d := cu.Device(dev)
// 	// if maxThreads, err = d.Attribute(cu.MaxThreadsPerBlock); err != nil {
// 	// 	return op.Do(inputs...) // resort to using slow methods if there was an error
// 	// }

// 	var memA, memB cu.DevicePtr
// 	if memA, err = valToDevicePointer(retVal); err != nil {
// 		err = errors.Wrapf(err, opDoFail)
// 		return
// 	}

// 	if memB, err = valToDevicePointer(b); err != nil {
// 		err = errors.Wrapf(err, opDoFail)
// 		return
// 	}

// 	// we don't want no leaks
// 	defer func(memA, memB cu.DevicePtr) {
// 		if err := cu.MemFree(memA); err != nil {
// 			cudaLogf("memfree(A): %v", err)
// 		}
// 		if err := cu.MemFree(memB); err != nil {
// 			cudaLogf("memfree(B): %v", err)
// 		}
// 	}(memA, memB)

// 	size := a.Shape().TotalSize()
// 	gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ := machine.ElemGridSize(size, int(dev))
// 	args := []unsafe.Pointer{
// 		unsafe.Pointer(&memA),
// 		unsafe.Pointer(&memB),
// 		unsafe.Pointer(&size),
// 	}

// 	cudaLogf("CUDADO %q, size %v", name, size)
// 	cudaLogf("%d, %d, %d, %d, %d, %d", gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ)
// 	if err = fn.LaunchAndSync(gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 0, cu.Stream(0), args); err != nil {
// 		return
// 	}

// 	err = devPtrToValue(retVal, memA)
// 	return

// }

func (op elemBinOp) CUDAFuncName() string {
	return ʘBinOpNames[op.binOpType()]
}
