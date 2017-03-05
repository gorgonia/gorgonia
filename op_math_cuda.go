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
	cudaLogf("CUDADoing %v", op)
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
		if mem, err = ctx.MemAlloc(memsize); err != nil {
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
		cudaLogf("a is val")
		cudaLogf("a %v", at.Data().([]float64)[0:3])
		memsize := int64(at.MemSize())
		size = int64(at.Shape().TotalSize())
		ctx.MemcpyHtoD(mem, at.Pointer(), memsize)
	case cu.DevicePtr:
		cudaLogf("a is Dev")
		memsize := int64(at.MemSize())
		size = memsize / int64(dt.Size())
		ctx.Memcpy(mem, at, memsize)
	}

	// blocks, threads := machine.(*tapeMachine).blockThread(int(size), int(dev))
	gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ := machine.ElemGridSize(int(size), int(dev))
	args := []unsafe.Pointer{
		unsafe.Pointer(&mem),
		unsafe.Pointer(&size),
	}
	// cudaLogf("threads: %d, blocks %d", threads, blocks)
	cudaLogf("gx %d, gy %d, gz %d | bx %d by %d, bz %d", gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ)
	cudaLogf("CUDADO %q, Mem: 0x%x size %v, args %v", name, mem, size, args)
	ctx.LaunchAndSync(fn, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 0, cu.Stream(0), args)
	// ctx.LaunchAndSync(fn, blocks, 1, 1, threads, 1, 1, 0, cu.Stream(0), args)
	return mem, nil
}

func (op elemUnaryOp) CUDAFuncName() string {
	return op.String()
}

func (op elemBinOp) CallsExtern() bool { return true }

func (op elemBinOp) CUDADo(extern External, dev Device, inputTypes hm.Types, prealloc Memory, inputs ...Memory) (retVal Memory, err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}

	cudaLogf("CUDADoing %v", op)
	enterLoggingContext()
	defer leaveLoggingContext()

	// check
	cudaLogf("checking if op is arith")
	if !op.isArith() {

	}

	a := inputs[0]
	b := inputs[1]
	var av, bv, pv Value
	var aok, bok, pok bool
	av, aok = a.(Value)
	bv, bok = b.(Value)
	pv, pok = prealloc.(Value)

	switch {
	case aok && bok:
		if av.Shape().IsScalar() || bv.Shape().IsScalar() {
			if pok {
				return op.UsePreallocDo(pv, av, bv)
			}
			return op.Do(av, bv)
		}
	case aok:
		// error
		err = errors.Errorf("Cannot perform op on CPU when `b` is non-value memory")
		return
	case bok:
		// error
		err = errors.Errorf("Cannot perform op on CPU when `a` is non-value memory")
		return
	}

	var dt tensor.Dtype
	if dt, err = dtypeOf(inputTypes[0]); err != nil {
		return
	}

	name := fmt.Sprintf("%v%d", op.CUDAFuncName(), int(dt.Size())*8)
	if !extern.HasFunc(name) {
		cudaLogf("extern %T has no function %q", extern, name)

		if pok {
			return op.UsePreallocDo(pv, av, bv)
		}
		return op.Do(av, bv)
	}

	machine := extern.(CUDAMachine)
	fn := machine.Functions()[name][int(dev)]
	ctx := machine.Contexts()[int(dev)]

	// allocate if necessary
	cudaLogf("allocate if necessary")
	var mem cu.DevicePtr
	switch pre := prealloc.(type) {
	case Value:
		memsize := int64(pre.MemSize())
		if mem, err = ctx.MemAlloc(memsize); err != nil {
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
	cudaLogf("copying a")
	var size int64
	switch at := a.(type) {
	case Value:
		cudaLogf("a is val")
		memsize := int64(at.MemSize())
		size = int64(at.Shape().TotalSize())
		ctx.MemcpyHtoD(mem, at.Pointer(), memsize)
	case cu.DevicePtr:
		cudaLogf("a is Dev")
		memsize := int64(at.MemSize())
		size = memsize / int64(dt.Size())
		ctx.Memcpy(mem, at, memsize)
	}

	var memB cu.DevicePtr
	switch bt := b.(type) {
	case Value:
		cudaLogf("b is val")
		memsize := int64(b.MemSize())
		if memB, err = ctx.MemAlloc(memsize); err != nil {
			err = errors.Wrapf(err, "Failed to allocate %v bytes", memsize)
			return
		}

		defer func(ctx *cu.BatchedContext, val Value, mem cu.DevicePtr) {
			err = devPtrToValue(ctx, val, mem)
			ctx.MemFree(mem)
		}(ctx, bt, memB)
	case cu.DevicePtr:
		memB = bt
	}

	gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ := machine.ElemGridSize(int(size), int(dev))
	args := []unsafe.Pointer{
		unsafe.Pointer(&mem),
		unsafe.Pointer(&memB),
		unsafe.Pointer(&size),
	}

	cudaLogf("CUDADO %q, size %v", name, size)
	cudaLogf("%d, %d, %d, %d, %d, %d", gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ)
	ctx.LaunchAndSync(fn, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 0, cu.Stream(0), args)
	return mem, nil
}

func (op elemBinOp) CUDAFuncName() string {
	return ʘBinOpNames[op.binOpType()]
}
