// +build cuda

package gorgonia

import (
	"fmt"
	"unsafe"

	"github.com/chewxy/cu"
	"github.com/chewxy/gorgonia/tensor"
	"github.com/pkg/errors"
)

func (op elemUnaryOp) CallsExtern() bool { return true }

func (op elemUnaryOp) CUDADo(extern External, dev Device, meta ExecutionMetadata, prealloc Memory, inputs ...Memory) (retVal Memory, err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}
	if err = meta.checkArity(op); err != nil {
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
	if dt, err = dtypeOf(meta.InputTypes[0]); err != nil {
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
		var m Memory
		if m, err = machine.Get(dev, uint(memsize)); err != nil {
			if _, ok := err.(NoOpError); !ok {
				return
			}
			cudaLogf("necessary to allocate %v", memsize)
			if mem, err = ctx.MemAlloc(memsize); err != nil {
				err = errors.Wrapf(err, "Failed to allocate %v bytes", memsize)
				return
			}
		} else {
			mem = m.(cu.DevicePtr)
		}
	case cu.DevicePtr:
		mem = pre
	}

	// copy
	cudaLogf("copying")
	var size int64
	switch at := a.(type) {
	case Value:
		cudaLogf("a is val: %v", at.Pointer())
		memsize := int64(at.MemSize())
		size = int64(at.Shape().TotalSize())
		ctx.MemcpyHtoD(mem, at.Pointer(), memsize)
	case cu.DevicePtr:
		cudaLogf("a is Dev: %v mem %v", at, mem)
		size = int64(meta.InputShapes[0].TotalSize())
		if at != mem {
			memsize := int64(dt.Size()) * size
			ctx.Memcpy(mem, at, memsize)
		}
	}

	// blocks, threads := machine.(*tapeMachine).blockThread(int(size), int(dev))
	gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ := machine.ElemGridSize(int(size), int(dev))
	args := []unsafe.Pointer{
		unsafe.Pointer(&mem),
		unsafe.Pointer(&size),
	}
	cudaLogf("gx %d, gy %d, gz %d | bx %d by %d, bz %d", gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ)
	cudaLogf("CUDADO %q, Mem: %v size %v, args %v", name, mem, size, args)
	cudaLogf("LaunchKernel Params. mem: %v. Size %v", mem, size)
	ctx.LaunchAndSync(fn, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 0, cu.Stream(0), args)
	// ctx.LaunchAndSync(fn, blocks, 1, 1, threads, 1, 1, 0, cu.Stream(0), args)
	return mem, nil
}

func (op elemUnaryOp) CUDAFuncName() string {
	return op.String()
}

func (op elemBinOp) CallsExtern() bool { return true }

func (op elemBinOp) CUDADo(extern External, dev Device, meta ExecutionMetadata, prealloc Memory, inputs ...Memory) (retVal Memory, err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}
	if err = meta.checkArity(op); err != nil {
		return
	}

	cudaLogf("CUDADoing %v", op)
	enterLoggingContext()
	defer leaveLoggingContext()

	// check
	var dt tensor.Dtype
	if dt, err = dtypeOf(meta.InputTypes[0]); err != nil {
		return
	}
	name := fmt.Sprintf("%v%d", op.CUDAFuncName(), int(dt.Size())*8)
	hasFn := extern.HasFunc(name)

	a := inputs[0]
	b := inputs[1]
	var av, bv, pv Value
	var aok, bok, pok bool
	av, aok = a.(Value)
	bv, bok = b.(Value)
	pv, pok = prealloc.(Value)

	switch {
	case aok && bok:
		if av.Shape().IsScalar() || bv.Shape().IsScalar() || !op.isArith() {
			if pok {
				return op.UsePreallocDo(pv, av, bv)
			}
			return op.Do(av, bv)
		}
		if !hasFn {
			cudaLogf("extern %T has no function %q", extern, name)

			if pok {
				return op.UsePreallocDo(pv, av, bv)
			}
			return op.Do(av, bv)
		}
	case aok:
		// error
		if !hasFn {
			err = errors.Errorf("Cannot perform op on CPU when `b` is non-value memory")
			return
		}
	case bok:
		// error
		if !hasFn {
			err = errors.Errorf("Cannot perform op on CPU when `a` is non-value memory")
			return
		}
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
		var m Memory
		if m, err = machine.Get(dev, uint(memsize)); err != nil {
			if _, ok := err.(NoOpError); !ok {
				return
			}

			cudaLogf("necessary to allocate %v", memsize)
			if mem, err = ctx.MemAlloc(memsize); err != nil {
				err = errors.Wrapf(err, "Failed to allocate %v bytes", memsize)
				return
			}
		} else {
			mem = m.(cu.DevicePtr)
		}
	case cu.DevicePtr:
		mem = pre
	}

	// copy
	cudaLogf("copying a")
	var size int64
	switch at := a.(type) {
	case Value:
		cudaLogf("a is val: %v", at.Pointer())
		memsize := int64(at.MemSize())
		size = int64(at.Shape().TotalSize())
		ctx.MemcpyHtoD(mem, at.Pointer(), memsize)
	case cu.DevicePtr:
		cudaLogf("a is Dev: %v mem %v", at, mem)
		size = int64(meta.InputShapes[0].TotalSize())
		if at != mem {
			memsize := int64(dt.Size()) * size
			ctx.Memcpy(mem, at, memsize)
		}
	}

	var memB cu.DevicePtr
	switch bt := b.(type) {
	case Value:
		cudaLogf("b is val: %v", bt.Pointer())
		memsize := int64(b.MemSize())

		var m Memory
		if m, err = machine.Get(dev, uint(memsize)); err != nil {
			cudaLogf("No memory found. Trying to alloc and copy")
			if _, ok := err.(NoOpError); !ok {
				return
			}

			if memB, err = ctx.AllocAndCopy(bt.Pointer(), memsize); err != nil {
				err = errors.Wrapf(err, "Failed to allocate %v bytes", memsize)
				return
			}
		} else {
			cudaLogf("Found. Copying")
			memB = m.(cu.DevicePtr)
			ctx.MemcpyHtoD(memB, bt.Pointer(), memsize)
		}

		// the reason why there are so many params to this defer is to prevent leaking of stuff into the heap.
		// FUTURE: come back when `go build -m` no longer indicates so.
		defer func(machine CUDAMachine, ctx *cu.BatchedContext, val Value, mem cu.DevicePtr, memsize int64) {
			if err = devPtrToValue(ctx, val, mem); err != nil {
				return
			}
			cudaLogf("Putting %v with size %v back", mem, memsize)
			machine.Put(dev, mem, uint(memsize))
		}(machine, ctx, bt, memB, memsize)
	case cu.DevicePtr:
		memB = bt
	}

	cudaLogf("%v mem %v, memB %v", op, mem, memB)
	gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ := machine.ElemGridSize(int(size), int(dev))
	args := []unsafe.Pointer{
		unsafe.Pointer(&mem),
		unsafe.Pointer(&memB),
		unsafe.Pointer(&size),
	}

	cudaLogf("CUDADO %q, size %v", name, size)
	cudaLogf("LaunchKernel params. mem: %v memB: %v size: %v", mem, memB, size)
	cudaLogf("%d, %d, %d, %d, %d, %d", gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ)
	ctx.LaunchAndSync(fn, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 0, cu.Stream(0), args)
	return mem, nil
}

func (op elemBinOp) CUDAFuncName() string {
	return ʘBinOpNames[op.binOpType()]
}
