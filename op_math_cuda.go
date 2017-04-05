// +build cuda

package gorgonia

import (
	"fmt"
	"unsafe"

	"github.com/chewxy/cu"
)

func (op elemUnaryOp) CallsExtern() bool { return true }

func (op elemUnaryOp) CUDADo(extern External, dev Device, prealloc Value, inputs ...Value) (retVal Value, err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}

	cudaLogf("CUDADoing %v | %v", op, inputs)
	enterLoggingContext()
	defer leaveLoggingContext()

	// check
	cudaLogf("checking if input is scalar")
	a := inputs[0]
	if a.Shape().IsScalar() {
		extern.Signal()
		<-extern.Sync()

		if retVal, err = op.do(a); err != nil {
			return
		}
		return Copy(prealloc, retVal)
	}

	dt := a.Dtype()

	name := fmt.Sprintf("%v%d", op.CUDAFuncName(), int(dt.Size())*8)
	if !extern.HasFunc(name) {
		cudaLogf("extern does not have func %q", name)
		extern.Signal()
		<-extern.Sync()
		return op.do(a)
	}

	machine := extern.(CUDAMachine)
	fn := machine.Functions()[name][int(dev)]
	ctx := machine.Contexts()[int(dev)]

	var mem cu.DevicePtr
	if prealloc.Uintptr() == a.Uintptr() && a.Shape().Eq(prealloc.Shape()) {
		mem = cu.DevicePtr(a.Uintptr())
	} else {
		mem = cu.DevicePtr(prealloc.Uintptr())
		memSize := int64(a.MemSize())
		memA := cu.DevicePtr(a.Uintptr())
		ctx.Memcpy(mem, memA, memSize)
	}
	size := logicalSize(a.Shape())

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
	return prealloc, nil
}

func (op elemUnaryOp) CUDAFuncName() string {
	return op.String()
}

func (op elemBinOp) CallsExtern() bool { return true }

func (op elemBinOp) CUDADo(extern External, dev Device, prealloc Value, inputs ...Value) (retVal Value, err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}

	cudaLogf("CUDADoing %v", op)
	enterLoggingContext()
	defer leaveLoggingContext()

	a := inputs[0]
	b := inputs[1]
	as := a.Shape()
	bs := b.Shape()

	dt := a.Dtype()
	name := fmt.Sprintf("%v%d", op.CUDAFuncName(), int(dt.Size())*8)
	hasFn := extern.HasFunc(name)

	if !hasFn {
		cudaLogf("NoFn")
		extern.Signal()
		<-extern.Sync()
		cudaLogf("DONE. Prealloc \n%v", prealloc)
		if prealloc != nil {
			return op.UsePreallocDo(prealloc, inputs...)
		}
		cudaLogf("Using DO")
		return op.Do(inputs...)
	}

	machine := extern.(CUDAMachine)
	fn := machine.Functions()[name][int(dev)]
	ctx := machine.Contexts()[int(dev)]

	var mem, memB cu.DevicePtr
	var size int64
	switch {
	case op.isArith() && !as.IsScalar() && !bs.IsScalar():
		if prealloc == nil {
			mem = cu.DevicePtr(a.Uintptr())
			retVal = a
			size = int64(logicalSize(a.Shape()))
		} else {
			mem = cu.DevicePtr(prealloc.Uintptr())
			memA := cu.DevicePtr(a.Uintptr())
			memSize := int64(a.MemSize())
			ctx.Memcpy(mem, memA, memSize)

			size = int64(logicalSize(prealloc.Shape()))
			retVal = prealloc
		}

		memB = cu.DevicePtr(b.Uintptr())
	default:
		cudaLogf("HELLO: hasFn %v, op.IsArith %v | ascalar %t, bscalar %t", hasFn, op.isArith(), as.IsScalar(), bs.IsScalar())
		extern.Signal()
		<-extern.Sync()
		if prealloc != nil {
			return op.UsePreallocDo(prealloc, inputs...)
		}
		cudaLogf("Using DO")
		return op.Do(inputs...)
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
	return
}

func (op elemBinOp) CUDAFuncName() string {
	return ʘBinOpNames[op.binOpType()]
}
