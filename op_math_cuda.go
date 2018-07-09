// +build cuda

package gorgonia

import (
	"fmt"
	"log"
	"unsafe"

	"gorgonia.org/cu"
)

// module names
const (
	elemBinOpMod   = "elembinop"
	elemUnaryOpMod = "elemunaryop"
)

func (op elemUnaryOp) CallsExtern() bool { return true }

func (op elemUnaryOp) CUDADo(extern External, dev Device, prealloc Value, inputs ...Value) (retVal Value, err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}

	cudaLogf("CUDADoing %v | prealloc %v | %v", op, prealloc, inputs)
	enterLogScope()
	defer leaveLogScope()

	// check
	cudaLogf("checking if input is scalar")
	a := inputs[0]
	dt := a.Dtype()

	// build name
	name := fmt.Sprintf("%v.%v_f%d", elemUnaryOpMod, op.unaryOpType(), int(dt.Size())*8)

	if !extern.HasFunc(name) {
		cudaLogf("extern does not have func %q", name)
		extern.Signal()

		if retVal, err = op.do(a); err != nil {
			return
		}
		if prealloc == nil {
			return
		}
		return Copy(prealloc, retVal)
	}

	machine := extern.(CUDAMachine)
	fn := machine.Functions()[name][int(dev)]
	ctx := machine.Contexts()[int(dev)]

	if prealloc == nil {
		prealloc = a
	}
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
	ctx.LaunchAndSync(fn, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 0, cu.NoStream, args)
	return prealloc, nil
}

func (op elemBinOp) CallsExtern() bool { return true }

func (op elemBinOp) CUDADo(extern External, dev Device, prealloc Value, inputs ...Value) (retVal Value, err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}
	cudaLogf("CUDADoing %v", op)
	enterLogScope()
	defer leaveLogScope()

	a := inputs[0]
	b := inputs[1]
	as := a.Shape()
	bs := b.Shape()

	var vv, vs, sv, ss bool

	dt := a.Dtype()
	opName := ʘBinOpNames[op.binOpType()]
	var name string
	switch {
	case as.IsScalar() && bs.IsScalar():
		ss = true
		name = fmt.Sprintf("%v.%v_ss_f%d", elemBinOpMod, opName, int(dt.Size())*8)
	case as.IsScalar() && !bs.IsScalar():
		sv = true
		name = fmt.Sprintf("%v.%v_sv_f%d", elemBinOpMod, opName, int(dt.Size())*8)
	case !as.IsScalar() && bs.IsScalar():
		vs = true
		name = fmt.Sprintf("%v.%v_vs_f%d", elemBinOpMod, opName, int(dt.Size())*8)
	case !as.IsScalar() && !bs.IsScalar():
		vv = true
		name = fmt.Sprintf("%v.%v_vv_f%d", elemBinOpMod, opName, int(dt.Size())*8)
	}

	machine := extern.(CUDAMachine)
	ctx := machine.Contexts()[int(dev)]
	var mem, memB cu.DevicePtr
	var size int64
	cudaLogf("a: 0x%x b 0x%x", a.Uintptr(), b.Uintptr())
	cudaLogf("a %v, b%v", a.Shape(), b.Shape())
	switch {
	case vv, vs, ss:
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
		cudaLogf("HERE")
	case sv:
		if prealloc == nil {
			mem = cu.DevicePtr(b.Uintptr())
			retVal = b
			size = int64(logicalSize(b.Shape()))
			memB = cu.DevicePtr(b.Uintptr())
		} else {
			mem = cu.DevicePtr(a.Uintptr())
			preallocMem := cu.DevicePtr(prealloc.Uintptr())

			B := cu.DevicePtr(b.Uintptr())
			memSize := int64(b.MemSize())
			ctx.Memcpy(preallocMem, B, memSize)

			size = int64(logicalSize(prealloc.Shape()))
			retVal = prealloc
			memB = preallocMem
		}
	}

	hasFn := extern.HasFunc(name)
	if !hasFn {
		cudaLogf("NoFn: %q", name)
		extern.Signal()
		cudaLogf("DONE. Prealloc \n%v", prealloc)
		if prealloc != nil {
			log.Printf("No func %v", name)
			log.Printf("Preallo %x", prealloc.Uintptr())
			return op.UsePreallocDo(prealloc, inputs...)
		}

		if op.retSame {
			return op.UsePreallocDo(retVal, inputs...)
		}

		cudaLogf("Using DO - Prealloc %v", retVal)
		return op.Do(inputs...)
	}

	fn := machine.Functions()[name][int(dev)]
	var args []unsafe.Pointer

	cudaLogf("%v mem %v, memB %v", op, mem, memB)
	gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ := machine.ElemGridSize(int(size), int(dev))
	args = []unsafe.Pointer{
		unsafe.Pointer(&mem),
		unsafe.Pointer(&memB),
		unsafe.Pointer(&size),
	}

	cudaLogf("CUDADO %q, size %v", name, size)
	cudaLogf("LaunchKernel params. mem: %v memB: %v size: %v", mem, memB, size)
	cudaLogf("%d, %d, %d, %d, %d, %d", gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ)
	ctx.LaunchAndSync(fn, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 0, cu.NoStream, args)
	return
}

// NewAddOp creates a new *ExternalOp that wraps a add op
func NewAddOp(a, b *Node, ctx ExecutionContext) *ExternalOp {
	add := newElemBinOp(addOpType, a, b)
	op := NewExternalOp(add, ctx, nil)
	if a.Device() == CPU && b.Device() == CPU {
		op.Device = CPU
		return op
	}

	if a.Device() != CPU {
		op.Device = a.Device()
		return op
	}

	if b.Device() != CPU {
		op.Device = b.Device()
		return op
	}

	return op
}

// NewSubOp creates a new *ExternalOp that wraps a sub op
func NewSubOp(a, b *Node, ctx ExecutionContext) *ExternalOp {
	sub := newEBOByType(subOpType, a.t, b.t)
	op := NewExternalOp(sub, ctx, nil)

	if a.Device() == CPU && b.Device() == CPU {
		op.Device = CPU
		return op
	}

	if a.Device() != CPU {
		op.Device = a.Device()
		return op
	}

	if b.Device() != CPU {
		op.Device = b.Device()
		return op
	}
	return op
}

func NewHadamardProdOp(a, b *Node, ctx ExecutionContext) *ExternalOp {
	mul := newEBOByType(mulOpType, a.t, b.t)
	op := NewExternalOp(mul, ctx, nil)

	if a.Device() == CPU && b.Device() == CPU {
		op.Device = CPU
		return op
	}

	if a.Device() != CPU {
		op.Device = a.Device()
		return op
	}

	if b.Device() != CPU {
		op.Device = b.Device()
		return op
	}
	return op
}
