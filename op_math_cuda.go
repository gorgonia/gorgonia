// +build cuda

package gorgonia

import (
	"fmt"
	"log"
	"unsafe"

	"github.com/pkg/errors"
	"gorgonia.org/cu"
	"gorgonia.org/tensor"
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

	cudaLogf("CUDADoing %v | prealloc %x | %x", op, prealloc.Uintptr(), inputs[0].Uintptr())
	enterLogScope()
	defer leaveLogScope()

	// check
	cudaLogf("checking if input is scalar")
	a := inputs[0]
	dt := a.Dtype()

	// build name
	name := fmt.Sprintf("%v.%v_f%d", elemUnaryOpMod, op.unaryOpType(), int(dt.Size())*8)

	machine := extern.(CUDAMachine)
	eng := machine.Engines()[int(dev)]
	if !eng.HasFunc(name) {
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
	fn := eng.Functions()[name]
	ctx := machine.Contexts()[int(dev)]

	retVal = prealloc
	if prealloc == nil {
		prealloc = a
		retVal = a
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

	log.Printf("retVal 0x%x", retVal.Uintptr())
	return
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

	if as.IsScalar() && bs.IsScalar() {
		return nil, errors.Errorf("NYI")
	}

	m := extern.(CUDAMachine)
	e := &m.Engines()[int(dev)]

	aT := a.(tensor.Tensor)
	bT := b.(tensor.Tensor)
	tensor.WithEngine(e)(aT)
	tensor.WithEngine(e)(bT)
	log.Printf("aT.Engine %T", aT.Engine())

	pT, ok := prealloc.(tensor.Tensor)
	if ok {
		tensor.WithEngine(e)(pT)
	}

	boType := op.binOpType()
	if fn := binOps[boType]; fn != nil {
		if ok {
			return (*fn)(aT, bT, tensor.WithReuse(pT))
		} else {
			return (*fn)(aT, bT, tensor.UseUnsafe())
		}
	}

	if fn := cmpOps[boType]; fn != nil {
		if ok {
			return (*fn)(aT, bT, tensor.WithReuse(pT))
		} else {
			return (*fn)(aT, bT, tensor.UseUnsafe())
		}
	}

	return nil, errors.Errorf("op %v cannot be done by CUDA", op)
}

/* LINEAR ALGEBRA STUFF */

func (op linAlgBinOp) CallsExtern() bool { return false }

// func (op linAlgBinOp) CUDADo(extern External, dev Device, prealloc Value, inputs ...Value) (retVal Value, err error) {
// 	return nil, errors.Errorf("NYI")
// }

/* API stuff  */

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
