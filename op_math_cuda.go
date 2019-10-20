// +build cuda

package gorgonia

import (
	"fmt"
	"unsafe"

	"github.com/pkg/errors"
	"gorgonia.org/cu"
	"gorgonia.org/gorgonia/cuda"
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

	m := extern.(CUDAMachine)
	e := &m.Engines()[int(dev)]

	if as.IsScalar() && bs.IsScalar() {
		return op.ssop(a, b, prealloc, e)
	}

	if aT, ok := a.(tensor.Tensor); ok {
		tensor.WithEngine(e)(aT)
	}
	if bT, ok := b.(tensor.Tensor); ok {
		tensor.WithEngine(e)(bT)
	}

	pT, toReuse := prealloc.(tensor.Tensor)
	if toReuse {
		tensor.WithEngine(e)(pT)
	}

	boType := op.binOpType()
	if fn := binOps[boType]; fn != nil {
		if toReuse {
			return (*fn)(a, b, tensor.WithReuse(pT))
		}
		return (*fn)(a, b, tensor.UseUnsafe())
	}

	if fn := cmpOps[boType]; fn != nil {
		if toReuse {
			return (*fn)(a, b, tensor.WithReuse(pT))
		}
		return (*fn)(a, b, tensor.UseUnsafe())
	}

	return nil, errors.Errorf("op %v cannot be done by CUDA", op)
}

func (op elemBinOp) ssop(a, b, prealloc Value, e *cuda.Engine) (retVal Value, err error) {
	dt := a.Dtype()
	ctx := e.Context()
	opName := ʘBinOpNames[op.binOpType()]
	name := fmt.Sprintf("%v.%v_ss_f%d", elemBinOpMod, opName, int(dt.Size())*8)
	var mem, memB cu.DevicePtr
	var size int64
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
	fn := e.Functions()[name]

	var args []unsafe.Pointer
	cudaLogf("%v mem %v, memB %v", op, mem, memB)
	gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ := e.ElemGridSize(int(size))
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

/* LINEAR ALGEBRA STUFF */

func (op linAlgBinOp) CallsExtern() bool { return true }

func (op linAlgBinOp) CUDADo(extern External, dev Device, prealloc Value, inputs ...Value) (retVal Value, err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}

	m := extern.(CUDAMachine)
	e := &m.Engines()[int(dev)]

	a := inputs[0]
	b := inputs[1]

	aT, ok := a.(tensor.Tensor)
	if !ok {
		return nil, errors.Errorf("Expected a a to be a Tensor. Got %T instead", a)
	}
	bT, ok := b.(tensor.Tensor)
	if !ok {
		return nil, errors.Errorf("Expected a b to be a Tensor. Got %T instead", b)
	}

	pT, ok := prealloc.(tensor.Tensor)
	if !ok {
		return nil, errors.Errorf("Expected a prealloc to be a Tensor. Got %T instead", prealloc)
	}
	tensor.WithEngine(e)(bT)
	tensor.WithEngine(e)(aT)
	tensor.WithEngine(e)(pT)

	if op.transA && op.āBinaryOperator != batchedMatMulOperator {
		if err = aT.T(); err != nil {
			return nil, errors.Wrap(err, tFail)
		}
		// untranspose
		defer aT.T()
	}

	if op.transB && op.āBinaryOperator != batchedMatMulOperator {
		if err = bT.T(); err != nil {
			return nil, errors.Wrap(err, tFail)
		}
		// untranspose
		defer bT.T()
	}

	switch op.āBinaryOperator {
	case matMulOperator:
		return tensor.MatMul(aT, bT, tensor.WithReuse(pT))
	case matVecMulOperator:
		return tensor.MatVecMul(aT, bT, tensor.WithReuse(pT))
	case vecDotOperator:
		return nil, errors.New("NYI")
	case outerProdOperator:
		return tensor.Outer(aT, bT, tensor.WithReuse(pT))
	case batchedMatMulOperator:
		return nil, errors.New("NYI")
	}
	panic("Unreachable")
}

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

// NewHadamardProdOp creates a new *ExternalOp that wraps a mul op
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
