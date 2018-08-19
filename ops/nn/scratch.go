package nnops

import (
	"fmt"
	"hash"

	"github.com/chewxy/hm"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// scratchOp is a dummy op. It exists so the VM is able to allocate spare memory.
//
// giving it a name makes it unique(r)
type scratchOp struct {
	shape tensor.Shape
	dt    tensor.Dtype
	name  string
}

func (op *scratchOp) Arity() int { return 0 }

func (op *scratchOp) Type() hm.Type {
	if op.shape.IsScalar() {
		return op.dt
	}
	tt := &gorgonia.TensorType{Dims: op.shape.Dims(), Of: op.dt}
	return tt
}

func (op *scratchOp) InferShape(...gorgonia.DimSizer) (tensor.Shape, error) { return op.shape, nil }
func (op *scratchOp) Do(...gorgonia.Value) (gorgonia.Value, error)          { panic("not implemented") }
func (op *scratchOp) ReturnsPtr() bool                                      { return true }
func (op *scratchOp) CallsExtern() bool                                     { return true }
func (op *scratchOp) OverwritesInput() int                                  { return -1 }
func (op *scratchOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "CPU Scratch %v of %v | %v", op.shape, op.dt, op.name)
}
func (op *scratchOp) Hashcode() uint32 { return simpleHash(op) }
func (op *scratchOp) String() string {
	return fmt.Sprintf("CPU Scratch %v of %v | %v", op.shape, op.dt, op.name)
}
func (op *scratchOp) UsePreallocDo(prealloc gorgonia.Value, inputs ...gorgonia.Value) (gorgonia.Value, error) {
	return prealloc, nil
}
