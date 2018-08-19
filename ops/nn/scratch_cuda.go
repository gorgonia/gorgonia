// +build cuda

package nnops

import (
	"fmt"
	"hash"

	"github.com/chewxy/hm"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// gpuScratchOp is a dummy op. It exists so the VM is able to allocate spare memory.
//
// giving it a name makes it unique(r)
type gpuScratchOp struct {
	scratchOp
}

func (op *gpuScratchOp) Arity() int { return 0 }

func (op *gpuScratchOp) Type() hm.Type {
	if op.shape.IsScalar() {
		return op.dt
	}
	tt := &gorgonia.TensorType{Dims: op.shape.Dims(), Of: op.dt}
	return tt
}

func (op *gpuScratchOp) InferShape(...gorgonia.DimSizer) (tensor.Shape, error) { return op.shape, nil }
func (op *gpuScratchOp) Do(...gorgonia.Value) (gorgonia.Value, error)          { panic("not implemented") }
func (op *gpuScratchOp) ReturnsPtr() bool                                      { return true }
func (op *gpuScratchOp) CallsExtern() bool                                     { return true }
func (op *gpuScratchOp) OverwritesInput() int                                  { return -1 }
func (op *gpuScratchOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "CUDA Scratch %v of %v | %v", op.shape, op.dt, op.name)
}
func (op *gpuScratchOp) Hashcode() uint32 { return simpleHash(op) }
func (op *gpuScratchOp) String() string {
	return fmt.Sprintf("CUDA Scratch %v of %v | %v", op.shape, op.dt, op.name)
}

func (op *gpuScratchOp) CUDADo(extern gorgonia.External, dev gorgonia.Device, prealloc gorgonia.Value, inputs ...gorgonia.Value) (retVal gorgonia.Value, err error) {
	return prealloc, nil
}
