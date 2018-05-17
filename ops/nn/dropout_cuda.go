// +build cuda

package nnops

import (
	"fmt"
	"hash"
	"time"

	"github.com/chewxy/hm"
	"gorgonia.org/cu/dnn"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type dropout struct {
	*cudnn.Dropout
	seed  uint64
	xDesc *cudnn.TensorDescriptor
}

func newDropout(x *gorgonia.Node, prob float64) (*dropout, error) {
	var xDesc *cudnn.TensorDescriptor
	if xDesc, err = t2cudnn.Describe(x); err != nil {
		return nil, err
	}

	internal, err := cudnn.NewDropout(prob)
	if err != nil {
		return nil, err
	}
	return &dropout{
		Dropout: internal,
		xDesc:   xDesc,
		seed:    uint64(time.Now().UnixNano()),
	}, nil
}

func (op *dropout) Arity() int { return 2 }

func (op *dropout) Type() hm.Type {
	return hm.NewFnType(hm.TypeVariable('a'), hm.TypeVariable('a'), hm.TypeVariable('a'))
}

func (op *dropout) InferShape(inputs ...gorgonia.DimSizer) (tensor.Shape, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, err
	}
	return inputs[0].(tensor.Shape).Clone(), nil
}

func (op *dropout) Do(...gorgonia.Value) (gorgonia.Value, error) { panic("not implemented") }
func (op *dropout) ReturnsPtr() bool                             { return true }
func (op *dropout) CallsExtern() bool                            { return true }
func (op *dropout) OverwritesInput() int                         { return -1 }
func (op *dropout) WriteHash(h hash.Hash)                        { fmt.Fprintf(h, "Dropout %v", op.Dropout.Dropout()) }
func (op *dropout) Hashcode() uint32                             { return simpleHash(op) }
func (op *dropout) String() string                               { fmt.Sprintf("Dropout %v", op.Dropout.Dropout()) }
func (op *dropout) DiffWRT(inputs int) []bool                    { return []bool{true, false} }

func (op *dropout) SymDiff(inputs gorgonia.Nodes, output *gorgonia.Node, grad *gorgonia.Node) (retVal gorgonia.Nodes, err error) {
	panic("not implemented")
}

func (op *dropout) DoDiff(ctx gorgonia.ExecutionContext, inputs gorgonia.Nodes, output *gorgonia.Node) error {
	panic("not implemented")
}

func (op *dropout) CUDADo(extern gorgonia.External, dev gorgonia.Device, prealloc gorgonia.Value, inputs ...gorgonia.Value) (retVal gorgonia.Value, err error) {
	if err = checkArity(c, len(inputs)); err != nil {
		return
	}

	x, s := inputs[0], inputs[1]
	machine := extern.(G.CUDAMachine)
	ctx := machine.CUDNNContext()
	if err = op.Use(ctx, s.(cudnn.Memory), op.seed); err != nil {
		return nil, err
	}
	err = cudnn.DropoutForward(op.Dropout, op.xDesc, x.(cudnn.Memory), op.xDesc, prealloc.(cudnn.Memory), s.(cudunn.Memory))
	return prealloc, err
}

// dropoutState is a dummy op. It's supposed to be like UniformRandomOp but doesn't actually do anything upon calling CUDADo. Instead it just returns the preallocated memory space.
type dropoutState struct {
	shape tensor.Shape
}

func (op *dropoutState) Arity() int { return 0 }

func (op *dropoutState) Type() hm.Type {
	if op.shape.IsScalar() {
		return op.dt
	}
	tt := &gorgonia.TensorType{Dims: op.shape.Dims(), Of: op.dt}
	return tt
}

func (op *dropoutState) InferShape(...gorgonia.DimSizer) (tensor.Shape, error) { return op.shape, nil }
func (op *dropoutState) Do(...gorgonia.Value) (gorgonia.Value, error)          { panic("not implemented") }
func (op *dropoutState) ReturnsPtr() bool                                      { return true }
func (op *dropoutState) CallsExtern() bool                                     { return true }
func (op *dropoutState) OverwritesInput() int                                  { return -1 }
func (op *dropoutState) WriteHash(h hash.Hash)                                 { fmt.Fprintf(h, "Spare Memory %v", op.shape) }
func (op *dropoutState) Hashcode() uint32                                      { return simpleHash(op) }
func (op *dropoutState) String() string                                        { return fmt.Sprintf("Spare Memory %v", op.shape) }

func (op *dropoutState) CUDADo(extern gorgonia.External, dev gorgonia.Device, prealloc gorgonia.Value, inputs ...gorgonia.Value) (retVal gorgonia.Value, err error) {
	return prealloc, nil
}
