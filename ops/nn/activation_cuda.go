// +build cuda

package nnops

import (
	"hash"

	"github.com/chewxy/hm"
	"gorgonia.org/cu/dnn"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type activation struct {
	*cudnn.Activation
	xDesc, yDesc *cudnn.TensorDescriptor
}

func newRelu() *activation {
	act := cudnn.NewActivation(cudnn.ReLU, cudnn.PropagateNan, 1.0)
	return &activation{act}
}

func (op *activation) Arity() int { return 1 }

func (op *activation) Type() hm.Type {
	return hm.NewFnType(hm.TypeVariable('a'), hm.TypeVariable('a'))
}

func (op *activation) InferShape(inputs ...gorgonia.DimSizer) (tensor.Shape, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, err
	}
	return inputs[0].(tensor.Shape).Clone(), nil
}

func (op *activation) Do(...gorgonia.Value) (gorgonia.Value, error) {
	panic("not implemented")
}

func (op *activation) ReturnsPtr() bool { return true }

func (op *activation) CallsExtern() bool { return true }

func (op *activation) OverwritesInput() int { return -1 }

func (op *activation) WriteHash(h hash.Hash) {
	panic("not implemented")
}

func (op *activation) Hashcode() uint32 {
	panic("not implemented")
}

func (op *activation) String() string {
	panic("not implemented")
}

func (op *activation) DiffWRT(inputs int) []bool { return []bool{true} }

func (op *activation) SymDiff(inputs gorgonia.Nodes, output *gorgonia.Node, grad *gorgonia.Node) (retVal gorgonia.Nodes, err error) {
	panic("not implemented")
}

func (op *activation) CUDADo(extern gorgonia.External, dev gorgonia.Device, prealloc gorgonia.Value, inputs ...gorgonia.Value) (retVal gorgonia.Value, err error) {
	if err = checkArity(c, len(inputs)); err != nil {
		return
	}

	x := inputs[0]

	if op.xDesc == nil {
		if op.xDesc, err = t2cudnn.Describe(x.(tensor.Tensor)); err != nil {
			return
		}
	}
	if op.yDesc == nil {
		if op.yDesc, err = t2cudnn.Describe(y.(tensor.Tensor)); err != nil {
			return
		}
	}

	machine := extern.(G.CUDAMachine)
	ctx := machine.CUDNNContext()
	err = ctx.ActivationForward(op.Activation, 1, op.xDesc, x.(cudnn.Memory), 0, op.yDesc, prealloc.(cudnn.Memory))
	return prealloc, err
}

type activationDiff struct {
	*activation
	dyDesc, dxDesc *cudnn.TensorDescriptor
}

func (op *activationDiff) Arity() int {
	return 4 // x, y, dy, dx
}

func (op *activationDiff) Type() hm.Type {
	return hm.NewFnType(hm.TypeVariable('a'), hm.TypeVariable('a'), hm.TypeVariable('a'), hm.TypeVariable('a'))
}

func (op *activationDiff) InferShape(inputs ...gorgonia.DimSizer) (tensor.Shape, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, err
	}
	return inputs[0].(tensor.Shape).Clone(), nil
}

func (op *activationDiff) Do(...gorgonia.Value) (gorgonia.Value, error) {
	panic("not implemented")
}

func (op *activationDiff) ReturnsPtr() bool { return true }

func (op *activationDiff) CallsExtern() bool { return true }

func (op *activationDiff) OverwritesInput() int { return -1 }

func (op *activationDiff) WriteHash(h hash.Hash) {
	panic("not implemented")
}

func (op *activationDiff) Hashcode() uint32 {
	panic("not implemented")
}

func (op *activationDiff) String() string {
	panic("not implemented")
}

func (op *activationDiff) CUDADo(extern gorgonia.External, dev gorgonia.Device, prealloc gorgonia.Value, inputs ...gorgonia.Value) (retVal gorgonia.Value, err error) {
	x, y, dy := inputs[0], inputs[1], inputs[2]
	if op.dxDesc == nil {
		if op.dxDesc, err = t2cudnn.Describe(prealloc.(tensor.Tensor)); err != nil {
			return
		}
	}
	if op.dyDesc == nil {
		if op.dyDesc, err = t2cudnn.Describe(dy.(tensor.Tensor)); err != nil {
			return
		}
	}
	machine := extern.(G.CUDAMachine)
	machine.Engines()[int(dev)].DoWork()
	ctx := machine.CUDNNContexts()[int(dev)]

	err = ctx.ActivationBackward(op.activation, 1,
		op.yDesc, y.(cudnn.Memory),
		op.dyDesc, dy.(cudnn.Memory),
		op.xDesc, x.(cudnn.Memory),
		0,
		op.dxDesc, prealloc.(cudnn.Memory))
	return prealloc, err
}
