// +build cuda

package nnops

import (
	"fmt"
	"hash"

	"github.com/chewxy/hm"
	cudnn "gorgonia.org/cu/dnn"
	t2cudnn "gorgonia.org/cu/dnn/interop"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var (
	_ G.Op       = &maxpool{}
	_ G.CUDADoer = &maxpool{}
	_ G.Op       = &maxpoolDiff{}
	_ G.CUDADoer = &maxpoolDiff{}
)

type maxpool struct {
	*cudnn.Pooling

	xDesc *cudnn.TensorDescriptor
	yDesc *cudnn.TensorDescriptor
}

func newMaxPoolOp(x *G.Node, kernel, pad, stride []int) (*maxpool, error) {
	var xDesc *cudnn.TensorDescriptor
	var err error
	if xDesc, err = t2cudnn.Describe(x); err != nil {
		return nil, err
	}

	var p *cudnn.Pooling
	if p, err = cudnn.NewPooling(cudnn.MaxPooling, cudnn.PropagateNan, kernel, stride, pad); err != nil {
		return nil, err
	}
	return &maxpool{
		Pooling: p,
		xDesc:   xDesc,
	}, nil
}

func (p *maxpool) Arity() int { return 1 }

func (p *maxpool) Type() hm.Type { return hm.NewFnType(hm.TypeVariable('a'), hm.TypeVariable('a')) }

func (p *maxpool) InferShape(inputs ...G.DimSizer) (tensor.Shape, error) {
	if err := checkArity(p, len(inputs)); err != nil {
		return nil, err
	}
	return p.OutputShape(p.xDesc, 2) // only maxpool2d for now
}

func (p *maxpool) Do(...G.Value) (G.Value, error) {
	panic("not implemented")
}

func (p *maxpool) ReturnsPtr() bool { return true }

func (p *maxpool) CallsExtern() bool { return true }

func (p *maxpool) OverwritesInput() int { return -1 }

func (p *maxpool) WriteHash(h hash.Hash) {
	xShape := p.xDesc.Shape()
	kernel := p.Shape()
	padding := p.Padding()
	strides := p.Strides()
	fmt.Fprintf(h, "MaxPool{%d, %d, %d, %d}(kernel: (%d, %d), pad: (%d, %d), stride: (%d, %d))",
		xShape[0], xShape[1], xShape[2], xShape[3],
		kernel[0], kernel[1],
		padding[0], padding[1],
		strides[0], strides[1])
}

func (p *maxpool) Hashcode() uint32 { return simpleHash(p) }

func (p *maxpool) String() string {
	xShape := p.xDesc.Shape()
	kernel := p.Shape()
	padding := p.Padding()
	strides := p.Strides()
	return fmt.Sprintf("MaxPool{%d, %d, %d, %d}(kernel: (%d, %d), pad: (%d, %d), stride: (%d, %d))",
		xShape[0], xShape[1], xShape[2], xShape[3],
		kernel[0], kernel[1],
		padding[0], padding[1],
		strides[0], strides[1])
}

func (p *maxpool) CUDADo(extern G.External, dev G.Device, prealloc G.Value, inputs ...G.Value) (retVal G.Value, err error) {
	if err = checkArity(p, len(inputs)); err != nil {
		return
	}
	in := inputs[0]

	if p.yDesc == nil {
		if p.yDesc, err = t2cudnn.Describe(prealloc.(tensor.Tensor)); err != nil {
			return
		}
	}

	machine := extern.(G.CUDAMachine)
	machine.Engines()[int(dev)].DoWork()
	ctx := machine.CUDNNContexts()[int(dev)]
	err = ctx.PoolingForward(p.Pooling, 1.0, p.xDesc, in.(cudnn.Memory), 0, p.yDesc, prealloc.(cudnn.Memory))
	return prealloc, err
}

func (p *maxpool) DiffWRT(inputs int) []bool { return []bool{true} }

func (p *maxpool) SymDiff(inputs G.Nodes, output *G.Node, grad *G.Node) (retVal G.Nodes, err error) {
	if err = checkArity(p, len(inputs)); err != nil {
		return
	}
	diff := (*maxpoolDiff)(p)
	x := inputs[0]

	retVal = make(G.Nodes, 1)
	retVal[0], err = G.ApplyOp(diff, x, output, grad)
	return
}

func (p *maxpool) DoDiff(ctx G.ExecutionContext, inputs G.Nodes, output *G.Node) error {
	panic("not implemented")
}

type maxpoolDiff maxpool

func (op *maxpoolDiff) Arity() int { return 3 }

func (op *maxpoolDiff) Type() hm.Type {
	return hm.NewFnType(hm.TypeVariable('a'), hm.TypeVariable('a'), hm.TypeVariable('a'), hm.TypeVariable('a'))
}

func (op *maxpoolDiff) InferShape(inputs ...G.DimSizer) (tensor.Shape, error) {
	return inputs[0].(tensor.Shape).Clone(), nil
}

func (op *maxpoolDiff) Do(...G.Value) (G.Value, error) { panic("not implemented") }

func (op *maxpoolDiff) ReturnsPtr() bool { return true }

func (op *maxpoolDiff) CallsExtern() bool { return true }

func (op *maxpoolDiff) OverwritesInput() int { return -1 }

func (op *maxpoolDiff) WriteHash(h hash.Hash) {
	xShape := op.xDesc.Shape()
	kernel := op.Shape()
	padding := op.Padding()
	strides := op.Strides()
	fmt.Fprintf(h, "MaxPoolDiff{%d, %d, %d, %d}(kernel: (%d, %d), pad: (%d, %d), stride: (%d, %d))",
		xShape[0], xShape[1], xShape[2], xShape[3],
		kernel[0], kernel[1],
		padding[0], padding[1],
		strides[0], strides[1])
}

func (op *maxpoolDiff) Hashcode() uint32 { return simpleHash(op) }

func (op *maxpoolDiff) String() string {
	xShape := op.xDesc.Shape()
	kernel := op.Shape()
	padding := op.Padding()
	strides := op.Strides()
	return fmt.Sprintf("MaxPoolDiff{%d, %d, %d, %d}(kernel: (%d, %d), pad: (%d, %d), stride: (%d, %d))",
		xShape[0], xShape[1], xShape[2], xShape[3],
		kernel[0], kernel[1],
		padding[0], padding[1],
		strides[0], strides[1])
}

func (op *maxpoolDiff) CUDADo(extern G.External, dev G.Device, prealloc G.Value, inputs ...G.Value) (retVal G.Value, err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}
	x, y, dy := inputs[0], inputs[1], inputs[2]

	machine := extern.(G.CUDAMachine)
	machine.Engines()[int(dev)].DoWork()
	ctx := machine.CUDNNContexts()[int(dev)]
	err = ctx.PoolingBackward(op.Pooling, 1.0, op.yDesc, y.(cudnn.Memory), op.yDesc, dy.(cudnn.Memory), op.xDesc, x.(cudnn.Memory), 0, op.xDesc, prealloc.(cudnn.Memory))
	return prealloc, err
}
