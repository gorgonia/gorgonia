// +build cuda

package nnops

import (
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func Conv2d(im, filter *G.Node, kernelShape tensor.Shape, pad, stride, dilation []int) (retVal *G.Node, err error) {
	var op *convolution
	if op, err = makeConvolutionOp(im, filter, kernelShape, pad, stride, dilation); err != nil {
		return nil, err
	}
	return G.ApplyOp(op, im, filter)
}

func MaxPool2D(x *G.Node, kernel tensor.Shape, pad, stride []int) (retVal *G.Node, err error) {
	var op *maxpool
	if op, err = newMaxPoolOp(x, kernel, pad, stride); err != nil {
		return nil, err
	}
	return G.ApplyOp(op, x)
}

func Dropout(x *G.Node, prob float64) (retVal *Node, err error) {
	var op *dropout
	if op, err = newDropout(x, prob); err != nil {
		return nil, err
	}

	states := &dropoutState{x.Shape().Clone()}
	m := G.NewUniqueNode(G.WithType(x.Type()), G.WithOp(op), G.In(x.Graph()), G.WithShape(states.shape...))

	return G.ApplyOp(op, x, m)
}
