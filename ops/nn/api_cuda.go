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

func MaxPool2D(x *G.Node, kernel tensor.Shape, pad, stride []int) (*G.Node, error) {
	var op *maxpool
	if op, err = newMaxPoolOp(x, kernel, pad, stride); err != nil {
		return nil, err
	}
	return G.ApplyOp(op, x)
}
