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

func Dropout(x *G.Node, prob float64) (retVal *G.Node, err error) {
	var op *dropout
	if op, err = newDropout(x, prob); err != nil {
		return nil, err
	}

	// states := &scratchOp{x.Shape().Clone(), x.Dtype(), ""}
	// m := G.NewUniqueNode(G.WithType(x.Type()), G.WithOp(states), G.In(x.Graph()), G.WithShape(states.shape...))

	retVal, err = G.ApplyOp(op, x)
	return
}

func Rectify(x *G.Node) (retVal *G.Node, err error) {
	var op *activation
	if op, err = newRelu(); err != nil {
		return nil, err
	}
	retVal, err = G.ApplyOp(op, x)
	return
}

// func BatchNorm(x *Node, momentum, epsilon float64, auto bool) (*G.Node, *BatchNormOp, error) {
// 	dt, err := dtypeOf(x.Type())
// 	if err != nil {
// 		return nil, nil, err
// 	}

// 	batches := x.Shape()[0]
// 	channels := x.Shape()[1]
// 	spatialDim := x.Shape().TotalSize() / (channels * batches)

// 	scale := &scratchOp{x.Shape().Clone(), dt, "scale"}
// 	bias := &scratchOp{x.Shape().Clone(), dt, "bias"}
// 	mean := &scratchOp{x.Shape().Clone(), dt, "mean"}
// 	variance := &scratchOp{x.Shape().Clone(), dt, "variance"}
// 	cacheMean := &scratchOp{x.Shape().Clone(), dt, "cacheMean"}
// 	cacheVariance := &scratchOp{x.Shape().Clone(), dt, "cacheVariance"}

// }
