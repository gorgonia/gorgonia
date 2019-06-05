// +build !cuda

package nnops

import (
	G "gorgonia"
	"gorgonia.org/tensor"
)

func Conv2d(im, filter *G.Node, kernelShape tensor.Shape, pad, stride, dilation []int) (retVal *G.Node, err error) {
	return G.Conv2d(im, filter, kernelShape, pad, stride, dilation)
}

func Conv1d(in, filter *G.Node, kernel, pad, stride, dilation int) (*G.Node, error) {
	return Conv2d(in, filter, tensor.Shape{1, kernel}, []int{0, pad}, []int{1, stride}, []int{1, dilation})
}

func MaxPool2D(x *G.Node, kernel tensor.Shape, pad, stride []int) (*G.Node, error) {
	return G.MaxPool2D(x, kernel, pad, stride)
}

func Dropout(x *G.Node, prob float64) (retVal *G.Node, err error) {
	return G.Dropout(x, prob)
}

func Rectify(x *G.Node) (retVal *G.Node, err error) {
	return G.Rectify(x)
}

func BatchNorm(x, scale, bias *G.Node, momentum, epsilon float64) (retVal, γ, β *G.Node, op *G.BatchNormOp, err error) {
	return G.BatchNorm(x, scale, bias, momentum, epsilon)
}
