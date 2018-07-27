// +build !cuda

package nnops

import (
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func Conv2d(im, filter *G.Node, kernelShape tensor.Shape, pad, stride, dilation []int) (retVal *G.Node, err error) {
	return G.Conv2d(im, filter, kernelShape, pad, stride, dilation)
}

func MaxPool2D(x *G.Node, kernel tensor.Shape, pad, stride []int) (*G.Node, error) {
	return G.MaxPool2D(x, kernel, pad, stride)
}

func Dropout(x *G.Node, prob float64) (retVal *G.Node, err error) {
	return G.Dropout(x, prob)
}

func BatchNorm(x *G.Node, momentum, epsilon float64, auto bool) (*G.Node, *G.BatchNormOp, error) {
	return G.BatchNorm(x, momentum, epsilon, auto)
}
