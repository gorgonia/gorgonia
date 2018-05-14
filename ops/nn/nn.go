package nnops

import (
	"github.com/pkg/errors"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func CheckConvolutionParams(pad, stride, dilation []int) error {
	// checks
	for _, s := range stride {
		if s <= 0 {
			return errors.Errorf("Cannot use strides of less than or equal 0: %v", stride)
		}
	}

	for _, p := range pad {
		if p < 0 {
			return errors.Errorf("Cannot use padding of less than 0: %v", pad)
		}
	}

	for _, d := range dilation {
		if d <= 0 {
			return errors.Errorf("Cannot use dilation less than or eq 0 %v", dilation)
		}
	}
	return nil
}

func Conv2d(im, filter *G.Node, kernelShape tensor.Shape, pad, stride, dilation []int) (retVal *G.Node, err error) {
	var op *convolution
	if op, err = makeConvolutionOp(im, filter, kernelShape, pad, stride, dilation); err != nil {
		return nil, err
	}
	return G.ApplyOp(op, im, filter)
}
