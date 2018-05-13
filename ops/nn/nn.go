package nnops

import (
	"github.com/pkg/errors"
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
