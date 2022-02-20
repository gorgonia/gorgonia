package gorgonia

import (
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

// this file provides utility functions for solvers

func doL1Reg(w, g tensor.Tensor, l1reg interface{}) (err error) {
	var l1regs tensor.Tensor
	if l1regs, err = tensor.Sign(w); err != nil {
		errors.Wrap(err, signFail)
	}
	if _, err = tensor.Mul(l1reg, l1regs, tensor.WithIncr(g)); err != nil {
		return errors.Wrap(err, pointWiseMulFail)
	}
	defer returnTensor(l1regs)
	return nil
}

func doL2Reg(w, g tensor.Tensor, l2reg interface{}) (err error) {
	if _, err = tensor.Mul(w, l2reg, tensor.WithIncr(g)); err != nil {
		return errors.Wrap(err, pointWiseMulFail)
	}
	return nil
}

func computeRecip(x float64, as tensor.Dtype) (retVal interface{}, err error) {
	switch as {
	case tensor.Float64:
		return 1.0 / x, nil
	case tensor.Float32:
		return float32(1.0) / float32(x), nil
	default:
		return 0.0, errors.Errorf("Unhandled Dtype %v for computeRecip", as)
	}
}

func divBatch(g tensor.Tensor, batch float64) (err error) {
	recip, err := computeRecip(batch, g.Dtype())
	if err != nil {
		return errors.Wrap(err, "In divBatch()")
	}

	_, err = tensor.Mul(g, recip, tensor.UseUnsafe())
	if err != nil {
		return errors.Wrap(err, "Cannot multiply by reciprocal of batch count")
	}
	return nil
}

func clipGrad(g tensor.Tensor, clip, negClip interface{}) (err error) {
	if _, err = tensor.Clamp(g, negClip, clip, tensor.UseUnsafe()); err != nil {
		return errors.Wrap(err, clampFail)
	}
	return nil
}
