package gtu

import (
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

// GetDenseTensor tries to extract a tensor.DenseTensor from a tensor.Tensor.
func GetDenseTensor(t tensor.Tensor) (tensor.DenseTensor, error) {
	switch tt := t.(type) {
	case tensor.DenseTensor:
		return tt, nil
	case tensor.Densor:
		return tt.Dense(), nil
	default:
		return nil, errors.Errorf("Tensor %T is not a DenseTensor", t)
	}
}
