package tensor

import "github.com/pkg/errors"

// this file handles matops. While by default most of these matops should already have been defined as part of the
// Tensor interface, not all are possible(for example, concatenating a sparse tensor), hence the need for the following functions

// Repeat repeats a Tensor along the axis and given the number of repeats.
func Repeat(t Tensor, axis int, repeats ...int) (retVal Tensor, err error) {
	switch T := t.(type) {
	case *Dense:
		return T.Repeat(axis, repeats...)
	}
	panic("Unreachable")
}

// T safely transposes a Tensor. It returns a tensor that is not a view of the input tensor - rather, the data is all copied.
func T(t Tensor, axes ...int) (retVal Tensor, err error) {
	switch tt := t.(type) {
	case *Dense:
		return tt.SafeT(axes...)
	}
	panic("Unreachable")
}

// Concat concatenates a list of Tensors. At the moment the operation only supports Tensors of the same type
// (*Dense can only be concatenated with a bunch of *Dense, CSCs can only be concatenated with a bunch of CSC, etc)
func Concat(axis int, t Tensor, others ...Tensor) (retVal Tensor, err error) {
	if len(others) == 0 {
		return t, nil
	}
	switch T := t.(type) {
	case *Dense:
		ts := make([]*Dense, len(others))
		for i, o := range others {
			if ot, ok := o.(*Dense); !ok {
				return nil, errors.Errorf("Expected all Tensors to be *Dense")
			} else {
				ts[i] = ot
			}
		}
		return T.Concat(axis, ts...)
	}
	panic("Unreachable")
}
