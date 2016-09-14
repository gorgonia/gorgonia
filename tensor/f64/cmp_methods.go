package tensorf64

import (
	tb "github.com/chewxy/gorgonia/tensor/b"
	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/pkg/errors"
)

type cmpOp byte

const (
	lt cmpOp = iota
	gt
	lte
	gte
	eq
	ne
)

func scalarCmpBacking(op cmpOp, left bool, val float64, data []float64) (backing []bool, err error) {
	backing = make([]bool, len(data))
	switch op {
	case lt:
		if left {
			for i, v := range data {
				backing[i] = v < val
			}
		} else {
			for i, v := range data {
				backing[i] = val < v
			}
		}
	case gt:
		if left {
			for i, v := range data {
				backing[i] = v > val
			}
		} else {
			for i, v := range data {
				backing[i] = val > v
			}
		}
	case lte:
		if left {
			for i, v := range data {
				backing[i] = v <= val
			}
		} else {
			for i, v := range data {
				backing[i] = val <= v
			}
		}
	case gte:
		if left {
			for i, v := range data {
				backing[i] = v >= val
			}
		} else {
			for i, v := range data {
				backing[i] = val >= v
			}
		}
	case eq:
		if left {
			for i, v := range data {
				backing[i] = v == val
			}
		} else {
			for i, v := range data {
				backing[i] = val == v
			}
		}
	case ne:
		if left {
			for i, v := range data {
				backing[i] = v != val
			}
		} else {
			for i, v := range data {
				backing[i] = val != v
			}
		}
	default:
		err = types.NewError(types.InvalidCmpOp, "Invalid Comparison Operator")
	}
	return
}

//scalarCmp compares a ndarray with a scalar value. left indicates if the ndarray is in left
func (t *Tensor) scalarCmp(op cmpOp, left bool, val float64) (retVal *tb.Tensor, err error) {
	var backing []bool
	if backing, err = scalarCmpBacking(op, left, val, t.data); err != nil {
		errors.Wrap(err, "Failed to compare backing with scalar")
		return
	}

	// we preserve any transpose info
	retVal = tb.NewTensor(tb.WithBacking(backing), tb.WithShape(t.Shape()...))
	if t.old != nil {
		retVal.T(t.transposeWith...)
	}
	return
}

// tensorCmp compares two ndarrays. It checks the "final" shape, that is to say post-transform. The return value willn't store any transform info
func (t *Tensor) tensorCmp(op cmpOp, other *Tensor, boolT bool) (retVal types.Tensor, err error) {
	// we compare the "final" shapes because that's what the shape of the retVal will take
	if !t.Shape().Eq(other.Shape()) {
		err = types.NewError(types.ShapeMismatch, "Cannot compare two tensors with %s. Got %s and %s", t.Shape(), other.Shape())
	}

	backing := make([]bool, len(t.data))
	switch op {
	case lt:
		for i, v := range t.data {
			backing[i] = v < other.data[i]
		}

	case gt:
		for i, v := range t.data {
			backing[i] = v > other.data[i]
		}

	case lte:
		for i, v := range t.data {
			backing[i] = v <= other.data[i]
		}

	case gte:
		for i, v := range t.data {
			backing[i] = v >= other.data[i]
		}

	case eq:
		for i, v := range t.data {
			backing[i] = v == other.data[i]
		}

	case ne:
		for i, v := range t.data {
			backing[i] = v != other.data[i]
		}

	default:
		err = types.NewError(types.InvalidCmpOp, "Invalid comparison operator %s", op)
		return
	}

	if !boolT {
		backingF := boolsToFloat64s(backing)
		retVal = NewTensor(WithBacking(backingF), WithShape(t.Shape()...))
	} else {
		retVal = tb.NewTensor(tb.WithBacking(backing), tb.WithShape(t.Shape()...))
	}
	return
}
