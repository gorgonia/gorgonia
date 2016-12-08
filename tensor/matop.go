package tensor

import (
	tb "github.com/chewxy/gorgonia/tensor/b"
	tf32 "github.com/chewxy/gorgonia/tensor/f32"
	tf64 "github.com/chewxy/gorgonia/tensor/f64"
	ti "github.com/chewxy/gorgonia/tensor/i"
	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/pkg/errors"
)

func Repeat(t types.Tensor, axis int, repeats ...int) (retVal types.Tensor, err error) {
	switch tt := t.(type) {
	case *tf64.Tensor:
		return tt.Repeat(axis, repeats...)
	case *tf32.Tensor:
		return tt.Repeat(axis, repeats...)
	case *ti.Tensor:
		return tt.Repeat(axis, repeats...)
	case *tb.Tensor:
		return tt.Repeat(axis, repeats...)
	default:
		panic("Not yet implemented")
	}
	panic("unreachable")
}

func T(t types.Tensor, axes ...int) (retVal types.Tensor, err error) {
	switch tt := t.(type) {
	case *tf64.Tensor:
		return tt.SafeT(axes...)
	case *tf32.Tensor:
		return tt.SafeT(axes...)
	case *ti.Tensor:
		return tt.SafeT(axes...)
	case *tb.Tensor:
		return tt.SafeT(axes...)
	default:
		panic("Not yet implemented")
	}
	panic("unreachable")
}

func Slice(t types.Tensor, slices ...types.Slice) (retVal types.Tensor, err error) {
	switch tt := t.(type) {
	case *tf64.Tensor:
		return tt.Slice(slices...)
	case *tf32.Tensor:
		return tt.Slice(slices...)
	case *ti.Tensor:
		return tt.Slice(slices...)
	case *tb.Tensor:
		return tt.Slice(slices...)
	default:
		panic("Not yet implemented")
	}
	panic("unreachable")
}

func Concat(axis int, t types.Tensor, others ...types.Tensor) (retVal types.Tensor, err error) {
	switch tt := t.(type) {
	case *tf64.Tensor:
		ts := make([]*tf64.Tensor, len(others))
		for i, o := range others {
			if ot, ok := o.(*tf64.Tensor); !ok {
				return nil, errors.Errorf("Expected all tensors to be *tf64.Tensor. Got %T in %dth index of slice", o, i)
			} else {
				ts[i] = ot
			}
		}

		return tt.Concat(axis, ts...)
	case *tf32.Tensor:
		ts := make([]*tf32.Tensor, len(others))
		for i, o := range others {
			if ot, ok := o.(*tf32.Tensor); !ok {
				return nil, errors.Errorf("Expected all tensors to be *tf32.Tensor. Got %T in %dth index of slice", o, i)
			} else {
				ts[i] = ot
			}
		}

		return tt.Concat(axis, ts...)
	case *ti.Tensor:
		ts := make([]*ti.Tensor, len(others))
		for i, o := range others {
			if ot, ok := o.(*ti.Tensor); !ok {
				return nil, errors.Errorf("Expected all tensors to be *ti.Tensor. Got %T in %dth index of slice", o, i)
			} else {
				ts[i] = ot
			}
		}

		return tt.Concat(axis, ts...)
	case *tb.Tensor:
		ts := make([]*tb.Tensor, len(others))
		for i, o := range others {
			if ot, ok := o.(*tb.Tensor); !ok {
				return nil, errors.Errorf("Expected all tensors to be *tb.Tensor. Got %T in %dth index of slice", o, i)
			} else {
				ts[i] = ot
			}
		}

		return tt.Concat(axis, ts...)
	default:
		panic("Not yet implemented")
	}
	panic("Unreachable")
}

func Argmax(t types.Tensor, axis int) (*ti.Tensor, error) {
	if am, ok := t.(Argmaxer); ok {
		return am.Argmax(axis)
	}
	return nil, types.NewError(types.DtypeMismatch, "Cannot argmax %T", t)
}

func Argmin(t types.Tensor, axis int) (*ti.Tensor, error) {
	if am, ok := t.(Argminer); ok {
		return am.Argmin(axis)
	}
	return nil, types.NewError(types.DtypeMismatch, "Cannot argmin %T", t)
}
