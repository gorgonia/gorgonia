package tensor

import (
	"fmt"

	tb "github.com/chewxy/gorgonia/tensor/b"
	tf32 "github.com/chewxy/gorgonia/tensor/f32"
	tf64 "github.com/chewxy/gorgonia/tensor/f64"
	ti "github.com/chewxy/gorgonia/tensor/i"
	"github.com/chewxy/gorgonia/tensor/types"
)

type Argmaxer interface {
	Argmax(int) (*ti.Tensor, error)
}

func Clone(t types.Tensor) types.Tensor {
	switch tt := t.(type) {
	case *tf64.Tensor:
		return tt.Clone()
	case *tf32.Tensor:
		return tt.Clone()
	case *ti.Tensor:
		return tt.Clone()
	case *tb.Tensor:
		return tt.Clone()
	default:
		panic(fmt.Sprintf("Cloning not yet implemented for %T", t))
	}
	panic("Unreachable")
}

func Copy(to, from types.Tensor) error {
	switch ft := from.(type) {
	case *tf64.Tensor:
		tt, ok := to.(*tf64.Tensor)
		if !ok {
			return types.NewError(types.DtypeMismatch, "%T and %T are not of the same type", from, to)
		}

		return ft.CopyTo(tt)
	case *tf32.Tensor:
		tt, ok := to.(*tf32.Tensor)
		if !ok {
			return types.NewError(types.DtypeMismatch, "%T and %T are not of the same type", from, to)
		}

		return ft.CopyTo(tt)
	case *ti.Tensor:
		tt, ok := to.(*ti.Tensor)
		if !ok {
			return types.NewError(types.DtypeMismatch, "%T and %T are not of the same type", from, to)
		}

		return ft.CopyTo(tt)
	case *tb.Tensor:
		tt, ok := to.(*tb.Tensor)
		if !ok {
			return types.NewError(types.DtypeMismatch, "%T and %T are not of the same type", from, to)
		}

		return ft.CopyTo(tt)
	default:
		return types.NewError(types.NotYetImplemented, "Copying data not yet implemented for %T", from)
	}
	return nil
}

func Ones(dt types.Dtype, sizes ...int) types.Tensor {
	switch dt {
	case types.Float64:
		return tf64.Ones(sizes...)
	case types.Float32:
		return tf32.Ones(sizes...)
	case types.Int:
		return ti.Ones(sizes...)
	case types.Bool:
		return tb.Ones(sizes...)
	default:
		panic(fmt.Sprintf("dt of %v not handled yet", dt))
	}
	panic("unreachabale")
}

func Zeroes(dt types.Dtype, sizes ...int) types.Tensor {
	switch dt {
	case types.Float64:
		return tf64.Zeroes(sizes...)
	case types.Float32:
		return tf32.Zeroes(sizes...)
	case types.Int:
		return ti.Zeroes(sizes...)
	case types.Bool:
		return tb.Zeroes(sizes...)
	default:
		panic(fmt.Sprintf("dt of %v not handled yet", dt))
	}
	panic("unreachable")
}
