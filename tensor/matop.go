package tensor

import (
	tb "github.com/chewxy/gorgonia/tensor/b"
	tf32 "github.com/chewxy/gorgonia/tensor/f32"
	tf64 "github.com/chewxy/gorgonia/tensor/f64"
	ti "github.com/chewxy/gorgonia/tensor/i"
	"github.com/chewxy/gorgonia/tensor/types"
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
}

func Argmax(t types.Tensor, axis int) (*ti.Tensor, error) {
	if am, ok := t.(Argmaxer); ok {
		return am.Argmax(axis)
	}
	return nil, types.NewError(types.DtypeMismatch, "Cannot argmax %T", t)
}
