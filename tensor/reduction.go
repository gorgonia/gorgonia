package tensor

import (
	tf32 "github.com/chewxy/gorgonia/tensor/f32"
	tf64 "github.com/chewxy/gorgonia/tensor/f64"
	"github.com/chewxy/gorgonia/tensor/types"
)

func Sum(t types.Tensor, along ...int) (retVal types.Tensor, err error) {
	switch T := t.(type) {
	case *tf64.Tensor:
		return T.Sum(along...)
	case *tf32.Tensor:
		return T.Sum(along...)
	default:
		err = types.NewError(types.NotYetImplemented, "Sum for %T", T)
		return
	}
	panic("Unreachable")
}
