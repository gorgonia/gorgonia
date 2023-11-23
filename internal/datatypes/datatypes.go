package datatypes

import (
	"fmt"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/dtype"
	gerrors "gorgonia.org/gorgonia/internal/errors"
	"gorgonia.org/gorgonia/types"
	"gorgonia.org/tensor"
)

// Tensor represents values that are acceptable in Gorgonia. At this point, it is implemented by:
//   - tensor.Basic[DT]
//   - exprgraph.Node
//
// There is an overlap with values.Value. The reason is semantic clarity. Values are Tensors. Tensors are Values.
type Tensor[DT any] interface {
	tensor.Basic[DT]
}

// TypeOf returns the type of a given Tensor
func TypeOf[DT any](t Tensor[DT]) hm.Type {
	switch tt := t.(type) {
	case tensor.Basic[DT]:
		if tt.Shape().IsScalar() {
			return tt.Dtype()
		}
		return types.MakeTensorType(tt.Dims(), tt.Dtype())
	case hm.Typer:
		return tt.Type()
	default:
		panic(fmt.Sprintf("%v of %T is currently unsupported", tt, tt))
	}
}

// DtypeOf returns the dtype of a given type.
//
// If the input hm.Type is not a parameterized type, or a Dtype, an error will be returned.
func DtypeOf(t hm.Type) (retVal dtype.Dtype, err error) {
	switch p := t.(type) {
	case dtype.Dtype:
		retVal = p
	case types.TensorType:
		return DtypeOf(p.Of)
	case hm.TypeVariable:
		err = errors.Errorf("instance %v does not have a dtype", p)
	default:
		err = gerrors.NYI(p)
		return
	}

	return
}
