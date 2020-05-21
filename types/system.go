package types

import (
	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	gerrors "gorgonia.org/gorgonia/internal/errors"
	"gorgonia.org/tensor"
)

// DtypeOf returns the dtype of a given type.
//
// If the input hm.Type is not a parameterized type, or a Dtype, an error will be returned.
func DtypeOf(t hm.Type) (retVal tensor.Dtype, err error) {
	switch p := t.(type) {
	case tensor.Dtype:
		retVal = p
	case TensorType:
		return DtypeOf(p.Of)
	case hm.TypeVariable:
		err = errors.Errorf("instance %v does not have a dtype", p)
	default:
		err = errors.Errorf(gerrors.NYITypeFail, "dtypeOf", p)
		return
	}

	return
}
