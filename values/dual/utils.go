package dual

import (
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/tensor/dense"
	"gorgonia.org/tensor/scalar"
)

func clone[DT any](v values.Value[DT]) values.Value[DT] {
	switch v := v.(type) {
	case *dense.Dense[DT]:
		return v.Clone()
	case scalar.Scalar[DT]:
		return scalar.S(v.V)
	default:
		panic("NYI")
	}
}
