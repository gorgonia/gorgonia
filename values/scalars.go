package values

import (
	"gorgonia.org/dtype"
	"gorgonia.org/tensor"
)

type Scalar struct {
	*tensor.Dense
}

func MakeScalar(v interface{}) Scalar {
	d := tensor.New(tensor.FromScalar(v))
	return Scalar{d}
}

func MakeScalarOf(dt dtype.Dtype, v int) Scalar {
	r, err := dtype.FromInt(v)
	if err != nil {
		panic(err)
	}
	return MakeScalar(r)
}

func (s Scalar) Clone() interface{} {
	return Scalar{s.Dense.Clone().(*tensor.Dense)}
}

func NewF64(f float64) Scalar { return MakeScalar(f) }

func NewF32(f float32) Scalar { return MakeScalar(f) }
