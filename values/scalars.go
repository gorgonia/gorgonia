package values

import "gorgonia.org/tensor"

type Scalar struct {
	*tensor.Dense
}

func MakeScalar(v interface{}) Scalar {
	d := tensor.New(tensor.FromScalar(v))
	return Scalar{d}
}

func (s Scalar) Clone() interface{} {
	return Scalar{s.Dense.Clone().(*tensor.Dense)}
}

func NewF64(f float64) Scalar { return MakeScalar(f) }
