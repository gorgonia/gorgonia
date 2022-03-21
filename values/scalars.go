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

// MakeScalarOf creates a scalar value from a given int. If the dtype doesn't have a constructor from an integer
// then it returns a scalar of type Int.
func MakeScalarOf(dt dtype.Dtype, v int) Scalar {
	r, err := dtype.FromInt(dt, v)
	if err != nil {
		return MakeScalar(v)
	}
	return MakeScalar(r)
}

// CopyScalarOf copies the scalar value
func CopyScalarOf(dt dtype.Dtype, to Scalar, v int) error {
	r, err := dtype.FromInt(dt, v)
	if err != nil {
		return err
	}
	to.Set(0, r)
	return nil
}

func (s Scalar) Clone() interface{} {
	return Scalar{s.Dense.Clone().(*tensor.Dense)}
}

func NewF64(f float64) Scalar { return MakeScalar(f) }

func NewF32(f float32) Scalar { return MakeScalar(f) }
