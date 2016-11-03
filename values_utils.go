package gorgonia

import "github.com/chewxy/gorgonia/tensor/types"

func TypeOf(v Value) Type {
	switch t := v.(type) {
	case types.Tensor:
		dt, dim := tensorInfo(t)
		shp := t.Shape()
		tt := newTensorType(dim, dt)
		tt.shape = shp
		return tt
	case Scalar:
		switch s := t.(type) {

		}
	default:

	}
}

func DtypeOf(v Value) Dtype             { return MAXDTYPE }
func ValueEq(a, b Value) bool           { return false }
func CloneValue(v Value) (Value, error) { return nil }
func ZeroValue(v Value) Value           { return nil }
