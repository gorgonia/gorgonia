package gorgonia

import (
	"testing"

	tf64 "github.com/chewxy/gorgonia/tensor/f64"
)

func TestBasicArithmeticDo(t *testing.T) {
	g := NewGraph()
	xb := tf64.NewTensor(tf64.WithBacking([]float64{1, 2, 3, 4}))
	x := NewVector(g, Float64, WithValue(xb))

	res := Must(Sub(onef64, x))

	v, err := res.op.Do(onef64.Value(), x.Value())
	if err != nil {
		t.Error(err)
	}
	t.Log(v)
}
