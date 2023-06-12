package gorgonia

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestEmbedding(t *testing.T) {
	var tests = []struct {
		dt                tensor.Dtype
		expected          interface{}
		expectedGrad      interface{}
		expectedInputGrad interface{}
	}{
		{Float64, []float64{1.0, 1.0, 1.0, 1.0, 1.0}, []float64{0.2, 0.2, 0.2, 0.2, 0.2}, []float64{0, 0, 0, 0, 0}},
		{Float64, []float64{1.25, 1.25, 1.25, 0.0, 0.0}, []float64{0.2, 0.2, 0.2, 0.2, 0.2}, []float64{1, 1, 1, 0.0, 0.0}},
		{Float64, []float64{2.0, 2.0, 0.0, 0.0, 0.0}, []float64{0.2, 0.2, 0.2, 0.2, 0.2}, []float64{0.4, 0.4, 0, 0, 0}},
		{Float32, []float32{1.25, 1.25, 1.25, 0.0, 0.0}, []float32{0.2, 0.2, 0.2, 0.2, 0.2}, []float32{1, 1, 1, 0, 0}},
		{Float32, []float32{2.0, 2.0, 0.0, 0.0, 0.0}, []float32{0.2, 0.2, 0.2, 0.2, 0.2}, []float32{0.4, 0.4, 0, 0, 0}},
	}

	for _, tt := range tests {
		name := fmt.Sprintf("%v", tt.dt)
		t.Run(name, func(t *testing.T) {
			g := NewGraph()
			emb := NewMatrix(g, tt.dt, WithShape(10, 2), WithName("x"), WithInit(RangedFrom(0)))
			x := NewVector(g, Int64, WithShape(5), WithName("x"), WithInit(RangedFromWithStep(0, 2)))

			y, err := ApplyOp(&embeddingOp{}, emb, x)
			assert.NoError(t, err)

			_ = y
			// cost, _ := Mean(y)
			// if _, err := Grad(cost, x); err != nil {
			// 	t.Fatal(err)
			// }

			// m := NewTapeMachine(g, BindDualValues())
			// defer m.Close()
			// defer runtime.GC()

			// require.NoError(t, m.RunAll())
			// assert.Equal(t, tt.expected, y.Value().Data())

			// yGrad, err := y.Grad()
			// require.NoError(t, err)
			// assert.Equal(t, tt.expectedGrad, yGrad.Data())

			// xGrad, err := x.Grad()
			// require.NoError(t, err)
			// assert.Equal(t, tt.expectedInputGrad, xGrad.Data())
		})
	}
}
