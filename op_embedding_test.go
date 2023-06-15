package gorgonia

import (
	"fmt"
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gorgonia.org/tensor"
)

func TestEmbedding(t *testing.T) {
	var tests = []struct {
		dt                tensor.Dtype
		w                 interface{}
		expected          interface{}
		expectedGrad      interface{}
		expectedInputGrad interface{}
	}{
		{
			Float64,
			[]float64{0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5},
			[]float64{0, 0.5, 2, 2.5, 4, 4.5, 6, 6.5, 8, 8.5},
			[]float64{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1},
			[]float64{0.1, 0.1, 0, 0, 0.1, 0.1, 0, 0, 0.1, 0.1, 0, 0, 0.1, 0.1, 0, 0, 0.1, 0.1, 0, 0},
		},
		{
			Float32,
			[]float32{0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5},
			[]float32{0, 0.5, 2, 2.5, 4, 4.5, 6, 6.5, 8, 8.5},
			[]float32{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1},
			[]float32{0.1, 0.1, 0, 0, 0.1, 0.1, 0, 0, 0.1, 0.1, 0, 0, 0.1, 0.1, 0, 0, 0.1, 0.1, 0, 0},
		},
	}

	for _, tt := range tests {
		name := fmt.Sprintf("%v", tt.dt)
		t.Run(name, func(t *testing.T) {
			var (
				g *ExprGraph = NewGraph()
				w *Node
				x *Node
			)
			w = NewMatrix(g, tt.dt, WithName("w"), WithValue(tensor.New(tensor.WithShape(10, 2), tensor.WithBacking(tt.w))))
			x = NewVector(g, tt.dt, WithName("x"), WithShape(5), WithInit(RangedFromWithStep(0, 2)))

			y, err := ApplyOp(&embeddingOp{}, w, x)
			assert.NoError(t, err)

			cost, _ := Mean(y)
			if _, err := Grad(cost, w); err != nil {
				t.Fatal(err)
			}

			m := NewTapeMachine(g, BindDualValues())
			defer m.Close()
			defer runtime.GC()

			require.NoError(t, m.RunAll())
			assert.Equal(t, tt.expected, y.Value().Data())

			yGrad, err := y.Grad()
			require.NoError(t, err)
			assert.Equal(t, tt.expectedGrad, yGrad.Data())

			wGrad, err := w.Grad()
			require.NoError(t, err)
			assert.Equal(t, tt.expectedInputGrad, wGrad.Data())
		})
	}
}
