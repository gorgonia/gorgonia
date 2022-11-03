package gorgonia

import (
	"testing"

	"github.com/stretchr/testify/require"
	"gorgonia.org/tensor"
)

func TestAvgPoolOp(t *testing.T) {
	testCases := []struct {
		desc           string
		input          tensor.Tensor
		kernelSize     tensor.Shape
		pad            []int
		stride         []int
		expectedOutput interface{}
		expectedShape  tensor.Shape
		expectedCost   interface{}
		PoolFunc       func(*Node, tensor.Shape, []int, []int) (*Node, error)
	}{
		{
			desc: "Example 1",
			input: tensor.New(
				tensor.WithShape(1, 1, 4, 4),
				tensor.WithBacking(tensor.Range(tensor.Float64, 0, 16)),
			),
			kernelSize:     []int{2, 2},
			pad:            []int{0, 0},
			stride:         []int{1, 1},
			expectedOutput: []float64{2.5, 3.5, 4.5, 6.5, 7.5, 8.5, 10.5, 11.5, 12.5},
			expectedShape:  tensor.Shape{1, 1, 3, 3},
			expectedCost:   53.583333333333336,
			PoolFunc:       AveragePool2D,
		},
		{
			desc: "Example 2",
			input: tensor.New(
				tensor.WithShape(1, 1, 4, 4),
				tensor.WithBacking(tensor.Range(tensor.Float32, 0, 16)),
			),
			kernelSize:     []int{2, 2},
			pad:            []int{0, 0},
			stride:         []int{1, 1},
			expectedOutput: []float32{2.5, 3.5, 4.5, 6.5, 7.5, 8.5, 10.5, 11.5, 12.5},
			expectedShape:  tensor.Shape{1, 1, 3, 3},
			expectedCost:   float32(53.583332),
			PoolFunc:       AveragePool2D,
		},
	}

	for _, tcase := range testCases {
		t.Run(tcase.desc, func(t *testing.T) {
			c := require.New(t)

			g := NewGraph()

			input := NewTensor(g, tcase.input.Dtype(), tcase.input.Shape().Dims(), WithName("input"), WithShape(tcase.input.Shape()...), WithValue(tcase.input))

			output, err := tcase.PoolFunc(input, tcase.kernelSize, tcase.pad, tcase.stride)
			c.NoError(err)

			t.Logf("%v output shape: %v", tcase.desc, output.Shape())
			t.Logf("%v input shape: %v", tcase.desc, input.Shape())

			y := NewTensor(g, output.Dtype(), output.Dims(), WithShape(output.Shape()...), WithInit(Ones()))

			cost := Must(Mean(Must(Square(Must(Sub(output, y)))))) // MSE

			_, err = Grad(cost, input)
			c.NoError(err)

			// logger := log.New(os.Stdout, "", 0)

			vm := NewTapeMachine(
				g,
				//WithLogger(logger),
				WithWatchlist(),
				BindDualValues(output),
				TraceExec(),
			)

			c.NoError(vm.RunAll())
			c.NoError(vm.Close())

			t.Logf("%v input:\n%v", tcase.desc, input.Value())
			t.Logf("%v result:\n%v", tcase.desc, output.Value())
			t.Logf("%v cost: %v", tcase.desc, cost.Value())

			c.Equal(tcase.expectedOutput, output.Value().Data())
			c.Equal(tcase.expectedShape, output.Shape())
			c.Equal(tcase.expectedCost, cost.Value().Data())
		})
	}
}
