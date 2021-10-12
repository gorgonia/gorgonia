package gorgonia

import (
	"testing"

	"github.com/stretchr/testify/require"
	"gorgonia.org/tensor"
)

func TestByIndicesOpDo(t *testing.T) {
	testCases := []struct {
		desc           string
		input          tensor.Tensor
		indices        tensor.Tensor
		axis           int
		expectedOutput []float64
		expectedShape  tensor.Shape
	}{
		{
			desc: "Example 1",
			input: tensor.New(
				tensor.WithShape(4, 2),
				tensor.WithBacking(tensor.Range(tensor.Float64, 0, 8)),
			),
			indices: tensor.New(
				tensor.WithShape(4),
				tensor.WithBacking([]int{0, 3, 2, 1}),
			),
			axis:           0,
			expectedOutput: []float64{0, 1, 6, 7, 4, 5, 2, 3},
			expectedShape:  tensor.Shape{4, 2},
		},
		{
			// 0 1 2
			// 3 4 5
			desc: "Example 2",
			input: tensor.New(
				tensor.WithShape(2, 3),
				tensor.WithBacking(tensor.Range(tensor.Float64, 0, 6)),
			),
			indices: tensor.New(
				tensor.WithShape(4),
				tensor.WithBacking([]int{0, 2, 1, 1}),
			),
			axis:           1,
			expectedOutput: []float64{0, 2, 1, 1, 3, 5, 4, 4},
			expectedShape:  tensor.Shape{2, 4},
		},
		{
			desc: "Example 3",
			input: tensor.New(
				tensor.WithShape(2, 5),
				tensor.WithBacking(tensor.Range(tensor.Float64, 0, 10)),
			),
			indices: tensor.New(
				tensor.WithShape(2),
				tensor.WithBacking([]int{1, 1}),
			),
			axis:           0,
			expectedOutput: []float64{5, 6, 7, 8, 9, 5, 6, 7, 8, 9},
			expectedShape:  tensor.Shape{2, 5},
		},
	}

	for _, tcase := range testCases {
		t.Run(tcase.desc, func(t *testing.T) {
			c := require.New(t)

			op := newByIndicesOp(tcase.axis)

			inputV, _, _, err := anyToValue(tcase.input)
			c.NoError(err)

			indicesV, _, _, err := anyToValue(tcase.indices)
			c.NoError(err)

			output, err := op.Do(inputV, indicesV)
			c.NoError(err)

			c.Equal(tcase.expectedOutput, output.Data())
			c.Equal(tcase.expectedShape, output.Shape())
		})
	}
}

func TestByIndicesOpFull(t *testing.T) {
	testCases := []struct {
		desc           string
		input          tensor.Tensor
		indices        tensor.Tensor
		axis           int
		expectedOutput []float64
		expectedShape  tensor.Shape
	}{
		{
			desc: "Example 1",
			input: tensor.New(
				tensor.WithShape(4, 2),
				tensor.WithBacking(tensor.Range(tensor.Float64, 0, 8)),
			),
			indices: tensor.New(
				tensor.WithShape(4),
				tensor.WithBacking([]int{0, 3, 2, 1}),
			),
			axis:           0,
			expectedOutput: []float64{0, 1, 6, 7, 4, 5, 2, 3},
			expectedShape:  tensor.Shape{4, 2},
		},
	}

	for _, tcase := range testCases {
		t.Run(tcase.desc, func(t *testing.T) {
			c := require.New(t)

			g := NewGraph()

			indices := NewTensor(g, tensor.Int, 1, WithName("indices"), WithShape(tcase.indices.Shape().TotalSize()), WithValue(tcase.indices))
			input := NewTensor(g, tensor.Float64, tcase.input.Shape().Dims(), WithName("input"), WithShape(tcase.input.Shape()...), WithValue(tcase.input))

			output, err := ByIndices(input, indices, tcase.axis)
			c.NoError(err)

			t.Logf("%v output shape: %v", tcase.desc, output.Shape())
			t.Logf("%v input shape: %v", tcase.desc, input.Shape())

			y := NewTensor(g, tensor.Float64, tcase.input.Shape().Dims(), WithName("target"), WithShape(tcase.input.Shape()...), WithValue(tcase.input))

			cost := Must(Mean(Must((Sub(output, y))))) // MSE

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

			t.Logf("%v input %v", tcase.desc, input.Value())
			t.Logf("%v result: %v", tcase.desc, output.Value())
			t.Logf("%v cost: %v", tcase.desc, cost.Value())

			c.Equal(tcase.expectedOutput, output.Value().Data())
			c.Equal(tcase.expectedShape, output.Shape())
			c.Equal(0.0, cost.Value().Data().(float64))
		})
	}
}
