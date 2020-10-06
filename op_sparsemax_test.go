package gorgonia

import (
	"testing"

	"github.com/stretchr/testify/require"
	"gorgonia.org/tensor"
)

var testCases = []struct {
	input    []float64
	expected []float64
}{
	{
		[]float64{0.3, 0.1, 1.2, 2.3}, []float64{1.3, 1.1, 2.2, 3.3},
	},
	{
		[]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, []float64{2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
	},
	{
		[]float64{0.1, 0.1, 0.1}, []float64{0.3666666666666667, 0.3666666666666667, 0.3666666666666667},
	},
	{
		[]float64{-0.1, 0.3, -1.1, 2.7}, []float64{0.9, 1.3, 0, 3.7},
	},
}

func TestSparsemaxDo(t *testing.T) {
	c := require.New(t)

	for i, testCase := range testCases {
		tt := tensor.New(tensor.Of(tensor.Float64), tensor.WithShape(len(testCase.input)), tensor.WithBacking(testCase.input))
		op := newSparsemaxOp(tt.Shape())

		out, err := op.Do(tt)
		c.NoError(err, "failed test case: %d", i)
		c.Equal(testCase.expected, out.Data())
	}
}

func TestSparsemaxFull(t *testing.T) {
	c := require.New(t)

	for i, testCase := range testCases {
		tt := tensor.New(tensor.Of(tensor.Float64), tensor.WithShape(len(testCase.input)), tensor.WithBacking(testCase.input))
		expected := tensor.New(tensor.Of(tensor.Float64), tensor.WithShape(len(testCase.expected)), tensor.WithBacking(testCase.expected))

		g := NewGraph()
		inp := NewTensor(g, tensor.Float64, 1, WithShape(len(testCase.input)), WithName("inp"))
		out := Must(Sparsemax(inp))

		vm := NewTapeMachine(g)
		err := Let(inp, tt)
		c.NoError(err, "failed assigning input on case %d", i)

		c.NoError(vm.RunAll())
		c.NoError(vm.Close())

		c.Equal(expected.Data(), out.Value().(*tensor.Dense).Data(), "output is not equal to expected value for case %d", i)
	}
}
