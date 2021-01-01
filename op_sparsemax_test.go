package gorgonia

import (
	"testing"

	"github.com/stretchr/testify/require"
	"gorgonia.org/tensor"
)

var testCasesSparseMaxDo = []struct {
	size     int
	input    interface{}
	expected interface{}
}{
	{
		4, []float64{0.3, 0.1, 1.2, 2.3}, []float64{1.3, 1.1, 2.2, 3.3},
	},
	{
		10, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, []float64{2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
	},
	{
		3, []float64{0.1, 0.1, 0.1}, []float64{0.3666666666666667, 0.3666666666666667, 0.3666666666666667},
	},
	{
		4, []float64{-0.1, 0.3, -1.1, 2.7}, []float64{0.9, 1.3, 0, 3.7},
	},
	{
		4, []float32{0.3, 0.1, 1.2, 2.3}, []float32{1.3, 1.1, 2.2, 3.3},
	},
	{
		10, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, []float32{2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
	},
	{
		3, []float32{0.1, 0.1, 0.1}, []float32{0.36666664, 0.36666664, 0.36666664},
	},
	{
		4, []float32{-0.1, 0.3, -1.1, 2.7}, []float32{0.9, 1.3, 0, 3.7},
	},
}

var testCasesSparseMaxDoDiff = []struct {
	shape tensor.Shape
	input interface{}
	grad  interface{}

	expected interface{}
}{
	{
		tensor.Shape{5},
		[]float64{0.0000, 0.0000, 0.0521, 0.2354, 0.7124},
		[]float64{0.2860, -0.0702, 0.8080, 0.9913, 1.4683},
		[]float64{0, 0, -0.2811999999999999, -0.09789999999999999, 0.3791},
	},
	{
		tensor.Shape{5},
		[]float64{0.0556, 0.0000, 0.7118, 0.2325, 0.0000},
		[]float64{0.1109, -1.4741, 0.7671, 0.2878, 0.0334},
		[]float64{-0.2777, -0.0000, 0.3785, -0.1008, -0.0000},
	},
	{
		tensor.Shape{5},
		[]float64{0.2841, 0.0000, 0.7159, 0.0000, 0.0000},
		[]float64{0.2094, -1.0000, 0.6411, -0.5032, -0.3909},
		[]float64{-0.21585000000000001, 0, 0.21585, 0, 0},
	},
	{
		tensor.Shape{5},
		[]float64{0.2592, 0.0000, 0.6909, 0.0498, 0.0000},
		[]float64{0.2094, -1.0000, 0.6411, 0.0000, -0.3909},
		[]float64{-0.07410000000000003, 0, 0.3576, -0.28350000000000003, 0},
	},
	{
		tensor.Shape{5},
		[]float32{0.0000, 0.0000, 0.0521, 0.2354, 0.7124},
		[]float32{0.2860, -0.0702, 0.8080, 0.9913, 1.4683},
		[]float32{-0, -0, -0.2812, -0.09790003, 0.37909997},
	},
	{
		tensor.Shape{5},
		[]float32{0.0556, 0.0000, 0.7118, 0.2325, 0.0000},
		[]float32{0.1109, -1.4741, 0.7671, 0.2878, 0.0334},
		[]float32{-0.2777, -0, 0.37849998, -0.10079998, -0},
	},
	{
		tensor.Shape{5},
		[]float32{0.2841, 0.0000, 0.7159, 0.0000, 0.0000},
		[]float32{0.2094, -1.0000, 0.6411, -0.5032, -0.3909},
		[]float32{-0.21585, -0, 0.21585, -0, -0},
	},
	{
		tensor.Shape{5},
		[]float32{0.2592, 0.0000, 0.6909, 0.0498, 0.0000},
		[]float32{0.2094, -1.0000, 0.6411, 0.0000, -0.3909},
		[]float32{-0.07409999, -0, 0.3576, -0.2835, -0},
	},
	{
		tensor.Shape{5, 1},
		[]float32{0.2592, 0.0000, 0.6909, 0.0498, 0.0000},
		[]float32{0.2094, -1.0000, 0.6411, 0.0000, -0.3909},
		[]float32{-0.07409999, -0, 0.3576, -0.2835, -0},
	},
	{
		tensor.Shape{2, 5},
		[]float32{0.2592, 0.0000, 0.6909, 0.0498, 0.0000, 0.2592, 0.0000, 0.6909, 0.0498, 0.0000},
		[]float32{0.2094, -1.0000, 0.6411, 0.0000, -0.3909, 0.2094, -1.0000, 0.6411, 0.0000, -0.3909},
		[]float32{-0.07409999, -0, 0.3576, -0.2835, -0, -0.07409999, -0, 0.3576, -0.2835, -0},
	},
}

func TestSparsemaxDo(t *testing.T) {
	c := require.New(t)

	for i, testCase := range testCasesSparseMaxDo {
		tt := tensor.New(tensor.Of(tensor.Float64), tensor.WithShape(testCase.size), tensor.WithBacking(testCase.input))
		op := newSparsemaxOp()

		out, err := op.Do(tt)
		c.NoError(err, "failed test case: %d", i)
		c.Equal(testCase.expected, out.Data(), "failed test case: %d", i)
	}
}

func TestSparsemaxDoDiff(t *testing.T) {
	c := require.New(t)

	for i, testCase := range testCasesSparseMaxDoDiff {
		g := NewGraph()
		a := NewTensor(g, Float64, 1, WithName("a"), WithShape(1))
		b := NewTensor(g, Float64, 1, WithName("b"), WithShape(1))

		op := newSparsemaxOp()
		r, err := ApplyOp(op, a)
		c.NoError(err)

		var backing interface{}

		switch testCase.input.(type) {
		case []float64:
			backing = make([]float64, testCase.shape.TotalSize())
		case []float32:
			backing = make([]float32, testCase.shape.TotalSize())
		}

		aT := tensor.New(tensor.WithShape(testCase.shape...), tensor.WithBacking(testCase.input))
		bT := tensor.New(tensor.WithShape(testCase.shape...), tensor.WithBacking(testCase.grad))
		rT := tensor.New(tensor.WithShape(testCase.shape...), tensor.WithBacking(backing))

		aVal, _, _, _ := anyToValue(aT)
		bVal, _, _, _ := anyToValue(bT)
		rVal, _, _, _ := anyToValue(rT)

		a.bind(dvUnit(aVal))
		b.bind(dvUnit(bVal))
		r.bind(dvUnitVar(rVal))

		err = op.DoDiff(ExecutionContext{}, Nodes{a, b}, r)
		c.NoError(err, "failed test case: %d", i)

		c.Equal(testCase.expected, r.boundTo.Data())
	}
}

func TestSparsemaxDoSymDiff(t *testing.T) {
	c := require.New(t)

	for i, testCase := range testCasesSparseMaxDoDiff {
		g := NewGraph()
		a := NewTensor(g, Float64, 1, WithName("a"), WithShape(1))
		b := NewTensor(g, Float64, 1, WithName("b"), WithShape(1))

		aT := tensor.New(tensor.WithShape(testCase.shape...), tensor.WithBacking(testCase.input))
		bT := tensor.New(tensor.WithShape(testCase.shape...), tensor.WithBacking(testCase.grad))

		aVal, _, _, _ := anyToValue(aT)
		bVal, _, _, _ := anyToValue(bT)

		a.bind(dvUnit(aVal))
		b.bind(dvUnit(bVal))

		op := newSparsemaxOp()
		diff, err := op.SymDiff(Nodes{a}, nil, b)
		c.NoError(err, "failed test case: %d", i)

		c.Len(diff, 1)

		vm := NewTapeMachine(g)

		c.NoError(vm.RunAll())
		c.NoError(vm.Close())

		c.Equal(testCase.expected, diff[0].boundTo.Data(), "failed test case: %d", i)
	}
}

func TestSparsemaxFull(t *testing.T) {
	c := require.New(t)

	for i, testCase := range testCasesSparseMaxDo {
		dtype := tensor.Float64

		if _, ok := testCase.input.([]float32); ok {
			dtype = tensor.Float32
		}

		tt := tensor.New(tensor.Of(dtype), tensor.WithShape(testCase.size), tensor.WithBacking(testCase.input))
		expected := tensor.New(tensor.Of(dtype), tensor.WithShape(testCase.size), tensor.WithBacking(testCase.expected))

		g := NewGraph()
		inp := NewTensor(g, dtype, 1, WithShape(testCase.size), WithName("inp"))
		out := Must(Sparsemax(inp))

		vm := NewTapeMachine(g)
		err := Let(inp, tt)
		c.NoError(err, "failed assigning input on case %d", i)

		c.NoError(vm.RunAll())
		c.NoError(vm.Close())

		c.Equal(expected.Data(), out.Value().(*tensor.Dense).Data(), "output is not equal to expected value for case %d", i)
	}
}
