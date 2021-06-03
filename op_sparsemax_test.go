package gorgonia

import (
	"testing"

	"github.com/stretchr/testify/require"
	"gorgonia.org/tensor"
)

var testCasesSparseMaxDo = []struct {
	size     tensor.Shape
	input    interface{}
	expected interface{}
	axis     int
}{
	{
		tensor.Shape{4}, []float64{0.3, 0.1, 1.2, 2.3}, []float64{0, 0, 0, 1.0}, -1,
	},
	{
		tensor.Shape{10}, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 1}, -1,
	},
	{
		tensor.Shape{3}, []float64{0.1, 0.1, 0.1}, []float64{0.3333333333333333, 0.3333333333333333, 0.3333333333333333}, -1,
	},
	{
		tensor.Shape{4}, []float64{-0.1, 0.3, -1.1, 2.7}, []float64{0, 0, 0, 1.0}, -1,
	},
	{
		tensor.Shape{4}, []float32{0.3, 0.1, 1.2, 2.3}, []float32{0, 0, 0, 1.0}, -1,
	},
	{
		tensor.Shape{10}, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, []float32{0, 0, 0, 0, 0, 0, 0, 0, 0, 1}, -1,
	},
	{
		tensor.Shape{3}, []float32{0.1, 0.1, 0.1}, []float32{0.33333334, 0.33333334, 0.33333334}, -1,
	},
	{
		tensor.Shape{4}, []float32{-0.1, 0.3, -1.1, 2.7}, []float32{0, 0, 0, 1.0}, -1,
	},
	{
		tensor.Shape{4}, []float64{0.9, 0.9, 0.9, 0.5}, []float64{0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.0000}, -1,
	},
	{
		tensor.Shape{6, 2},
		[]float64{-1.0000, -1.0000, 1.0000, 1.0000, -0.9998, -0.9998, 0.9998, 0.9998, 0.9945, 0.9945, -0.9945, -0.9945},
		[]float64{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5},
		-1,
	},
	// {
	// 	tensor.Shape{6, 2},
	// 	[]float64{-1.0, -1.0, 1.0, 1.0, -0.9998, -0.9998, 0.9998, 0.9998, 0.9945, 0.9945, -0.9945, -0.9945},
	// 	[]float64{0.0000, 0.0000, 0.3352, 0.3352, 0.0000, 0.0000, 0.3350, 0.3350, 0.3297, 0.3297, 0.0000, 0.0000},
	// 	0, // TODO
	// },
	{
		tensor.Shape{6, 2},
		[]float32{-1.0000, -1.0000, 1.0000, 1.0000, -0.9998, -0.9998, 0.9998, 0.9998, 0.9945, 0.9945, -0.9945, -0.9945},
		[]float32{0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000},
		-1,
	},
}

var testCasesSparseMaxDoDiff = []struct {
	shape tensor.Shape
	input interface{}
	grad  interface{}

	expected      interface{}
	expectedShape tensor.Shape
}{
	{
		tensor.Shape{5},
		[]float64{1.9968e-05, 1.9968e-05, 5.2120e-02, 2.3542e-01, 7.1242e-01},
		[]float64{0.2860, -0.0702, 0.8080, 0.9913, 1.4683},
		[]float64{-0.41068, -0.76688, 0.11132000000000009, 0.29462, 0.77162},
		tensor.Shape{5},
	},
	{
		tensor.Shape{5},
		[]float64{5.5620e-02, 2.0027e-05, 7.1182e-01, 2.3252e-01, 2.0027e-05},
		[]float64{0.1109, -1.4741, 0.7671, 0.2878, 0.0334},
		[]float64{0.16588, -1.41912, 0.82208, 0.34278, 0.08837999999999999},
		tensor.Shape{5},
	},
	{
		tensor.Shape{5},
		[]float64{0.0369, 0.3210, 0.0000, 0.3210, 0.3210},
		[]float64{0.2094, -1.0000, 0.6411, -0.5032, -0.3909},
		[]float64{0.630575, -0.5788249999999999, 0, -0.08202499999999996, 0.030274999999999996},
		tensor.Shape{5},
	},
	{
		tensor.Shape{5},
		[]float64{0.2592, 0.0000, 0.6909, 0.0498, 0.0000},
		[]float64{0.2094, -1.0000, 0.6411, 0.0000, -0.3909},
		[]float64{-0.07410000000000003, 0, 0.3576, -0.28350000000000003, 0},
		tensor.Shape{5},
	},
	{
		tensor.Shape{5},
		[]float32{0.0000, 0.0000, 0.0521, 0.2354, 0.7124},
		[]float32{0.2860, -0.0702, 0.8080, 0.9913, 1.4683},
		[]float32{-0, -0, -0.2812, -0.09790003, 0.37909997},
		tensor.Shape{5},
	},
	{
		tensor.Shape{5},
		[]float32{0.0556, 0.0000, 0.7118, 0.2325, 0.0000},
		[]float32{0.1109, -1.4741, 0.7671, 0.2878, 0.0334},
		[]float32{-0.2777, -0, 0.37849998, -0.10079998, -0},
		tensor.Shape{5},
	},
	{
		tensor.Shape{5},
		[]float32{0.2841, 0.0000, 0.7159, 0.0000, 0.0000},
		[]float32{0.2094, -1.0000, 0.6411, -0.5032, -0.3909},
		[]float32{-0.21585, -0, 0.21585, -0, -0},
		tensor.Shape{5},
	},
	{
		tensor.Shape{5},
		[]float32{0.2592, 0.0000, 0.6909, 0.0498, 0.0000},
		[]float32{0.2094, -1.0000, 0.6411, 0.0000, -0.3909},
		[]float32{-0.07409999, -0, 0.3576, -0.2835, -0},
		tensor.Shape{5},
	},
	{
		tensor.Shape{5, 1},
		[]float32{1, 1, 1, 1, 1},
		[]float32{0.2094, -1.0000, 0.6411, -0.5032, -0.3909},
		[]float32{1.253, 0.043599963, 1.6847, 0.54039997, 0.65269995, 1.253, 0.043599963, 1.6847, 0.54039997, 0.65269995, 1.253, 0.043599963, 1.6847, 0.54039997, 0.65269995, 1.253, 0.043599963, 1.6847, 0.54039997, 0.65269995, 1.253, 0.043599963, 1.6847, 0.54039997, 0.65269995},
		tensor.Shape{5, 5},
	},
}

func TestSparsemaxDo(t *testing.T) {
	c := require.New(t)

	for i, testCase := range testCasesSparseMaxDo {
		dtype := tensor.Float64

		switch testCase.input.(type) {
		case []float32:
			dtype = tensor.Float32
		}

		tt := tensor.New(tensor.Of(dtype), tensor.WithShape(testCase.size...), tensor.WithBacking(testCase.input))
		op := newSparsemaxOp(testCase.axis)

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
			backing = make([]float64, testCase.expectedShape.TotalSize())
		case []float32:
			backing = make([]float32, testCase.expectedShape.TotalSize())
		}

		aT := tensor.New(tensor.WithShape(testCase.shape...), tensor.WithBacking(testCase.input))
		bT := tensor.New(tensor.WithShape(testCase.shape.TotalSize()), tensor.WithBacking(testCase.grad))
		rT := tensor.New(tensor.WithShape(testCase.expectedShape...), tensor.WithBacking(backing))

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
		bT := tensor.New(tensor.WithShape(testCase.shape.TotalSize()), tensor.WithBacking(testCase.grad))

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

		tt := tensor.New(tensor.Of(dtype), tensor.WithShape(testCase.size...), tensor.WithBacking(testCase.input))
		expected := tensor.New(tensor.Of(dtype), tensor.WithShape(testCase.size...), tensor.WithBacking(testCase.expected))

		g := NewGraph()
		inp := NewTensor(g, dtype, testCase.size.Dims(), WithShape(testCase.size...), WithName("inp"))
		out := Must(Sparsemax(inp, testCase.axis))

		vm := NewTapeMachine(g)
		err := Let(inp, tt)
		c.NoError(err, "failed assigning input on case %d", i)

		c.NoError(vm.RunAll())
		c.NoError(vm.Close())

		c.Equal(expected.Data(), out.Value().(*tensor.Dense).Data(), "output is not equal to expected value for case %d", i)
	}
}
