package gorgonia

import (
	"log"
	"testing"

	"github.com/stretchr/testify/require"
	"gorgonia.org/tensor"
)

var testCasesSparseMaxDo = []struct {
	size     tensor.Shape
	input    interface{}
	weights  interface{}
	expected interface{}
	axis     int
}{
	{
		tensor.Shape{2, 3}, []float64{-2.1714, 0.0000, 0.0000, -0.4233, 0.0000, -1.2849}, []float64{0.3, 0.0, 1.0, 0.7, 0.0, 1.0}, []float64{0.17428999999999994, 0.8257099999999999, 1, 0}, -1,
	},
	{
		tensor.Shape{3, 3}, []float32{-3.1437, -0.5651, 0.0000, -0.7925, 0.0000, -0.5319, -0.0313, -1.1569, 0.0000}, []float32{0, 0.21744996, 0.78255, 0, 0.76594996, 0.23404998, 0.48434997, 0, 0.51565}, []float32{1, 0, 0, 0.45735168, 0.5426483, 0, 0.64763314, 0, 0.3523669}, -1,
	},
	{
		tensor.Shape{6, 2},
		[]float32{-1.0000, -1.0000, 1.0000, 1.0000, -0.9998, -0.9998, 0.9998, 0.9998, 0.9945, 0.9945, -0.9945, -0.9945},
		[]float32{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}, []float32{0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667},
		-1,
	},
}

func TestSparsemaxFull(t *testing.T) {
	c := require.New(t)

	for i, testCase := range testCasesSparseMaxDo {
		var err error

		dtype := tensor.Float64

		if _, ok := testCase.input.([]float32); ok {
			dtype = tensor.Float32
		}

		tt := tensor.New(tensor.Of(dtype), tensor.WithShape(testCase.size...), tensor.WithBacking(testCase.input))

		weightsT := tensor.New(tensor.Of(dtype), tensor.WithShape(testCase.size[1], testCase.size[0]), tensor.WithBacking(testCase.weights))

		expected := tensor.New(tensor.Of(dtype), tensor.WithShape(testCase.size[0], testCase.size[0]), tensor.WithBacking(testCase.expected))

		g := NewGraph()
		inp := NewTensor(g, dtype, testCase.size.Dims(), WithShape(testCase.size...), WithName("inp"))

		weights := NewTensor(g, dtype, 2, WithValue(weightsT), WithName("weights"))

		fc := Must(Mul(inp, weights))
		out := Must(Sparsemax(fc, testCase.axis))
		cost := Must(Mean(out))

		_, err = Grad(cost, weights, inp)
		c.NoError(err)

		vm := NewTapeMachine(g, BindDualValues(weights))
		err = Let(inp, tt)
		c.NoError(err, "failed assigning input on case %d", i)

		c.NoError(vm.RunAll())
		c.NoError(vm.Close())

		c.Equal(expected.Data(), out.Value().(*tensor.Dense).Data(), "output is not equal to expected value for case %d", i)

		outGrad, _ := out.Grad()
		log.Printf("output grad: %v", outGrad)

		inpGrad, _ := inp.Grad()
		log.Printf("input grad: %v", inpGrad)
	}
}
