package gorgonia

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gorgonia.org/tensor"
)

var testCasesSoftMaxDo = []struct {
	input    []float64
	expected []float64
}{
	{
		[]float64{0.2094, -1.0, 0.6411, 0.0, -0.3909}, []float64{0.2382105379413429, 0.07107636737487558, 0.36681399568548617, 0.19320559786800362, 0.13069350113029174},
	},
	{
		[]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, []float64{7.801341612780742e-05, 0.00021206245143623275, 0.0005764455082375902, 0.0015669413501390804, 0.004259388198344144, 0.0115782175399118, 0.031472858344688034, 0.08555209892803112, 0.23255471590259755, 0.6321492583604866},
	},
	{
		[]float64{0.1, 0.1, 0.1}, []float64{0.3333333333333333, 0.3333333333333333, 0.3333333333333333},
	},
	{
		[]float64{-0.1, 0.3, -1.1, 2.7}, []float64{0.05180179352659075, 0.07727919496508177, 0.019056814854240642, 0.8518621966540868},
	},
}

func TestSoftmaxDo(t *testing.T) {
	c := require.New(t)

	for i, testCase := range testCasesSoftMaxDo {
		tt := tensor.New(tensor.Of(tensor.Float64), tensor.WithShape(len(testCase.input)), tensor.WithBacking(testCase.input))
		op := newSoftmaxOp(tt.Shape())

		out, err := op.Do(tt)
		c.NoError(err, "failed test case: %d", i)
		c.Equal(testCase.expected, out.Data(), "failed test case: %d", i)
	}
}

func TestSoftmaxKernel(t *testing.T) {
	// this test is used for migrating to a new algorithm for softmax
	assert := assert.New(t)
	a := tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{-0.1, 0.3, -1.1, 2.7, 3.14, 0.1}))
	op := newSoftmaxOp(a.Shape())
	op.axis = 0
	b0, _ := op.Do(a)
	op.axis = 1
	b1, _ := op.Do(a)

	// across axis 0
	out := make([]float64, 6)
	op.doF64s(tensor.Shape{2, 3}, 0, a.Data().([]float64), out)
	assert.True(floatsEqual64(out, b0.Data().([]float64)))
	t.Logf("\n%v\n%v", out, b0.Data())

	// acros axis 1
	out = make([]float64, 6)
	op.doF64s(tensor.Shape{2, 3}, 1, a.Data().([]float64), out)
	assert.True(floatsEqual64(out, b1.Data().([]float64)))
	/*
		// super large
		a = tensor.New(tensor.WithShape(10, 1024, 2048, 30), tensor.WithBacking(Uniform64(-1, 1, 10, 1024, 2048, 30)))
		op = newSoftmaxOp(a.Shape())
		op.axis = 0
		b, _ := op.Do(a)

		out = make([]float64, 10*1024*2048*30)
		op.doF64s(tensor.Shape{10, 1024, 2048, 30}, 0, a.Data().([]float64), out)
		assert.True(floatsEqual64(out, b.Data().([]float64)))
	*/
}

func BenchmarkSoftmaxLargeOldAxis0(b *testing.B) {
	b.StopTimer()
	a := tensor.New(tensor.WithShape(10, 1024, 2048, 30), tensor.WithBacking(Uniform64(-1, 1, 10, 1024, 2048, 30)))
	op := newSoftmaxOp(a.Shape())
	op.axis = 0
	var v Value

	b.ResetTimer()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		v, _ = op.Do(a)
	}
	_ = v
}

func BenchmarkSoftmaxLargeNewAxis0(b *testing.B) {
	b.StopTimer()
	a := tensor.New(tensor.WithShape(10, 1024, 2048, 30), tensor.WithBacking(Uniform64(-1, 1, 10, 1024, 2048, 30)))
	op := newSoftmaxOp(a.Shape())
	op.axis = 0
	out := make([]float64, len(a.Data().([]float64)))

	b.ResetTimer()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		op.doF64s(a.Shape(), 0, a.Data().([]float64), out)
	}

}
