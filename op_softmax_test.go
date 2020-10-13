package gorgonia

import (
	"fmt"
	"runtime"
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

var testCasesSoftMax = []struct {
	input            []float64
	inputShape       tensor.Shape
	axis             []int
	expectedGradient []float64
}{
	{
		[]float64{0.1, 0.2, -0.3, 0.4, 0.5},
		tensor.Shape{5},
		[]int{},
		[]float64{0.178025447751409, 0.1967485475322529, 0.11933402633223977, 0.24030921861990098, 0.2655827597641975},
	},
	{
		[]float64{0.1, 0.2, -0.3, 0.4, 0.5},
		tensor.Shape{5},
		[]int{0},
		[]float64{0.178025447751409, 0.1967485475322529, 0.11933402633223977, 0.24030921861990098, 0.2655827597641975},
	},
	{
		[]float64{0.1, 0.2, -0.3, 0.4, 0.5, -0.6},
		tensor.Shape{3, 2},
		[]int{1},
		[]float64{0, 0, 0, 0, 0.7502601055951175, 0.24973989440488234},
	},
}

func TestSoftMax(t *testing.T) {
	defer runtime.GC()

	for _, tc := range testCasesSoftMax {
		t.Run(fmt.Sprintf("SoftMax(%v,%v)", tc.input, tc.axis), func(t *testing.T) {
			c := assert.New(t)

			g := NewGraph()
			xT := tensor.New(tensor.WithShape(tc.inputShape...), tensor.WithBacking(tc.input))
			x := NewTensor(g, Float64, xT.Dims(), WithValue(xT))
			sm := Must(SoftMax(x, tc.axis...))
			logsm := Must(Neg(Must(Log(sm))))
			cost := Must(Slice(logsm, S(2)))

			if cost.Dims() > 0 {
				cost = Must(Slice(cost, S(0)))
			}

			_, err := Grad(cost, x)
			c.NoError(err)

			m := NewTapeMachine(g, TraceExec())
			defer m.Close()

			err = m.RunAll()
			c.NoError(err)

			xG, err := x.Grad()
			c.NoError(err)
			c.NotNil(xG)

			// machine 2, graph 2
			h := NewGraph()
			xT2 := tensor.New(tensor.WithShape(tc.inputShape...), tensor.WithBacking(tc.input))
			x2 := NewTensor(h, Float64, xT2.Dims(), WithValue(xT2))
			sm2 := Must(SoftMax(x2, tc.axis...))
			logsm2 := Must(Neg(Must(Log(sm2))))

			cost2 := Must(Slice(logsm2, S(2)))
			if cost2.Dims() > 0 {
				cost2 = Must(Slice(cost2, S(0)))
			}

			m2 := NewLispMachine(h)
			defer m2.Close()

			err = m2.RunAll()
			c.NoError(err)

			x2G, err := x2.Grad()
			c.NoError(err)

			c.Equal(tc.expectedGradient, xG.Data())
			c.Equal(tc.expectedGradient, x2G.Data())
		})
	}
}

func Benchmark(b *testing.B) {
	c := require.New(b)

	tt := tensor.New(tensor.Of(tensor.Float64), tensor.WithShape(5), tensor.WithBacking([]float64{0.1, 0.1, -0.1, 0.3, 0.2}))

	for i := 0; i < b.N; i++ {
		op := newSoftmaxOp(tt.Shape())
		_, err := op.Do(tt)

		c.NoError(err)
	}
}
