package gorgonia

import (
	"fmt"
	"io/ioutil"
	"log"
	"testing"

	"github.com/pkg/errors"
	"github.com/stretchr/testify/assert"
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

func TestSoftMaxFull(t *testing.T) {
	testCases := []struct {
		Dtype    tensor.Dtype
		XInit    InitWFn
		XShape   tensor.Shape
		Expected tensor.Tensor
	}{
		{
			Dtype:  tensor.Float64,
			XInit:  RangedFromWithStep(0.0, 0.01),
			XShape: tensor.Shape{2, 3},
		},
	}
	for i, tC := range testCases {
		t.Run(fmt.Sprintf("#%d %v", i+1, tC.XShape), func(t *testing.T) {
			c := assert.New(t)

			g := NewGraph()

			x := NewTensor(g, tC.Dtype, 2, WithShape(tC.XShape...), WithInit(tC.XInit), WithName("x"))
			w := NewTensor(g, tC.Dtype, 2, WithShape(tC.XShape...), WithInit(RangedFromWithStep(-0.05, 0.03)), WithName("w"))

			optim := NewAdamSolver(WithLearnRate(0.1))

			wT := Must(Transpose(w, 1, 0))

			log.Printf("wT: %v", wT.Shape())

			output := Must(Mul(x, wT))

			var fcVal Value
			Read(output, &fcVal)

			output = Must(SoftMax(output))

			cost := Must(Mean(output))

			_, err := Grad(cost, x, w)
			c.NoError(err)

			vm := NewTapeMachine(g, BindDualValues(w))
			c.NoError(vm.RunAll())

			log.Printf("dx: %v", x.Deriv().Value())

			c.NoError(optim.Step(NodesToValueGrads(Nodes{w})))

			log.Printf("output: %v", output.Value())
			log.Printf("FC Val: %v", fcVal)
			log.Printf("cost: %v", cost.Value())
			log.Printf("w: %v", w.Value())
		})
	}
}

func TestSoftmaxDo(t *testing.T) {
	assert := assert.New(t)

	for i, testCase := range testCasesSoftMaxDo {
		tt := tensor.New(tensor.Of(tensor.Float64), tensor.WithShape(len(testCase.input)), tensor.WithBacking(testCase.input))
		op := newSoftmaxOp(tt.Shape())

		out, err := op.Do(tt)

		log.Printf("out: %v", out)

		assert.NoError(err, "failed test case: %d", i)
		assert.InDeltaSlice(testCase.expected, out.Data().([]float64), 1e-7)
	}
}

// func TestSoftmaxKernel(t *testing.T) {
// 	// this test is used for migrating to a new algorithm for softmax
// 	assert := assert.New(t)
// 	a := tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{-0.1, 0.3, -1.1, 2.7, 3.14, 0.1}))
// 	op := newSoftmaxOp(a.Shape())
// 	op.axis = 0
// 	b0, _ := op.Do(a)
// 	op.axis = 1
// 	b1, _ := op.Do(a)

// 	// across axis 0
// 	out := make([]float64, 6)
// 	op.do(tensor.Shape{2, 3}, 0, a.Data().([]float64), out)
// 	assert.True(floatsEqual64(out, b0.Data().([]float64)))
// 	t.Logf("\n%v\n%v", out, b0.Data())

// 	// acros axis 1
// 	out = make([]float64, 6)
// 	op.do(tensor.Shape{2, 3}, 1, a.Data().([]float64), out)
// 	assert.True(floatsEqual64(out, b1.Data().([]float64)))
// 	/*
// 		// super large
// 		a = tensor.New(tensor.WithShape(10, 1024, 2048, 30), tensor.WithBacking(Uniform64(-1, 1, 10, 1024, 2048, 30)))
// 		op = newSoftmaxOp(a.Shape())
// 		op.axis = 0
// 		b, _ := op.Do(a)

// 		out = make([]float64, 10*1024*2048*30)
// 		op.doF64s(tensor.Shape{10, 1024, 2048, 30}, 0, a.Data().([]float64), out)
// 		assert.True(floatsEqual64(out, b.Data().([]float64)))
// 	*/
// }

func oldsoftmax(a *Node, axes ...int) (retVal *Node, err error) {
	aShape := a.Shape()
	axis := aShape.Dims() - 1 // default: last dim
	if a.IsColVec() || (a.IsVector() && !a.IsRowVec()) {
		axis = 0
	}

	if len(axes) > 0 {
		if axes[0] >= axis+1 || axes[0] < 0 {
			return nil, errors.Errorf("Cannot perform SoftMax on axis %d. Input has shape %v", axes[0], a.Shape())
		}
		axis = axes[0]
	}

	var exp, sum *Node
	if exp, err = Exp(a); err != nil {
		return nil, errors.Wrap(err, operationError)
	}
	if sum, err = Sum(exp, axis); err != nil {
		return nil, errors.Wrap(err, operationError)
	}

	if sum.IsScalar() {
		return HadamardDiv(exp, sum)
	}

	// reshape if necessary
	ss := sum.Shape()
	diff := exp.Shape().Dims() - ss.Dims()

	// TODO: multirank softmax
	if diff > 0 {
		newShape := tensor.Shape(tensor.BorrowInts(ss.Dims() + diff))
		copy(newShape, ss)
		copy(newShape[axis+1:], newShape[axis:])
		newShape[axis] = 1

		if sum, err = Reshape(sum, newShape); err != nil {
			return nil, errors.Wrap(err, "Failed to reshape")
		}
	}

	return BroadcastHadamardDiv(exp, sum, nil, []byte{byte(axis)})
}

func TestOld_NewSoftmax(t *testing.T) {
	a := tensor.New(tensor.WithBacking([]float64{0.1, 0.1, 0.3, 0.1, 0.4}))

	g := NewGraph()
	A := NodeFromAny(g, a, WithName("A"))
	sm := Must(SoftMax(A))
	sum := Must(Sum(sm))
	if _, err := Grad(sum, A); err != nil {
		t.Fatal(err)
	}

	h := NewGraph()
	A2 := NodeFromAny(h, a, WithName("A"))
	sm2 := Must(oldsoftmax(A2))
	sum2 := Must(Sum(sm2))
	if _, err := Grad(sum2, A2); err != nil {
		t.Fatal(err)
	}

	m1 := NewTapeMachine(g, TraceExec(), BindDualValues())
	if err := m1.RunAll(); err != nil {
		t.Fatalf("m1 %v", err)
	}

	m2 := NewTapeMachine(h, TraceExec(), BindDualValues())
	if err := m2.RunAll(); err != nil {
		t.Fatalf("m2 %v", err)
	}

	Agrad, err := A.Grad()
	if err != nil {
		t.Fatalf("No grad for A %v", err)
	}

	A2grad, err := A2.Grad()
	if err != nil {
		t.Fatalf("No grad for A2 %v", err)
	}

	t.Logf("\n%v\n%v", sm.Value(), sm2.Value())
	t.Logf("\n%v\n%v", Agrad, A2grad)

	ioutil.WriteFile("oldsm.dot", []byte(h.ToDot()), 0644)
	ioutil.WriteFile("newsm.dot", []byte(g.ToDot()), 0644)

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

// func BenchmarkSoftmaxLargeNewAxis0(b *testing.B) {
// 	b.StopTimer()
// 	a := tensor.New(tensor.WithShape(10, 1024, 2048, 30), tensor.WithBacking(Uniform64(-1, 1, 10, 1024, 2048, 30)))
// 	op := newSoftmaxOp(a.Shape())
// 	op.axis = 0
// 	out := make([]float64, len(a.Data().([]float64)))

// 	b.ResetTimer()
// 	b.StartTimer()
// 	for i := 0; i < b.N; i++ {
// 		op.do(a.Shape(), 0, a.Data().([]float64), out)
// 	}

// }

// func BenchmarkSoftmaxMedOldAxis0(b *testing.B) {
// 	b.StopTimer()
// 	a := tensor.New(tensor.WithShape(1200, 2500), tensor.WithBacking(Uniform64(-1, 1, 1200, 2500)))
// 	op := newSoftmaxOp(a.Shape())
// 	op.axis = 0
// 	var v Value

// 	b.ResetTimer()
// 	b.StartTimer()
// 	for i := 0; i < b.N; i++ {
// 		v, _ = op.Do(a)
// 	}
// 	_ = v
// }

// func BenchmarkSoftmaxMedNewAxis0(b *testing.B) {
// 	b.StopTimer()
// 	a := tensor.New(tensor.WithShape(1200, 2500), tensor.WithBacking(Uniform64(-1, 1, 1200, 2500)))
// 	op := newSoftmaxOp(a.Shape())
// 	op.axis = 0
// 	out := make([]float64, len(a.Data().([]float64)))

// 	b.ResetTimer()
// 	b.StartTimer()
// 	for i := 0; i < b.N; i++ {
// 		op.do(a.Shape(), 0, a.Data().([]float64), out)
// 	}

// }
