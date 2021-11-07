package gorgonia

import (
	"fmt"
	"io/ioutil"
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gorgonia.org/tensor"
)

func TestApplyOp(t *testing.T) {
	assert := assert.New(t)
	g := NewGraph()

	var cpi *Node
	var ct *Node
	var op Op

	t.Log("Simple Constant Scalar test")
	cpi = NewConstant(3.1415, WithName("constantPi"))
	cpi = g.AddNode(cpi)

	t.Logf("g: %v", cpi.g)

	op = newElemBinOp(addOpType, cpi, cpi)
	added, err := ApplyOpWithName(op, "+ pi pi", cpi, cpi)
	if err != nil {
		t.Fatal(err)
	}
	assert.Equal(g, added.g)
	assert.Equal(Float64, added.t)

	ct = NewConstant(tensor.Ones(tensor.Float64, 3, 3)) // no graph set for ct
	op = newElemBinOp(addOpType, cpi, ct)
	if added, err = ApplyOpWithName(op, "+ pi constTensor(3,3)_ones", cpi, ct); err != nil {
		t.Error(err)
	}
}

var mulTests = []struct {
	name   string
	xshape tensor.Shape
	wshape tensor.Shape

	gradX []float64
	gradW []float64
}{
	{"x vector", tensor.Shape{2}, tensor.Shape{2, 3}, []float64{3, 12}, []float64{0, 0, 0, 1, 1, 1}},
	{"x mat", tensor.Shape{3, 2}, tensor.Shape{2, 3}, []float64{3, 12, 3, 12, 3, 12}, []float64{6, 6, 6, 9, 9, 9}},
	{"x_vec_w_vec", tensor.Shape{6}, tensor.Shape{6}, []float64{0, 1, 2, 3, 4, 5}, []float64{0, 1, 2, 3, 4, 5}},
}

func TestMul(t *testing.T) {
	defer runtime.GC()
	assert := assert.New(t)
	for _, mts := range mulTests {
		g := NewGraph()
		x := NewTensor(g, Float64, mts.xshape.Dims(), WithName(mts.name), WithShape(mts.xshape...), WithInit(RangedFrom(0)))
		w := NewTensor(g, Float64, mts.wshape.Dims(), WithName("w"), WithShape(mts.wshape...), WithInit(RangedFrom(0)))

		xw, err := Mul(x, w)
		if err != nil {
			t.Errorf("Error when testing %q. Err: %v", mts.name, err)
			continue
		}

		if mts.xshape.IsVector() && mts.wshape.IsVector() {
			if _, err = Grad(xw, x, w); err != nil {
				t.Errorf("Error while differentiating %q, Err: %v", mts.name, err)
				continue
			}
		} else {
			cost, err := Sum(xw)
			if err != nil {
				t.Errorf("Error when summing %q. Err: %v", mts.name, err)
				continue
			}

			if _, err = Grad(cost, x, w); err != nil {
				t.Errorf("Error while differentiating %q, Err: %v", mts.name, err)
				continue
			}
		}

		m := NewTapeMachine(g)
		if err = m.RunAll(); err != nil {
			t.Errorf("Error while executing %q. Err: %v", mts.name, err)
			continue
		}

		gradX, err := x.Grad()
		if err != nil {
			t.Errorf("Error while getting gradient of x %q. Err: %v", mts.name, err)
		}

		gradW, err := w.Grad()
		if err != nil {
			t.Errorf("Error while getting gradient of w %q. Err: %v", mts.name, err)
		}

		assert.Equal(mts.gradX, gradX.Data().([]float64))
		assert.Equal(mts.gradW, gradW.Data().([]float64))
		assert.True(mts.xshape.Eq(gradX.Shape()))
		assert.True(mts.wshape.Eq(gradW.Shape()))
		m.Close()
	}

	t.Logf("Testing Mul with LispMachine")
	for _, mts := range mulTests {
		g := NewGraph()
		x := NewTensor(g, Float64, mts.xshape.Dims(), WithName(mts.name), WithShape(mts.xshape...), WithInit(RangedFrom(0)))
		w := NewTensor(g, Float64, mts.wshape.Dims(), WithName("w"), WithShape(mts.wshape...), WithInit(RangedFrom(0)))

		xw, err := Mul(x, w)
		if err != nil {
			t.Errorf("Error when testing %q. Err: %v", mts.name, err)
			continue
		}

		if mts.xshape.IsVector() && mts.wshape.IsVector() {

		} else {
			if _, err = Sum(xw); err != nil {
				t.Errorf("Error when summing %q. Err: %v", mts.name, err)
				continue
			}
		}

		m := NewLispMachine(g)

		if err = m.RunAll(); err != nil {
			// ioutil.WriteFile(fmt.Sprintf("fullGraph_%v.dot", mts.name), []byte(g.ToDot()), 0644)
			t.Errorf("Error while executing %q. Err: %v", mts.name, err)
			continue
		}

		gradX, err := x.Grad()
		if err != nil {
			t.Errorf("Error while getting gradient of x %q. Err: %v", mts.name, err)
		}

		gradW, err := w.Grad()
		if err != nil {
			t.Errorf("Error while getting gradient of w %q. Err: %v", mts.name, err)
		}

		assert.Equal(mts.gradX, gradX.Data().([]float64))
		assert.Equal(mts.gradW, gradW.Data().([]float64))
		assert.True(mts.xshape.Eq(gradX.Shape()))
		assert.True(mts.wshape.Eq(gradW.Shape()))
		m.Close()
	}
}

var gtTests = []struct {
	a, b    Value
	retSame bool

	expected Value
	err      bool
}{
	// s-s
	{NewF64(float64(1)), NewF64(float64(0)), true, NewF64(1.0), false},
	{NewF64(float64(0)), NewF64(float64(1)), true, NewF64(0.0), false},
	{NewF64(float64(1)), NewF64(float64(0)), false, NewB(true), false},
	{NewF32(float32(0)), NewF32(float32(1)), false, NewB(false), false},

	// s-t
	{
		NewF64(float64(1)), tensor.New(tensor.WithShape(2), tensor.WithBacking([]float64{0, 2})),
		true,
		tensor.New(tensor.WithShape(2), tensor.WithBacking([]float64{1, 0})),
		false,
	},

	{
		NewF32(float32(1)), tensor.New(tensor.WithShape(2), tensor.WithBacking([]float32{0, 2})),
		false,
		tensor.New(tensor.WithShape(2), tensor.WithBacking([]bool{true, false})),
		false,
	},

	// t-s
	{
		tensor.New(tensor.WithShape(2), tensor.WithBacking([]float64{0, 2})), NewF64(float64(1)),
		true,
		tensor.New(tensor.WithShape(2), tensor.WithBacking([]float64{0, 1})),
		false,
	},

	{
		tensor.New(tensor.WithShape(2), tensor.WithBacking([]float32{0, 2})), NewF32(float32(1)),
		false,
		tensor.New(tensor.WithShape(2), tensor.WithBacking([]bool{false, true})),
		false,
	},

	// t-t
	{
		tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{0, 1, 2, 3, 4, 5})),
		tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{5, 4, 3, 2, 1, 0})),
		true,

		tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{0, 0, 0, 1, 1, 1})),
		false,
	},

	{
		tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{0, 1, 2, 3, 4, 5})),
		tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{5, 4, 3, 2, 1, 0})),
		false,

		tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]bool{false, false, false, true, true, true})),
		false,
	},

	// stupids

	// different shapes
	{
		tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(2)), tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(4)),
		true, nil, true,
	},

	// different dtypes
	{
		tensor.New(tensor.Of(tensor.Float64), tensor.WithShape(2)), tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(2)),
		true, nil, true,
	},
}

func TestGt(t *testing.T) {
	defer runtime.GC()
	for i, gtts := range gtTests {
		// if i != 11 {
		// 	continue
		// }
		g := NewGraph()
		a := NodeFromAny(g, gtts.a, WithName("a"))
		b := NodeFromAny(g, gtts.b, WithName("b"))

		var ret *Node
		var err error
		ret, err = Gt(a, b, gtts.retSame)

		switch {
		case gtts.err:
			if err == nil {
				t.Errorf("Expected an error in Test %d", i)
			}
			continue
		case !gtts.err && err != nil:
			t.Errorf("Test %d: %+v", i, err)
			continue
		}

		if gtts.retSame {
			cost := Must(Sum(ret))
			Grad(cost, a, b)
		}

		m1 := NewTapeMachine(g)
		if err = m1.RunAll(); err != nil {
			ioutil.WriteFile("fail.dot", []byte(g.ToDot()), 0644)
			t.Errorf("%v", m1.Prog())
			t.Errorf("Test %d: %+v", i, err)
			continue
		}

		if !ValueEq(gtts.expected, ret.Value()) {
			t.Errorf("Test %d Expected %v. Got %v", i, gtts.expected, ret.Value())
		}

		// Test LispMachine implementation
		h := NewGraph()
		x := NodeFromAny(h, gtts.a, WithName("x"))
		y := NodeFromAny(h, gtts.b, WithName("y"))
		ret2, _ := Gt(x, y, gtts.retSame)

		var m2 VM
		if gtts.retSame {
			Must(Sum(ret2))
			m2 = NewLispMachine(h)
		} else {
			m2 = NewLispMachine(h, ExecuteFwdOnly())
		}
		if err = m2.RunAll(); err != nil {
			t.Errorf("Test %d LispMachine: %+v", i, err)
			continue
		}

		if !ValueEq(ret.Value(), ret2.Value()) {
			t.Errorf("Test %d. Expected %v. Got  %v", i, ret.Value(), ret2.Value())
		}
		m1.Close()
		m2.Close()
		runtime.GC()
	}

	// other special cases
	g := NewGraph()
	c := NewConstant(F64(1))
	// T := NewTensor(g, Float64, 1, WithShape(2), WithInit(RangedFrom(0)))
	T := UniformRandomNode(g, Float64, 0, 1, 2)

	var gt *Node
	var err error
	if gt, err = Gt(c, T, true); err != nil {
		t.Error(err)
	}
	cost := Must(Sum(gt))
	Grad(cost, T)

	m1 := NewTapeMachine(g)
	defer m1.Close()
	if err = m1.RunAll(); err != nil {
		t.Error(err)
	}

	if (TensorType{Dims: 1, Of: Float64}) != TypeOf(gt.Value()) {
		t.Error("Expected a tensor type of float64")
	}

	// Same test as above, but using *lispMachine

	h := NewGraph()
	d := NewConstant(F64(1))
	U := UniformRandomNode(h, Float64, 0, 1, 2)
	var gt2 *Node
	if gt2, err = Gt(d, U, true); err != nil {
		t.Error(err)
	}
	Must(Sum(gt2))

	m2 := NewLispMachine(h)
	defer m2.Close()
	if err = m2.RunAll(); err != nil {
		t.Error(err)
	}

	if (TensorType{Dims: 1, Of: Float64}) != TypeOf(gt2.Value()) {
		t.Error("Expected a tensor type of float64")
	}

	t.Logf("%v", gt2.Value())
	runtime.GC()

}

func TestMisha(t *testing.T) {
	defer runtime.GC()
	assert := assert.New(t)
	g := NewGraph()
	var err error
	var x0, x1, x2, f0, f1, f2 *Node
	var grad0, grad1, grad2 Nodes

	x0 = NewScalar(g, Float64, WithName("x0"))
	x1 = NewScalar(g, Float64, WithName("x1"))
	x2 = NewScalar(g, Float64, WithName("x2"))

	Let(x0, -2.5)
	Let(x1, -2.2)
	Let(x2, 1.0)

	f0 = Must(Mish(x0))
	f1 = Must(Mish(x1))
	f2 = Must(Mish(x2))

	if grad0, err = Grad(f0, x0); err != nil {
		t.Error(err)
	}
	if grad1, err = Grad(f1, x1); err != nil {
		t.Error(err)
	}
	if grad2, err = Grad(f2, x2); err != nil {
		t.Error(err)
	}

	machine := NewTapeMachine(g)
	defer machine.Close()
	if err = machine.RunAll(); err != nil {
		t.Error(err)
	}

	// assert non-monotonicity of Mish
	// x0 < x1 < x2 && f0 > f1 < f2
	assert.Less(extractF64(x0.Value()), extractF64(x1.Value()))
	assert.Less(extractF64(x1.Value()), extractF64(x2.Value()))
	assert.Greater(extractF64(f0.Value()), extractF64(f1.Value()))
	assert.Less(extractF64(f1.Value()), extractF64(f2.Value()))

	// assert non-monotonocity of Mish'
	assert.Greater(extractF64(grad0[0].Value()), extractF64(grad1[0].Value()))
	assert.Less(extractF64(grad1[0].Value()), extractF64(grad2[0].Value()))
}

func TestSoftMax(t *testing.T) {
	defer runtime.GC()
	g := NewGraph()
	xT := tensor.New(tensor.WithBacking([]float64{0.1, 0.2, -0.3, 0.4, 0.5}))
	x := NewVector(g, Float64, WithShape(5), WithValue(xT))
	sm := Must(SoftMax(x))
	logsm := Must(Neg(Must(Log(sm))))
	cost := Must(Slice(logsm, S(2)))

	if _, err := Grad(cost, x); err != nil {
		t.Error(err)
	}

	m := NewTapeMachine(g, TraceExec())
	defer m.Close()
	if err := m.RunAll(); err != nil {
		t.Error(err)
	}
	ioutil.WriteFile("fullGraph.dot", []byte(g.ToDot()), 0644)
	var xG Value
	var err error
	if xG, err = x.Grad(); err != nil {
		t.Error(err)
	}

	// machine 2, graph 2
	h := NewGraph()
	xT2 := tensor.New(tensor.WithBacking([]float64{0.1, 0.2, -0.3, 0.4, 0.5}))
	x2 := NewVector(h, Float64, WithShape(5), WithValue(xT2))
	sm2 := Must(SoftMax(x2))
	logsm2 := Must(Neg(Must(Log(sm2))))
	Must(Slice(logsm2, S(2)))

	m2 := NewLispMachine(h)
	defer m2.Close()
	if err = m2.RunAll(); err != nil {
		t.Error(err)
	}

	var x2G Value
	if x2G, err = x2.Grad(); err != nil {
		t.Error(err)
	}

	if !floatsEqual64(xG.Data().([]float64), x2G.Data().([]float64)) {
		t.Errorf("Expected both gradients of X to be the same.")
	}
	t.Logf("\n%v\n%v\n%v", sm.Value(), logsm.Value(), cost.Value())
	correctXGrad := []float64{
		0.178025447751409, 0.1967485475322529, -0.8806659736677602, 0.24030921861990098, 0.2655827597641975,
	}

	if !floatsEqual64(correctXGrad, x2G.Data().([]float64)) {
		t.Errorf("Expected results to be %v. Got %v.", correctXGrad, x2G.Data())
	}
	if !floatsEqual64(correctXGrad, xG.Data().([]float64)) {
		t.Errorf("Expected results to be %v. Got %v.", correctXGrad, xG.Data())
	}
}

var sliceTests = []struct {
	name   string
	shape  tensor.Shape
	slices []tensor.Slice

	expected tensor.Shape
	data     interface{}
	err      bool
}{
	{"vec[0]", tensor.Shape{2}, []tensor.Slice{S(0)}, scalarShape, float64(0), false},
	{"vec[0:2]", tensor.Shape{2}, []tensor.Slice{S(0, 2)}, tensor.Shape{2}, []float64{0, 1}, false},
	{"Mat[0]", tensor.Shape{2, 3}, []tensor.Slice{S(0)}, tensor.Shape{3}, []float64{0, 1, 2}, false},
	{"Mat[:, 0]", tensor.Shape{2, 3}, []tensor.Slice{nil, S(0)}, tensor.Shape{2}, []float64{0, 3}, false},
	{"3Tensor[0]", tensor.Shape{2, 3, 4}, []tensor.Slice{S(0)}, tensor.Shape{3, 4}, tensor.Range(tensor.Float64, 0, 12), false},
	{"3Tensor[0:2]", tensor.Shape{2, 3, 4}, []tensor.Slice{S(0, 2)}, tensor.Shape{2, 3, 4}, tensor.Range(tensor.Float64, 0, 24), false},
	{"3Tensor[:, 0]", tensor.Shape{2, 3, 4}, []tensor.Slice{nil, S(0)}, tensor.Shape{2, 4}, []float64{0, 1, 2, 3, 12, 13, 14, 15}, false},
	{"3Tensor[0, :, 0]", tensor.Shape{2, 3, 4}, []tensor.Slice{S(0), nil, S(0)}, tensor.Shape{3}, []float64{0, 4, 8}, false},

	{"vec[:, 0]", tensor.Shape{2}, []tensor.Slice{nil, S(0)}, nil, nil, true},
}

func TestSlice(t *testing.T) {
	defer runtime.GC()
	for _, sts := range sliceTests {
		g := NewGraph()
		x := NewTensor(g, Float64, len(sts.shape), WithShape(sts.shape...), WithInit(RangedFrom(0)))
		sliced, err := Slice(x, sts.slices...)
		switch {
		case sts.err:
			if err == nil {
				t.Errorf("Expected an error while running test %q", sts.name)
			}
			continue
		case !sts.err && err != nil:
			t.Errorf("Error in %q: %+v", sts.name, err)
			continue
		}

		// test expected shapes:
		if !sts.expected.Eq(sliced.shape) {
			t.Errorf("Test %q - Expected %v. Got %v instead", sts.name, sts.expected, sliced.shape)
			continue
		}

		// test forwards and backwards prop
		cost := Must(Sum(sliced))
		if _, err := Grad(cost, x); err != nil {
			t.Errorf("Test %q failed to backprop: %+v", sts.name, err)
			continue
		}

		m1 := NewTapeMachine(g)
		if err = m1.RunAll(); err != nil {
			t.Errorf("Test %q Runtime error %+v ", sts.name, err)
			continue
		}

		sV := sliced.Value()
		if !sts.expected.Eq(sV.Shape()) {
			t.Errorf("Test %q For TapeMachine. Expected sliced value to have the shape %v. Got %v instead", sts.name, sts.expected, sV.Shape())
		}

		assert.Equal(t, sts.data, sV.Data(), "Test %q For TapeMachine data expected %v, Got %v instead. Formatted:\n %+v", sts.name, sts.data, sV.Data(), sV)

		// Test Lisp Machine for equivalence of gradients

		h := NewGraph()
		a := NewTensor(g, Float64, len(sts.shape), WithShape(sts.shape...), WithInit(RangedFrom(0)))
		sliced2 := Must(Slice(a, sts.slices...))
		Must(Sum(sliced2))

		m2 := NewLispMachine(h)
		if err = m2.RunAll(); err != nil {
			t.Errorf("Test %q Lispmachine Runtime error: %+v", sts.name, err)
			continue
		}

		s2V := sliced2.Value()
		if !sts.expected.Eq(s2V.Shape()) {
			t.Errorf("Test %q For LispMachine. Expected sliced value to have the shape %v. Got %v instead", sts.name, sts.expected, s2V.Shape())
		}

		assert.Equal(t, sts.data, s2V.Data(), "Test %q For TapeMachine data expected %v, Got %v instead. Formatted:\n %+v", sts.name, sts.data, s2V.Data(), s2V)

		sG, err := sliced.Grad()
		if err != nil {
			t.Errorf("Test %q sliced has no grad: %+v", sts.name, err)
			continue
		}

		s2G, err := sliced2.Grad()
		if err != nil {
			t.Errorf("Test %q sliced2 has no grad: %+v", sts.name, err)
			continue
		}

		if !ValueEq(sG, s2G) {
			t.Errorf("Test %q - Expected sG and s2G to have the same value", sts.name)
		}

		m1.Close()
		m2.Close()

		// For visual checks
		// xG, err := x.Grad()
		// t.Logf("Test  %q x: \n%+v,\n%+v", sts.name, x.Value(), xG)
	}

	// special cases with UnsafeLet
	g := NewGraph()
	x := NewTensor(g, Float64, 2, WithShape(2, 3), WithInit(RangedFrom(0)))
	sliced, _ := Slice(x, S(0))
	cost := Must(Slice(sliced, S(0)))
	Grad(cost, x)

	m := NewTapeMachine(g)
	defer m.Close()
	// mutate the graph before running
	UnsafeLet(sliced, S(1))
	UnsafeLet(cost, S(2))
	if err := m.RunAll(); err != nil {
		t.Fatal(err)
	}

	xG, err := x.Grad()
	if err != nil {
		t.Fatal(err)
	}

	// ioutil.WriteFile("blah.dot", []byte(g.ToDot()), 0644)
	assert.Equal(t, []float64{0, 0, 0, 0, 0, 1}, xG.Data())
	// visual inspection
	// t.Logf("x: \n%+v,\n%+v", x.Value(), xG)

}

var sumTests = []struct {
	name  string
	shape tensor.Shape
	along []int

	expectedShape tensor.Shape
	expectedVal   Value
	expectedGrad  Value
	err           bool
}{
	{"Sum(vec)", tensor.Shape{2}, nil, scalarShape, NewF64(1.0), NewF64(1.0), false},
	{"Sum(vec, 0)", tensor.Shape{2}, []int{0}, scalarShape, NewF64(1), NewF64(1.0), false},
	{"Sum(Mat)", tensor.Shape{2, 3}, nil, scalarShape, NewF64(15.0), tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 1, 1, 1, 1, 1})), false},
	{"Sum(Mat, 0)", tensor.Shape{2, 3}, []int{0}, tensor.Shape{3},
		tensor.New(tensor.WithShape(3), tensor.WithBacking([]float64{3, 5, 7})),
		tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 1, 1, 1, 1, 1})), false,
	},
	{"Sum(Mat, 1)", tensor.Shape{2, 3}, []int{1}, tensor.Shape{2},
		tensor.New(tensor.WithShape(2), tensor.WithBacking([]float64{3, 12})),
		tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 1, 1, 1, 1, 1})), false,
	},

	// TODO: tests for 3-Tensors
	// TODO: negative and stupids cases.
}

func TestSum(t *testing.T) {
	defer runtime.GC()
	for _, sts := range sumTests {
		g := NewGraph()
		x := NewTensor(g, Float64, len(sts.shape), WithShape(sts.shape...), WithInit(RangedFrom(0)))
		var s *Node
		var err error

		if len(sts.along) == 0 {
			s, err = Sum(x)
		} else {
			s, err = Sum(x, sts.along...)
		}

		switch {
		case sts.err:
			if err == nil {
				t.Errorf("Expected an error in %q", sts.name)
			}
			continue
		case !sts.err && err != nil:
			t.Errorf("Test %q errored while Sum() %+v", sts.name, err)
			continue
		}

		if !sts.expectedShape.Eq(s.shape) {
			t.Errorf("Test %q has wrong shape. Want %v, got %v instead", sts.name, sts.expectedShape, s.shape)
			continue
		}

		cost := s
		if len(sts.along) < len(sts.shape) && len(sts.along) > 0 {
			cost = Must(Sum(s))
		}

		if _, err = Grad(cost, x); err != nil {
			t.Errorf("Test %q - Unable to back prop. Err : %+v", sts.name, err)
			continue
		}

		m := NewTapeMachine(g)
		if err = m.RunAll(); err != nil {
			t.Errorf("Test %q - Runtime error: %v", sts.name, err)
			continue
		}

		if !ValueEq(sts.expectedVal, s.Value()) {
			t.Errorf("Test %q Expected %v. Got %v", sts.name, sts.expectedVal, s.Value())
		}

		sG, err := s.Grad()
		if err != nil {
			t.Errorf("Test %q Grad() error: %+v", sts.name, err)
			continue
		}

		// LISP MACHINE TO TEST GRAD EQUIVALENCE
		h := NewGraph()
		a := NewTensor(h, Float64, len(sts.shape), WithShape(sts.shape...), WithInit(RangedFrom(0)))
		var b *Node
		if len(sts.along) == 0 {
			b = Must(Sum(a))
		} else {
			b = Must(Sum(a, sts.along...))
		}

		if len(sts.along) < len(sts.shape) && len(sts.along) > 0 {
			Must(Sum(b))
		}

		m2 := NewLispMachine(h)
		if err = m2.RunAll(); err != nil {
			t.Errorf("Test %q Lisp machine runtime error %+v", sts.name, err)
			continue
		}

		if !ValueEq(sts.expectedVal, b.Value()) {
			t.Errorf("Test %q LispMachine Run. Expected %v. Got %v instead", sts.name, sts.expectedVal, b.Value())
		}

		bG, err := b.Grad()
		if err != nil {
			t.Errorf("Test %q Grad() err in lispmachine run %+v", sts.name, err)
			continue
		}

		if !ValueEq(sG, bG) {
			t.Errorf("Expected the values of the partial derivatives of both machines to be the same")
		}

		m.Close()
		m2.Close()
	}
}

func TestNorm(t *testing.T) {
	assert := assert.New(t)
	g := NewGraph()
	x := NewMatrix(g, Float64, WithShape(3, 3))
	norm, err := Norm(x, 0, 2)
	if err != nil {
		t.Error(err)
		return
	}
	m := NewLispMachine(g, ExecuteFwdOnly())
	defer m.Close()

	xT := tensor.New(tensor.WithShape(3, 3), tensor.WithBacking(tensor.Range(tensor.Float64, 0, 9)))
	Let(x, xT)
	m.RunAll()

	correct := []float64{6.708203932499369, 8.12403840463596, 9.643650760992955}
	assert.Equal(correct, extractF64s(norm.Value()))

}

func TestMean(t *testing.T) {
	g := NewGraph()
	x := NewMatrix(g, Float64, WithShape(3, 3))
	m, err := Mean(x)
	if err != nil {
		t.Fatal(err)
	}

	if !m.IsScalar() {
		t.Error("Expected result to be scalar")
	}
}

func TestTensordot(t *testing.T) {
	assert := assert.New(t)

	// Scalars
	g := NewGraph()

	a := NewTensor(g, Float64, 0, WithName("a"), WithShape(1), WithInit(RangedFrom(2)))
	b := NewTensor(g, Float64, 0, WithName("b"), WithShape(1), WithInit(RangedFrom(21)))
	c := NewTensor(g, Float64, 0, WithName("c"), WithShape(1), WithInit(ValuesOf(1.0)))

	tensordot, err := Tensordot([]int{0}, []int{0}, a, b)
	if err == nil {
		t.Fatal("Expected scalars to fail")
	}

	// Scalar-like
	g = NewGraph()
	a = NewTensor(g, Float64, 1, WithName("a"), WithShape(1), WithInit(RangedFrom(2)))
	b = NewTensor(g, Float64, 1, WithName("b"), WithShape(1), WithInit(RangedFrom(21)))
	c = NewTensor(g, Float64, 1, WithName("c"), WithShape(1), WithInit(ValuesOf(1.0)))

	tensordot, err = Tensordot([]int{0}, []int{0}, a, b)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("SHAPE a %v b %v c %v tensordot %v", a.Shape(), b.Shape(), c.Shape(), tensordot.Shape())

	dtensordot, err := Backpropagate(Nodes{tensordot}, Nodes{c}, Nodes{a, b})

	if err != nil {
		t.Fatalf("%+v", err)
	}

	m := NewTapeMachine(g)
	defer m.Close()
	if err = m.RunAll(); err != nil {
		t.Fatal(err)
	}

	correctScalarlike := []float64{42.0}
	value := tensordot.Value().Data()
	assert.Equal(correctScalarlike, value)

	dtensordotCorrectScalarlike0 := []float64{21}
	dtensordotCorrectScalarlike1 := []float64{2}

	assert.Equal(dtensordotCorrectScalarlike0, dtensordot[0].Value().Data())
	assert.Equal(dtensordotCorrectScalarlike1, dtensordot[1].Value().Data())

	// Vectors

	g = NewGraph()
	a = NewTensor(g, Float64, 1, WithName("a"), WithShape(2), WithInit(RangedFrom(1)))
	b = NewTensor(g, Float64, 1, WithName("b"), WithShape(2), WithInit(RangedFrom(3)))
	c = NewTensor(g, Float64, 0, WithName("c"), WithShape(), WithInit(ValuesOf(1.0)))

	if tensordot, err = Tensordot([]int{0}, []int{0}, a, b); err != nil {
		t.Fatal(err)
	}

	if dtensordot, err = Backpropagate(Nodes{tensordot}, Nodes{c}, Nodes{a, b}); err != nil {
		t.Fatalf("%+v", err)
	}

	// Need to multiply dtensordot with identiy matrix, otherwise the transpose action in symdiff is not performed
	id := NewConstant(tensor.I(Float64, 2, 2, 0))

	dtensordot0 := Must(Mul(id, dtensordot[0]))
	dtensordot1 := Must(Mul(id, dtensordot[1]))

	m = NewTapeMachine(g)
	defer m.Close()
	if err = m.RunAll(); err != nil {
		t.Fatal(err)
	}

	t.Logf("TensorDot %v | %v", tensordot.Value().Shape(), tensordot.Type())
	correctScalarlike = []float64{11}
	assert.Equal(correctScalarlike, tensordot.Value().Data())

	dcorrect0 := []float64{3, 4}
	dcorrect1 := []float64{1, 2}

	assert.Equal(dcorrect0, extractF64s(dtensordot[0].Value()))
	assert.Equal(dcorrect1, extractF64s(dtensordot[1].Value()))

	// Vector and Matrix
	g = NewGraph()
	a = NewTensor(g, Float64, 2, WithName("a"), WithShape(2, 2), WithInit(RangedFrom(0)))
	b = NewTensor(g, Float64, 1, WithName("b"), WithShape(2), WithInit(RangedFrom(0)))

	c = NewTensor(g, Float64, 1, WithName("c"), WithShape(2), WithInit(ValuesOf(1.0)))

	if tensordot, err = Tensordot([]int{1}, []int{0}, a, b); err != nil {
		t.Fatal(err)
	}

	if dtensordot, err = Backpropagate(Nodes{tensordot}, Nodes{c}, Nodes{a, b}); err != nil {
		t.Fatal(err)
	}

	// Need to multiply dtensordot with identiy matrix, otherwise the transpose action in symdiff is not performed
	id = NewConstant(tensor.I(Float64, 2, 2, 0))

	if dtensordot0, err = Mul(id, dtensordot[0]); err != nil {
		t.Fatal(err)
	}
	if dtensordot1, err = Mul(id, dtensordot[1]); err != nil {
		t.Fatal(err)
	}

	m = NewTapeMachine(g)
	defer m.Close()
	if err = m.RunAll(); err != nil {
		t.Fatal(err)
	}

	correct := []float64{1, 3}
	assert.Equal(correct, extractF64s(tensordot.Value()))

	dcorrect0 = []float64{0, 1, 0, 1}
	dcorrect1 = []float64{2, 4}

	assert.Equal(dcorrect0, extractF64s(dtensordot0.Value()))
	assert.Equal(dcorrect1, extractF64s(dtensordot1.Value()))

	// Matrices
	g = NewGraph()

	a = NewTensor(g, Float64, 2, WithName("a"), WithShape(2, 2), WithInit(RangedFrom(0)))
	b = NewTensor(g, Float64, 2, WithName("b"), WithShape(2, 2), WithInit(RangedFrom(0)))

	c = NewTensor(g, Float64, 2, WithName("c"), WithShape(2, 2), WithInit(ValuesOf(1.0)))

	if tensordot, err = Tensordot([]int{1}, []int{1}, a, b); err != nil {
		t.Fatal(err)
	}

	if dtensordot, err = Backpropagate(Nodes{tensordot}, Nodes{c}, Nodes{a, b}); err != nil {
		t.Fatal(err)
	}

	// Need to multiply dtensordot with identiy matrix, otherwise the transpose action in symdiff is not performed
	id = NewConstant(tensor.I(Float64, 2, 2, 0))

	if dtensordot0, err = Mul(id, dtensordot[0]); err != nil {
		t.Fatal(err)
	}
	if dtensordot1, err = Mul(id, dtensordot[1]); err != nil {
		t.Fatal(err)
	}

	m = NewTapeMachine(g)
	if err = m.RunAll(); err != nil {
		t.Fatal(err)
	}

	correct = []float64{1, 3, 3, 13}
	assert.Equal(correct, extractF64s(tensordot.Value()))

	dcorrect := []float64{2, 4, 2, 4}
	assert.Equal(dcorrect, extractF64s(dtensordot0.Value()))
	assert.Equal(dcorrect, extractF64s(dtensordot1.Value()))

	// Total matrix contraction
	g = NewGraph()

	a = NewTensor(g, Float64, 2, WithName("a"), WithShape(2, 2), WithInit(RangedFrom(0)))
	b = NewTensor(g, Float64, 2, WithName("b"), WithShape(2, 2), WithInit(RangedFrom(0)))

	c = NewTensor(g, Float64, 0, WithName("c"), WithShape(), WithInit(ValuesOf(1.0)))

	if tensordot, err = Tensordot([]int{0, 1}, []int{0, 1}, a, b); err != nil {
		t.Fatal(err)
	}

	if dtensordot, err = Backpropagate(Nodes{tensordot}, Nodes{c}, Nodes{a, b}); err != nil {
		t.Fatal(err)
	}

	// Need to multiply dtensordot with identiy matrix, otherwise the transpose action in symdiff is not performed
	id = NewConstant(tensor.I(Float64, 2, 2, 0))

	if dtensordot0, err = Mul(id, dtensordot[0]); err != nil {
		t.Fatal(err)
	}
	if dtensordot1, err = Mul(id, dtensordot[1]); err != nil {
		t.Fatal(err)
	}

	m = NewTapeMachine(g)
	defer m.Close()
	if err = m.RunAll(); err != nil {
		t.Fatal(err)
	}

	correctScalarlike = []float64{14}
	assert.Equal(correctScalarlike, tensordot.Value().Data())

	dcorrect = []float64{0, 1, 2, 3}
	assert.Equal(dcorrect, extractF64s(dtensordot0.Value()))
	assert.Equal(dcorrect, extractF64s(dtensordot1.Value()))

}

var reshapeTests = []struct {
	testName string
	input    tensor.Shape
	to       tensor.Shape
	output   tensor.Shape
	err      bool
}{
	{"simple", tensor.Shape{2, 2}, tensor.Shape{4}, tensor.Shape{4}, false},
	{"simple big tensor", tensor.Shape{200, 200}, tensor.Shape{200 * 200}, tensor.Shape{200 * 200}, false},
	{"negative dim1 1", tensor.Shape{3, 2}, tensor.Shape{6, -1}, tensor.Shape{6, 1}, false},
	{"negative dim1 2", tensor.Shape{3, 2}, tensor.Shape{2, -1}, tensor.Shape{2, 3}, false},
	{"negative dim0 1", tensor.Shape{3, 2}, tensor.Shape{-1, 3}, tensor.Shape{2, 3}, false},
	{"negative dims0.1 with error", tensor.Shape{3, 2}, tensor.Shape{-1, -1}, nil, true},
	{"devative dim0 with error", tensor.Shape{3, 2}, tensor.Shape{4, -1}, nil, true},
}

func TestReshape(t *testing.T) {
	for _, rst := range reshapeTests {
		g := NewGraph()
		T := NewTensor(g, Float64, len(rst.input), WithShape(rst.input.Clone()...))
		T2, err := Reshape(T, rst.to.Clone())
		t.Log(T2)
		switch {
		case rst.err && err == nil:
			t.Fatalf("Expected Error when testing %v", rst)
		case rst.err:
			continue
		case err != nil:
			t.Fatal(err)
		default:
			assert.True(t, rst.output.Eq(T2.Shape()), "expected both to be the same")
		}

	}
}
func TestReshape_Dense(t *testing.T) {
	for _, rst := range reshapeTests {
		g := NewGraph()
		tT := tensor.New(tensor.Of(tensor.Float64), tensor.WithShape(rst.input.Clone()...))
		T := NodeFromAny(g, tT)
		T2, err := Reshape(T, rst.to.Clone())
		switch {
		case rst.err && err == nil:
			t.Fatalf("Expected Error when testing %v", rst)
		case rst.err:
			continue
		case err != nil:
			t.Fatal(err)
		default:
			assert.True(t, rst.output.Eq(T2.Shape()), "expected both to be the same")
		}
		m := NewTapeMachine(g)
		if err := m.RunAll(); err != nil {
			t.Errorf("Error while executing %q. Err: %v", rst.testName, err)
			continue
		}

	}
}

func TestReshapeRuntime(t *testing.T) {
	g := NewGraph()
	x := NewMatrix(g, tensor.Float64, WithName("x"), WithShape(28, 28), WithInit(GlorotU(1)))
	w := NewMatrix(g, tensor.Float64, WithName("W"), WithShape(50, 784), WithInit(GlorotU(1)))
	x2 := Must(Reshape(x, tensor.Shape{784}))
	wx := Must(Mul(w, x2))
	wx2 := Must(Reshape(wx, tensor.Shape{5, 10}))

	cost := Must(Sum(wx2))
	if _, err := Grad(cost, w); err != nil {
		t.Fatal(err)
	}
	m := NewTapeMachine(g)
	if err := m.RunAll(); err != nil {
		t.Fatal(err)
	}

	if !x.Value().Shape().Eq(tensor.Shape{28, 28}) {
		t.Errorf("A mutation of shape has occurred")
	}
}

var ravelTests = []struct {
	input  tensor.Shape
	output tensor.Shape
}{
	{
		tensor.Shape{3, 3},
		tensor.Shape{9},
	},
	{
		tensor.Shape{2, 3},
		tensor.Shape{6},
	},
	{
		tensor.Shape{2, 1, 3},
		tensor.Shape{6},
	},
	{
		tensor.Shape{1, 1, 1},
		tensor.Shape{1},
	},
}

func TestRavel(t *testing.T) {
	c := require.New(t)

	for i, rst := range ravelTests {
		g := NewGraph()
		t := NewTensor(g, Float64, len(rst.input), WithShape(rst.input...))
		t2, err := Ravel(t)

		c.NoError(err)
		c.Equal(rst.output, t2.Shape(), "expected to be flatten in test case: %d", i)
	}
}

func TestAuto(t *testing.T) {
	testCases := []struct {
		desc          string
		shapeA        tensor.Shape
		shapeB        tensor.Shape
		expectedShape tensor.Shape
		expectedErr   string
	}{
		{
			desc:        "Example 0",
			shapeA:      tensor.Shape{12},
			shapeB:      tensor.Shape{1, 11},
			expectedErr: "shapes (12) and (1, 11) should have the same dimensions",
		},
		{
			desc:          "Example 1",
			shapeA:        tensor.Shape{12, 1},
			shapeB:        tensor.Shape{12, 11},
			expectedShape: tensor.Shape{12, 11},
			expectedErr:   "",
		},
		{
			desc:          "Example 2",
			shapeA:        tensor.Shape{1, 12},
			shapeB:        tensor.Shape{11, 12},
			expectedShape: tensor.Shape{11, 12},
			expectedErr:   "",
		},
		{
			desc:          "Example 3",
			shapeA:        tensor.Shape{2, 3, 5},
			shapeB:        tensor.Shape{2, 3, 1},
			expectedShape: tensor.Shape{2, 3, 5},
			expectedErr:   "",
		},
		{
			desc:          "Example 4",
			shapeA:        tensor.Shape{2, 1, 5},
			shapeB:        tensor.Shape{2, 3, 5},
			expectedShape: tensor.Shape{2, 3, 5},
			expectedErr:   "",
		},
		{
			desc:          "Example 5",
			shapeA:        tensor.Shape{2, 1, 1},
			shapeB:        tensor.Shape{2, 5, 3},
			expectedShape: tensor.Shape{2, 5, 3},
			expectedErr:   "",
		},
	}
	for _, tC := range testCases {
		t.Run(tC.desc, func(t *testing.T) {
			c := require.New(t)

			g := NewGraph()
			a := NewTensor(g, Float64, tC.shapeA.Dims(), WithShape(tC.shapeA...), WithInit(RangedFrom(0)))
			b := NewTensor(g, Float64, tC.shapeB.Dims(), WithShape(tC.shapeB...), WithInit(RangedFrom(0)))

			out, err := Auto(BroadcastHadamardProd, a, b)

			if tC.expectedErr != "" {
				c.Error(err)
				c.Equal(tC.expectedErr, err.Error())
				return
			} else {
				c.NoError(err)
			}

			c.Equal(tC.expectedShape, out.Shape())

			out, err = Auto(BroadcastHadamardProd, b, a)
			c.NoError(err)
			c.Equal(tC.expectedShape, out.Shape())
		})
	}
}

func TestSliceBNConcat(t *testing.T) {
	testCases := []struct {
		XInit             InitWFn
		XShape            tensor.Shape
		WeightsInit       InitWFn
		ExpectedScale     []float64
		ExpectedScaleGrad []float64
		ExpectedOutput    []float64
		ExpectedInputGrad []float64
	}{
		{
			XInit:             RangedFromWithStep(0.1, 2),
			XShape:            tensor.Shape{4, 2, 2, 2},
			WeightsInit:       RangedFromWithStep(-0.05, 3),
			ExpectedScale:     []float64{-0.05, 2.95},
			ExpectedScaleGrad: []float64{3.6115753308647314, 3.611575330864615},
			ExpectedOutput:    []float64{17.605945678258838, 27.882593114861223, 389.7811912407563, 936.3045438041536, 17.605945678259026, 27.882593114861688, 389.78119124075647, 936.3045438041542},
			ExpectedInputGrad: []float64{0.004972459049631834, 0.0007851252852762937, -0.003402208479079247, -0.007589542243434793, -0.29337508392827627, -0.046322391831299575, 0.2007303002656771, 0.44778299236265373, 0.007589542243434797, 0.0034022084790792566, -0.0007851252852762831, -0.004972459049631828, -0.4477829923626526, -0.20073030026567584, 0.046322391831300797, 0.2933750839282775, 0.004972459049631834, 0.0007851252852762937, -0.003402208479079247, -0.007589542243434793, -0.29337508392827627, -0.046322391831299575, 0.2007303002656771, 0.44778299236265373, 0.007589542243434797, 0.0034022084790792566, -0.0007851252852762831, -0.004972459049631828, -0.4477829923626526, -0.20073030026567584, 0.046322391831300797, 0.2933750839282775},
		},
	}

	for i, tC := range testCases {
		t.Run(fmt.Sprintf("#%d %v", i+1, tC.XShape), func(t *testing.T) {
			c := require.New(t)

			g := NewGraph()

			input := NewTensor(g, Float64, tC.XShape.Dims(), WithShape(tC.XShape...), WithInit(tC.XInit), WithName("x"))

			scale := NewTensor(g, Float64, 4, WithShape(1, 2, 1, 1), WithInit(tC.WeightsInit), WithName("scale"))
			bias := NewTensor(g, Float64, 4, WithShape(1, 2, 1, 1), WithInit(tC.WeightsInit), WithName("bias"))

			sl1 := Must(Slice(input, S(2, 4)))
			w1 := NewTensor(g, Float64, 2, WithShape(2, 8), WithInit(tC.WeightsInit), WithName("w1"))

			sl2 := Must(Slice(input, S(0, 2)))
			w2 := NewTensor(g, Float64, 2, WithShape(2, 8), WithInit(tC.WeightsInit), WithName("w2"))

			slShape := tensor.Shape{sl1.Shape()[0], tensor.Shape(sl1.Shape()[1:]).TotalSize()}

			bn1, _, _, _, err := BatchNorm(sl1, scale, bias, 0.1, 1e-5)
			c.NoError(err)

			bn1 = Must(Reshape(bn1, slShape))

			y1 := Must(Mul(bn1, Must(Transpose(w1, 1, 0))))

			bn2, _, _, _, err := BatchNorm(sl2, scale, bias, 0.1, 1e-5)
			c.NoError(err)
			bn2 = Must(Reshape(bn2, slShape))

			y2 := Must(Mul(bn2, Must(Transpose(w2, 1, 0))))

			y := Must(Concat(0, y1, y2))

			cost := Must(Mean(y))

			_, err = Grad(cost, input, scale)
			c.NoError(err)

			vm := NewTapeMachine(g) //, TraceExec())
			c.NoError(vm.RunAll())

			t.Logf("y: %v", y.Value())
			t.Logf("dx: %v", input.Deriv().Value())
			t.Logf("scale: %v", scale.Value())
			t.Logf("dScale: %v", scale.Deriv().Value())

			c.InDeltaSlice(tC.ExpectedScale, scale.Value().Data(), 1e-5, "expected: %v\ngot: %#v", tC.ExpectedScale, scale.Value().Data())
			c.InDeltaSlice(tC.ExpectedScaleGrad, scale.Deriv().Value().Data(), 1e-5, "expected: %v\ngot: %#v", tC.ExpectedScaleGrad, scale.Deriv().Value().Data())

			c.InDeltaSlice(tC.ExpectedOutput, y.Value().Data(), 1e-5, "expected: %v\ngot: %#v", tC.ExpectedOutput, y.Value().Data())
			c.InDeltaSlice(tC.ExpectedInputGrad, input.Deriv().Value().Data(), 1e-5, "expected: %#v\ngot: %#v", tC.ExpectedInputGrad, input.Deriv().Value().Data())
		})
	}
}
