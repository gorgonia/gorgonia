package gorgonia

import (
	"io/ioutil"
	"testing"

	"github.com/chewxy/gorgonia/tensor"
	tf64 "github.com/chewxy/gorgonia/tensor/f64"
	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/stretchr/testify/assert"
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
	added, err := applyOpWithName(op, "+ pi pi", cpi, cpi)
	if err != nil {
		t.Fatal(err)
	}
	assert.Equal(g, added.g)
	assert.Equal(Float64, added.t)

	ct = NewConstant(tf64.Ones(3, 3)) // no graph set for ct
	op = newElemBinOp(addOpType, cpi, ct)
	added, err = applyOpWithName(op, "+ pi constTensor(3,3)_ones", cpi, ct)
	if err != nil {
		t.Error(err)
	}
}

var mulTests = []struct {
	name   string
	xshape types.Shape
	wshape types.Shape

	gradX []float64
	gradW []float64
}{
	{"x vector", types.Shape{2}, types.Shape{2, 3}, []float64{3, 12}, []float64{0, 0, 0, 1, 1, 1}},
	{"x mat", types.Shape{3, 2}, types.Shape{2, 3}, []float64{3, 12, 3, 12, 3, 12}, []float64{6, 6, 6, 9, 9, 9}},
	{"x_vec_w_vec", types.Shape{6}, types.Shape{6}, []float64{0, 1, 2, 3, 4, 5}, []float64{0, 1, 2, 3, 4, 5}},
}

func TestMul(t *testing.T) {
	assert := assert.New(t)

	t.Logf("Testing Mul with TapeMachine")
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

		prog, locMap, err := Compile(g)
		if err != nil {
			t.Errorf("Error while compiling %q. Err: %v", mts.name, err)
			continue
		}

		m := NewTapeMachine(prog, locMap)
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
	}
}

var gtTests = []struct {
	a, b    Value
	retSame bool

	expected Value
	err      bool
}{
	// s-s
	{F64(1), F64(0), true, F64(1), false},
	{F64(0), F64(1), true, F64(0), false},
	{F64(1), F64(0), false, B(true), false},
	{F32(0), F32(1), false, B(false), false},

	// s-t
	{
		F64(1), tensor.New(types.Float64, tensor.WithShape(2), tensor.WithBacking([]float64{0, 2})),
		true,
		tensor.New(types.Float64, tensor.WithShape(2), tensor.WithBacking([]float64{1, 0})),
		false,
	},

	{
		F32(1), tensor.New(types.Float32, tensor.WithShape(2), tensor.WithBacking([]float32{0, 2})),
		false,
		tensor.New(types.Bool, tensor.WithShape(2), tensor.WithBacking([]bool{true, false})),
		false,
	},

	// t-s
	{
		tensor.New(types.Float64, tensor.WithShape(2), tensor.WithBacking([]float64{0, 2})), F64(1),
		true,
		tensor.New(types.Float64, tensor.WithShape(2), tensor.WithBacking([]float64{0, 1})),
		false,
	},

	{
		tensor.New(types.Float32, tensor.WithShape(2), tensor.WithBacking([]float32{0, 2})), F32(1),
		false,
		tensor.New(types.Bool, tensor.WithShape(2), tensor.WithBacking([]bool{false, true})),
		false,
	},

	// t-t
	{
		tensor.New(types.Float64, tensor.WithShape(2, 3), tensor.WithBacking([]float64{0, 1, 2, 3, 4, 5})),
		tensor.New(types.Float64, tensor.WithShape(2, 3), tensor.WithBacking([]float64{5, 4, 3, 2, 1, 0})),
		true,

		tensor.New(types.Float64, tensor.WithShape(2, 3), tensor.WithBacking([]float64{0, 0, 0, 1, 1, 1})),
		false,
	},

	{
		tensor.New(types.Float64, tensor.WithShape(2, 3), tensor.WithBacking([]float64{0, 1, 2, 3, 4, 5})),
		tensor.New(types.Float64, tensor.WithShape(2, 3), tensor.WithBacking([]float64{5, 4, 3, 2, 1, 0})),
		false,

		tensor.New(types.Bool, tensor.WithShape(2, 3), tensor.WithBacking([]bool{false, false, false, true, true, true})),
		false,
	},

	// stupids

	// different shapes
	{
		tensor.New(types.Float32, tensor.WithShape(2)), tensor.New(types.Float32, tensor.WithShape(4)),
		true, nil, true,
	},

	// different dtypes
	{
		tensor.New(types.Float64, tensor.WithShape(2)), tensor.New(types.Float32, tensor.WithShape(2)),
		true, nil, true,
	},
}

func TestGt(t *testing.T) {
	for i, gtts := range gtTests {
		g := NewGraph()
		a := NodeFromAny(g, gtts.a, WithName("a"))
		b := NodeFromAny(g, gtts.b, WithName("b"))

		var ret *Node
		var err error
		ret, err = Gt(a, b, gtts.retSame)

		switch {
		case gtts.err:
			if err == nil {
				t.Error("Expected an error")
			}
			continue
		case !gtts.err && err != nil:
			t.Errorf("Test %d: %+v", i, err)
			continue

		}

		prog, locMap, err := Compile(g)
		m1 := NewTapeMachine(prog, locMap)
		if err = m1.RunAll(); err != nil {
			t.Errorf("Test %d: %+v", i, err)
			continue
		}

		if !ValueEq(gtts.expected, ret.Value()) {
			t.Errorf("Test %d Expected %v. Got %v", i, gtts.expected, ret.Value())
		}

		if i == 6 {
			ioutil.WriteFile("fullGraph.dot", []byte(g.ToDot()), 0644)
		}
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

	prog, locMap, err := Compile(g)
	m1 := NewTapeMachine(prog, locMap)
	if err = m1.RunAll(); err != nil {
		t.Error(err)
	}

	if (TensorType{d: 1, of: Float64}) != TypeOf(gt.Value()) {
		t.Error("Expected a tensor type of float64")
	}

}

func TestSoftMax(t *testing.T) {
	assert := assert.New(t)
	g := NewGraph()
	xT := tf64.NewTensor(tf64.WithBacking([]float64{0.1, 0.2, -0.3, 0.4, 0.5}))
	x := NewVector(g, Float64, WithShape(5), WithValue(xT))
	sm := Must(SoftMax(x))
	logsm := Must(Neg(Must(Log(sm))))
	cost := Must(Slice(logsm, S(2)))

	if _, err := Grad(cost, x); err != nil {
		t.Error(err)
	}
	prog, locMap, err := Compile(g)
	if err != nil {
		t.Error(err)
	}

	m := NewTapeMachine(prog, locMap)
	err = m.RunAll()
	if err != nil {
		t.Error(err)
	}

	var smg, xG Value
	if smg, err = sm.Grad(); err != nil {
		t.Error(err)
	}

	if xG, err = x.Grad(); err != nil {
		t.Error(err)
	}

	// machine 2, graph 2

	g2 := NewGraph()
	xT2 := tf64.NewTensor(tf64.WithBacking([]float64{0.1, 0.2, -0.3, 0.4, 0.5}))
	x2 := NewVector(g, Float64, WithShape(5), WithValue(xT2))
	sm2 := Must(SoftMax(x2))
	logsm2 := Must(Neg(Must(Log(sm2))))
	Must(Slice(logsm2, S(2)))

	m2 := NewLispMachine(g2)
	err = m2.RunAll()
	if err != nil {
		t.Error(err)
	}

	var sm2g, x2G Value
	if sm2g, err = sm2.Grad(); err != nil {
		t.Error(err)
	}

	if x2G, err = x2.Grad(); err != nil {
		t.Error(err)
	}

	assert.Equal(smg, sm2g)
	assert.Equal(xG, x2G)
}

var sliceTests = []struct {
	name   string
	shape  types.Shape
	slices []types.Slice

	expected types.Shape
	data     interface{}
	err      bool
}{
	{"vec[0]", types.Shape{2}, []types.Slice{S(0)}, scalarShape, float64(0), false},
	{"vec[0:2]", types.Shape{2}, []types.Slice{S(0, 2)}, types.Shape{2}, []float64{0, 1}, false},
	{"Mat[0]", types.Shape{2, 3}, []types.Slice{S(0)}, types.Shape{3}, []float64{0, 1, 2}, false},
	{"Mat[:, 0]", types.Shape{2, 3}, []types.Slice{nil, S(0)}, types.Shape{2}, []float64{0, 1, 2, 3}, false},
	{"3Tensor[0]", types.Shape{2, 3, 4}, []types.Slice{S(0)}, types.Shape{3, 4}, tf64.RangeFloat64(0, 12), false},
	{"3Tensor[0:2]", types.Shape{2, 3, 4}, []types.Slice{S(0, 2)}, types.Shape{2, 3, 4}, tf64.RangeFloat64(0, 24), false},
	{"3Tensor[:, 0]", types.Shape{2, 3, 4}, []types.Slice{nil, S(0)}, types.Shape{2, 4}, tf64.RangeFloat64(0, 16), false},
	{"3Tensor[0, :, 0]", types.Shape{2, 3, 4}, []types.Slice{S(0), nil, S(0)}, types.Shape{3}, tf64.RangeFloat64(0, 9), false},

	{"vec[:, 0]", types.Shape{2}, []types.Slice{nil, S(0)}, nil, nil, true},
}

func TestSlice(t *testing.T) {
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
		prog, locMap, err := Compile(g)
		if err != nil {
			t.Errorf("Test %q failed to compile: %+v", sts.name, err)
			continue
		}
		m1 := NewTapeMachine(prog, locMap)
		if err = m1.RunAll(); err != nil {
			t.Errorf("Test %q Runtime error %+v ", sts.name, err)
			continue
		}

		sV := sliced.Value()
		if !sts.expected.Eq(sV.Shape()) {
			t.Errorf("Test %q. Expected sliced value to have the shape %v. Got %v instead", sts.name, sts.expected, sV.Shape())
		}

		assert.Equal(t, sts.data, sV.Data(), "Test %q data expected %v, Got %v instead. Formatted:\n %+v", sts.name, sts.data, sV.Data(), sV)

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
			t.Errorf("Test %q. Expected sliced value to have the shape %v. Got %v instead", sts.name, sts.expected, s2V.Shape())
		}

		assert.Equal(t, sts.data, s2V.Data(), "Test %q data expected %v, Got %v instead. Formatted:\n %+v", sts.name, sts.data, s2V.Data(), s2V)

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
	prog, locMap, err := Compile(g)
	m := NewTapeMachine(prog, locMap)

	// mutate the graph before running
	UnsafeLet(sliced, S(1))
	UnsafeLet(cost, S(2))
	if err = m.RunAll(); err != nil {
		t.Fatal(err)
	}

	xG, err := x.Grad()
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(t, []float64{0, 0, 0, 0, 0, 1}, xG.Data())
	// visual inspection
	// t.Logf("x: \n%+v,\n%+v", x.Value(), xG)

}

var sumTests = []struct {
	name  string
	shape types.Shape
	along []int

	expectedShape types.Shape
	expectedVal   Value
	expectedGrad  Value
	err           bool
}{
	{"Sum(vec)", types.Shape{2}, nil, scalarShape, F64(1), F64(1), false},
	{"Sum(vec, 0)", types.Shape{2}, []int{0}, scalarShape, F64(1), F64(1), false},
	{"Sum(Mat)", types.Shape{2, 3}, nil, scalarShape, F64(15), tensor.New(Float64.TensorDtype(), tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 1, 1, 1, 1, 1})), false},
	{"Sum(Mat, 0)", types.Shape{2, 3}, []int{0}, types.Shape{3},
		tensor.New(Float64.TensorDtype(), tensor.WithShape(3), tensor.WithBacking([]float64{3, 5, 7})),
		tensor.New(Float64.TensorDtype(), tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 1, 1, 1, 1, 1})), false,
	},
	{"Sum(Mat, 1)", types.Shape{2, 3}, []int{1}, types.Shape{2},
		tensor.New(types.Float64, tensor.WithShape(2), tensor.WithBacking([]float64{3, 12})),
		tensor.New(Float64.TensorDtype(), tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 1, 1, 1, 1, 1})), false,
	},

	// TODO: tests for 3-Tensors
	// TODO: negative and stupids cases.
}

func TestSum(t *testing.T) {
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
			t.Errorf("Test %q has wrong shape. Want %v, got %v instead", sts.expectedShape, s.shape)
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

		prog, locMap, err := Compile(g)
		if err != nil {
			t.Errorf("Test %q - Unable to compile. Err: %+v", sts.name, err)
			continue
		}

		m := NewTapeMachine(prog, locMap)
		if err = m.RunAll(); err != nil {
			t.Errorf("Test %q - Runtime error", sts.name, err)
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

	xT := tf64.NewTensor(tf64.WithShape(3, 3), tf64.WithBacking(tf64.RangeFloat64(0, 9)))
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
