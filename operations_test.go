package gorgonia

import (
	"testing"

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

func TestSoftMax(t *testing.T) {
	assert := assert.New(t)
	g := NewGraph()
	xT := tf64.NewTensor(tf64.WithBacking([]float64{0.1, 0.2, -0.3, 0.4, 0.5}))
	x := NewVector(g, Float64, WithShape(5), WithValue(xT))
	sm := Must(SoftMax(x))
	logsm := Must(Neg(Must(Log(sm))))
	cost := Must(Slice(logsm, S(2)))

	grads, _ := Grad(cost, sm)
	prog, locMap, err := Compile(g)
	if err != nil {
		t.Error(err)
	}

	m := NewTapeMachine(prog, locMap)
	err = m.RunAll()
	if err != nil {
		t.Error(err)
	}
	var smg Value
	smg, err = sm.Grad()
	if err != nil {
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

	smg, err = sm2.Grad()
	if err != nil {
		t.Error(err)
	}
	assert.Equal(smg, grads[0].Value())
}

func TestSlice(t *testing.T) {
	assert := assert.New(t)
	g := NewGraph()
	x := NewMatrix(g, Float64, WithShape(3, 3))
	x0, err := Slice(x, S(0))
	if err != nil {
		t.Log(err)
	}
	WithName("x[0]")(x0)

	x_0, err := Slice(x, nil, S(0))
	if err != nil {
		t.Log(err)
	}
	WithName("x[:, 0]")(x_0)

	x00, err := Slice(x, S(0), S(0))
	if err != nil {
		t.Log(err)
	}
	WithName("x[0,0]")(x00)

	xV := tf64.NewTensor(tf64.WithShape(3, 4), tf64.WithBacking(tf64.RangeFloat64(0, 12)))
	Let(x, xV)
	m := NewLispMachine(g, ExecuteFwdOnly())
	if err = m.RunAll(); err != nil {
		t.Fatal(err)
	}

	assert.Equal([]float64{0, 1, 2, 3}, x0.Value().(types.Tensor).Data())
	assert.Equal(tf64.RangeFloat64(0, 9), x_0.Value().(types.Tensor).Data()) // but it is [0,4,8]
	assert.Equal(0.0, x00.Value().(Scalar).Data())

	t.Logf("x: %+v", xV)
	t.Logf("x[0]: %v ", x0.Value())
	t.Logf("x[:, 0]: %v ", x_0.Value())
	t.Logf("x[0, 0]: %v ", x00.Value())
}

func TestSum(t *testing.T) {
	assert := assert.New(t)
	var g *ExprGraph
	var x, y, z, sz *Node
	var m *lispMachine
	var xBack, yBack []float64
	var xT, yT *tf64.Tensor
	var err error

	// scalar:
	g, x, y, z = simpleEqn()
	if sz, err = Sum(z); err != nil {
		t.Error(err)
		goto vectest
	}
	Let(x, 3.14)
	Let(y, 5.0)
	m = NewLispMachine(g)
	if err = m.RunAll(); err != nil {
		t.Error(err)
		goto vectest
	}

	assert.Equal(8.14, extractF64(sz.Value()))

	// vector sum

vectest:
	t.Log("vectest1")
	g, x, y, z = simpleVecEqn()
	if sz, err = Sum(z); err != nil {
		t.Error(err)
		goto vectest2
	}

	xBack = []float64{0.5, -0.1}
	xT = tf64.NewTensor(tf64.WithShape(2), tf64.WithBacking(xBack))

	yBack = []float64{-3.1, 1.1}
	yT = tf64.NewTensor(tf64.WithShape(2), tf64.WithBacking(yBack))

	Let(x, xT)
	Let(y, yT)

	m = NewLispMachine(g)
	if err = m.RunAll(); err != nil {
		t.Error(err)
		goto vectest2
	}

	assert.Equal(0.5-0.1-3.1+1.1, extractF64(sz.Value()))
vectest2:

	t.Log("vectest2 : Row vecs")
	g, x, y, z = simpleVecEqn()
	x.shape = types.Shape{1, 2}
	y.shape = types.Shape{1, 2}
	z.shape = types.Shape{1, 2}
	if sz, err = Sum(z); err != nil {
		t.Error(err)
		goto mattest1
	}

	xBack = []float64{0.5, -0.1}
	xT = tf64.NewTensor(tf64.WithShape(1, 2), tf64.WithBacking(xBack))

	yBack = []float64{-3.1, 1.1}
	yT = tf64.NewTensor(tf64.WithShape(1, 2), tf64.WithBacking(yBack))

	Let(x, xT)
	Let(y, yT)

	m = NewLispMachine(g)
	if err = m.RunAll(); err != nil {
		t.Error(err)
		goto mattest1
	}

	assert.Equal(0.5-0.1-3.1+1.1, extractF64(sz.Value()))

	// matsum
mattest1:

	t.Log("mattest1")
	g, x, y, z = simpleMatEqn()
	if sz, err = Sum(z); err != nil {
		t.Error(err)
		goto mattest2
	}

	xBack = []float64{0.5, -0.1, 1.1, -2.2}
	xT = tf64.NewTensor(tf64.WithShape(2, 2), tf64.WithBacking(xBack))

	yBack = []float64{-3.1, 1.1, 0.01, 0.2}
	yT = tf64.NewTensor(tf64.WithShape(2, 2), tf64.WithBacking(yBack))

	Let(x, xT)
	Let(y, yT)

	m = NewLispMachine(g)
	if err = m.RunAll(); err != nil {
		t.Error(err)
		goto mattest2
	}

	assert.Equal(0.5-0.1+1.1-2.2-3.1+1.1+0.01+0.2, extractF64(sz.Value()))

mattest2:
	t.Log("mattest2")
	g, x, y, z = simpleMatEqn()
	if sz, err = Sum(z, 1); err != nil {
		t.Error(err)
		goto mattest3
	}

	xBack = []float64{0.5, -0.1, 1.1, -2.2}
	xT = tf64.NewTensor(tf64.WithShape(2, 2), tf64.WithBacking(xBack))

	yBack = []float64{-3.1, 1.1, 0.01, 0.2}
	yT = tf64.NewTensor(tf64.WithShape(2, 2), tf64.WithBacking(yBack))

	Let(x, xT)
	Let(y, yT)

	m = NewLispMachine(g, ExecuteFwdOnly())
	if err = m.RunAll(); err != nil {
		t.Error(err)
		goto mattest3
	}

	assert.Equal([]float64{-1.6, -0.8899999999999999}, extractF64s(sz.Value()))

mattest3:
	t.Log("mattest3")
	g, x, y, z = simpleMatEqn()
	if sz, err = Sum(z, 0); err != nil {
		t.Fatal(err)
	}

	xBack = []float64{0.5, -0.1, 1.1, -2.2}
	xT = tf64.NewTensor(tf64.WithShape(2, 2), tf64.WithBacking(xBack))

	yBack = []float64{-3.1, 1.1, 0.01, 0.2}
	yT = tf64.NewTensor(tf64.WithShape(2, 2), tf64.WithBacking(yBack))

	Let(x, xT)
	Let(y, yT)

	m = NewLispMachine(g, ExecuteFwdOnly())
	if err = m.RunAll(); err != nil {
		t.Fatal(err)
	}

	assert.Equal([]float64{-1.49, -1}, extractF64s(sz.Value()))

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
