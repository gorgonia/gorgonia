package gorgonia

import (
	"math"
	"math/rand"
	"runtime"
	"testing"

	"github.com/chewxy/gorgonia/tensor"
	"github.com/chewxy/math32"
	"github.com/stretchr/testify/assert"
)

func unaryOpTest(t *testing.T, dt tensor.Dtype, shape tensor.Shape, fn func(*Node) (*Node, error)) (x, y, a *Node, v, bV Value, err error) {

	var xV, aV Value
	var b *Node
	var any interface{}
	if shape.IsScalar() {
		if dt == tensor.Float64 {
			any = rand.ExpFloat64()
		} else {
			any = float32(rand.ExpFloat64())
		}
	} else {
		any = tensor.New(tensor.WithBacking(tensor.Random(dt, shape.TotalSize())))
	}
	if v, _, _, err = anyToValue(any); err != nil {
		t.Errorf("anyToValue failed %v", err)
		return
	}
	if xV, err = CloneValue(v); err != nil {
		t.Errorf("Clone to xV failed %v", err)
		return
	}

	g := NewGraph()
	x = NodeFromAny(g, xV, WithName("x"))
	y = Must(fn(x))
	Must(Sum(y))

	var grads Nodes
	h := NewGraph()
	a = NodeFromAny(h, xV, WithName("x"))
	b = Must(fn(a))
	cost := Must(Sum(b))
	Read(b, &bV)
	if grads, err = Grad(cost, a); err != nil {
		t.Errorf("Unable to get gradient %v", err)
		return
	}

	if aV, err = CloneValue(v); err != nil {
		t.Errorf("Clone to aV failed: %v", err)
		return
	}

	m0 := NewLispMachine(g)
	m1 := NewTapeMachine(h)

	Let(x, xV)
	if err = m0.RunAll(); err != nil {
		t.Errorf("m0 failed:", err)
		return
	}

	Let(a, aV)
	if err = m1.RunAll(); err != nil {
		t.Errorf("m1 failed:", err)
		return
	}

	var yV, xG, aG Value
	yV = y.Value()
	if xG, err = x.Grad(); err != nil {
		t.Errorf("x has no grad: %v", err)
		return
	}

	if aG, err = a.Grad(); err != nil {
		t.Errorf("a has no grad: %v", err)
		t.Logf("a.deriv %p | %p", a.deriv, grads[0])
		return
	}

	if !ValueClose(yV, bV) {
		t.Errorf("Expected yV and bV to be close. yV: %v, bV: %v", yV, bV)
	}

	if !ValueClose(aG, xG) {
		t.Errorf("Expected aG and xG to be close. aG: %v, xG %v", aG, xG)
	}

	return
}

func unaryOpDiffTest(op ʘUnaryOperatorType) (xRandVal float64, x, y, xT, yT *Node, err error) {
	_, x, y = simpleUnaryEqn()

	xRandVal = rand.ExpFloat64()
	fn := *(sf64UnaryOperators[op])
	diff := ʘUnaryOpDiffFns[op]

	// let the first stone be cast!
	Let(x, xRandVal)
	v, _, _, _ := anyToValue(fn(xRandVal)) // as if the graph has been executed upon
	ydv := variableDV(v)

	if err = y.bind(ydv); err != nil {
		return
	}

	if err = x.bind(dvUnit(x.boundTo)); err != nil {
		return
	}

	if err = diff(x, y); err != nil {
		return
	}

	// Tensor edition
	_, xT, yT = simpleUnaryVecEqn()

	xBack := []float64{-xRandVal, xRandVal}
	yBack := []float64{fn(-xRandVal), fn(xRandVal)}
	Let(xT, tensor.New(tensor.WithShape(2, 1), tensor.WithBacking(xBack)))
	vT, _, _, _ := anyToValue(tensor.New(tensor.WithShape(2, 1), tensor.WithBacking(yBack)))
	yTdv := variableDV(vT)

	if err = yT.bind(yTdv); err != nil {
		return
	}

	if err = xT.bind(dvUnit(xT.boundTo)); err != nil {
		return
	}

	if err = diff(xT, yT); err != nil {
		return
	}
	return
}

func TestAbs(t *testing.T) {
	defer runtime.GC()
	assert := assert.New(t)

	var x, y, a *Node
	var v Value
	var yV, xG, bV, aG Value
	var err error

	/* FLOAT 64 Scalar */

	x, y, a, v, bV, err = unaryOpTest(t, Float64, tensor.Shape{}, Abs)
	if err != nil {
		t.Fatal(err)
	}

	yV = y.Value()
	if xG, err = x.Grad(); err != nil {
		t.Errorf("x has no grad: %v", err)
		return
	}

	if aG, err = a.Grad(); err != nil {
		t.Errorf("a has no grad: %v", err)
	}

	correctF64 := math.Abs(v.Data().(float64))
	assert.True(ValueClose(newF64(correctF64), yV))
	assert.True(ValueClose(newF64(correctF64), bV))
	assert.True(ValueClose(newF64(1.0), xG))
	assert.True(ValueClose(newF64(1.0), aG))

	/* FLOAT 32 Scalar */

	x, y, a, v, bV, err = unaryOpTest(t, Float32, tensor.Shape{}, Abs)
	if err != nil {
		t.Fatal(err)
	}

	yV = y.Value()
	if xG, err = x.Grad(); err != nil {
		t.Errorf("x has no grad: %v", err)
		return
	}

	if aG, err = a.Grad(); err != nil {
		t.Errorf("a has no grad: %v", err)
	}

	correctF32 := math32.Abs(v.Data().(float32))
	assert.True(ValueClose(newF32(correctF32), yV))
	assert.True(ValueClose(newF32(correctF32), bV))
	assert.True(ValueClose(newF32(1.0), xG))
	assert.True(ValueClose(newF32(1.0), aG))

	/* FLOAT64 Vector */

	x, y, a, v, bV, err = unaryOpTest(t, Float64, tensor.Shape{10}, Abs)
	if err != nil {
		t.Fatal(err)
	}

	yV = y.Value()
	if xG, err = x.Grad(); err != nil {
		t.Errorf("x has no grad: %v", err)
		return
	}

	if aG, err = a.Grad(); err != nil {
		t.Errorf("a has no grad: %v", err)
	}

	absF64s := v.Data().([]float64)
	backingGrad64 := make([]float64, len(absF64s))
	for i, v := range absF64s {
		absF64s[i] = math.Abs(v)
		if v > 0 {
			backingGrad64[i] = 1
		} else {
			backingGrad64[i] = -1
		}
	}
	correctVecF64 := tensor.New(tensor.WithBacking(absF64s))
	gradF64s := tensor.New(tensor.WithBacking(backingGrad64))

	assert.True(ValueClose(correctVecF64, yV))
	assert.True(ValueClose(correctVecF64, bV))
	assert.True(ValueClose(gradF64s, xG), "xG %v", xG)
	assert.True(ValueClose(gradF64s, aG), "aG %v", aG)

	/* FLOAT32 Vector */

	x, y, a, v, bV, err = unaryOpTest(t, Float32, tensor.Shape{10}, Abs)
	if err != nil {
		t.Fatal(err)
	}

	yV = y.Value()
	if xG, err = x.Grad(); err != nil {
		t.Errorf("x has no grad: %v", err)
		return
	}

	if aG, err = a.Grad(); err != nil {
		t.Errorf("a has no grad: %v", err)
	}

	absF32s := v.Data().([]float32)
	backingGrad32 := make([]float32, len(absF32s))
	for i, v := range absF32s {
		absF32s[i] = math32.Abs(v)
		if v > 0 {
			backingGrad32[i] = 1
		} else {
			backingGrad32[i] = -1
		}
	}
	correctVecF32 := tensor.New(tensor.WithBacking(absF32s))
	gradF32s := tensor.New(tensor.WithBacking(backingGrad32))

	assert.True(ValueClose(correctVecF32, yV))
	assert.True(ValueClose(correctVecF32, bV), "bV %v", bV)
	assert.True(ValueClose(gradF32s, xG), "xG %v", xG)
	assert.True(ValueClose(gradF32s, aG), "aG %v", aG)

}

func TestSinDiff(t *testing.T) {
	assert := assert.New(t)
	v, x, _, xT, _, err := unaryOpDiffTest(sinOpType)
	if err != nil {
		t.Error(err)
	}

	correct := math.Cos(v)
	assert.Equal(correct, x.boundTo.(*dualValue).d.Data())

	// Tensor edition
	xdvd := xT.boundTo.(*dualValue).d.(*tensor.Dense)
	correctT := []float64{math.Cos(-v), math.Cos(v)}
	assert.Equal(correctT, xdvd.Data())
}

func TestCosDiff(t *testing.T) {
	assert := assert.New(t)

	v, x, _, xT, _, err := unaryOpDiffTest(cosOpType)
	if err != nil {
		t.Error(err)
	}

	assert.Equal(-math.Sin(v), x.boundTo.(*dualValue).d.Data())

	// Tensor edition
	xdvd := xT.boundTo.(*dualValue).d.(*tensor.Dense)
	correct := []float64{-math.Sin(-v), -math.Sin(v)}
	assert.Equal(correct, xdvd.Data())
}

func TestExpDiff(t *testing.T) {
	assert := assert.New(t)
	_, x, y, xT, yT, err := unaryOpDiffTest(expOpType)
	if err != nil {
		t.Error(err)
	}

	assert.Equal(y.boundTo.(*dualValue).Value, x.boundTo.(*dualValue).d)

	// Tensor edition
	xdvd := xT.boundTo.(*dualValue).d.(*tensor.Dense)
	ydvd := yT.boundTo.(*dualValue).Value.(*tensor.Dense)
	assert.Equal(ydvd.Data(), xdvd.Data())
}

func TestLnDiff(t *testing.T) {
	assert := assert.New(t)
	var err error
	v, x, _, xT, _, err := unaryOpDiffTest(lnOpType)
	if err != nil {
		t.Error(err)
	}
	correct := 1.0 / v
	assert.Equal(correct, x.boundTo.(*dualValue).d.Data(), "v was %v", v)

	// Tensor edition
	xdvd := xT.boundTo.(*dualValue).d.(*tensor.Dense)
	correctT := []float64{1.0 / -v, 1.0 / v}
	assert.Equal(correctT, xdvd.Data())
}

func TestLog2Diff(t *testing.T) {
	assert := assert.New(t)
	v, x, _, xT, _, err := unaryOpDiffTest(log2OpType)
	if err != nil {
		t.Error(err)
	}
	correct := 1.0 / (v * math.Ln2)
	assert.Equal(correct, x.boundTo.(*dualValue).d.Data())

	// Tensor edition
	xdvd := xT.boundTo.(*dualValue).d.(*tensor.Dense)
	correctT := []float64{1.0 / (-v * math.Ln2), 1.0 / (v * math.Ln2)}
	assert.Equal(correctT, xdvd.Data())
}

func TestSquareDiff(t *testing.T) {
	assert := assert.New(t)
	var err error
	v, x, _, xT, _, err := unaryOpDiffTest(squareOpType)
	if err != nil {
		t.Error(err)
	}

	assert.Equal(2*v, x.boundTo.(*dualValue).d.Data())

	// Tensor edition
	xdvd := xT.boundTo.(*dualValue).d.(*tensor.Dense)
	correct := []float64{2 * -v, 2 * v}
	assert.Equal(correct, xdvd.Data())
}

func TestSqrtDiff(t *testing.T) {
	assert := assert.New(t)
	v, x, _, xT, _, err := unaryOpDiffTest(sqrtOpType)
	if err != nil {
		t.Error(err)
	}

	assert.Equal(1.0/(2*math.Sqrt(v)), x.boundTo.(*dualValue).d.Data())

	// Tensor edition
	xdvd := xT.boundTo.(*dualValue).d.(*tensor.Dense)
	correct := []float64{1.0 / (2 * math.Sqrt(-v)), 1.0 / (2 * math.Sqrt(v))}
	got := xdvd.Data().([]float64)
	if !math.IsNaN(got[0]) && math.IsNaN(correct[0]) {
		t.Error("Expected NaN for the first value")
	}
	if got[1] != correct[1] {
		t.Error("Different second values")
	}
}

func TestInverseDiff(t *testing.T) {
	assert := assert.New(t)
	v, x, _, xT, _, err := unaryOpDiffTest(inverseOpType)
	if err != nil {
		t.Error(err)
	}

	correct := -((1 / v) * (1 / v))
	assert.Equal(correct, x.boundTo.(*dualValue).d.Data())

	// Tensor edition
	xdvd := xT.boundTo.(*dualValue).d.(*tensor.Dense)
	correctT := []float64{correct, correct}
	assert.Equal(correctT, xdvd.Data())
}

func TestCubeDiff(t *testing.T) {
	assert := assert.New(t)
	v, x, _, xT, _, err := unaryOpDiffTest(cubeOpType)
	if err != nil {
		t.Error(err)
	}

	correct := 3 * v * v
	xG, err := x.Grad()
	if err != nil {
		t.Error(err)
	}

	assert.True(closeF64(correct, extractF64(xG)), "%v != %v", xG, correct)

	// Tensor edition
	xdvd := xT.boundTo.(*dualValue).d
	correctT := []float64{correct, correct}
	assert.True(floatsEqual64(correctT, extractF64s(xdvd)))
}

func TestTanhDiff(t *testing.T) {
	assert := assert.New(t)
	v, x, _, xT, _, err := unaryOpDiffTest(tanhOpType)
	if err != nil {
		t.Error(err)
	}

	correct := 1.0 - (math.Tanh(v) * math.Tanh(v)) // I'm surprised Golang doesn't have a secant function!
	assert.Equal(correct, x.boundTo.(*dualValue).d.Data())

	// Tensor edition
	xdvd := xT.boundTo.(*dualValue).d.(*tensor.Dense)
	assert.Equal([]float64{correct, correct}, xdvd.Data())
}

func TestSigmoidDiff(t *testing.T) {
	assert := assert.New(t)
	v, x, _, xT, _, err := unaryOpDiffTest(sigmoidOpType)
	if err != nil {
		t.Error(err)
	}

	correct := math.Exp(-v) / ((1 + math.Exp(-v)) * (1 + math.Exp(-v)))
	xG := x.boundTo.(*dualValue).d
	assert.True(closeF64(correct, extractF64(xG)))

	// Tensor edition
	xdvd := xT.boundTo.(*dualValue).d
	negCorrect := math.Exp(v) / ((1 + math.Exp(v)) * (1 + math.Exp(v)))
	corrects := []float64{negCorrect, correct}
	assert.True(floatsEqual64(corrects, extractF64s(xdvd)))
}

func TestLog1pDiff(t *testing.T) {
	assert := assert.New(t)
	v, x, _, xT, _, err := unaryOpDiffTest(log1pOpType)
	if err != nil {
		t.Error(err)
	}

	correct := 1 / (1.0 + v)
	assert.Equal(correct, x.boundTo.(*dualValue).d.Data())

	// Tensor edition
	xdvd := xT.boundTo.(*dualValue).d.(*tensor.Dense)
	correct0 := 1 / (1.0 - v)
	assert.Equal([]float64{correct0, correct}, xdvd.Data())
}

func TestExpm1Diff(t *testing.T) {
	assert := assert.New(t)
	v, x, _, xT, _, err := unaryOpDiffTest(expm1OpType)
	if err != nil {
		t.Error(err)
	}

	correct := math.Exp(v)
	assert.Equal(correct, x.boundTo.(*dualValue).d.Data())

	// Tensor edition
	xdvd := xT.boundTo.(*dualValue).d.(*tensor.Dense)
	correct0 := math.Exp(-v)
	assert.Equal([]float64{correct0, correct}, xdvd.Data())
}

func TestSoftplus(t *testing.T) {
	defer runtime.GC()
	assert := assert.New(t)

	var x, y, a *Node
	var v Value
	var xV, yV, xG, bV, aG Value
	var err error

	/* FLOAT64 SCALAR */

	if x, y, a, v, bV, err = unaryOpTest(t, Float64, tensor.Shape{}, Softplus); err != nil {
		t.Fatal(err)
	}

	xV = x.Value()
	yV = y.Value()
	if xG, err = x.Grad(); err != nil {
		t.Errorf("x has no grad: %v", err)
		return
	}

	if aG, err = a.Grad(); err != nil {
		t.Errorf("a has no grad: %v", err)
	}

	correctVF64 := softplusf64(v.Data().(float64))
	correctDF64 := sigmoidf64(xV.Data().(float64))
	assert.True(ValueClose(newF64(correctVF64), yV))
	assert.True(ValueClose(newF64(correctVF64), bV))
	assert.True(ValueClose(newF64(correctDF64), xG))
	assert.True(ValueClose(newF64(correctDF64), aG))

	/* FLOAT32 SCALAR */

	if x, y, a, v, bV, err = unaryOpTest(t, Float32, tensor.Shape{}, Softplus); err != nil {
		t.Fatal(err)
	}

	xV = x.Value()
	yV = y.Value()
	if xG, err = x.Grad(); err != nil {
		t.Errorf("x has no grad: %v", err)
		return
	}

	if aG, err = a.Grad(); err != nil {
		t.Errorf("a has no grad: %v", err)
	}

	correctVF32 := softplusf32(v.Data().(float32))
	correctDF32 := sigmoidf32(xV.Data().(float32))
	assert.True(ValueClose(newF32(correctVF32), yV))
	assert.True(ValueClose(newF32(correctVF32), bV))
	assert.True(ValueClose(newF32(correctDF32), xG))
	assert.True(ValueClose(newF32(correctDF32), aG))

	/* FLOAT64 Vector */

	if x, y, a, v, bV, err = unaryOpTest(t, Float64, tensor.Shape{10}, Softplus); err != nil {
		t.Fatal(err)
	}

	xV = x.Value()
	yV = y.Value()
	if xG, err = x.Grad(); err != nil {
		t.Errorf("x has no grad: %v", err)
		return
	}

	if aG, err = a.Grad(); err != nil {
		t.Errorf("a has no grad: %v", err)
	}

	correctVF64s := v.Data().([]float64)
	correctDF64s := xV.Data().([]float64)

	for i, v := range correctVF64s {
		correctVF64s[i] = softplusf64(v)
		correctDF64s[i] = sigmoidf64(correctDF64s[i])
	}
	assert.True(floatsEqual64(correctVF64s, yV.Data().([]float64)))
	assert.True(floatsEqual64(correctVF64s, bV.Data().([]float64)))
	assert.True(floatsEqual64(correctDF64s, xG.Data().([]float64)))
	assert.True(floatsEqual64(correctDF64s, aG.Data().([]float64)))

	/* FLOAT32 Vector */

	if x, y, a, v, bV, err = unaryOpTest(t, Float32, tensor.Shape{10}, Softplus); err != nil {
		t.Fatal(err)
	}

	xV = x.Value()
	yV = y.Value()
	if xG, err = x.Grad(); err != nil {
		t.Errorf("x has no grad: %v", err)
		return
	}

	if aG, err = a.Grad(); err != nil {
		t.Errorf("a has no grad: %v", err)
	}

	correctVF32s := v.Data().([]float32)
	correctDF32s := xV.Data().([]float32)

	for i, v := range correctVF32s {
		correctVF32s[i] = softplusf32(v)
		correctDF32s[i] = sigmoidf32(correctDF32s[i])
	}
	assert.True(floatsEqual32(correctVF32s, yV.Data().([]float32)))
	assert.True(floatsEqual32(correctVF32s, bV.Data().([]float32)))
	assert.True(floatsEqual32(correctDF32s, xG.Data().([]float32)))
	assert.True(floatsEqual32(correctDF32s, aG.Data().([]float32)))
}
