package gorgonia

import (
	"math"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"

	tf64 "github.com/chewxy/gorgonia/tensor/f64"
)

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
	Let(xT, tf64.NewTensor(tf64.WithShape(2, 1), tf64.WithBacking(xBack)))
	vT, _, _, _ := anyToValue(tf64.NewTensor(tf64.WithShape(2, 1), tf64.WithBacking(yBack)))
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

func TestAbsDiff(t *testing.T) {
	assert := assert.New(t)
	_, x, _, xT, _, err := unaryOpDiffTest(absOpType)
	if err != nil {
		t.Error(err)
	}

	assert.Equal(1.0, x.boundTo.(*dualValue).d.(Scalar).Any())

	// Tensor edition
	xdvd := xT.boundTo.(*dualValue).d.(*tf64.Tensor)
	assert.Equal([]float64{-1, 1}, xdvd.Data())
}

func TestSinDiff(t *testing.T) {
	assert := assert.New(t)
	v, x, _, xT, _, err := unaryOpDiffTest(sinOpType)
	if err != nil {
		t.Error(err)
	}

	correct := math.Cos(v)
	assert.Equal(correct, x.boundTo.(*dualValue).d.(Scalar).Any())

	// Tensor edition
	xdvd := xT.boundTo.(*dualValue).d.(*tf64.Tensor)
	correctT := []float64{math.Cos(-v), math.Cos(v)}
	assert.Equal(correctT, xdvd.Data())
}

func TestCosDiff(t *testing.T) {
	assert := assert.New(t)

	v, x, _, xT, _, err := unaryOpDiffTest(cosOpType)
	if err != nil {
		t.Error(err)
	}

	assert.Equal(-math.Sin(v), x.boundTo.(*dualValue).d.(Scalar).Any())

	// Tensor edition
	xdvd := xT.boundTo.(*dualValue).d.(*tf64.Tensor)
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
	xdvd := xT.boundTo.(*dualValue).d.(*tf64.Tensor)
	ydvd := yT.boundTo.(*dualValue).Value.(*tf64.Tensor)
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
	assert.Equal(correct, x.boundTo.(*dualValue).d.(Scalar).Any(), "v was %v", v)

	// Tensor edition
	xdvd := xT.boundTo.(*dualValue).d.(*tf64.Tensor)
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
	assert.Equal(correct, x.boundTo.(*dualValue).d.(Scalar).Any())

	// Tensor edition
	xdvd := xT.boundTo.(*dualValue).d.(*tf64.Tensor)
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

	assert.Equal(2*v, x.boundTo.(*dualValue).d.(Scalar).Any())

	// Tensor edition
	xdvd := xT.boundTo.(*dualValue).d.(*tf64.Tensor)
	correct := []float64{2 * -v, 2 * v}
	assert.Equal(correct, xdvd.Data())
}

func TestSqrtDiff(t *testing.T) {
	assert := assert.New(t)
	v, x, _, xT, _, err := unaryOpDiffTest(sqrtOpType)
	if err != nil {
		t.Error(err)
	}

	assert.Equal(1.0/(2*math.Sqrt(v)), x.boundTo.(*dualValue).d.(Scalar).Any())

	// Tensor edition
	xdvd := xT.boundTo.(*dualValue).d.(*tf64.Tensor)
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
	assert.Equal(correct, x.boundTo.(*dualValue).d.(Scalar).Any())

	// Tensor edition
	xdvd := xT.boundTo.(*dualValue).d.(*tf64.Tensor)
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
	assert.True(floatEquals(correct, extractF64(xG)), "%v != %v", xG, correct)

	// Tensor edition
	xdvd := xT.boundTo.(*dualValue).d
	correctT := []float64{correct, correct}
	assert.True(floatsEqual(correctT, extractF64s(xdvd)))
}

func TestTanhDiff(t *testing.T) {
	assert := assert.New(t)
	v, x, _, xT, _, err := unaryOpDiffTest(tanhOpType)
	if err != nil {
		t.Error(err)
	}

	correct := 1.0 - (math.Tanh(v) * math.Tanh(v)) // I'm surprised Golang doesn't have a secant function!
	assert.Equal(correct, x.boundTo.(*dualValue).d.(Scalar).Any())

	// Tensor edition
	xdvd := xT.boundTo.(*dualValue).d.(*tf64.Tensor)
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
	assert.True(floatEquals(correct, extractF64(xG)))

	// Tensor edition
	xdvd := xT.boundTo.(*dualValue).d
	negCorrect := math.Exp(v) / ((1 + math.Exp(v)) * (1 + math.Exp(v)))
	corrects := []float64{negCorrect, correct}
	assert.True(floatsEqual(corrects, extractF64s(xdvd)))
}

func TestLog1pDiff(t *testing.T) {
	assert := assert.New(t)
	v, x, _, xT, _, err := unaryOpDiffTest(log1pOpType)
	if err != nil {
		t.Error(err)
	}

	correct := 1 / (1.0 + v)
	assert.Equal(correct, x.boundTo.(*dualValue).d.(Scalar).Any())

	// Tensor edition
	xdvd := xT.boundTo.(*dualValue).d.(*tf64.Tensor)
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
	assert.Equal(correct, x.boundTo.(*dualValue).d.(Scalar).Any())

	// Tensor edition
	xdvd := xT.boundTo.(*dualValue).d.(*tf64.Tensor)
	correct0 := math.Exp(-v)
	assert.Equal([]float64{correct0, correct}, xdvd.Data())
}

func TestSoftplusDiff(t *testing.T) {
	assert := assert.New(t)
	v, x, _, xT, _, err := unaryOpDiffTest(softplusOpType)
	if err != nil {
		t.Error(err)
	}

	correct := sigmoidf64(v)
	assert.Equal(correct, x.boundTo.(*dualValue).d.(Scalar).Any())

	// Tensor edition
	xdvd := xT.boundTo.(*dualValue).d.(*tf64.Tensor)
	correct0 := sigmoidf64(-v)
	assert.Equal([]float64{correct0, correct}, xdvd.Data())
}
