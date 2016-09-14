package gorgonia

import (
	"math/rand"
	"testing"

	tf64 "github.com/chewxy/gorgonia/tensor/f64"
	"github.com/stretchr/testify/assert"
)

func ssBinOpDiffTest(op ʘBinaryOperatorType) (randX, randY float64, x, y, z *Node, err error) {
	_, x, y, z = simpleEqn()

	randX = rand.ExpFloat64()
	randY = rand.ExpFloat64()

	binOp := newElemBinOp(op, x, y)
	diff := ʘBinOpDiffFns[op]

	Let(x, randX)
	Let(y, randY)

	var v Value
	if v, err = binOp.Do(x.boundTo, y.boundTo); err != nil {
		return
	}
	zdv := variableDV(v)

	if err = z.bind(zdv); err != nil {
		return
	}

	if err = x.bind(dvUnit(x.boundTo)); err != nil {
		return
	}

	if err = y.bind(dvUnit(y.boundTo)); err != nil {
		return
	}

	err = diff(x, y, z)
	if err != nil {
		return
	}
	return
}

func stBinOpDiffTest(op ʘBinaryOperatorType) (randX, randY float64, x, yT, zT *Node, err error) {
	var g *ExprGraph
	g, _, yT, zT = simpleVecEqn()
	x = NewScalar(g, Float64, WithName("x"))

	randX = rand.ExpFloat64()
	randY = rand.ExpFloat64()

	binOp := newElemBinOp(op, x, yT)
	diff := ʘBinOpDiffFns[op]

	yBack := []float64{-randY, randY}
	Let(x, randX)
	Let(yT, tf64.NewTensor(tf64.WithShape(2, 1), tf64.WithBacking(yBack)))

	var v Value
	if v, err = binOp.Do(x.boundTo, yT.boundTo); err != nil {
		return
	}
	zdv := variableDV(v)

	if err = zT.bind(zdv); err != nil {
		return
	}

	if err = x.bind(dvUnit(x.boundTo)); err != nil {
		return
	}

	if err = yT.bind(dvUnit(yT.boundTo)); err != nil {
		return
	}

	err = diff(x, yT, zT)
	if err != nil {
		return
	}
	return
}

func tsBinOpDiffTest(op ʘBinaryOperatorType) (randX, randY float64, xT, y, zT *Node, err error) {
	var g *ExprGraph
	g, xT, _, zT = simpleVecEqn()
	y = NewScalar(g, Float64, WithName("y"))

	randX = rand.ExpFloat64()
	randY = rand.ExpFloat64()

	binOp := newElemBinOp(op, xT, y)
	diff := ʘBinOpDiffFns[op]

	xBack := []float64{randX, -randX}
	Let(xT, tf64.NewTensor(tf64.WithShape(2, 1), tf64.WithBacking(xBack)))
	Let(y, randY)

	var v Value
	if v, err = binOp.Do(xT.boundTo, y.boundTo); err != nil {
		return
	}
	zdv := variableDV(v)

	if err = zT.bind(zdv); err != nil {
		return
	}

	if err = xT.bind(dvUnit(xT.boundTo)); err != nil {
		return
	}

	if err = y.bind(dvUnit(y.boundTo)); err != nil {
		return
	}

	err = diff(xT, y, zT)
	if err != nil {
		return
	}
	return
}

func ttBinOpDiffTest(op ʘBinaryOperatorType) (randX, randY float64, x, y, z *Node, err error) {
	_, x, y, z = simpleVecEqn()

	randX = rand.ExpFloat64()
	randY = rand.ExpFloat64()

	binOp := newElemBinOp(op, x, y)
	diff := ʘBinOpDiffFns[op]

	xBack := []float64{-randX, randX}
	yBack := []float64{randY, -randY}
	Let(x, tf64.NewTensor(tf64.WithShape(2, 1), tf64.WithBacking(xBack)))
	Let(y, tf64.NewTensor(tf64.WithShape(2, 1), tf64.WithBacking(yBack)))

	var v Value
	if v, err = binOp.Do(x.boundTo, y.boundTo); err != nil {
		return
	}
	zdv := variableDV(v)

	if err = z.bind(zdv); err != nil {
		return
	}

	if err = x.bind(dvUnit(x.boundTo)); err != nil {
		return
	}

	if err = y.bind(dvUnit(y.boundTo)); err != nil {
		return
	}

	err = diff(x, y, z)
	if err != nil {
		return
	}
	return
}

func TestAddDiff(t *testing.T) {
	assert := assert.New(t)
	op := addOpType

	// s $ s version
	_, _, x, y, z, err := ssBinOpDiffTest(op)
	if err != nil {
		t.Error(err)
	}
	xdv := x.boundTo.(*dualValue)
	ydv := y.boundTo.(*dualValue)
	zdv := z.boundTo.(*dualValue)

	assert.Equal(zdv.d, xdv.d)
	assert.Equal(zdv.d, ydv.d)

	// s $ T ver
	_, _, x, y, z, err = stBinOpDiffTest(op)
	xdv = x.boundTo.(*dualValue)
	ydv = y.boundTo.(*dualValue)
	zdv = z.boundTo.(*dualValue)

	assert.Equal(zdv.d, xdv.d)
	assert.Equal(zdv.d, ydv.d)

	// T $ s ver
	_, _, x, y, z, err = tsBinOpDiffTest(op)
	xdv = x.boundTo.(*dualValue)
	ydv = y.boundTo.(*dualValue)
	zdv = z.boundTo.(*dualValue)

	assert.Equal(zdv.d, xdv.d)
	assert.Equal(zdv.d, ydv.d)

	// T $ T ver
	_, _, x, y, z, err = ttBinOpDiffTest(op)
	xdv = x.boundTo.(*dualValue)
	ydv = y.boundTo.(*dualValue)
	zdv = z.boundTo.(*dualValue)

	assert.Equal(zdv.d, xdv.d)
	assert.Equal(zdv.d, ydv.d)

}

func TestSubDiff(t *testing.T) {
	assert := assert.New(t)
	op := subOpType

	// s $ s version
	_, _, x, y, z, err := ssBinOpDiffTest(op)
	if err != nil {
		t.Error(err)
	}
	xdv := x.boundTo.(*dualValue)
	ydv := y.boundTo.(*dualValue)
	zdv := z.boundTo.(*dualValue)

	neg := newElemUnaryOp(negOpType, z)
	dzdy, _ := neg.Do(zdv.d)

	assert.Equal(zdv.d, xdv.d)
	assert.Equal(dzdy, ydv.d)

	// s $ T version
	_, _, x, y, z, err = stBinOpDiffTest(op)
	if err != nil {
		t.Error(err)
	}
	xdv = x.boundTo.(*dualValue)
	ydv = y.boundTo.(*dualValue)
	zdv = z.boundTo.(*dualValue)

	neg = newElemUnaryOp(negOpType, z)
	dzdy, _ = neg.Do(zdv.d)

	assert.Equal(zdv.d, xdv.d)
	assert.Equal(dzdy, ydv.d)

	// T $ s version
	_, _, x, y, z, err = tsBinOpDiffTest(op)
	if err != nil {
		t.Error(err)
	}
	xdv = x.boundTo.(*dualValue)
	ydv = y.boundTo.(*dualValue)
	zdv = z.boundTo.(*dualValue)

	neg = newElemUnaryOp(negOpType, z)
	dzdy, _ = neg.Do(zdv.d)

	assert.Equal(zdv.d, xdv.d)
	assert.Equal(dzdy, ydv.d)

	// T $ T version
	_, _, x, y, z, err = ttBinOpDiffTest(op)
	if err != nil {
		t.Error(err)
	}
	xdv = x.boundTo.(*dualValue)
	ydv = y.boundTo.(*dualValue)
	zdv = z.boundTo.(*dualValue)

	neg = newElemUnaryOp(negOpType, z)
	dzdy, _ = neg.Do(zdv.d)

	assert.Equal(zdv.d, xdv.d)
	assert.Equal(dzdy, ydv.d)

}

func TestHadamardProdDiff(t *testing.T) {
	assert := assert.New(t)
	op := mulOpType

	// s $ s version
	_, _, x, y, _, err := ssBinOpDiffTest(op)
	if err != nil {
		t.Error(err)
	}
	xdv := x.boundTo.(*dualValue)
	ydv := y.boundTo.(*dualValue)

	assert.Equal(ydv.Value, xdv.d)
	assert.Equal(xdv.Value, ydv.d)

	// s $ T version
	var randX float64
	randX, _, x, y, _, err = stBinOpDiffTest(op)
	if err != nil {
		t.Error(err)
	}
	xdv = x.boundTo.(*dualValue)
	ydv = y.boundTo.(*dualValue)

	dzdy, err := anyToValue(tf64.NewTensor(tf64.WithShape(2, 1), tf64.WithBacking([]float64{randX, randX})))
	if err != nil {
		t.Error(err)
	}
	assert.Equal(ydv.Value, xdv.d)
	assert.Equal(dzdy, ydv.d)

	// T $ s version
	var randY float64
	_, randY, x, y, _, err = tsBinOpDiffTest(op)
	if err != nil {
		t.Error(err)
	}
	xdv = x.boundTo.(*dualValue)
	ydv = y.boundTo.(*dualValue)

	dzdx, err := anyToValue(tf64.NewTensor(tf64.WithShape(2, 1), tf64.WithBacking([]float64{randY, randY})))
	if err != nil {
		t.Error(err)
	}

	assert.Equal(dzdx, xdv.d)
	assert.Equal(xdv.Value, ydv.d)

	// T $ T version
	_, _, x, y, _, err = ttBinOpDiffTest(op)
	if err != nil {
		t.Error(err)
	}
	xdv = x.boundTo.(*dualValue)
	ydv = y.boundTo.(*dualValue)

	assert.Equal(ydv.Value, xdv.d)
	assert.Equal(xdv.Value, ydv.d)
}

func TestHadamardDivDiff(t *testing.T) {
	assert := assert.New(t)
	op := divOpType

	// s $ s version
	randX, randY, x, y, _, err := ssBinOpDiffTest(op)
	if err != nil {
		t.Error(err)
	}
	xdv := x.boundTo.(*dualValue)
	ydv := y.boundTo.(*dualValue)

	randZ := randX / randY

	assert.True(floatEquals(1.0/randY, extractF64(xdv.d)))
	assert.True(floatEquals(-randZ/randY, extractF64(ydv.d)))

	// s $ T version
	randX, randY, x, y, _, err = stBinOpDiffTest(op)
	if err != nil {
		t.Error(err)
	}
	xdv = x.boundTo.(*dualValue)
	ydv = y.boundTo.(*dualValue)

	randZs := []float64{randX / -randY, randX / randY}
	dzdx := []float64{1 / -randY, 1 / randY}
	dzdy := []float64{-randZs[0] / -randY, -randZs[1] / randY}

	assert.True(floatsEqual(dzdx, extractF64s(xdv.d)))
	assert.True(floatsEqual(dzdy, extractF64s(ydv.d)))

	// T $ s version
	randX, randY, x, y, _, err = tsBinOpDiffTest(op)
	if err != nil {
		t.Error(err)
	}
	xdv = x.boundTo.(*dualValue)
	ydv = y.boundTo.(*dualValue)

	randZs = []float64{randX / randY, -randX / randY}
	dzdx = []float64{1 / randY, 1 / randY}
	dzdy = []float64{-randZs[0] / randY, -randZs[1] / randY}

	assert.True(floatsEqual(dzdx, extractF64s(xdv.d)))
	assert.True(floatsEqual(dzdy, extractF64s(ydv.d)))

	// T $ T version
	randX, randY, x, y, _, err = ttBinOpDiffTest(op)
	if err != nil {
		t.Error(err)
	}
	xdv = x.boundTo.(*dualValue)
	ydv = y.boundTo.(*dualValue)

	randZs = []float64{-randX / randY, randX / -randY}
	dzdx = []float64{1 / randY, 1 / -randY}
	dzdy = []float64{-randZs[0] / randY, -randZs[1] / -randY}

	assert.True(floatsEqual(dzdx, extractF64s(xdv.d)))
	assert.True(floatsEqual(dzdy, extractF64s(ydv.d)))

}
