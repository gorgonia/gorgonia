package gorgonia

import (
	"log"
	"testing"

	tf64 "github.com/chewxy/gorgonia/tensor/f64"
	"github.com/stretchr/testify/assert"
)

func matMulDiffTest(op linAlgBinOp, t *testing.T) (X, Y, Z, A, B, C *Node, err error) {
	// setup
	g := NewGraph()
	A = NewMatrix(g, Float64, WithShape(2, 2), WithName("A"))
	B = NewMatrix(g, Float64, WithShape(2, 2), WithName("B"))
	C = Must(Mul(A, B))

	X = NewMatrix(g, Float64, WithShape(2, 2), WithName("X"))
	Y = NewMatrix(g, Float64, WithShape(2, 2), WithName("Y"))
	Z = Must(Mul(X, Y))

	// aBack := Gaussian64(0.0, 0.08, 2, 2)
	// bBack := Gaussian64(0.0, 0.08, 2, 2)
	aBack := []float64{-2, -1, 1, 2}
	bBack := []float64{5, 4, 3, 2}
	aT := tf64.NewTensor(tf64.WithShape(2, 2), tf64.WithBacking(aBack))
	bT := tf64.NewTensor(tf64.WithShape(2, 2), tf64.WithBacking(bBack))

	// autodiff
	Let(A, aT)
	Let(B, bT)
	var cT Value
	cT, err = op.Do(A.boundTo, B.boundTo)
	if err != nil {
		return
	}
	cTdv := variableDV(cT)

	if err = C.bind(cTdv); err != nil {
		return
	}

	if err = A.bind(dvUnit(A.boundTo)); err != nil {
		return
	}

	if err = B.bind(dvUnit(B.boundTo)); err != nil {
		return
	}

	if err = matMulDiff(op.transA, op.transB, A, B, C); err != nil {
		return
	}

	// symdiff
	Z.op = op
	ns, err := op.SymDiff(Nodes{X, Y}, Z, onef64)
	if err != nil {
		return
	}
	dZdX := ns[0]
	dZdY := ns[1]

	dZdX.derivOf = Nodes{X}
	dZdY.derivOf = Nodes{Y}
	X.deriv = dZdX
	Y.deriv = dZdY

	// run the whole graph
	sg := g.SubgraphRoots(dZdX, dZdY)
	prog, locMap, err := CompileFunctionNEW(sg, Nodes{X, Y}, ns)
	if err != nil {
		return
	}

	m := NewTapeMachine(prog, locMap)
	Let(X, aT)
	Let(Y, bT)
	err = m.RunAll()
	if err != nil {
		log.Printf("prog: %v", prog)
		log.Printf("Tapemachine error")
		return
	}
	return
}

// check that two implementations (autodiff and symdiff) yields the same result
func TestMatMulDiffImplEquality(t *testing.T) {
	assert := assert.New(t)
	var op linAlgBinOp
	var X, Y, Z, A, B, C *Node
	var xG, yG, aG, bG Value
	var err error

	// mul, false, false
	op = linAlgBinOp{matMulOperator, false, false}
	X, Y, Z, A, B, C, err = matMulDiffTest(op, t)
	if err != nil {
		t.Fatal(err)
	}

	if xG, err = X.Grad(); err != nil {
		t.Error(err)
	}
	if yG, err = Y.Grad(); err != nil {
		t.Error(err)
	}
	if aG, err = A.Grad(); err != nil {
		t.Error(err)
	}
	if bG, err = B.Grad(); err != nil {
		t.Error(err)
	}

	assert.Equal(X.Value(), A.Value(), "\nE: %+#3.3s\nG: %+#3.3s", X.Value(), A.Value())
	assert.Equal(Y.Value(), B.Value(), "\nE: %+#3.3s\nG: %+#3.3s", Y.Value(), B.Value())
	assert.Equal(Z.Value(), C.Value(), "\nE: %+#3.3s\nG: %+#3.3s", Z.Value(), C.Value())
	assert.Equal(xG, aG)
	assert.Equal(yG, bG)

	op = linAlgBinOp{matMulOperator, true, false}
	X, Y, Z, A, B, C, err = matMulDiffTest(op, t)
	if err != nil {
		t.Error(err)
	}
	if xG, err = X.Grad(); err != nil {
		t.Error(err)
	}
	if yG, err = Y.Grad(); err != nil {
		t.Error(err)
	}
	if aG, err = A.Grad(); err != nil {
		t.Error(err)
	}
	if bG, err = B.Grad(); err != nil {
		t.Error(err)
	}
	assert.Equal(X.Value(), A.Value(), "\nE: %+#3.3s\nG: %+#3.3s", X.Value(), A.Value())
	assert.Equal(Y.Value(), B.Value(), "\nE: %+#3.3s\nG: %+#3.3s", Y.Value(), B.Value())
	assert.Equal(Z.Value(), C.Value(), "\nE: %+#3.3s\nG: %+#3.3s", Z.Value(), C.Value())
	assert.Equal(xG, aG)
	assert.Equal(yG, bG)

	op = linAlgBinOp{matMulOperator, false, true}
	X, Y, Z, A, B, C, err = matMulDiffTest(op, t)
	if err != nil {
		t.Error(err)
	}
	if xG, err = X.Grad(); err != nil {
		t.Error(err)
	}
	if yG, err = Y.Grad(); err != nil {
		t.Error(err)
	}
	if aG, err = A.Grad(); err != nil {
		t.Error(err)
	}
	if bG, err = B.Grad(); err != nil {
		t.Error(err)
	}
	assert.Equal(X.Value(), A.Value(), "\nE: %+#3.3s\nG: %+#3.3s", X.Value(), A.Value())
	assert.Equal(Y.Value(), B.Value(), "\nE: %+#3.3s\nG: %+#3.3s", Y.Value(), B.Value())
	assert.Equal(Z.Value(), C.Value(), "\nE: %+#3.3s\nG: %+#3.3s", Z.Value(), C.Value())
	assert.Equal(xG, aG)
	assert.Equal(yG, bG)

	op = linAlgBinOp{matMulOperator, true, true}
	X, Y, Z, A, B, C, err = matMulDiffTest(op, t)
	if err != nil {
		t.Error(err)
	}
	if xG, err = X.Grad(); err != nil {
		t.Error(err)
	}
	if yG, err = Y.Grad(); err != nil {
		t.Error(err)
	}
	if aG, err = A.Grad(); err != nil {
		t.Error(err)
	}
	if bG, err = B.Grad(); err != nil {
		t.Error(err)
	}
	assert.Equal(X.Value(), A.Value(), "\nE: %+#3.3s\nG: %+#3.3s", X.Value(), A.Value())
	assert.Equal(Y.Value(), B.Value(), "\nE: %+#3.3s\nG: %+#3.3s", Y.Value(), B.Value())
	assert.Equal(Z.Value(), C.Value(), "\nE: %+#3.3s\nG: %+#3.3s", Z.Value(), C.Value())
	assert.Equal(xG, aG)
	assert.Equal(yG, bG)
}
