package gorgonia

import (
	"testing"

	tf64 "github.com/chewxy/gorgonia/tensor/f64"
	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/stretchr/testify/assert"
)

func TestRepeatOp(t *testing.T) {
	assert := assert.New(t)
	g := NewGraph()

	rep := NewScalarValue(2) // the number of times an axis will be repeated
	repN := NewScalar(g, Int, WithValue(rep))

	// test repeat tensor:

	// repeat on axis 1
	T := tf64.NewTensor(tf64.WithBacking([]float64{1, 2, 3, 4}), tf64.WithShape(2, 2))
	TT := FromTensor(T)
	TN := NewMatrix(g, Float64, WithValue(TT))

	repeat := newRepeatOp([]int{1}, Nodes{TN, repN})
	correct := tf64.NewTensor(tf64.WithBacking([]float64{1, 1, 2, 2, 3, 3, 4, 4}), tf64.WithShape(2, 4))

	res, err := repeat.Do(TT, rep)
	if err != nil {
		t.Error(err)
	}
	if !correct.Eq(res.(Tensor).Tensor) {
		t.Error("Something wrong has happend. Failed to repeat correctly")
	}

	// repeat on axis 0
	repeat = newRepeatOp([]int{0}, Nodes{TN, repN})
	correct = tf64.NewTensor(tf64.WithBacking([]float64{1, 2, 1, 2, 3, 4, 3, 4}), tf64.WithShape(4, 2))

	res, err = repeat.Do(TT, rep)
	if err != nil {
		t.Error(err)
	}
	if !correct.Eq(res.(Tensor).Tensor) {
		t.Error("Failed to repeat correctly")
	}

	// test repeat vector
	T = tf64.NewTensor(tf64.WithBacking([]float64{1, 2}), tf64.WithShape(2, 1))
	TT = FromTensor(T)
	TN = NewVector(g, Float64, WithValue(TT))

	// repeat on the 0th axis
	repeat = newRepeatOp([]int{0}, Nodes{TN, repN})
	correct = tf64.NewTensor(tf64.WithBacking([]float64{1, 1, 2, 2}), tf64.WithShape(4, 1))
	res, err = repeat.Do(TT, rep)
	if err != nil {
		t.Error(err)
	}
	if !correct.Eq(res.(Tensor).Tensor) {
		t.Error("Failed to repeat a vector correctly")
		t.Errorf("%v", res)
	}

	// repeat on the 1st axis
	repeat = newRepeatOp([]int{1}, Nodes{TN, repN})
	correct = tf64.NewTensor(tf64.WithBacking([]float64{1, 1, 2, 2}), tf64.WithShape(2, 2))
	res, err = repeat.Do(TT, rep)
	if err != nil {
		t.Error(err)
	}
	if !correct.Eq(res.(Tensor).Tensor) {
		t.Error("Failed to repeat a vector correctly")
		t.Errorf("%v", res)
	}

	// test repeat scalar
	s := NewScalarValue(3.1415)
	sn := NewScalar(g, Float64, WithValue(s))
	repeat = newRepeatOp([]int{0}, Nodes{sn, repN})
	correct = tf64.NewTensor(tf64.WithBacking([]float64{3.1415, 3.1415}))
	res, err = repeat.Do(s, rep)
	if err != nil {
		t.Error(err)
	}
	if !correct.Eq(res.(Tensor).Tensor) {
		t.Error("Failed to repeat a scalar correctly")
		t.Errorf("%v", res)
	}

	/* IDIOTS CHOICE AWARD */

	// impossible axes
	T = tf64.NewTensor(tf64.WithBacking([]float64{1, 2, 3, 4}), tf64.WithShape(2, 2))
	TT = FromTensor(T)
	repeat = newRepeatOp([]int{3}, Nodes{TN, repN})
	fails := func() { repeat.Do(TT, rep) }
	assert.Panics(fails)
}

func repeatOpDiff(repeatOn int, shape types.Shape, xV, yV interface{}) (g *ExprGraph, x, y *Node, err error) {
	g = NewGraph()
	switch shape.Dims() {
	case 0:
		x = NewScalar(g, Float64, WithName("x"))
	case 1:
		// vanilla vector
		x = NewVector(g, Float64, WithName("x"), WithShape(shape...))
	case 2:
		x = NewMatrix(g, Float64, WithName("x"), WithShape(shape...))
	default:
		//matrix and tensors
		x = NewTensor(g, Float64, shape.Dims(), WithName("x"), WithShape(shape...))
	}

	repN := NewScalar(g, Float64, WithValue(2.0))
	repeat := newRepeatOp([]int{repeatOn}, Nodes{x, repN})
	if y, err = applyOp(repeat, x, repN); err != nil {
		return
	}
	xVal, _ := anyToValue(xV)
	yVal, _ := anyToValue(yV)
	x.bind(dvUnit(xVal))
	y.bind(dvUnitVar(yVal))
	if err = repeat.DoDiff(Nodes{x, repN}, y); err != nil {
		return
	}
	return
}

func TestRepeatOpDoDiff(t *testing.T) {
	assert := assert.New(t)
	// var g *ExprGraph
	// var x, y, repN *Node
	// var repeat *repeatOp
	var x *Node
	var err error

	var xG Value
	var xT, yT *tf64.Tensor

	yT = tf64.NewTensor(tf64.WithShape(2), tf64.WithBacking([]float64{3.14, 3.14}))

	// scalar repeated into a vec/colvec
	if _, x, _, err = repeatOpDiff(0, scalarShape, 3.14, yT); err != nil {
		t.Fatal(err)
	}
	xG, _ = x.Grad()
	assert.Equal(2.0, extractF64(xG))

	// scalar repeated into a rowvec
	if _, x, _, err = repeatOpDiff(1, scalarShape, 3.14, yT); err != nil {
		t.Fatal(err)
	}
	xG, _ = x.Grad()
	assert.Equal(2.0, extractF64(xG))

	// vector repeated unto itself
	xT = tf64.NewTensor(tf64.WithShape(2), tf64.WithBacking([]float64{3.14, 3.14}))
	yT = tf64.NewTensor(tf64.WithShape(4), tf64.WithBacking([]float64{3.14, 3.14, 3.14, 3.14}))
	if _, x, _, err = repeatOpDiff(0, types.Shape{2}, xT, yT); err != nil {
		t.Fatal(err)
	}
	xG, _ = x.Grad()
	assert.Equal([]float64{2, 2}, extractF64s(xG))

	// colvec repeated unto itself
	xT = tf64.NewTensor(tf64.WithShape(2, 1), tf64.WithBacking([]float64{3.14, 3.14}))
	yT = tf64.NewTensor(tf64.WithShape(4, 1), tf64.WithBacking([]float64{3.14, 3.14, 3.14, 3.14}))
	if _, x, _, err = repeatOpDiff(0, types.Shape{2, 1}, xT, yT); err != nil {
		t.Fatal(err)
	}
	xG, _ = x.Grad()
	assert.Equal([]float64{2, 2}, extractF64s(xG))

	// rowvec repeated unto itself
	xT = tf64.NewTensor(tf64.WithShape(1, 2), tf64.WithBacking([]float64{3.14, 3.14}))
	yT = tf64.NewTensor(tf64.WithShape(1, 4), tf64.WithBacking([]float64{3.14, 3.14, 3.14, 3.14}))
	if _, x, _, err = repeatOpDiff(1, types.Shape{1, 2}, xT, yT); err != nil {
		t.Fatal(err)
	}
	xG, _ = x.Grad()
	assert.Equal([]float64{2, 2}, extractF64s(xG))

	// matrix on axis 0
	xT = tf64.NewTensor(tf64.WithShape(2, 2), tf64.WithBacking([]float64{3.14, 2.718, 1.618, 1.414}))
	yT = tf64.NewTensor(tf64.WithShape(4, 2), tf64.WithBacking([]float64{3.14, 2.718, 3.14, 2.718, 1.618, 1.414, 1.618, 1.414}))
	if _, x, _, err = repeatOpDiff(0, types.Shape{1, 2}, xT, yT); err != nil {
		t.Fatal(err)
	}
	xG, _ = x.Grad()
	assert.Equal([]float64{2, 2, 2, 2}, extractF64s(xG))

	// matrix on axis 1
	xT = tf64.NewTensor(tf64.WithShape(2, 2), tf64.WithBacking([]float64{3.14, 2.718, 1.618, 1.414}))
	yT = tf64.NewTensor(tf64.WithShape(4, 2), tf64.WithBacking([]float64{3.14, 2.718, 3.14, 2.718, 1.618, 1.414, 1.618, 1.414}))
	if _, x, _, err = repeatOpDiff(1, types.Shape{1, 2}, xT, yT); err != nil {
		t.Fatal(err)
	}
	xG, _ = x.Grad()
	assert.Equal([]float64{2, 2, 2, 2}, extractF64s(xG))

}

func TestSliceOp(t *testing.T) {
	assert := assert.New(t)
	var T *tf64.Tensor
	var TT Tensor
	var v Value
	var slice sliceOp
	var shape types.Shape
	var err error

	var n, done *Node
	var grads Nodes

	g := NewGraph()

	// T[0] -> Scalar
	T = tf64.NewTensor(tf64.WithShape(2), tf64.WithBacking([]float64{1, 2}))
	TT = FromTensor(T)
	slice = newSliceOp(S(0), 0, T.Dims())

	n = newNode(withGraph(g), withType(TT.Type()), WithShape(TT.Shape()...))
	if shape, err = slice.inferShape(TT.Type(), n); err != nil {
		t.Error(err)
	}

	assert.Equal(scalarShape, shape)

	if v, err = slice.Do(TT); err != nil {
		t.Fatal(err)
	}

	assert.Equal(1.0, extractF64(v))

	done = newNode(withGraph(g), withType(Float64), WithShape())

	if grads, err = slice.SymDiff(Nodes{n}, done, onef64); err != nil {
		t.Fatal(err)
	}
	assert.Equal(1, len(grads))
	assert.IsType(sliceIncrOp{}, grads[0].op)
	assert.Equal(2, len(grads[0].children))
	assert.Equal(n, grads[0].children[0])
	assert.Equal(onef64.Hashcode(), grads[0].children[1].Hashcode())

	// T[0] -> Scalar (again, but this time, with a colvec)
	T = tf64.NewTensor(tf64.WithShape(2, 1), tf64.WithBacking([]float64{1, 2}))
	TT = FromTensor(T)
	slice = newSliceOp(S(0), 0, T.Dims())

	n = newNode(withGraph(g), withType(TT.Type()), WithShape(TT.Shape()...))
	if shape, err = slice.inferShape(TT.Type(), n); err != nil {
		t.Error(err)
	}

	assert.Equal(scalarShape, shape)

	if v, err = slice.Do(TT); err != nil {
		t.Fatal(err)
	}

	assert.Equal(1.0, extractF64(v))

	// T[0] again, but this time, with a rowvec, and on axis 0
	T = tf64.NewTensor(tf64.WithShape(1, 2), tf64.WithBacking([]float64{1, 2}))
	TT = FromTensor(T)
	slice = newSliceOp(S(0), 0, T.Dims())

	n = newNode(withGraph(g), withType(TT.Type()), WithShape(TT.Shape()...))
	if shape, err = slice.inferShape(TT.Type(), n); err != nil {
		t.Error(err)
	}

	assert.Equal(types.Shape{2}, shape)

	if v, err = slice.Do(TT); err != nil {
		t.Fatal(err)
	}

	assert.Equal([]float64{1, 2}, extractF64s(v))

	// T[0] again, but this time, with a rowvec, this time along axis 1. this should yield a scalar
	T = tf64.NewTensor(tf64.WithShape(1, 2), tf64.WithBacking([]float64{1, 2}))
	TT = FromTensor(T)
	slice = newSliceOp(S(0), 1, T.Dims())

	n = newNode(withGraph(g), withType(TT.Type()), WithShape(TT.Shape()...))
	if shape, err = slice.inferShape(TT.Type(), n); err != nil {
		t.Error(err)
	}

	assert.Equal(types.Shape{2}, shape)

	if v, err = slice.Do(TT); err != nil {
		t.Fatal(err)
	}

	assert.Equal(1.0, extractF64(v))

	/*
		// T[:, 1:2, :]
		T = tf64.NewTensor(tf64.WithBacking(tf64.RangeFloat64(0, 24)), tf64.WithShape(2, 3, 4))
		TT = FromTensor(T)
		slice = newSliceOp(1, 2, 1, T.Dims())

		v, err = slice.Do(TT)
		if err != nil {
			t.Error(err)
		}

		vt, ok := v.(Tensor)
		if !ok {
			t.Error("Expected result to be a TensorType")
		}

		correctShape := types.Shape{2, 4}
		correctStride := []int{12, 1}
		correctData := tf64.RangeFloat64(4, 20)
		assert.Equal(correctShape, vt.Tensor.Shape())
		assert.Equal(correctStride, vt.Tensor.Strides())
		assert.Equal(correctData, vt.Data())

		backing := tf64.RandomFloat64(3 * 4)
		backingLW := make([]float64, len(backing))
		backingLD := make([]float64, len(backing))
		for i, v := range backing {
			backingLW[i] = v
			backingLD[i] = 0.0
		}
		G := lstm.NewGraph()
		L := new(lstm.Layer)
		L.W = mat64.NewDense(3, 4, backingLW)
		L.D = mat64.NewDense(3, 4, backingLD)
		pr := G.PluckRow(L, 1)

		T = tf64.NewTensor(tf64.WithBacking(backing), tf64.WithShape(3, 4))
		g := NewGraph()
		x := NewMatrix(g, Float64, WithValue(T), WithShape(3, 4))
		sl := Must(Slice(x, S(1)))

		machine := NewLispMachine(g)
		machine.dontExecBwd()
		err = machine.RunAll()
		if err != nil {
			t.Fatal(err)
		}
		vvv := sl.boundTo.(*dualValue).Value
		log.Printf("%#v \n%v", vvv.(Tensor), pr.W.RawMatrix().Data)
	*/
}

func TestSliceOpDiff(t *testing.T) {
	assert := assert.New(t)
	g := NewGraph()
	A := NewMatrix(g, Float64, WithShape(2, 2), WithInit(RangedFrom(0)))
	sli := Must(Slice(A, nil, S(1))) // A[:, 1]
	x := Must(Sum(Must(Mul(sli, twof64))))

	_, err := Grad(x, A)
	if err != nil {
		t.Error(err)
	}

	prog, locMap, err := Compile(g)
	if err != nil {
		t.Error(err)
	}

	machine := NewTapeMachine(prog, locMap)
	err = machine.RunAll()
	if err != nil {
		t.Error(err)
	}

	T := A.Value().(Tensor).Tensor
	aG, _ := A.Grad()

	G := aG.(Tensor).Tensor.(*tf64.Tensor)
	assert.NotEqual(T, G)

	correct := []float64{0, 2, 0, 2}
	assert.Equal(correct, G.Data())

	// t.Logf("Visual confirmation")
	// t.Logf("%+v", A.Value())
	// t.Logf("%+v", A.Grad())

	/* Lisp machine version */
	g2 := NewGraph()
	A = NewMatrix(g2, Float64, WithShape(2, 2), WithInit(RangedFrom(0)))
	sli = Must(Slice(A, nil, S(1))) // A[:, 1]
	x = Must(Sum(Must(Mul(sli, twof64))))

	m2 := NewLispMachine(g2)
	err = m2.RunAll()
	if err != nil {
		t.Error(err)
	}

	// t.Logf("Visual confirmation")
	// t.Logf("%+v", A.Value())
	// t.Logf("%+v", A.Grad())
}

func TestTransposeOp(t *testing.T) {
	assert := assert.New(t)
	g := NewGraph()
	A := NewMatrix(g, Float64, WithShape(2, 3), WithInit(RangedFrom(0)))
	AT := Must(Transpose(A))
	Must(Sum(AT))

	m := NewLispMachine(g)
	if err := m.RunAll(); err != nil {
		t.Error(err)
	}

	assert.Equal(types.Shape{3, 2}, AT.shape)
}
