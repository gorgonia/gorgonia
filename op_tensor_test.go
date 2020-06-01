package gorgonia

import (
	"fmt"
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

var repeatOpTests = []struct {
	name string
	rep  int
	axes int
	val  Value

	correct       Value
	expectedShape tensor.Shape
	err           bool
}{
	{
		"repeat matrix on axis 0", 2, 0,
		tensor.New(tensor.WithBacking([]float64{1, 2, 3, 4}), tensor.WithShape(2, 2)),
		tensor.New(tensor.WithBacking([]float64{1, 2, 1, 2, 3, 4, 3, 4}), tensor.WithShape(4, 2)),
		tensor.Shape{4, 2}, false,
	},

	{
		"repeat matrix on axis 1", 2, 1,
		tensor.New(tensor.WithBacking([]float64{1, 2, 3, 4}), tensor.WithShape(2, 2)),
		tensor.New(tensor.WithBacking([]float64{1, 1, 2, 2, 3, 3, 4, 4}), tensor.WithShape(2, 4)),
		tensor.Shape{2, 4}, false,
	},

	{
		"repeat col vec on axis 0", 2, 0,
		tensor.New(tensor.WithBacking([]float64{1, 2}), tensor.WithShape(2, 1)),
		tensor.New(tensor.WithBacking([]float64{1, 1, 2, 2}), tensor.WithShape(4, 1)),
		tensor.Shape{4, 1}, false,
	},

	{
		"repeat col vec on axis 1", 2, 1,
		tensor.New(tensor.WithBacking([]float64{1, 2}), tensor.WithShape(2, 1)),
		tensor.New(tensor.WithBacking([]float64{1, 1, 2, 2}), tensor.WithShape(2, 2)),
		tensor.Shape{2, 2}, false,
	},

	{
		"repeat row vec on axis 0", 2, 0,
		tensor.New(tensor.WithBacking([]float64{1, 2}), tensor.WithShape(1, 2)),
		tensor.New(tensor.WithBacking([]float64{1, 2, 1, 2}), tensor.WithShape(2, 2)),
		tensor.Shape{2, 2}, false,
	},

	{
		"repeat row vec on axis 1", 2, 1,
		tensor.New(tensor.WithBacking([]float64{1, 2}), tensor.WithShape(1, 2)),
		tensor.New(tensor.WithBacking([]float64{1, 1, 2, 2}), tensor.WithShape(1, 4)),
		tensor.Shape{1, 4}, false,
	},

	{
		"repeat vector on axis 0", 2, 0,
		tensor.New(tensor.WithBacking([]float64{1, 2}), tensor.WithShape(2)),
		tensor.New(tensor.WithBacking([]float64{1, 1, 2, 2}), tensor.WithShape(4)),
		tensor.Shape{4}, false,
	},

	{
		"repeat vector on axis 1", 2, 1,
		tensor.New(tensor.WithBacking([]float64{1, 2}), tensor.WithShape(2)),
		tensor.New(tensor.WithBacking([]float64{1, 1, 2, 2}), tensor.WithShape(2, 2)),
		tensor.Shape{2, 2}, false,
	},

	{
		"repeat scalar", 2, 0,
		newF64(3.14), tensor.New(tensor.WithBacking([]float64{3.14, 3.14}), tensor.WithShape(2)),
		tensor.Shape{2}, false,
	},
}

func TestRepeatOp(t *testing.T) {
	// assert := assert.New(t)

	for _, rots := range repeatOpTests {
		// if rots.name != "repeat matrix on axis 1" {
		// 	continue
		// }
		g := NewGraph()
		var res Value
		var err error
		var repeat *repeatOp

		rep := newI(rots.rep)
		n := NodeFromAny(g, rots.val)

		repeat = newRepeatOp(rots.axes, n)

		res, err = repeat.Do(rots.val, rep)
		switch {
		case rots.err:
			if err == nil {
				t.Errorf("Test %q: Expected an error", rots.name)
			}
			goto infershape
		case !rots.err && err != nil:
			t.Errorf("%+v", err)
			goto infershape
		}

		if !ValueEq(res, rots.correct) {
			t.Errorf("Test %q: Expected %v. Got %v", rots.name, rots.correct, res)
		}

	infershape:
		var s tensor.Shape
		size := sizeOp{axis: rots.axes, val: rots.rep}
		s, err = repeat.InferShape(rots.val.Shape(), size)
		switch {
		case rots.err:
			if err == nil {
				t.Error("Expected an error")
			}
			continue
		case !rots.err && err != nil:
			t.Errorf("Test %q %+v", rots.name, err)
			continue
		}

		if !rots.expectedShape.Eq(s) {
			t.Errorf("Test %q InferShape: Expected %v. Got %v instead", rots.name, rots.expectedShape, s)
		}
	}
}

func repeatOpDiff(repeatOn int, shape tensor.Shape, xV, yV interface{}) (g *ExprGraph, x, y *Node, err error) {
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

	repOp := sizeOp{axis: repeatOn, val: 2}
	repN := NewScalar(g, Float64, WithName("REPCONST"), WithOp(repOp), WithValue(2.0))
	repeat := newRepeatOp(repeatOn, x)

	if y, err = ApplyOp(repeat, x, repN); err != nil {
		return
	}
	xVal, _, _, _ := anyToValue(xV)
	yVal, _, _, _ := anyToValue(yV)
	x.bind(dvUnit(xVal))
	y.bind(dvUnitVar(yVal))
	if err = repeat.DoDiff(ExecutionContext{}, Nodes{x, repN}, y); err != nil {
		return
	}
	return
}

func TestRepeatOpDoDiff(t *testing.T) {
	//t.SkipNow()
	assert := assert.New(t)
	// var g *ExprGraph
	// var x, y, repN *Node
	// var repeat *repeatOp
	var x *Node
	var err error

	var xG Value
	var xT, yT *tensor.Dense

	yT = tensor.New(tensor.WithShape(2), tensor.WithBacking([]float64{3.14, 3.14}))

	// scalar repeated into a vec/colvec
	if _, x, _, err = repeatOpDiff(0, scalarShape, 3.14, yT); err != nil {
		t.Fatal(err)
	}
	xG, _ = x.Grad()
	assert.Equal(2.0, extractF64(xG))

	// scalar repeated into a rowvec
	// if _, x, _, err = repeatOpDiff(1, scalarShape, 3.14, yT); err != nil {
	// 	t.Fatal(err)
	// }
	// xG, _ = x.Grad()
	// assert.Equal(2.0, extractF64(xG))

	// vector repeated unto itself
	xT = tensor.New(tensor.WithShape(2), tensor.WithBacking([]float64{3.14, 3.14}))
	yT = tensor.New(tensor.WithShape(4), tensor.WithBacking([]float64{3.14, 3.14, 3.14, 3.14}))
	if _, x, _, err = repeatOpDiff(0, tensor.Shape{2}, xT, yT); err != nil {
		t.Fatal(err)
	}
	xG, _ = x.Grad()
	assert.Equal([]float64{2, 2}, extractF64s(xG))

	// colvec repeated unto itself
	xT = tensor.New(tensor.WithShape(2, 1), tensor.WithBacking([]float64{3.14, 3.14}))
	yT = tensor.New(tensor.WithShape(4, 1), tensor.WithBacking([]float64{3.14, 3.14, 3.14, 3.14}))
	if _, x, _, err = repeatOpDiff(0, tensor.Shape{2}, xT, yT); err != nil {
		t.Fatal(err)
	}
	xG, _ = x.Grad()
	assert.Equal([]float64{2, 2}, extractF64s(xG))

	// rowvec repeated unto itself
	xT = tensor.New(tensor.WithShape(1, 2), tensor.WithBacking([]float64{3.14, 3.14}))
	yT = tensor.New(tensor.WithShape(1, 4), tensor.WithBacking([]float64{3.14, 3.14, 3.14, 3.14}))
	if _, x, _, err = repeatOpDiff(1, tensor.Shape{1, 2}, xT, yT); err != nil {
		t.Fatal(err)
	}
	xG, _ = x.Grad()
	assert.Equal([]float64{2, 2}, extractF64s(xG))

	// matrix on axis 0
	xT = tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{3.14, 2.718, 1.618, 1.414}))
	yT = tensor.New(tensor.WithShape(4, 2), tensor.WithBacking([]float64{3.14, 2.718, 3.14, 2.718, 1.618, 1.414, 1.618, 1.414}))
	if _, x, _, err = repeatOpDiff(0, tensor.Shape{1, 2}, xT, yT); err != nil {
		t.Fatal(err)
	}
	xG, _ = x.Grad()
	assert.Equal([]float64{2, 2, 2, 2}, extractF64s(xG))

	// matrix on axis 1
	xT = tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{3.14, 2.718, 1.618, 1.414}))
	yT = tensor.New(tensor.WithShape(4, 2), tensor.WithBacking([]float64{3.14, 2.718, 3.14, 2.718, 1.618, 1.414, 1.618, 1.414}))
	if _, x, _, err = repeatOpDiff(1, tensor.Shape{1, 2}, xT, yT); err != nil {
		t.Fatal(err)
	}
	xG, _ = x.Grad()
	assert.Equal([]float64{2, 2, 2, 2}, extractF64s(xG))

}

func TestTransposeOp(t *testing.T) {
	assert := assert.New(t)
	g := NewGraph()
	A := NewMatrix(g, Float64, WithShape(2, 3), WithInit(RangedFrom(0)))
	AT := Must(Transpose(A))
	cost1 := Must(Sum(AT))

	var m VM
	var err error

	m = NewLispMachine(g)
	defer m.Close()
	if err = m.RunAll(); err != nil {
		t.Error(err)
	}

	assert.Equal(tensor.Shape{3, 2}, AT.shape)

	h := NewGraph()
	B := NewMatrix(h, Float64, WithShape(2, 3), WithInit(RangedFrom(0)))
	BT := Must(Transpose(B))
	cost2 := Must(Sum(BT))
	Grad(cost2, B)

	m = NewTapeMachine(h)
	defer m.Close()
	if err = m.RunAll(); err != nil {
		t.Error(err)
	}
	assert.Equal(tensor.Shape{3, 2}, BT.shape)

	var ag, bg Value
	if ag, err = A.Grad(); err != nil {
		t.Fatalf("Cannot get grad of A. Err: %v", err)
	}

	if bg, err = B.Grad(); err != nil {
		t.Fatalf("Cannot get grad of B. Err: %v", err)
	}

	var costGrad1, costGrad2 Value
	if costGrad1, err = cost1.Grad(); err != nil {
		t.Fatalf("Cannot get grad of Cost1. Err %v", err)
	}

	if costGrad2, err = cost2.Grad(); err != nil {
		t.Fatalf("Cannot get grad of Cost2. Err %v", err)
	}

	t.Logf("%v %v", cost1.Value(), cost2.Value())
	t.Logf("%v %v", costGrad1, costGrad2)

	assert.True(ValueEq(ag, bg))
}

type concatOpTest struct {
	name string
	axes int
	vals []Value

	correct Value
}

var concatOpTests = []concatOpTest{
	{
		"concat 2 vectors",
		0,
		[]Value{
			tensor.New(tensor.WithBacking([]float64{1, 2}), tensor.WithShape(2)),
			tensor.New(tensor.WithBacking([]float64{3, 4}), tensor.WithShape(2)),
		},
		tensor.New(tensor.WithBacking([]float64{1, 2, 3, 4}), tensor.WithShape(4)),
	},
	{
		"concat 2 matrices on dim with size of 1",
		1,
		[]Value{
			tensor.New(tensor.WithBacking([]float64{1, 2}), tensor.WithShape(2, 1)),
			tensor.New(tensor.WithBacking([]float64{3, 4}), tensor.WithShape(2, 1)),
		},
		tensor.New(tensor.WithBacking([]float64{1, 3, 2, 4}), tensor.WithShape(2, 2)),
	},
}

func TestConcatOp(t *testing.T) {
	for _, cot := range concatOpTests {
		t.Run(cot.name, func(t *testing.T) {
			testConcatOp(t, cot)
		})
	}
}

func testConcatOp(t *testing.T, cot concatOpTest) {
	defer runtime.GC()

	as := assert.New(t)
	g1 := NewGraph()
	g2 := NewGraph()

	var n1, n2 Nodes
	for i, v := range cot.vals {
		n1 = append(n1, NodeFromAny(g1, v, WithName(fmt.Sprintf("n1_%d", i))))
		n2 = append(n2, NodeFromAny(g2, v, WithName(fmt.Sprintf("n2_%d", i))))
	}

	xx, err := Concat(cot.axes, n1...)
	if err != nil {
		t.Fatalf("%+v", err)
	}
	aa, err := Concat(cot.axes, n2...)
	if err != nil {
		t.Fatalf("%+v", err)
	}

	cost1 := Must(Sum(xx))
	_, err = Grad(cost1, n1...)
	if err != nil {
		t.Fatalf("%+v", err)
	}
	Must(Sum(aa)) // cost

	m1 := NewTapeMachine(g1)
	if err = m1.RunAll(); err != nil {
		t.Fatal(err)
	}
	defer m1.Close()
	m2 := NewLispMachine(g2)
	if err = m2.RunAll(); err != nil {
		t.Fatal(err)
	}
	defer m2.Close()

	xG, err := xx.Grad()
	if err != nil {
		t.Fatalf("%+v", err)
	}
	aG, err := aa.Grad()
	if err != nil {
		t.Fatalf("%+v", err)
	}

	as.True(ValueEq(cot.correct, xx.Value()))
	as.True(ValueEq(xG, aG))

	// Grads must have the same shapes as the input values
	for i, v := range cot.vals {
		g1, err := n1[i].Grad()
		if err != nil {
			t.Fatalf("%+v", err)
		}
		as.Equal(v.Shape(), g1.Shape())

		g2, err := n1[i].Grad()
		if err != nil {
			t.Fatalf("%+v", err)
		}
		as.Equal(v.Shape(), g2.Shape())
	}
}

func TestUnconcatConcatOpSequence(t *testing.T) {
	defer runtime.GC()

	as := assert.New(t)
	g := NewGraph()

	x := NewTensor(g, tensor.Float64, 3, WithShape(2, 3, 3), WithName("x"), WithValue(tensor.New(tensor.WithShape(2, 3, 3), tensor.WithBacking([]float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,

		10, 11, 12,
		13, 14, 15,
		16, 17, 18,
	}))))

	ux, err := Unconcat(x, 2, 3)
	if err != nil {
		t.Fatalf("%+v", err)
	}

	as.Len(ux, 3)

	for i, n := range ux {
		ux[i], err = Reshape(n, tensor.Shape{2, 3, 1})
		if err != nil {
			t.Fatalf("%+v", err)
		}
	}

	cx, err := Concat(2, ux...)
	if err != nil {
		t.Fatalf("%+v", err)
	}

	cost := Must(Sum(cx))
	_, err = Grad(cost, x)
	if err != nil {
		t.Fatalf("%+v", err)
	}

	m := NewTapeMachine(g)
	if err = m.RunAll(); err != nil {
		t.Fatalf("%+v", err)
	}
	defer m.Close()

	xG, err := x.Grad()
	if err != nil {
		t.Fatalf("%+v", err)
	}

	as.True(ValueEq(x.Value(), cx.Value()))
	as.Equal([]float64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, xG.Data())
	as.True(tensor.Shape{2, 3, 3}.Eq(xG.Shape()))
}
