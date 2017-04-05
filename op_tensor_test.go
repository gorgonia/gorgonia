package gorgonia

import (
	"runtime"
	"testing"

	"github.com/chewxy/gorgonia/tensor"
	"github.com/stretchr/testify/assert"
)

var repeatOpTests = []struct {
	name string
	rep  int
	axes []int
	val  Value

	correct       Value
	expectedShape tensor.Shape
	err           bool
}{
	{
		"repeat matrix on axis 0", 2, []int{0},
		tensor.New(tensor.WithBacking([]float64{1, 2, 3, 4}), tensor.WithShape(2, 2)),
		tensor.New(tensor.WithBacking([]float64{1, 2, 1, 2, 3, 4, 3, 4}), tensor.WithShape(4, 2)),
		tensor.Shape{4, 2}, false,
	},

	{
		"repeat matrix on axis 1", 2, []int{1},
		tensor.New(tensor.WithBacking([]float64{1, 2, 3, 4}), tensor.WithShape(2, 2)),
		tensor.New(tensor.WithBacking([]float64{1, 1, 2, 2, 3, 3, 4, 4}), tensor.WithShape(2, 4)),
		tensor.Shape{2, 4}, false,
	},

	{
		"repeat col vec on axis 0", 2, []int{0},
		tensor.New(tensor.WithBacking([]float64{1, 2}), tensor.WithShape(2, 1)),
		tensor.New(tensor.WithBacking([]float64{1, 1, 2, 2}), tensor.WithShape(4, 1)),
		tensor.Shape{4, 1}, false,
	},

	{
		"repeat col vec on axis 1", 2, []int{1},
		tensor.New(tensor.WithBacking([]float64{1, 2}), tensor.WithShape(2, 1)),
		tensor.New(tensor.WithBacking([]float64{1, 1, 2, 2}), tensor.WithShape(2, 2)),
		tensor.Shape{2, 2}, false,
	},

	{
		"repeat row vec on axis 0", 2, []int{0},
		tensor.New(tensor.WithBacking([]float64{1, 2}), tensor.WithShape(1, 2)),
		tensor.New(tensor.WithBacking([]float64{1, 2, 1, 2}), tensor.WithShape(2, 2)),
		tensor.Shape{2, 2}, false,
	},

	{
		"repeat row vec on axis 1", 2, []int{1},
		tensor.New(tensor.WithBacking([]float64{1, 2}), tensor.WithShape(1, 2)),
		tensor.New(tensor.WithBacking([]float64{1, 1, 2, 2}), tensor.WithShape(1, 4)),
		tensor.Shape{1, 4}, false,
	},

	{
		"repeat vector on axis 0", 2, []int{0},
		tensor.New(tensor.WithBacking([]float64{1, 2}), tensor.WithShape(2)),
		tensor.New(tensor.WithBacking([]float64{1, 1, 2, 2}), tensor.WithShape(4)),
		tensor.Shape{4}, false,
	},

	{
		"repeat vector on axis 1", 2, []int{1},
		tensor.New(tensor.WithBacking([]float64{1, 2}), tensor.WithShape(2)),
		tensor.New(tensor.WithBacking([]float64{1, 1, 2, 2}), tensor.WithShape(2, 2)),
		tensor.Shape{2, 2}, false,
	},

	{
		"repeat scalar", 2, []int{0},
		newF64(3.14), tensor.New(tensor.WithBacking([]float64{3.14, 3.14}), tensor.WithShape(2)),
		tensor.Shape{2}, false,
	},
}

func TestRepeatOp(t *testing.T) {
	// assert := assert.New(t)

	for _, rots := range repeatOpTests {
		g := NewGraph()
		var res Value
		var err error
		var repeat *repeatOp

		rep := newI(rots.rep)
		repN := NodeFromAny(g, rep)
		n := NodeFromAny(g, rots.val)

		repeat = newRepeatOp(rots.axes, Nodes{n, repN})

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
		size := sizeOp{val: rots.rep}
		s, err = repeat.InferShape(rots.val.Shape(), size)
		switch {
		case rots.err:
			if err == nil {
				t.Error("Expected an error")
			}
			continue
		case !rots.err && err != nil:
			t.Errorf("%+v", err)
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

	repN := NewScalar(g, Float64, WithValue(2.0))
	repeat := newRepeatOp([]int{repeatOn}, Nodes{x, repN})
	if y, err = applyOp(repeat, x, repN); err != nil {
		return
	}
	xVal, _, _, _ := anyToValue(xV)
	yVal, _, _, _ := anyToValue(yV)
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
	var xT, yT *tensor.Dense

	yT = tensor.New(tensor.WithShape(2), tensor.WithBacking([]float64{3.14, 3.14}))

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
	Must(Sum(AT))

	m := NewLispMachine(g)
	if err := m.RunAll(); err != nil {
		t.Error(err)
	}

	assert.Equal(tensor.Shape{3, 2}, AT.shape)
}

func TestConcatOp(t *testing.T) {
	defer runtime.GC()

	assert := assert.New(t)
	g := NewGraph()
	x := NewVector(g, Float64, WithShape(2))
	xx, err := Concat(0, x, x)
	if err != nil {
		t.Fatalf("%+v", err)
	}

	cost := Must(Sum(xx))
	Grad(cost, x)

	g2 := NewGraph()
	a := NewVector(g2, Float64, WithShape(2))
	aa, err := Concat(0, a, a)
	if err != nil {
		t.Fatalf("%+v", err)
	}
	Must(Sum(aa)) // cost

	aBack := []float64{1, 2}
	aT := tensor.New(tensor.WithShape(2), tensor.WithBacking(aBack))

	xBack := []float64{1, 2}
	xT := tensor.New(tensor.WithShape(2), tensor.WithBacking(xBack))

	Let(a, aT)
	Let(x, xT)
	m1 := NewTapeMachine(g)
	m2 := NewLispMachine(g2)

	if err = m1.RunAll(); err != nil {
		t.Fatal(err)
	}

	if err = m2.RunAll(); err != nil {
		t.Fatalf("%+v", err)
	}

	xG, _ := x.Grad()
	aG, _ := a.Grad()
	assert.True(ValueEq(xG, aG))
	assert.True(ValueEq(xx.Value(), aa.Value()))

}
