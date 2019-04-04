package gorgonia

import (
	"fmt"
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestSumOp(t *testing.T) {
	assert := assert.New(t)
	// var g *ExprGraph
	var z, sz *Node
	var grads Nodes
	var err error
	var op sumOp

	_, _, _, z = simpleVecEqn()
	sz = Must(Sum(z))
	// t.Logf(" %v  %v %v %v", g, x, y, z)

	diffWRT := sz.diffWRT()
	assert.Equal([]bool{true}, diffWRT)

	op = sz.op.(sumOp)
	grads, err = op.SymDiff(Nodes{z}, sz, onef64)
	assert.Nil(err)
	assert.Equal(1, len(grads))
	t.Logf("%v", grads[0])

}

func TestSumOpDiff(t *testing.T) {
	defer runtime.GC()
	assert := assert.New(t)
	var g, g2 *ExprGraph
	var x, y, z, a, b, c *Node
	// var x, y, a, b *Node
	var xG, yG, aG, bG Value
	// var xG, aG Value
	// var prog *program
	// var locMap map[*Node]register
	var m *tapeMachine
	var m2 *lispMachine
	var err error

	// Basic Test case: a vector is summed

	g = NewGraph()
	x = NewVector(g, Float64, WithName("x"), WithShape(5), WithInit(RangedFrom(0)))
	y = Must(Sum(x))
	WithName("y")(y)

	Grad(y, x)

	// ioutil.WriteFile("SumOp.dot", []byte(g.ToDot()), 0644)

	m = NewTapeMachine(g)
	defer m.Close()
	if err = m.RunAll(); err != nil {
		t.Error(err)
	}

	g2 = NewGraph()
	a = NewVector(g2, Float64, WithShape(5), WithInit(RangedFrom(0)))
	b = Must(Sum(a))

	m2 = NewLispMachine(g2, WithWatchlist())
	defer m2.Close()
	if err = m2.RunAll(); err != nil {
		t.Error(err)
	}

	if aG, err = a.Grad(); err != nil {
		t.Error(err)
	}

	if xG, err = x.Grad(); err != nil {
		t.Error(err)
	}

	if bG, err = b.Grad(); err != nil {
		t.Error(err)
	}

	if yG, err = y.Grad(); err != nil {
		t.Error(err)
	}

	assert.True(ValueEq(x.Value(), a.Value()))
	assert.True(ValueEq(xG, aG))
	assert.True(ValueEq(y.Value(), b.Value()))
	assert.True(ValueEq(yG, bG))

	// long standing bug: sometimes the derivation will get executed in the machine first
	// for example, the deriv of y is 1, and occasionally, the machine will choose to
	// execute const 1 into register 0
	// It would then fail to bind to y's boundTo, because at that point in time, y is still unknown.

	// assert.Equal(y.Grad(), b.Grad())

	// Slightly more advanced test case: A matrix is summed
	g = NewGraph()
	x = NewMatrix(g, Float64, WithName("x"), WithShape(11, 7), WithInit(RangedFrom(0)))
	y = Must(Sum(x))
	WithName("y")(y)

	Grad(y, x)

	m = NewTapeMachine(g)
	defer m.Close()
	if err = m.RunAll(); err != nil {
		t.Error(err)
	}

	g2 = NewGraph()
	a = NewMatrix(g2, Float64, WithName("x"), WithShape(11, 7), WithInit(RangedFrom(0)))
	b = Must(Sum(a))

	m2 = NewLispMachine(g2)
	defer m2.Close()
	if err = m2.RunAll(); err != nil {
		t.Error(err)
	}

	if aG, err = a.Grad(); err != nil {
		t.Error(err)
	}

	if xG, err = x.Grad(); err != nil {
		t.Error(err)
	}
	if bG, err = b.Grad(); err != nil {
		t.Error(err)
	}

	if yG, err = y.Grad(); err != nil {
		t.Error(err)
	}
	assert.True(ValueEq(x.Value(), a.Value()))
	assert.True(ValueEq(xG, aG))
	assert.True(ValueEq(y.Value(), b.Value()))
	assert.True(ValueEq(yG, bG))

	/* Sum is not the root node */

	g = NewGraph()
	x = NewMatrix(g, Float64, WithName("x"), WithShape(11, 7), WithInit(RangedFrom(0)))
	y = Must(Sum(x))
	z = Must(Add(y, twof64))

	if _, err = Grad(z, x); err != nil {
		t.Fatal(err)
	}

	m = NewTapeMachine(g)
	defer m.Close()
	if err = m.RunAll(); err != nil {
		t.Errorf("%v", m.Prog())
		t.Error(err)
	}

	g2 = NewGraph()
	a = NewMatrix(g2, Float64, WithName("x"), WithShape(11, 7), WithInit(RangedFrom(0)))
	b = Must(Sum(a))
	c = Must(Add(b, twof64))

	m2 = NewLispMachine(g2)
	defer m2.Close()
	if err = m2.RunAll(); err != nil {
		t.Fatalf("%+v", err)
	}

	if aG, err = a.Grad(); err != nil {
		t.Error(err)
	}

	if xG, err = x.Grad(); err != nil {
		t.Error(err)
	}

	if bG, err = b.Grad(); err != nil {
		t.Error(err)
	}

	if yG, err = b.Grad(); err != nil {
		t.Error(err)
	}

	assert.True(ValueEq(x.Value(), a.Value()))
	assert.True(ValueEq(xG, aG))
	assert.True(ValueEq(y.Value(), b.Value()))
	assert.True(ValueEq(yG, bG))
	assert.True(ValueEq(z.Value(), c.Value()))

	runtime.GC()
}

func TestMaxOp(t *testing.T) {
	subTests := []struct {
		along    []int
		expected interface{}
	}{
		{along: []int{0}, expected: []float32{5, 6}},
		{along: []int{1}, expected: []float32{2, 4, 6}},
		{along: []int{}, expected: float32(6)},
		{along: []int{0, 1}, expected: float32(6)},
		{along: []int{1, 0}, expected: float32(6)},
	}

	dt := Float32
	for _, subTest := range subTests {
		t.Run(fmt.Sprintf("along %d", subTest.along), func(t *testing.T) {
			g := NewGraph()
			Xn := NewTensor(g, dt, 2, WithShape(2, 2))
			max := Must(Max(Xn, subTest.along...))
			// resShape, errShape := max.op.(*maxOp).InferShape(Xn.Shape())
			// fmt.Printf("along %d inferShape, %s", resShape, errShape)

			xT := tensor.New(tensor.WithShape(3, 2), tensor.WithBacking([]float32{
				1, 2,
				3, 4,
				5, 6}))
			vm := NewTapeMachine(g)
			defer vm.Close()
			vm.Let(Xn, xT)
			err := vm.RunAll()
			if err != nil {
				t.Error(err)
			}
			assert := assert.New(t)
			assert.Equal(subTest.expected, max.Value().(*tensor.Dense).Data())
		})
	}
}
