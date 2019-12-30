package gorgonia

import (
	"fmt"
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestSumOpGrad(t *testing.T) {
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

func TestSumOpFakeVec(t *testing.T) {
	g := NewGraph()

	xv := tensor.New(tensor.WithBacking([]float64{1, 2}), tensor.WithShape(2, 1))
	yv := tensor.New(tensor.WithBacking([]float64{10, 20}), tensor.WithShape(1, 2))
	x := NewMatrix(g, Float64, WithName("x"), WithShape(2, 1), WithValue(xv))
	y := NewMatrix(g, Float64, WithName("y"), WithShape(1, 2), WithValue(yv))
	sx, _ := Sum(x)
	sy, _ := Sum(y)

	assert.True(t, sx.Shape().Eq(tensor.ScalarShape()))
	assert.True(t, sy.Shape().Eq(tensor.ScalarShape()))

	sx2, _ := Sum(x, 1)
	assert.True(t, sx2.Shape().Eq(tensor.Shape{2}))

	vm := NewTapeMachine(g)
	vm.RunAll()

	assert.Equal(t, 3.0, sx.Value().Data(), "Expected sx to be 3.0")
	assert.Equal(t, 30.0, sy.Value().Data(), "Expected sy to be 30.0")
	assert.Equal(t, []float64{1, 2}, sx2.Value().Data(), "sx2 should be a flat array")
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
	subTests := []reductionTest{
		{dt: Float32, inShape: []int{3, 2}, inData: []float32{1, 2, 3, 4, 5, 6}, op: Max, along: []int{0}, wantShape: []int{2}, wantData: []float32{5, 6}},
		{dt: Float32, inShape: []int{3, 2}, inData: []float32{1, 2, 3, 4, 5, 6}, op: Max, along: []int{1}, wantShape: []int{3}, wantData: []float32{2, 4, 6}},
		{dt: Float32, inShape: []int{3, 2}, inData: []float32{1, 2, 3, 4, 5, 6}, op: Max, along: []int{}, wantShape: []int{}, wantData: float32(6)},
		{dt: Float32, inShape: []int{3, 2}, inData: []float32{1, 2, 3, 4, 5, 6}, op: Max, along: []int{0, 1}, wantShape: []int{}, wantData: float32(6)},
		{dt: Float32, inShape: []int{3, 2}, inData: []float32{1, 2, 3, 4, 5, 6}, op: Max, along: []int{1, 0}, wantShape: []int{}, wantData: float32(6)},
		{
			dt:        Float32,
			inShape:   []int{2, 2, 2, 2},
			inData:    []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			op:        Max,
			along:     []int{0, 1, 2, 3},
			wantShape: []int{},
			wantData:  float32(16),
		},
		{
			dt:        Float32,
			inShape:   []int{2, 2, 2, 2},
			inData:    []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			op:        Max,
			along:     []int{},
			wantShape: []int{},
			wantData:  float32(16),
		},
		{
			dt:        Float32,
			inShape:   []int{2, 2, 2, 2},
			inData:    []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			op:        Max,
			along:     []int{0},
			wantShape: []int{1, 2, 2, 2},
			wantData:  []float32{9, 10, 11, 12, 13, 14, 15, 16},
		},
		{
			dt:        Float32,
			inShape:   []int{2, 2, 2, 2},
			inData:    []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			op:        Max,
			along:     []int{1},
			wantShape: []int{2, 1, 2, 2},
			wantData:  []float32{5, 6, 7, 8, 13, 14, 15, 16},
		},
		{
			dt:        Float32,
			inShape:   []int{2, 2, 2, 2},
			inData:    []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			op:        Max,
			along:     []int{2},
			wantShape: []int{2, 2, 1, 2},
			wantData:  []float32{3, 4, 7, 8, 11, 12, 15, 16},
		},
		{
			dt:        Float32,
			inShape:   []int{2, 2, 2, 2},
			inData:    []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			op:        Max,
			along:     []int{3},
			wantShape: []int{2, 2, 2, 1},
			wantData:  []float32{2, 4, 6, 8, 10, 12, 14, 16},
		},
		{
			dt:        Float32,
			inShape:   []int{2, 2, 2, 2},
			inData:    []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			op:        Max,
			along:     []int{1, 3},
			wantShape: []int{2, 1, 2, 1},
			wantData:  []float32{6, 8, 14, 16},
		},
		{
			dt:        Float32,
			inShape:   []int{2, 2, 2, 2},
			inData:    []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			op:        Max,
			along:     []int{0, 2, 3},
			wantShape: []int{2},
			wantData:  []float32{12, 16},
		},
	}

	for _, subTest := range subTests {
		t.Run(fmt.Sprintf("along %v", subTest.along), func(t *testing.T) {
			testReductionOp(t, subTest)
		})
	}
}

func TestSumOp(t *testing.T) {
	subTests := []reductionTest{
		{dt: Float32, inShape: []int{3, 2}, inData: []float32{1, 2, 3, 4, 5, 6}, op: Sum, along: []int{0}, wantShape: []int{2}, wantData: []float32{9, 12}},
		{dt: Float32, inShape: []int{3, 2}, inData: []float32{1, 2, 3, 4, 5, 6}, op: Sum, along: []int{1}, wantShape: []int{3}, wantData: []float32{3, 7, 11}},
		{dt: Float32, inShape: []int{3, 2}, inData: []float32{1, 2, 3, 4, 5, 6}, op: Sum, along: []int{}, wantShape: []int{}, wantData: float32(21)},
		{dt: Float32, inShape: []int{3, 2}, inData: []float32{1, 2, 3, 4, 5, 6}, op: Sum, along: []int{0, 1}, wantShape: []int{}, wantData: float32(21)},
		{dt: Float32, inShape: []int{3, 2}, inData: []float32{1, 2, 3, 4, 5, 6}, op: Sum, along: []int{1, 0}, wantShape: []int{}, wantData: float32(21)},
		{
			dt:        Float32,
			inShape:   []int{2, 2, 2, 2},
			inData:    []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			op:        Sum,
			along:     []int{0, 1, 2, 3},
			wantShape: []int{},
			wantData:  float32(136),
		},
		{
			dt:        Float32,
			inShape:   []int{2, 2, 2, 2},
			inData:    []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			op:        Sum,
			along:     []int{},
			wantShape: []int{},
			wantData:  float32(136),
		},
		{
			dt:        Float32,
			inShape:   []int{2, 2, 2, 2},
			inData:    []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			op:        Sum,
			along:     []int{0},
			wantShape: []int{1, 2, 2, 2},
			wantData:  []float32{10, 12, 14, 16, 18, 20, 22, 24},
		},
		{
			dt:        Float32,
			inShape:   []int{2, 2, 2, 2},
			inData:    []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			op:        Sum,
			along:     []int{1},
			wantShape: []int{2, 1, 2, 2},
			wantData:  []float32{6, 8, 10, 12, 22, 24, 26, 28},
		},
		{
			dt:        Float32,
			inShape:   []int{2, 2, 2, 2},
			inData:    []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			op:        Sum,
			along:     []int{2},
			wantShape: []int{2, 2, 1, 2},
			wantData:  []float32{4, 6, 12, 14, 20, 22, 28, 30},
		},
		{
			dt:        Float32,
			inShape:   []int{2, 2, 2, 2},
			inData:    []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			op:        Sum,
			along:     []int{3},
			wantShape: []int{2, 2, 2, 1},
			wantData:  []float32{3, 7, 11, 15, 19, 23, 27, 31},
		},
		{
			dt:        Float32,
			inShape:   []int{2, 2, 2, 2},
			inData:    []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			op:        Sum,
			along:     []int{1, 3},
			wantShape: []int{2, 1, 2, 1},
			wantData:  []float32{14, 22, 46, 54},
		},
		{
			dt:        Float32,
			inShape:   []int{2, 2, 2, 2},
			inData:    []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			op:        Sum,
			along:     []int{0, 2, 3},
			wantShape: []int{2},
			wantData:  []float32{52, 84},
		},
	}

	for _, subTest := range subTests {
		t.Run(fmt.Sprintf("along %v", subTest.along), func(t *testing.T) {
			testReductionOp(t, subTest)
		})
	}
}

type reductionTest struct {
	dt        tensor.Dtype
	inShape   tensor.Shape
	inData    interface{}
	op        func(*Node, ...int) (*Node, error)
	along     []int
	wantShape tensor.Shape
	wantData  interface{}
}

func testReductionOp(t *testing.T, test reductionTest) {
	g := NewGraph()
	Xn := NewTensor(g, test.dt, len(test.inShape), WithShape(test.inShape...))
	got := Must(test.op(Xn, test.along...))

	xT := tensor.New(tensor.WithShape(test.inShape...), tensor.WithBacking(test.inData))
	vm := NewTapeMachine(g)
	defer vm.Close()
	vm.Let(Xn, xT)
	err := vm.RunAll()
	if err != nil {
		t.Fatal(err)
	}
	assert := assert.New(t)
	assert.Equal(test.wantShape, got.Value().Shape(), "shape mismatch")
	assert.Equal(test.wantData, got.Value().Data(), "data mismatch")
}

// TestFollowupOp confirms that an element-wise binary op will work as expected after a sum/max.
// The underlying reduction on the tensor changes the number of dimensions, but the gorgonia node does not.
// We therefore confirm that the resulting nodes actually work.
func TestFollowupOp(t *testing.T) {
	g := NewGraph()
	Xn := NewTensor(g, tensor.Float64, 4, WithShape(2, 2, 2, 2), WithInit(RangedFrom(1)))
	mx := Must(Max(Xn, 1, 2))
	sx := Must(Sum(Xn, 1, 2))
	y := NewTensor(g, tensor.Float64, 4, WithShape(2, 1, 1, 2), WithInit(RangedFrom(1)))

	amx := Must(Add(mx, y))
	asx := Must(Add(sx, y))
	assert.Equal(t, amx.Shape(), tensor.Shape{2, 1, 1, 2})
	assert.Equal(t, asx.Shape(), tensor.Shape{2, 1, 1, 2})
	vm := NewTapeMachine(g)
	defer vm.Close()
	err := vm.RunAll()
	if err != nil {
		t.Error(err)
	}
	assert.Equal(t, []float64{8, 10, 18, 20}, amx.Value().Data(), "data mismatch")
	assert.Equal(t, []float64{17, 22, 51, 56}, asx.Value().Data(), "data mismatch")
}
