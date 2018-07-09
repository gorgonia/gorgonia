package gorgonia

import (
	"log"
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

type binOpTest struct {
	binOp func(*Node, *Node) (*Node, error)
	a, b  Value

	correct       Value
	correctDerivA Value
	correctDerivB Value
	correctShape  tensor.Shape
}

var binOpTests = []binOpTest{

	{Add,
		tensor.New(tensor.WithBacking([]float64{1, 2, 3, 4})),
		tensor.New(tensor.WithBacking([]float64{1, 2, 3, 4})),

		tensor.New(tensor.WithBacking([]float64{2, 4, 6, 8})),
		tensor.New(tensor.WithBacking([]float64{1, 1, 1, 1})),
		tensor.New(tensor.WithBacking([]float64{1, 1, 1, 1})),
		tensor.Shape{4},
	},

	{Add,
		tensor.New(tensor.WithBacking([]float64{1, 2, 3, 4})),
		newF64(1.0),

		tensor.New(tensor.WithBacking([]float64{2, 3, 4, 5})),
		tensor.New(tensor.WithBacking([]float64{1, 1, 1, 1})),
		newF64(4.0),
		tensor.Shape{4},
	},

	{Add,
		newF64(1.0),
		tensor.New(tensor.WithBacking([]float64{1, 2, 3, 4})),

		tensor.New(tensor.WithBacking([]float64{2, 3, 4, 5})),
		newF64(4.0),
		tensor.New(tensor.WithBacking([]float64{1, 1, 1, 1})),
		tensor.Shape{4},
	},

	{Add,
		newF64(1.0),
		newF64(1.0),

		newF64(2.0),
		newF64(1.0),
		newF64(1.0),
		scalarShape,
	},

	{Sub,
		tensor.New(tensor.WithBacking([]float64{1, 2, 3, 4})),
		tensor.New(tensor.WithBacking([]float64{1, 2, 3, 4})),

		tensor.New(tensor.WithBacking([]float64{0, 0, 0, 0})),
		tensor.New(tensor.WithBacking([]float64{1, 1, 1, 1})),
		tensor.New(tensor.WithBacking([]float64{-1, -1, -1, -1})),
		tensor.Shape{4},
	},

	{Sub,
		tensor.New(tensor.WithBacking([]float64{1, 2, 3, 4})),
		newF64(1.0),

		tensor.New(tensor.WithBacking([]float64{0, 1, 2, 3})),
		tensor.New(tensor.WithBacking([]float64{1, 1, 1, 1})),
		newF64(-4.0),
		tensor.Shape{4},
	},

	{Sub,
		newF64(1.0),
		tensor.New(tensor.WithBacking([]float64{1, 2, 3, 4})),

		tensor.New(tensor.WithBacking([]float64{0, -1, -2, -3})),
		newF64(4.0),
		tensor.New(tensor.WithBacking([]float64{-1, -1, -1, -1})),
		tensor.Shape{4},
	},

	{Sub,
		newF64(1.0),
		newF64(1.0),

		newF64(0.0),
		newF64(1.0),
		newF64(-1.0),
		scalarShape,
	},

	{HadamardProd,
		tensor.New(tensor.WithBacking([]float64{1, 2, 3, 4})),
		tensor.New(tensor.WithBacking([]float64{1, 2, 3, 4})),

		tensor.New(tensor.WithBacking([]float64{1, 4, 9, 16})),
		tensor.New(tensor.WithBacking([]float64{1, 2, 3, 4})),
		tensor.New(tensor.WithBacking([]float64{1, 2, 3, 4})),
		tensor.Shape{4},
	},

	{Mul,
		tensor.New(tensor.WithBacking([]float64{1, 2, 3, 4})),
		newF64(1.0),

		tensor.New(tensor.WithBacking([]float64{1, 2, 3, 4})),
		tensor.New(tensor.WithBacking([]float64{1, 1, 1, 1})),
		newF64(10),
		tensor.Shape{4},
	},

	{Mul,
		newF64(1.0),
		tensor.New(tensor.WithBacking([]float64{1, 2, 3, 4})),

		tensor.New(tensor.WithBacking([]float64{1, 2, 3, 4})),
		newF64(10),
		tensor.New(tensor.WithBacking([]float64{1, 1, 1, 1})),
		tensor.Shape{4},
	},

	{Mul,
		newF64(1.0),
		newF64(1.0),

		newF64(1.0),
		newF64(1.0),
		newF64(1.0),
		scalarShape,
	},

	{HadamardDiv,
		tensor.New(tensor.WithBacking([]float64{1, 2, 3, 4})),
		tensor.New(tensor.WithBacking([]float64{1, 1, 1, 1})),

		tensor.New(tensor.WithBacking([]float64{1, 2, 3, 4})),
		tensor.New(tensor.WithBacking([]float64{1, 1, 1, 1})),
		tensor.New(tensor.WithBacking([]float64{-1, -2, -3, -4})),
		tensor.Shape{4},
	},

	{Div,
		tensor.New(tensor.WithBacking([]float64{1, 2, 3, 4})),
		newF64(1.0),

		tensor.New(tensor.WithBacking([]float64{1, 2, 3, 4})),
		tensor.New(tensor.WithBacking([]float64{1, 1, 1, 1})),
		newF64(-10),
		tensor.Shape{4},
	},

	{Div,
		newF64(1),
		tensor.New(tensor.WithBacking([]float64{1, 1, 1, 1})),

		tensor.New(tensor.WithBacking([]float64{1, 1, 1, 1})),
		newF64(4),
		tensor.New(tensor.WithBacking([]float64{-1, -1, -1, -1})),
		tensor.Shape{4},
	},

	{Div,
		newF64(1.0),
		newF64(1.0),

		newF64(1.0),
		newF64(1.0),
		newF64(-1.0),
		scalarShape,
	},

	// Float32

	{Add,
		tensor.New(tensor.WithBacking([]float32{1, 2, 3, 4})),
		tensor.New(tensor.WithBacking([]float32{1, 2, 3, 4})),

		tensor.New(tensor.WithBacking([]float32{2, 4, 6, 8})),
		tensor.New(tensor.WithBacking([]float32{1, 1, 1, 1})),
		tensor.New(tensor.WithBacking([]float32{1, 1, 1, 1})),
		tensor.Shape{4},
	},

	{Add,
		tensor.New(tensor.WithBacking([]float32{1, 2, 3, 4})),
		newF32(1.0),

		tensor.New(tensor.WithBacking([]float32{2, 3, 4, 5})),
		tensor.New(tensor.WithBacking([]float32{1, 1, 1, 1})),
		newF32(4.0),
		tensor.Shape{4},
	},

	{Add,
		newF32(1.0),
		tensor.New(tensor.WithBacking([]float32{1, 2, 3, 4})),

		tensor.New(tensor.WithBacking([]float32{2, 3, 4, 5})),
		newF32(4.0),
		tensor.New(tensor.WithBacking([]float32{1, 1, 1, 1})),
		tensor.Shape{4},
	},

	{Add,
		newF32(1.0),
		newF32(1.0),

		newF32(2.0),
		newF32(1.0),
		newF32(1.0),
		scalarShape,
	},

	{Sub,
		tensor.New(tensor.WithBacking([]float32{1, 2, 3, 4})),
		tensor.New(tensor.WithBacking([]float32{1, 2, 3, 4})),

		tensor.New(tensor.WithBacking([]float32{0, 0, 0, 0})),
		tensor.New(tensor.WithBacking([]float32{1, 1, 1, 1})),
		tensor.New(tensor.WithBacking([]float32{-1, -1, -1, -1})),
		tensor.Shape{4},
	},

	{Sub,
		tensor.New(tensor.WithBacking([]float32{1, 2, 3, 4})),
		newF32(1.0),

		tensor.New(tensor.WithBacking([]float32{0, 1, 2, 3})),
		tensor.New(tensor.WithBacking([]float32{1, 1, 1, 1})),
		newF32(-4.0),
		tensor.Shape{4},
	},

	{Sub,
		newF32(1.0),
		tensor.New(tensor.WithBacking([]float32{1, 2, 3, 4})),

		tensor.New(tensor.WithBacking([]float32{0, -1, -2, -3})),
		newF32(4.0),
		tensor.New(tensor.WithBacking([]float32{-1, -1, -1, -1})),
		tensor.Shape{4},
	},

	{Sub,
		newF32(1.0),
		newF32(1.0),

		newF32(0.0),
		newF32(1.0),
		newF32(-1.0),
		scalarShape,
	},

	{HadamardProd,
		tensor.New(tensor.WithBacking([]float32{1, 2, 3, 4})),
		tensor.New(tensor.WithBacking([]float32{1, 2, 3, 4})),

		tensor.New(tensor.WithBacking([]float32{1, 4, 9, 16})),
		tensor.New(tensor.WithBacking([]float32{1, 2, 3, 4})),
		tensor.New(tensor.WithBacking([]float32{1, 2, 3, 4})),
		tensor.Shape{4},
	},

	{Mul,
		tensor.New(tensor.WithBacking([]float32{1, 2, 3, 4})),
		newF32(1.0),

		tensor.New(tensor.WithBacking([]float32{1, 2, 3, 4})),
		tensor.New(tensor.WithBacking([]float32{1, 1, 1, 1})),
		newF32(10),
		tensor.Shape{4},
	},

	{Mul,
		newF32(1.0),
		tensor.New(tensor.WithBacking([]float32{1, 2, 3, 4})),

		tensor.New(tensor.WithBacking([]float32{1, 2, 3, 4})),
		newF32(10),
		tensor.New(tensor.WithBacking([]float32{1, 1, 1, 1})),
		tensor.Shape{4},
	},

	{Mul,
		newF32(1.0),
		newF32(1.0),

		newF32(1.0),
		newF32(1.0),
		newF32(1.0),
		scalarShape,
	},

	{HadamardDiv,
		tensor.New(tensor.WithBacking([]float32{1, 2, 3, 4})),
		tensor.New(tensor.WithBacking([]float32{1, 1, 1, 1})),

		tensor.New(tensor.WithBacking([]float32{1, 2, 3, 4})),
		tensor.New(tensor.WithBacking([]float32{1, 1, 1, 1})),
		tensor.New(tensor.WithBacking([]float32{-1, -2, -3, -4})),
		tensor.Shape{4},
	},

	{Div,
		tensor.New(tensor.WithBacking([]float32{1, 2, 3, 4})),
		newF32(1.0),

		tensor.New(tensor.WithBacking([]float32{1, 2, 3, 4})),
		tensor.New(tensor.WithBacking([]float32{1, 1, 1, 1})),
		newF32(-10),
		tensor.Shape{4},
	},

	{Div,
		newF32(1),
		tensor.New(tensor.WithBacking([]float32{1, 1, 1, 1})),

		tensor.New(tensor.WithBacking([]float32{1, 1, 1, 1})),
		newF32(4),
		tensor.New(tensor.WithBacking([]float32{-1, -1, -1, -1})),
		tensor.Shape{4},
	},

	{Div,
		newF32(1.0),
		newF32(1.0),

		newF32(1.0),
		newF32(1.0),
		newF32(-1.0),
		scalarShape,
	},
}

func TestBasicArithmetic(t *testing.T) {
	assert := assert.New(t)
	for i, bot := range binOpTests {
		// log.Printf("TEST %d", i)
		// if i != 22 {
		// 	continue
		// }
		g := NewGraph()
		xV, _ := CloneValue(bot.a)
		yV, _ := CloneValue(bot.b)
		x := NodeFromAny(g, xV, WithName("x"))
		y := NodeFromAny(g, yV, WithName("y"))

		var ret *Node
		var retVal Value
		var err error
		if ret, err = bot.binOp(x, y); err != nil {
			t.Errorf("Test %d: %v", i, err)
			continue
		}
		Read(ret, &retVal)

		cost := Must(Sum(ret))
		var grads Nodes
		if grads, err = Grad(cost, x, y); err != nil {
			t.Errorf("Test %d: error while symbolic op: %v", i, err)
			continue
		}

		m1 := NewTapeMachine(g)
		// log.Printf("%v", m1.Prog())
		if err = m1.RunAll(); err != nil {
			log.Printf("m.buf \n%v", m1.buf.String())
			t.Errorf("Test %d: error while running %v", i, err)
			runtime.GC()
			continue
		}

		as := newAssertState(assert)
		as.Equal(bot.correct.Data(), retVal.Data(), "Test %d result", i)
		as.True(bot.correctShape.Eq(ret.Shape()))
		as.Equal(2, len(grads))
		as.Equal(bot.correctDerivA.Data(), grads[0].Value().Data(), "Test %v xgrad", i)
		as.Equal(bot.correctDerivB.Data(), grads[1].Value().Data(), "Test %v ygrad. Expected %v. Got %v", i, bot.correctDerivB, grads[1].Value())
		if !as.cont {
			prog := m1.Prog()
			t.Logf("Test %d failed. Prog: %v", i, prog)
			t.Logf("Exec Log \n%v", m1.buf.String())
		}

		if assertGraphEngine(t, g, stdengType); t.Failed() {
			t.Errorf("BasicArithmetic - TapeMachine failure")
			t.FailNow()
		}
		m1.Close()
		runtime.GC()
	}

	for i, bot := range binOpTests {
		// log.Printf("i: %d", i)
		// if i != 1 {
		// 	continue
		// }
		g := NewGraph()
		xV, _ := CloneValue(bot.a)
		yV, _ := CloneValue(bot.b)
		x := NodeFromAny(g, xV, WithName("x"))
		y := NodeFromAny(g, yV, WithName("y"))

		var ret *Node
		var retVal Value
		var err error
		if ret, err = bot.binOp(x, y); err != nil {
			t.Errorf("Test %d: %v", i, err)
			continue
		}
		Read(ret, &retVal)

		if xV.Shape().IsScalar() && yV.Shape().IsScalar() {

		} else {
			Must(Sum(ret))
		}

		m1 := NewLispMachine(g)
		if err = m1.RunAll(); err != nil {
			t.Errorf("Test %d: error while running %+v", i, err)
			runtime.GC()
			continue
		}

		as := newAssertState(assert)
		as.Equal(bot.correct.Data(), retVal.Data(), "Test %d result", i)
		as.True(bot.correctShape.Eq(ret.Shape()))

		var xG, yG Value
		if xG, err = x.Grad(); err != nil {
			t.Errorf("Test %d: error while getting grad of x: %v", i, err)
			runtime.GC()
			continue
		}

		if yG, err = y.Grad(); err != nil {
			t.Errorf("Test %d: error while getting grad of x: %v", i, err)
			runtime.GC()
			continue
		}

		as.Equal(bot.correctDerivA.Data(), xG.Data(), "Test %v xgrad", i)
		as.Equal(bot.correctDerivB.Data(), yG.Data(), "Test %v ygrad. Expected %v. Got %v", i, bot.correctDerivB, yG)
		if !as.cont {
			t.Errorf("an error occurred")
		}

		if assertGraphEngine(t, g, stdengType); t.Failed() {
			t.Errorf("Test %d  - LispMachine failure in test", i)
			t.FailNow()
		}
		m1.Close()
		runtime.GC()
	}
}

func TestTensordotOpDoDiff(t *testing.T) {
	assert := assert.New(t)

	// Scalars
	g := NewGraph()
	a := NewTensor(g, Float64, 1, WithName("a"), WithShape(1))
	b := NewTensor(g, Float64, 1, WithName("b"), WithShape(1))

	tensordot := tensordotOp{
		aAxes:   []int{0},
		bAxes:   []int{0},
		aDims:   0,
		bDims:   0,
		retDims: 0,
	}

	c, err := ApplyOp(tensordot, a, b)

	if err != nil {
		log.Fatal("scalars: Cannot ApplyOp:", err)
		return
	}

	aT := tensor.New(tensor.WithShape(1), tensor.WithBacking([]float64{2}))
	bT := tensor.New(tensor.WithShape(1), tensor.WithBacking([]float64{21}))
	cT := tensor.New(tensor.WithShape(1), tensor.WithBacking([]float64{1})) // Backing doesn't matter as long as it is set

	aVal, _, _, _ := anyToValue(aT)
	bVal, _, _, _ := anyToValue(bT)
	cVal, _, _, _ := anyToValue(cT)

	a.bind(dvUnit(aVal))
	b.bind(dvUnit(bVal))
	c.bind(dvUnitVar(cVal)) // Will set Output derivative to all ones

	if err := tensordot.DoDiff(ExecutionContext{}, Nodes{a, b}, c); err != nil {
		log.Fatal("scalars: Cannot DoDiff:", err)
		return
	}

	aG, _ := a.Grad()
	aGfloat := aG.Data()

	bG, _ := b.Grad()
	bGfloat := bG.Data()

	aGcorrect := 21.0
	bGcorrect := 2.0

	assert.Equal(aGcorrect, aGfloat)
	assert.Equal(bGcorrect, bGfloat)

	// Vectors

	g = NewGraph()
	a = NewTensor(g, Float64, 1, WithName("a"), WithShape(2))
	b = NewTensor(g, Float64, 1, WithName("b"), WithShape(2))

	tensordot = tensordotOp{
		aAxes:   []int{0},
		bAxes:   []int{0},
		aDims:   1,
		bDims:   1,
		retDims: 1,
	}

	if c, err = ApplyOp(tensordot, a, b); err != nil {
		log.Fatal("vectors: Cannot ApplyOp:", err)
		return
	}

	aT = tensor.New(tensor.WithShape(2), tensor.WithBacking([]float64{1, 2}))
	bT = tensor.New(tensor.WithShape(2), tensor.WithBacking([]float64{3, 4}))
	cT = tensor.New(tensor.WithShape(1), tensor.WithBacking([]float64{1})) // Backing doesn't matter as long as it is set

	aVal, _, _, _ = anyToValue(aT)
	bVal, _, _, _ = anyToValue(bT)
	cVal, _, _, _ = anyToValue(cT)

	a.bind(dvUnit(aVal))
	b.bind(dvUnit(bVal))
	c.bind(dvUnitVar(cVal)) // Will set Output derivative to all ones

	if err := tensordot.DoDiff(ExecutionContext{}, Nodes{a, b}, c); err != nil {
		log.Fatal("vectors: Cannot DoDiff:", err)
		return
	}

	aG, _ = a.Grad()
	bG, _ = b.Grad()

	aGfloats := extractF64s(aG)
	bGfloats := extractF64s(bG)

	aGcorrectFloats := []float64{3, 4}
	bGcorrectFloats := []float64{1, 2}

	assert.Equal(aGcorrectFloats, aGfloats)
	assert.Equal(bGcorrectFloats, bGfloats)

	// Matrix and Vector

	g = NewGraph()
	a = NewTensor(g, Float64, 2, WithName("a"), WithShape(2, 2))
	b = NewTensor(g, Float64, 1, WithName("b"), WithShape(2))

	tensordot = tensordotOp{
		aAxes:   []int{1},
		bAxes:   []int{0},
		aDims:   2,
		bDims:   1,
		retDims: 1,
	}

	if c, err = ApplyOp(tensordot, a, b); err != nil {
		log.Fatal("matrix vector: Cannot ApplyOp:", err)
		return
	}

	aT = tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1, 2, 3, 4}))
	bT = tensor.New(tensor.WithShape(2), tensor.WithBacking([]float64{1, 2}))
	cT = tensor.New(tensor.WithShape(2), tensor.WithBacking([]float64{1, 1})) // Backing doesn't matter as long as it is set

	aVal, _, _, _ = anyToValue(aT)
	bVal, _, _, _ = anyToValue(bT)
	cVal, _, _, _ = anyToValue(cT)

	a.bind(dvUnit(aVal))
	b.bind(dvUnit(bVal))
	c.bind(dvUnitVar(cVal)) // Will set Output derivative to all ones

	if err := tensordot.DoDiff(ExecutionContext{}, Nodes{a, b}, c); err != nil {
		log.Fatal("matrix vector: Cannot DoDiff:", err)
		return
	}

	aG, _ = a.Grad()
	bG, _ = b.Grad()

	aGfloats = extractF64s(aG)
	bGfloats = extractF64s(bG)

	aGcorrectFloats = []float64{1, 2, 1, 2}
	bGcorrectFloats = []float64{4, 6}

	assert.Equal(aGcorrectFloats, aGfloats)
	assert.Equal(bGcorrectFloats, bGfloats)

	// Matrix multiplication

	g = NewGraph()

	a = NewTensor(g, Float64, 2, WithName("a"), WithShape(2, 2))
	b = NewTensor(g, Float64, 2, WithName("b"), WithShape(2, 2))

	tensordot = tensordotOp{
		aAxes:   []int{1},
		bAxes:   []int{0},
		aDims:   2,
		bDims:   2,
		retDims: 2,
	}

	if c, err = ApplyOp(tensordot, a, b); err != nil {
		log.Fatal("matrices: Cannot ApplyOp:", err)
		return
	}

	aT = tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1, 2, 3, 4}))
	bT = tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1, 2, 3, 4}))
	cT = tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1, 1, 1, 1})) // Backing doesn't matter as long as it is set

	aVal, _, _, _ = anyToValue(aT)
	bVal, _, _, _ = anyToValue(bT)
	cVal, _, _, _ = anyToValue(cT)

	a.bind(dvUnit(aVal))
	b.bind(dvUnit(bVal))
	c.bind(dvUnitVar(cVal)) // Will set Output derivative to all ones

	if err := tensordot.DoDiff(ExecutionContext{}, Nodes{a, b}, c); err != nil {
		log.Fatal("matrices: Cannot DoDiff:", err)
		return
	}

	aG, _ = a.Grad()
	bG, _ = b.Grad()

	aGfloats = extractF64s(aG)
	bGfloats = extractF64s(bG)

	aGcorrectFloats = []float64{3, 7, 3, 7}
	bGcorrectFloats = []float64{4, 4, 6, 6}

	assert.Equal(aGcorrectFloats, aGfloats)
	assert.Equal(bGcorrectFloats, bGfloats)

	// Total matrix contraction

	g = NewGraph()

	a = NewTensor(g, Float64, 2, WithName("a"), WithShape(2, 2))
	b = NewTensor(g, Float64, 2, WithName("b"), WithShape(2, 2))

	tensordot = tensordotOp{
		aAxes:   []int{1, 0},
		bAxes:   []int{0, 1},
		aDims:   2,
		bDims:   2,
		retDims: 1,
	}

	if c, err = ApplyOp(tensordot, a, b); err != nil {
		log.Fatal("matrices total contraction: Cannot ApplyOp:", err)
		return
	}

	aT = tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1, 2, 3, 4}))
	bT = tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{5, 6, 7, 8}))
	cT = tensor.New(tensor.WithShape(1), tensor.WithBacking([]float64{1})) // Backing doesn't matter as long as it is set

	aVal, _, _, _ = anyToValue(aT)
	bVal, _, _, _ = anyToValue(bT)
	cVal, _, _, _ = anyToValue(cT)

	a.bind(dvUnit(aVal))
	b.bind(dvUnit(bVal))
	c.bind(dvUnitVar(cVal)) // Will set Output derivative to all ones

	if err := tensordot.DoDiff(ExecutionContext{}, Nodes{a, b}, c); err != nil {
		log.Fatal("matrices total contraction: Cannot DoDiff:", err)
		return
	}

	aG, _ = a.Grad()
	bG, _ = b.Grad()

	aGfloats = extractF64s(aG)
	bGfloats = extractF64s(bG)

	aGcorrectFloats = []float64{5, 7, 6, 8}
	bGcorrectFloats = []float64{1, 3, 2, 4}

	assert.Equal(aGcorrectFloats, aGfloats)
	assert.Equal(bGcorrectFloats, bGfloats)

}

func TestLinearAlgebraOps(t *testing.T) {
	g := NewGraph()
	x := NewMatrix(g, Float64, WithShape(2, 3), WithName("x"))
	y := NewMatrix(g, Float64, WithShape(3, 5), WithName("y"))
	if _, err := Mul(x, y); err != nil {
		t.Fatal(err)
	}

	if _, err := Mul(y, x); err == nil {
		t.Error("Expect an error")
	}

}
