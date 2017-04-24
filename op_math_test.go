package gorgonia

import (
	"io/ioutil"
	"log"
	"runtime"
	"testing"

	"github.com/chewxy/gorgonia/tensor"
	"github.com/stretchr/testify/assert"
)

var binOpTests = []struct {
	binOp func(*Node, *Node) (*Node, error)
	a, b  Value

	correct       Value
	correctDerivA Value
	correctDerivB Value
	correctShape  tensor.Shape
}{

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
	// for i, bot := range binOpTests {
	// 	g := NewGraph()
	// 	xV, _ := CloneValue(bot.a)
	// 	yV, _ := CloneValue(bot.b)
	// 	x := NodeFromAny(g, xV, WithName("x"))
	// 	y := NodeFromAny(g, yV, WithName("y"))

	// 	var ret *Node
	// 	var retVal Value
	// 	var err error
	// 	if ret, err = bot.binOp(x, y); err != nil {
	// 		t.Errorf("Test %d: %v", i, err)
	// 		continue
	// 	}
	// 	Read(ret, &retVal)

	// 	cost := Must(Sum(ret))
	// 	var grads Nodes
	// 	if grads, err = Grad(cost, x, y); err != nil {
	// 		t.Errorf("Test %d: error while symbolic op: %v", i, err)
	// 		continue
	// 	}

	// 	m1 := NewTapeMachine(g)
	// 	if err = m1.RunAll(); err != nil {
	// 		t.Errorf("Test %d: error while running %v", i, err)
	// 		runtime.GC()
	// 		continue
	// 	}

	// 	as := newAssertState(assert)
	// 	as.Equal(bot.correct.Data(), retVal.Data(), "Test %d result", i)
	// 	as.True(bot.correctShape.Eq(ret.Shape()))
	// 	as.Equal(2, len(grads))
	// 	as.Equal(bot.correctDerivA.Data(), grads[0].Value().Data(), "Test %v xgrad", i)
	// 	as.Equal(bot.correctDerivB.Data(), grads[1].Value().Data(), "Test %v ygrad. Expected %v. Got %v", i, bot.correctDerivB, grads[1].Value())
	// 	if !as.cont {
	// 		prog := m1.Prog()
	// 		t.Logf("Test %d failed. Prog: %v", i, prog)
	// 	}

	// 	runtime.GC()
	// }

	for i, bot := range binOpTests {
		log.Printf("i: %d", i)
		// if i != 25 {
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

		ioutil.WriteFile("BLAH.dot", []byte(g.ToDot()), 0644)

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
			t.Errorf("an error occured")
		}

		runtime.GC()
	}
}
