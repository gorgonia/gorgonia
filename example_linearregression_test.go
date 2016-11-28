package gorgonia_test

import (
	"fmt"
	"log"
	"math/rand"

	. "github.com/chewxy/gorgonia"
	"github.com/chewxy/gorgonia/tensor"
	"github.com/chewxy/gorgonia/tensor/types"
)

// manually generate a fake dataset which is y=2x+random
func xy(dt Dtype) (x types.Tensor, y types.Tensor) {
	var xBack, yBack interface{}
	switch dt {
	case Float32:
		xBack = []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
		yBack = []float32{2.5, 4.2, 6.1, 8, 9.992, 11.7, 15.1, 16, 18.1, 19.89}
	case Float64:
		xBack = []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
		yBack = []float64{2.5, 4.2, 6.1, 8, 9.992, 11.7, 15.1, 16, 18.1, 19.89}
	}

	x = tensor.New(types.Dtype(dt), tensor.WithBacking(xBack), tensor.WithShape(10))
	y = tensor.New(types.Dtype(dt), tensor.WithBacking(yBack), tensor.WithShape(10))
	return
}

func random(dt Dtype) Value {
	rand.Seed(13370)
	switch dt {
	case Float32:
		return F32(rand.Float32())
	case Float64:
		return F64(rand.Float64())
	default:
		panic("Unhandled dtype")
	}
}

func linearRegression(Float Dtype) {
	var xT, yT Value
	xT, yT = xy(Float)

	g := NewGraph()
	x := NewVector(g, Float, WithShape(10), WithName("x"), WithValue(xT))
	y := NewVector(g, Float, WithShape(10), WithName("y"), WithValue(yT))
	m := NewScalar(g, Float, WithName("m"), WithValue(random(Float)))
	c := NewScalar(g, Float, WithName("c"), WithValue(random(Float)))

	pred := Must(Add(Must(Mul(x, m)), c))
	se := Must(Square(Must(Sub(pred, y))))
	cost := Must(Mean(se))

	_, err := Grad(cost, m, c)
	prog, locMap, err := Compile(g)

	machine := NewTapeMachine(prog, locMap, BindDualValues())
	// machine := NewLispMachine(g)  // you can use a LispMachine, but it'll be VERY slow.
	model := Nodes{m, c}
	solver := NewVanillaSolver(WithLearnRate(0.001), WithClip(5)) // good idea to clip

	for i := 0; i < 10000; i++ {
		if err = machine.RunAll(); err != nil {
			log.Fatal(err)
		}

		if err = solver.Step(model); err != nil {
			log.Fatal(err)
		}

		machine.Reset() // Reset is necessary in a loop like this
	}

	fmt.Printf("%v: y = %3.3fx + %3.3f\n", Float, m.Value(), c.Value())
}

// Linear Regression Example
//
// The formula for a straight line is
//		y = mx + c
// We want to find an `m` and a `c` that fits the equation well. We'll do it in both float32 and float64 to showcase the extensibility of Gorgonia
func Example_linearRegression() {
	// Float32
	linearRegression(Float32)

	// Float64
	linearRegression(Float64)

	// Output:
	// Float32: y = 1.975x + 0.302
	// Float64: y = 1.975x + 0.302
}
