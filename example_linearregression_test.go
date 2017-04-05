package gorgonia_test

import (
	"fmt"
	"log"
	"math/rand"
	"runtime"

	. "github.com/chewxy/gorgonia"
	"github.com/chewxy/gorgonia/tensor"
)

// manually generate a fake dataset which is y=2x+random
func xy(dt tensor.Dtype) (x tensor.Tensor, y tensor.Tensor) {
	var xBack, yBack interface{}
	switch dt {
	case Float32:
		xBack = []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
		yBack = []float32{2.5, 4.2, 6.1, 8, 9.992, 11.7, 15.1, 16, 18.1, 19.89}
	case Float64:
		xBack = []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
		yBack = []float64{2.5, 4.2, 6.1, 8, 9.992, 11.7, 15.1, 16, 18.1, 19.89}
	}

	x = tensor.New(tensor.WithBacking(xBack), tensor.WithShape(10))
	y = tensor.New(tensor.WithBacking(yBack), tensor.WithShape(10))
	return
}

func random(dt tensor.Dtype) interface{} {
	rand.Seed(13370)
	switch dt {
	case tensor.Float32:
		return rand.Float32()
	case tensor.Float64:
		return rand.Float64()
	default:
		panic("Unhandled dtype")
	}
}

func linearRegression(Float tensor.Dtype) {
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
	machine := NewTapeMachine(g, BindDualValues())
	defer runtime.GC()

	// machine := NewLispMachine(g)  // you can use a LispMachine, but it'll be VERY slow.
	model := Nodes{m, c}
	solver := NewVanillaSolver(WithLearnRate(0.001), WithClip(5)) // good idea to clip

	if CUDA {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
	}
	for i := 0; i < 10000; i++ {
		if err = machine.RunAll(); err != nil {
			fmt.Printf("Error during iteration: %v: %v\n", i, err)
			break
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
	// float32: y = 1.975x + 0.302
	// float64: y = 1.975x + 0.302
}
