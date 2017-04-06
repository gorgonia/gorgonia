package gorgonia_test

import (
	"fmt"
	"log"
	"math/rand"
	"runtime"

	. "github.com/chewxy/gorgonia"
	"github.com/chewxy/gorgonia/tensor"
)

const (
	vecSize = 1000000
)

// manually generate a fake dataset which is y=2x+random
func xy(dt tensor.Dtype) (x tensor.Tensor, y tensor.Tensor) {
	var xBack, yBack interface{}
	switch dt {
	case Float32:
		xBack = tensor.Range(tensor.Float32, 1, vecSize+1).([]float32)
		yBackC := tensor.Range(tensor.Float32, 1, vecSize+1).([]float32)

		for i, v := range yBackC {
			yBackC[i] = v*2 + rand.Float32()
		}
		yBack = yBackC
	case Float64:
		xBack = tensor.Range(tensor.Float64, 1, vecSize+1).([]float64)
		yBackC := tensor.Range(tensor.Float64, 1, vecSize+1).([]float64)

		for i, v := range yBackC {
			yBackC[i] = v*2 + rand.Float64()
		}
		yBack = yBackC
	}

	x = tensor.New(tensor.WithBacking(xBack), tensor.WithShape(vecSize))
	y = tensor.New(tensor.WithBacking(yBack), tensor.WithShape(vecSize))
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
	x := NewVector(g, Float, WithShape(vecSize), WithName("x"), WithValue(xT))
	y := NewVector(g, Float, WithShape(vecSize), WithName("y"), WithValue(yT))
	m := NewScalar(g, Float, WithName("m"), WithValue(random(Float)))
	c := NewScalar(g, Float, WithName("c"), WithValue(random(Float)))

	pred := Must(Add(Must(Mul(x, m)), c))
	se := Must(Square(Must(Sub(pred, y))))
	cost := Must(Mean(se))

	_, err := Grad(cost, m, c)

	// logger := log.New(os.Stderr, "", 0)
	// machine := NewTapeMachine(g, BindDualValues(m, c), WithLogger(logger))
	machine := NewTapeMachine(g, BindDualValues(m, c))
	log.Printf("%v", machine.Prog())
	defer runtime.GC()

	// machine := NewLispMachine(g)  // you can use a LispMachine, but it'll be VERY slow.
	model := Nodes{m, c}
	solver := NewVanillaSolver(WithLearnRate(0.001), WithClip(5)) // good idea to clip

	if CUDA {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
	}
	for i := 0; i < 500; i++ {
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
	// float32: y = 2.001x + 2.001
	// float64: y = 2.001x + 2.001
}
