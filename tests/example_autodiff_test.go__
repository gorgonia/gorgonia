package gorgonia_test

import (
	"fmt"
	"log"

	. "gorgonia.org/gorgonia"
)

// Autodiff showcases automatic differentiation
func Example_autodiff() {
	g := NewGraph()

	var x, y, z *Node
	var err error

	// define the expression
	x = NewScalar(g, Float64, WithName("x"))
	y = NewScalar(g, Float64, WithName("y"))
	if z, err = Add(x, y); err != nil {
		log.Fatal(err)
	}

	// set initial values then run
	Let(x, 2.0)
	Let(y, 2.5)

	// by default, LispMachine performs forward mode and backwards mode execution
	m := NewLispMachine(g)
	defer m.Close()
	if err = m.RunAll(); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("z: %v\n", z.Value())

	if xgrad, err := x.Grad(); err == nil {
		fmt.Printf("dz/dx: %v\n", xgrad)
	}

	if ygrad, err := y.Grad(); err == nil {
		fmt.Printf("dz/dy: %v\n", ygrad)
	}

	// Output:
	// z: 4.5
	// dz/dx: 1
	// dz/dy: 1
}
