package gorgonia_test

import (
	"fmt"
	"log"

	. "github.com/chewxy/gorgonia"
)

// SymbolicDiff showcases symbolic differentiation
func Example_symbolicDiff() {
	g := NewGraph()

	var x, y, z *Node
	var err error

	// define the expression
	x = NewScalar(g, Float64, WithName("x"))
	y = NewScalar(g, Float64, WithName("y"))
	if z, err = Add(x, y); err != nil {
		log.Fatal(err)
	}

	// symbolically differentiate z with regards to x and y
	// this adds the gradient nodes to the graph g
	var grads Nodes
	if grads, err = Grad(z, x, y); err != nil {
		log.Fatal(err)
	}

	// compile into a program
	prog, locMap, err := Compile(g)
	if err != nil {
		log.Fatal(err)
	}

	// create a VM to run the program on
	machine := NewTapeMachine(prog, locMap)

	// set initial values then run
	Let(x, 2.0)
	Let(y, 2.5)
	if err = machine.RunAll(); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("z: %v\n", z.Value())
	if xgrad, err := x.Grad(); err == nil {
		fmt.Printf("dz/dx: %v | %v\n", xgrad, grads[0].Value())
	}

	if ygrad, err := y.Grad(); err == nil {
		fmt.Printf("dz/dy: %v | %v\n", ygrad, grads[1].Value())
	}

	// Output:
	// z: 4.5
	// dz/dx: 1 | 1
	// dz/dy: 1 | 1
}
