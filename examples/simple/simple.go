package main

import (
	"fmt"
	"log"

	"gorgonia.org/gorgonia"
	"gorgonia.org/gorgonia/debugger/dot"
)

func main() {
	g := gorgonia.NewGraph()

	// define the expression
	x := g.NewScalar(gorgonia.Float64, gorgonia.WithName("x"))
	g.AddNode(x)
	y := g.NewScalar(gorgonia.Float64, gorgonia.WithName("y"))
	g.AddNode(y)
	z, err := gorgonia.Add(x, y)
	if err != nil {
		log.Fatal(err)
	}
	b, err := dot.Marshal(g)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(string(b))

	// create a VM to run the program on
	machine := gorgonia.NewTapeMachine(g)

	// set initial values then run
	gorgonia.Let(x, 2.0)
	gorgonia.Let(y, 2.5)
	err = machine.RunAll()
	if err != nil {
		log.Fatal(err)
	}

	log.Printf("result: %v", z.Value())
}
