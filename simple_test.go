package gorgonia

import (
	"testing"
)

func TestAdd(t *testing.T) {
	g := NewGraph()

	// Build the graph
	x := g.NewScalar(Float64, WithName("x"))
	g.AddNode(x)
	y := g.NewScalar(Float64, WithName("y"))
	g.AddNode(y)
	z := g.NewNode().(*Node)
	g.AddNode(z)
	g.SetWeightedEdge(g.NewWeightedEdge(z, x, 1.0))
	g.SetWeightedEdge(g.NewWeightedEdge(z, y, 2.0))

	// Apply an operation on the node z
	err := g.ApplyOp(PowOp, z)
	if err != nil {
		t.Fatal(err)
	}

	// create a VM to run the program on
	machine := NewTapeMachine(g)

	// set initial values then run
	Let(x, 2.0)
	Let(y, 3.0)

	// Run the program
	err = machine.RunAll()
	if err != nil {
		t.Fatal(err)
	}
	if z.Value().Data().(float64) != float64(9) {
		t.Fatalf("1 result: %v", z.Value())
	}
	// change the order of the inputs

	/*
		machine.Reset()
		// Run the program
		err = machine.RunAll()
		if err != nil {
			t.Fatal(err)
		}
		if z.Value().Data().(float64) != float64(9) {
			t.Fatalf("2 result: %v", z.Value())
		}
	*/
}
