package gorgonia

import (
	"testing"

	"gorgonia.org/gorgonia/internal/engine"
)

func TestAdd(t *testing.T) {
	g := NewGraph()

	// Build the graph
	x := NewScalar(g, engine.Float64, WithName("x"))
	y := NewScalar(g, engine.Float64, WithName("y"))
	z, err := Add(g, x, y)
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
	if z.Value().Data().(float64) != float64(5) {
		t.Fatalf("1 result: %v", z.Value())
	}
}
