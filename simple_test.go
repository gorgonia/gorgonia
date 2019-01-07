package gorgonia

import (
	"testing"
)

func TestAdd(t *testing.T) {
	g := NewGraph()

	// define the expression
	x := g.NewScalar(Float64, WithName("x"))
	g.AddNode(x)
	y := g.NewScalar(Float64, WithName("y"))
	g.AddNode(y)
	z := g.NewNode().(*Node)
	g.AddNode(z)
	g.SetWeightedEdge(g.NewWeightedEdge(z, x, 1.0))
	g.SetWeightedEdge(g.NewWeightedEdge(z, y, 2.0))
	err := g.ApplyOp(newElemBinOp(addOpType, x, y), z)
	//z, err := gorgonia.Add(x, y)

	if err != nil {
		t.Fatal(err)
	}

	// create a VM to run the program on
	machine := NewTapeMachine(g)

	// set initial values then run
	Let(x, 2.0)
	Let(y, 2.5)
	err = machine.RunAll()
	if err != nil {
		t.Fatal(err)
	}
	if z.Value().Data().(float64) != float64(4.5) {
		t.Fatalf("result: %v", z.Value())
	}

}
