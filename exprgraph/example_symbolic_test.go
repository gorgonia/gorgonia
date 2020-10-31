package exprgraph_test

import (
	"fmt"

	"gorgonia.org/gorgonia/exprgraph"
	"gorgonia.org/tensor"
)

// SymbolicEngine is a engine that performs symbolic operations.
type SymbolicEngine struct {
	tensor.StdEng
	g *exprgraph.Graph
}

func (e *SymbolicEngine) Graph() *exprgraph.Graph { return e.g }

func (e *SymbolicEngine) SetGraph(g *exprgraph.Graph) { e.g = g }

// ExampleSymbolic demonstrates how to implement a symbolic engine that does perform symbolic operations
func Example_symbolicEngine() {
	g := exprgraph.NewGraph(&SymbolicEngine{})

	x := exprgraph.NewNode(g, "x", tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	y := exprgraph.NewNode(g, "y", tensor.WithShape(3, 2), tensor.WithBacking([]float64{6, 5, 4, 3, 2, 1}))
	z := exprgraph.NewNode(g, "z", tensor.WithShape(), tensor.Of(tensor.Float64))

	xy, err := MatMul(x, y)
	if err != nil {
		fmt.Println(err)
	}
	xypz, err := Add(xy, z)
	if err != nil {
		fmt.Println(err)
	}

	fmt.Printf("x:\n%s\ny:\n%s\nxy:\n%s\nxy+z:\n%s\n", x, y, xy, xypz)

	// TODO
	// Output:
	// x:
	//x
	//y:
	//y
	//xy:
	//x×y
	//xy+z:
	//x×y+z

}
