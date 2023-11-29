package exprgraph_test

import (
	"fmt"

	"gorgonia.org/gorgonia/exprgraph"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/dense"
)

func ExampleSort() {
	engine := &SymbolicEngine{}
	g := exprgraph.NewGraph(engine)
	engine.g = g

	x := exprgraph.New[float64](g, "x", tensor.WithShape(2, 2))
	w := exprgraph.New[float64](g, "w", tensor.WithShape(2, 2))
	b := exprgraph.New[float64](g, "b", tensor.WithShape())

	xw, err := MatMul[float64, *dense.Dense[float64]](x, w)
	if err != nil {
		fmt.Println(err)
	}
	_, err = Add[float64, *dense.Dense[float64]](xw, b)
	if err != nil {
		fmt.Println(err)
	}

	sorted, err := exprgraph.Sort(g)
	if err != nil {
		fmt.Println(err)
	}
	fmt.Printf("Sorted: %s\n", sorted)

	// Output:
	// Sorted: [x×w+b x×w b w x]

}
