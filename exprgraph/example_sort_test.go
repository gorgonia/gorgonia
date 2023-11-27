package exprgraph_test

import (
	"fmt"

	"gorgonia.org/gorgonia/exprgraph"
	"gorgonia.org/tensor"
)

func ExampleSort() {
	engine := &SymbolicEngine{}
	g := exprgraph.NewGraph(engine)
	engine.g = g

	x := exprgraph.New[float64](g, "x", tensor.WithShape(2, 2))
	w := exprgraph.New[float64](g, "w", tensor.WithShape(2, 2))
	b := exprgraph.New[float64](g, "b", tensor.WithShape())

	xw, err := MatMul(x, w)
	if err != nil {
		fmt.Println(err)
	}
	_, err = Add(xw, b)
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
