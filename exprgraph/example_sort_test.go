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

	x := exprgraph.NewNode(g, "x", tensor.WithShape(2, 3), tensor.Of(tensor.Float64))
	w := exprgraph.NewNode(g, "w", tensor.WithShape(3, 2), tensor.Of(tensor.Float64))
	b := exprgraph.NewNode(g, "b", tensor.WithShape(), tensor.Of(tensor.Float64))

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
