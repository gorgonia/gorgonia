package exprgraph_test

import (
	"fmt"

	"gorgonia.org/gorgonia/exprgraph"
	"gorgonia.org/gorgonia/internal/encoding"
	"gorgonia.org/tensor"
)

func Example_encoding() {
	engine := &SymbolicEngine{}
	g := exprgraph.NewGraph(engine)
	engine.g = g

	x := exprgraph.New[float64](g, "x", tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	y := exprgraph.New[float64](g, "y", tensor.WithShape(3, 2), tensor.WithBacking([]float64{6, 5, 4, 3, 2, 1}))
	z := exprgraph.New[float64](g, "z", tensor.WithShape())

	xy, err := MatMul(x, y)
	if err != nil {
		fmt.Println(err)
	}
	xypz, err := Add(xy, z)
	if err != nil {
		fmt.Println(err)
	}
	grp := encoding.NewGroup("BASIC")
	g.SetGroup(xy, grp)
	g.SetGroup(xypz, grp)

	fmt.Printf("Group[0] of xy: %v\n", g.GroupsOf(xy)[0].Name)
	fmt.Printf("Group[0] of xypz: %v\n", g.GroupsOf(xypz)[0].Name)

	// Output:
	// Group[0] of xy: BASIC
	// Group[0] of xypz: BASIC

}
