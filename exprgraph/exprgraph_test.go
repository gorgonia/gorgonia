package exprgraph

import (
	"fmt"

	"gorgonia.org/tensor"
)

func Example() {
	g := NewGraph()

	x := New(g, "x", tensor.WithShape(2, 3), tensor.Of(tensor.Float64))
	y := New(g, "y", tensor.WithShape(3, 2), tensor.Of(tensor.Float64))
	z := New(g, "z", tensor.WithShape(), tensor.Of(tensor.Float64))

	xy, err := MatMul(x, y)
	if err != nil {
		fmt.Println(err)
	}
	xypz, err := Add(xy, z)
	if err != nil {
		fmt.Println(err)
	}

	fmt.Printf("x:\n%v\ny:\n%v\nxy:\n%v\nxy+z", x, y, xy, xypz)
	fmt.Printf("Nodes: %v", g.Nodes())

	// Output:
	// x:
	// ⎡0  0  0⎤
	// ⎣0  0  0⎦
	//
	// y:
	// ⎡0  0⎤
	// ⎢0  0⎥
	// ⎣0  0⎦
	//
	// xy:
	// ⎡0  0⎤
	// ⎣0  0⎦
	//
	// Nodes: [x y x×y]
}
