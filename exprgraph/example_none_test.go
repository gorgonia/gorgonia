package exprgraph_test

import (
	"fmt"

	"gorgonia.org/tensor"
	"gorgonia.org/tensor/dense"
)

// This is an example of how Gorgonia's APIs will still be compatible with tensor's APIs.
func Example_nograph() {
	x := dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	y := dense.New[float64](tensor.WithShape(3, 2), tensor.WithBacking([]float64{6, 5, 4, 3, 2, 1}))
	z := dense.New[float64](tensor.WithShape(), tensor.WithBacking([]float64{1}))
	xy, err := MatMul(x, y)
	if err != nil {
		fmt.Println(err)
		return
	}

	xypz, err := Add(xy, z)
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Printf("x:\n%v\ny:\n%v\nxy:\n%v\nxy+z:\n%v\n", x, y, xy, xypz)

	// Output:
	// x:
	// ⎡1  2  3⎤
	// ⎣4  5  6⎦
	//
	// y:
	// ⎡6  5⎤
	// ⎢4  3⎥
	// ⎣2  1⎦
	//
	// xy:
	// ⎡20  14⎤
	// ⎣56  41⎦
	//
	// xy+z:
	// ⎡21  15⎤
	// ⎣57  42⎦
}
