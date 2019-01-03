package dot

import (
	"fmt"
	"log"

	"gorgonia.org/gorgonia"
)

func ExampleMarshal() {
	g := gorgonia.NewGraph()

	var x, y, z *Node
	var err error

	// define the expression
	x = gorgonia.NewScalar(g, Float64, WithName("x"))
	y = gorgonia.NewScalar(g, Float64, WithName("y"))
	z, err = gorgonia.Add(x, y)
	if err != nil {
		log.Fatal(err)
	}

	b, err := Marshal(g)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(string(b))
	// Output: 4.5
}
