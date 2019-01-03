package dot

import (
	"fmt"
	"log"

	"gorgonia.org/gorgonia"
)

func ExampleMarshal() {
	g := gorgonia.NewGraph()

	var x, y *gorgonia.Node

	// define the expression
	x = gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("x"))
	y = gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("y"))
	_, err := gorgonia.Add(x, y)
	if err != nil {
		log.Fatal(err)
	}

	b, err := Marshal(g)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(string(b))
}
