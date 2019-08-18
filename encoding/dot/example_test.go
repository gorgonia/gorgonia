package dot

import (
	"fmt"
	"log"

	"gorgonia.org/gorgonia"
)

func ExampleMarshal() {
	g := gorgonia.NewGraph()

	var x, y *gorgonia.Node
	var err error

	// define the expression
	x = gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("x"))
	y = gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("y"))
	if _, err = gorgonia.Add(x, y); err != nil {
		log.Fatal(err)
	}
	if b, err := Marshal(g); err == nil {
		fmt.Println(string(b))
	}
}
