package tests

import (
	"fmt"
	"log"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func main() {
	g := gorgonia.NewGraph()

	mat1 := gorgonia.NewMatrix(g, tensor.Float32, gorgonia.WithShape(2, 2), gorgonia.WithName("mat1"), gorgonia.WithInit(gorgonia.GlorotU(1.0)))
	mat2 := gorgonia.NewMatrix(g, tensor.Float32, gorgonia.WithShape(2, 2), gorgonia.WithName("mat2"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	mat3 := gorgonia.NewMatrix(g, tensor.Float32, gorgonia.WithShape(2, 2), gorgonia.WithName("mat3"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))

	fmt.Println(mat1.Value().Data())
	fmt.Println(mat2.Value().Data())

	z, err := gorgonia.Mul(mat1, mat2)
	if err != nil {
		log.Fatal(err)
	}

	var zOutput gorgonia.Value
	gorgonia.Read(z, &zOutput)

	z2, err := gorgonia.Mul(z, mat3)
	if err != nil {
		log.Fatal(err)
	}

	var z2Output gorgonia.Value
	gorgonia.Read(z2, &z2Output)

	// create a VM to run the program on
	// logger := log.New(os.Stderr, "", 0)
	machine := gorgonia.NewTapeMachine(g)
	// machine := gorgonia.NewTapeMachine(g, gorgonia.WithLogger(logger))
	// machine := gorgonia.NewTapeMachine(g, gorgonia.WithLogger(logger), gorgonia.WithWatchlist(), gorgonia.TraceExec())
	// prog := machine.Prog()
	// fmt.Printf("%v\n", prog)

	// set initial values then run
	fmt.Println(zOutput)
	if err := machine.RunAll(); err != nil {
		log.Fatal(err)
	}

	fmt.Println(zOutput)
}
