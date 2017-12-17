package main

import (
	"fmt"
	"log"

	T "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func main() {
	g := T.NewGraph()
	x := T.NewMatrix(g, T.Float32, T.WithName("x"), T.WithShape(100, 100))
	y := T.NewMatrix(g, T.Float32, T.WithName("y"), T.WithShape(100, 100))
	xpy := T.Must(T.Add(x, y))
	xpy2 := T.Must(T.Tanh(xpy))

	m := T.NewTapeMachine(g, T.UseCudaFor("tanh"))

	T.Let(x, tensor.New(tensor.WithShape(100, 100), tensor.WithBacking(tensor.Random(tensor.Float32, 100*100))))
	T.Let(y, tensor.New(tensor.WithShape(100, 100), tensor.WithBacking(tensor.Random(tensor.Float32, 100*100))))

	for i := 0; i < 1000; i++ {
		if err := m.RunAll(); err != nil {
			log.Fatalf("iteration: %d. Err: %v", i, err)
		}
	}

	fmt.Printf("%1.1f", xpy2.Value())
}
