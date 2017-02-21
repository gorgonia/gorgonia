package main

import (
	"fmt"
	"log"
	"runtime"

	T "github.com/chewxy/gorgonia"
	"github.com/chewxy/gorgonia/tensor"
)

func main() {
	g := T.NewGraph()
	x := T.NewMatrix(g, T.Float32, T.WithName("x"), T.WithShape(100, 100))
	y := T.NewMatrix(g, T.Float32, T.WithName("y"), T.WithShape(100, 100))
	xpy := T.Must(T.Add(x, y))
	xpy2 := T.Must(T.Tanh(xpy))

	prog, locMap, _ := T.Compile(g)
	m := T.NewTapeMachine(prog, locMap, T.UseCudaFor("tanh"))

	T.Let(x, tensor.New(tensor.WithShape(100, 100), tensor.WithBacking(tensor.Random(tensor.Float32, 100*100))))
	T.Let(y, tensor.New(tensor.WithShape(100, 100), tensor.WithBacking(tensor.Random(tensor.Float32, 100*100))))

	runtime.LockOSThread()
	for i := 0; i < 1000; i++ {
		if err := m.RunAll(); err != nil {
			log.Fatalf("iteration: %d. Err: %v", i, err)
		}
	}
	runtime.UnlockOSThread()

	fmt.Printf("%1.1f", xpy2.Value())
}
