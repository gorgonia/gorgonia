package nnops_test

import (
	"fmt"
	"log"
	"runtime"

	"gorgonia.org/gorgonia"
	nnops "gorgonia.org/gorgonia/ops/nn"
	"gorgonia.org/tensor"
)

func ExampleConv2d() {
	g := gorgonia.NewGraph()
	x := gorgonia.NodeFromAny(g, tensor.New(
		tensor.WithShape(1, 1, 7, 5),
		tensor.WithBacking([]float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34})))
	filter := gorgonia.NodeFromAny(g, tensor.New(
		tensor.WithShape(1, 1, 3, 3),
		tensor.WithBacking([]float32{1, 1, 1, 1, 1, 1, 1, 1, 1})))
	y := gorgonia.Must(nnops.Conv2d(x, filter, []int{3, 3}, []int{0, 0}, []int{2, 2}, []int{1, 1}))
	m := gorgonia.NewTapeMachine(g)
	runtime.LockOSThread()
	for i := 0; i < 1000; i++ {
		if err := m.RunAll(); err != nil {
			log.Fatalf("iteration: %d. Err: %v", i, err)
		}
	}
	runtime.UnlockOSThread()

	fmt.Printf("%1.1f", y.Value())
	// output:
	// ⎡ 54.0   72.0⎤
	// ⎢144.0  162.0⎥
	// ⎣234.0  252.0⎦
}
