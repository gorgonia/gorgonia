package xvm

import (
	"context"
	"testing"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func BenchmarkMachine_Run(b *testing.B) {
	g := gorgonia.NewGraph()
	xV := tensor.New(tensor.WithShape(1, 1, 5, 5), tensor.WithBacking([]float32{
		0, 0, 0, 0, 0,
		1, 1, 1, 1, 1,
		2, 2, 2, 2, 2,
		3, 3, 3, 3, 3,
		4, 4, 4, 4, 4,
	}))
	kernelV := tensor.New(tensor.WithShape(1, 1, 3, 3), tensor.WithBacking([]float32{
		1, 1, 1,
		1, 1, 1,
		1, 1, 1,
	}))

	x := gorgonia.NewTensor(g, gorgonia.Float32, 4, gorgonia.WithShape(1, 1, 5, 5), gorgonia.WithValue(xV), gorgonia.WithName("x"))
	w := gorgonia.NewTensor(g, gorgonia.Float32, 4, gorgonia.WithShape(1, 1, 3, 3), gorgonia.WithValue(kernelV), gorgonia.WithName("w"))

	_, err := gorgonia.Conv2d(x, w, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1})
	if err != nil {
		b.Fatal(err)
	}
	// logger := log.New(os.Stderr, "", 0)
	// vm := NewTapeMachine(g, WithLogger(logger), WithWatchlist(), WithValueFmt("%#v"))

	vm := NewMachine(g)
	ctx := context.Background()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		//ctx, cancel := context.WithTimeout(context.Background(), 900*time.Millisecond)
		if err := vm.Run(ctx); err != nil {
			b.Fatal(err)
		}
	}
	vm.Close()
}
func BenchmarkMachine_RunTapeMachine(b *testing.B) {
	g := gorgonia.NewGraph()
	xV := tensor.New(tensor.WithShape(1, 1, 5, 5), tensor.WithBacking([]float32{
		0, 0, 0, 0, 0,
		1, 1, 1, 1, 1,
		2, 2, 2, 2, 2,
		3, 3, 3, 3, 3,
		4, 4, 4, 4, 4,
	}))
	kernelV := tensor.New(tensor.WithShape(1, 1, 3, 3), tensor.WithBacking([]float32{
		1, 1, 1,
		1, 1, 1,
		1, 1, 1,
	}))

	x := gorgonia.NewTensor(g, gorgonia.Float32, 4, gorgonia.WithShape(1, 1, 5, 5), gorgonia.WithValue(xV), gorgonia.WithName("x"))
	w := gorgonia.NewTensor(g, gorgonia.Float32, 4, gorgonia.WithShape(1, 1, 3, 3), gorgonia.WithValue(kernelV), gorgonia.WithName("w"))

	_, err := gorgonia.Conv2d(x, w, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1})
	if err != nil {
		b.Fatal(err)
	}
	// logger := log.New(os.Stderr, "", 0)
	// vm := NewTapeMachine(g, WithLogger(logger), WithWatchlist(), WithValueFmt("%#v"))

	vm := gorgonia.NewTapeMachine(g)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		if err := vm.RunAll(); err != nil {
			b.Fatal(err)
		}
		vm.Reset()
	}
	vm.Close()
}
