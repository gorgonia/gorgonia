package gorgonia_test

import (
	"fmt"

	. "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func Example_KeepDims() {
	g := NewGraph()
	a := NodeFromAny(g, tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6})))
	m1, _ := Mean(a, 1)
	m2, _ := KeepDims(a, false, func(a *Node) (*Node, error) { return Mean(a, 1) })
	m3, _ := Mean(a, 0)
	m4, _ := KeepDims(a, true, func(a *Node) (*Node, error) { return Mean(a, 0) })
	_, err := KeepDims(a, true, func(a *Node) (*Node, error) { return Mean(a) })
	if err == nil {
		panic("expected an error: you can't keep dims on a scalar result")
	}
	vm := NewTapeMachine(g)
	if err := vm.RunAll(); err != nil {
		panic(err)
	}

	fmt.Printf("a:\n%v\n", a.Value())
	fmt.Printf("m1 (shape: %v):\n%v\n", m1.Value().Shape(), m1.Value())
	fmt.Printf("m2 (shape: %v):\n%v\n", m2.Value().Shape(), m2.Value())
	fmt.Printf("m3 (shape: %v):\n%v\n", m3.Value().Shape(), m3.Value())
	fmt.Printf("m4 (shape: %v):\n%v\n", m4.Value().Shape(), m4.Value())

	// Output:
	// a:
	// ⎡1  2  3⎤
	// ⎣4  5  6⎦
	//
	// m1 (shape: (2)):
	// [2  5]
	// m2 (shape: (2, 1)):
	// C[2  5]
	// m3 (shape: (3)):
	// [2.5  3.5  4.5]
	// m4 (shape: (1, 3)):
	// R[2.5  3.5  4.5]

}
