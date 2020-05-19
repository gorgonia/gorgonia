package gorgonia_test

import (
	"fmt"

	. "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func Example_keepDims() {
	g := NewGraph()
	a := NodeFromAny(g, tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6})))
	m1, _ := Mean(a, 1)
	m2, _ := KeepDims(a, false, func(a *Node) (*Node, error) { return Mean(a, 1) })
	m3, _ := Mean(a, 0)
	m4, _ := KeepDims(a, true, func(a *Node) (*Node, error) { return Mean(a, 0) })
	m5, _ := KeepDims(a, true, func(a *Node) (*Node, error) { return Mean(a) })

	// these reads are necessary as the VM may feel free to clobber the underlying data.
	// e.g. if m1.Value() is used in the print statement below, the answer will be wrong.
	// This is because before the VM executes the operations, a check is done to see if unsafe
	// operations may be done. Unsafe operations are useful in saving memory.
	// In this example, Reshape can be unsafely done if no other node is "using" m1,
	// so m1.Value() will have its shape clobbered. Thus if m1.Value() is read after the VM has run,
	// there is no guarantee that the data is correct. The only way around this is to "use" m1, by the Read() function.
	var m1v, m2v, m3v, m4v Value
	Read(m1, &m1v)
	Read(m2, &m2v)
	Read(m3, &m3v)
	Read(m4, &m4v)

	vm := NewTapeMachine(g)
	if err := vm.RunAll(); err != nil {
		panic(err)
	}

	fmt.Printf("a:\n%v\n", a.Value())
	fmt.Printf("m1 (shape: %v):\n%v\n", m1.Value().Shape(), m1v)
	fmt.Printf("m2 (shape: %v):\n%v\n", m2.Value().Shape(), m2v)
	fmt.Printf("m3 (shape: %v):\n%v\n", m3.Value().Shape(), m3v)
	fmt.Printf("m4 (shape: %v):\n%v\n", m4.Value().Shape(), m4v)
	fmt.Printf("m5 (shape: %v):\n%v\n", m5.Value().Shape(), m5.Value())

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
	// m5 (shape: (1, 1)):
	// ⎡3.5⎤

}
