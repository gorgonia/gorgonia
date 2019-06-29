package gorgonia_test

import (
	"fmt"
	"log"

	. "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// By default, Gorgonia operations do not perform broadcasting.
// To do broadcasting, you would need to manually specify the operation
func Example_broadcasting1() {
	g := NewGraph()
	a := NewVector(g, tensor.Float64, WithShape(2), WithName("a"), WithValue(tensor.New(tensor.WithBacking([]float64{100, 100}))))
	b := NewMatrix(g, tensor.Float64, WithShape(2, 2), WithName("b"), WithValue(tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1, 1, 2, 2}))))

	fmt.Printf("a = %v\nb =\n%v\n", a.Value(), b.Value())

	_, err := Add(a, b)
	fmt.Printf("a + b yields an error: %v\n\n", err)

	// Note here the broadcasting of a is on the first axis, not the zeroth axis. Simply put, assume that it's already a (2,1) matrix.
	ab, err := BroadcastAdd(a, b, []byte{1}, nil)
	if err != nil {
		fmt.Printf("uh oh, something went wrong: %v\n", err)
	}

	ba, err := BroadcastAdd(b, a, nil, []byte{1})
	if err != nil {
		fmt.Printf("uh oh, something went wrong: %v\n", err)
	}

	// Now, let's run the program
	machine := NewTapeMachine(g)
	defer machine.Close()
	if err = machine.RunAll(); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("a +⃗ b =\n%v\n", ab.Value())
	fmt.Printf("b +⃗ a =\n%v", ba.Value())

	// Output:
	// a = [100  100]
	// b =
	// ⎡1  1⎤
	// ⎣2  2⎦
	//
	// a + b yields an error: Failed to infer shape. Op: + false: Shape mismatch: (2) and (2, 2)
	//
	// a +⃗ b =
	// ⎡101  101⎤
	// ⎣102  102⎦
	//
	// b +⃗ a =
	// ⎡101  101⎤
	// ⎣102  102⎦

}
