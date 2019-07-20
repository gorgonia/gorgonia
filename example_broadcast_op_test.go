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

func Example_creatingTriangleMatrices() {
	// Broadcasting is useful. We can create triangular dense matrices simply

	g := NewGraph()
	a := NewMatrix(g, tensor.Float64, WithShape(3, 1), WithName("a"), WithInit(RangedFrom(0)))
	b := NewMatrix(g, tensor.Float64, WithShape(1, 4), WithName("b"), WithInit(RangedFrom(0)))
	tl, err := BroadcastGte(a, b, true, []byte{1}, []byte{0})
	if err != nil {
		log.Fatalf("uh oh. Something went wrong %v", err)
	}

	tu, err := BroadcastLt(a, b, true, []byte{1}, []byte{0})
	if err != nil {
		log.Fatalf("uh oh. Something went wrong %v", err)
	}

	m := NewTapeMachine(g)

	// PEDAGOGICAL:
	// Uncomment the following code if you want to see what happens behind the scenes
	// m.Close()
	// logger := log.New(os.Stderr, "",0)
	// m = NewTapeMachine(g, WithLogger(logger), WithWatchlist())

	defer m.Close()
	if err = m.RunAll(); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("triangular, lower:\n%v\n", tl.Value())
	fmt.Printf("triangular, upper:\n%v\n", tu.Value())

	// Output:
	// triangular, lower:
	// ⎡1  0  0  0⎤
	// ⎢1  1  0  0⎥
	// ⎣1  1  1  0⎦
	//
	// triangular, upper:
	// ⎡0  1  1  1⎤
	// ⎢0  0  1  1⎥
	// ⎣0  0  0  1⎦

}
