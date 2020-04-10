package gorgonia_test

import (
	"fmt"

	. "gorgonia.org/gorgonia"
)

// This example showcases the reasons for the more confusing functions.
func Example_monad_raison_detre() {
	// The main reason for the following function is to make it easier to create APIs.
	// Gorgonia;s APIs are very explicit hence not very user friendly.

	const (
		n        = 32
		features = 784
		size     = 100
	)

	// The following is an example of how to set up a neural network

	// First, we set up the components
	g := NewGraph()
	w1 := NewMatrix(g, Float32, WithShape(features, size), WithName("w"), WithInit(GlorotU(1)))
	b1 := NewMatrix(g, Float32, WithShape(1, size), WithName("b"), WithInit(Zeroes()))
	x1 := NewMatrix(g, Float32, WithShape(n, features), WithName("x"))

	// Then we write the expression:
	var xw, xwb, act *Node
	var err error
	if xw, err = Mul(x1, w1); err != nil {
		fmt.Printf("Err while Mul: %v\n", err)
	}
	if xwb, err = BroadcastAdd(xw, b1, nil, []byte{0}); err != nil {
		fmt.Printf("Err while Add: %v\n", err)
	}
	if act, err = Tanh(xwb); err != nil {
		fmt.Printf("Err while Tanh: %v\n", err)
	}
	fmt.Printf("act is a %T\n", act)

	// The following is how to set up the exact same network

	// First we set up our environment
	//
	// These LiftXXX functions transforms Gorgonia's default API into functions that return `Result`
	var mul = Lift2(Mul)                   // Lift2 turns a func(*Node, *Node) (*Node, error)
	var tanh = Lift1(Tanh)                 // Lift1 turns a func(*Node) (*Node, error)
	var add = Lift2Broadcast(BroadcastAdd) // Lift2Broadcast turns a func(*Node, *Node, []byte, []byte) (*Nide, error)

	// First we set up the components
	h := NewGraph()
	w2 := NewMatrix(h, Float32, WithShape(features, size), WithName("w"), WithInit(GlorotU(1)))
	b2 := NewMatrix(h, Float32, WithShape(1, size), WithName("b"), WithInit(Zeroes()))
	x2 := NewMatrix(h, Float32, WithShape(n, features), WithName("x"))

	// Then we write the expression
	act2 := tanh(add(mul(x2, w2), b2, nil, []byte{0}))
	fmt.Printf("act2 is a %T (note it's wrapped in the `Result` type)\n", act2)
	fmt.Println()
	// both g and h are the same graph but the expression is easier to write for act2
	fmt.Printf("Both g and h are the same graph:\ng: %v\nh: %v\n", g.AllNodes(), h.AllNodes())

	// Output:
	// act is a *gorgonia.Node
	// act2 is a *gorgonia.Node (note it's wrapped in the `Result` type)
	//
	// Both g and h are the same graph:
	// g: [w, b, x, A × B(%2, %0), Reshape(1, 100)(%1), SizeOf=32(%3), Repeat0(%4, %5), + false(%3, %6), tanh(%7)]
	// h: [w, b, x, A × B(%2, %0), Reshape(1, 100)(%1), SizeOf=32(%3), Repeat0(%4, %5), + false(%3, %6), tanh(%7)]
}

// This example showcases dealing with errors. This is part 2 of the raison d'être of the more complicated functions - dealing with errors
func Example_monad_raison_detre_errors() {
	// Observe that in a similar example, errors are manually controllable in the original case,
	// but automated in the second case
	const (
		n        = 32
		features = 784
		size     = 100
	)

	// The following is an example of how to set up a neural network

	// First, we set up the components
	g := NewGraph()
	w1 := NewMatrix(g, Float32, WithShape(features, size), WithName("w"), WithInit(GlorotU(1)))
	b1 := NewMatrix(g, Float32, WithShape(1, size), WithName("b"), WithInit(Zeroes()))
	x1 := NewMatrix(g, Float32, WithShape(n, features), WithName("x"))

	// Then we write the expression:
	var xw, xwb, act *Node
	var err error
	if xw, err = Mul(x1, w1); err != nil {
		fmt.Printf("Err while Mul: %v\n", err)
	}
	// we introduce an error here - it should be []byte{0}
	if xwb, err = BroadcastAdd(xw, b1, nil, []byte{1}); err != nil {
		fmt.Printf("Err while Add: %v\n", err)
		goto case2
	}
	if act, err = Tanh(xwb); err != nil {
		fmt.Printf("Err while Tanh: %v\n", err)
	}
	_ = act // will never happen

case2:

	// The following is how to set up the exact same network

	// First we set up our environment
	//
	// Now, remember all these functions no longer return (*Node, error). Instead they return `Result`
	var mul = Lift2(Mul)
	var tanh = Lift1(Tanh)
	var add = Lift2Broadcast(BroadcastAdd)

	// First we set up the components
	h := NewGraph()
	w2 := NewMatrix(h, Float32, WithShape(features, size), WithName("w"), WithInit(GlorotU(1)))
	b2 := NewMatrix(h, Float32, WithShape(1, size), WithName("b"), WithInit(Zeroes()))
	x2 := NewMatrix(h, Float32, WithShape(n, features), WithName("x"))

	// Then we write the expression
	act2 := tanh(add(mul(x2, w2), b2, nil, []byte{1}))

	// REMEMBER: act2 is not a *Node! It is a Result
	fmt.Printf("act2: %v\n", act2)

	// To extract error, use CheckOne
	fmt.Printf("error: %v\n", CheckOne(act2))

	// If you extract the *Node from an error, you get nil
	fmt.Printf("Node: %v\n", act2.Node())

	// Output:
	// Err while Add: Failed to infer shape. Op: + false: Shape mismatch: (32, 100) and (1, 10000)
	// act2: Failed to infer shape. Op: + false: Shape mismatch: (32, 100) and (1, 10000)
	// error: Failed to infer shape. Op: + false: Shape mismatch: (32, 100) and (1, 10000)
	// Node: <nil>
}
