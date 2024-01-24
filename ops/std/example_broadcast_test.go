//go:build ignoreexp
// +build ignoreexp

package stdops

import (
	"context"
	"fmt"

	"github.com/chewxy/hm"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
)

func ExampleBroadcast() {
	var a, b, c values.Value
	a = tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	b = tensor.New(tensor.WithShape(1, 3), tensor.WithBacking([]float64{10, 20, 30}))

	op, err := Auto(Add(a, b), a, b)
	if err != nil {
		fmt.Printf("Cannot create Broadcast Op. Err %v\n", err)
		return
	}

	var expectedType hm.Type
	var expectedShape shapes.Shape

	// type and shape checks
	if expectedType, err = typecheck(op, a, b); err != nil {
		fmt.Printf("Expected Broadcast to pass type checking. Err: %v\n", err)
		return
	}
	if expectedShape, err = shapecheck(op, a, b); err != nil {
		fmt.Printf("Expected Broadcast to pass shape checking. Err: %v\n", err)
	}
	fmt.Printf("Expected Type: %v\nExpected Shape: %v\n", expectedType, expectedShape)

	if c, err = op.Do(context.Background(), a, b); err != nil {
		fmt.Printf("Err: %v", err)
	}
	fmt.Printf("a:\n%v\nb: %v\nc = a %v b:\n%v\n", a, b, op, c)

	// Output:
	// Expected Type: Matrix float64
	// Expected Shape: (2, 3)
	// a:
	// ⎡1  2  3⎤
	// ⎣4  5  6⎦
	//
	// b: R[10  20  30]
	// c = a ¨+ b:
	// ⎡11  22  33⎤
	// ⎣14  25  36⎦

}
