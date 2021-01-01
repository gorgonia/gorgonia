package gorgonia

import (
	"fmt"
)

/*
func ExampleTensordot_scalar() {
	// Scalars
	g := NewGraph()
	a := NewScalar(g, Float64, WithValue(2.0), WithName("a"))
	b := NewScalar(g, Float64, WithValue(21.0), WithName("b"))
	c, err := Tensordot([]int{0}, []int{0}, a, b)
	if err != nil {
		fmt.Printf("Cannot call Tensordot. Error: %v\n", err)
		return
	}

	vm := NewTapeMachine(g)
	if err := vm.RunAll(); err != nil {
		fmt.Printf("Cannot perform scalars. Error %v\n", err)
	}
	fmt.Printf("c: %v (%v) of %v", c.Value(), c.Value().Dtype(), c.Value().Shape())

	// Output:
	//...
}
*/
func ExampleTensordot_vectors() {
	g := NewGraph()
	a := NewVector(g, Float64, WithName("a"), WithShape(2), WithInit(RangedFrom(2)))
	b := NewVector(g, Float64, WithName("b"), WithShape(2), WithInit(RangedFrom(21)))

	c, err := Tensordot([]int{0}, []int{0}, a, b)
	if err != nil {
		fmt.Printf("Cannot call Tensordot. Error: %v\n", err)
		return
	}

	vm := NewTapeMachine(g)
	if err := vm.RunAll(); err != nil {
		fmt.Printf("Cannot perform tensordot on vectors. Error %v\n", err)
	}
	fmt.Printf("a %v b %v ", a.Value(), b.Value())
	fmt.Printf("c: %v (%v) of %v", c.Value(), c.Type(), c.Value().Shape())

	// Output:
	// a [2  3] b [21  22] c: [108] (float64) of (1)

}
