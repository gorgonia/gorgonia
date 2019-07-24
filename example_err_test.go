package gorgonia_test

import (
	"fmt"

	. "gorgonia.org/gorgonia"
)

// Gorgonia provides an API that is fairly idiomatic - most of the functions in in the API return (T, error).
// This is useful for many cases, such as an interactive shell for deep learning.
// However, it must also be acknowledged that this makes composing functions together a bit cumbersome.
//
// To that end, Gorgonia provides two alternative methods.
func Example_errorHandling() {
	// Do
	g := NewGraph()
	x := NewMatrix(g, Float32, WithShape(2, 3), WithInit(RangedFrom(0)), WithName("a"))
	y := NewMatrix(g, Float32, WithShape(2, 3), WithInit(ValuesOf(float32(2))), WithName("b"))
	z := NewMatrix(g, Float64, WithShape(2, 3), WithInit(RangedFrom(0)), WithName("c"))

	xy := Do2(Add, x, y)   // OK
	xy2 := Do1(Square, xy) // OK
	xyz := Do2(Add, xy, z) // type error

	fmt.Printf("xy: %v\nxy^2: %v\n", xy, xy2)
	fmt.Printf("An error occurs: %v\n", xyz.Err())

	// Lift
	g2 := NewGraph()
	x2 := NewMatrix(g2, Float32, WithShape(2, 3), WithInit(RangedFrom(0)), WithName("a"))
	y2 := NewMatrix(g2, Float32, WithShape(2, 3), WithInit(ValuesOf(float32(2))), WithName("b"))
	z2 := NewMatrix(g2, Float64, WithShape(2, 3), WithInit(RangedFrom(0)), WithName("c"))

	add := Lift2(Add)
	sq := Lift1(Square)
	xyz2 := add(sq(add(x2, y2)), z2)
	fmt.Printf("An error occurs: %v\n", xyz2.Err())
	// note: add(x2, y2) yields the correct answer
	// note: square(...) also yields the correct answer.
	// The point of the Lift pattern is to provide a naturally composable function

	// Must()
	h := NewGraph()
	a := NewMatrix(h, Float32, WithShape(2, 3), WithInit(RangedFrom(0)), WithName("a"))
	b := NewMatrix(h, Float32, WithShape(2, 3), WithInit(ValuesOf(float32(2))), WithName("b"))
	c := NewMatrix(h, Float64, WithShape(2, 3), WithInit(RangedFrom(0)), WithName("c"))

	defer func() {
		if r := recover(); r != nil {
			fmt.Printf("An error occurs: %v\n", r)
		}
	}()
	ab := Must(Add(a, b))    // No Panic
	sqab := Must(Square(ab)) // No Panic
	fmt.Printf("ab: %v\nsqab: %v\n", ab, sqab)
	Must(Add(ab, c)) // Panic

	// Output:
	// xy: + false(%0, %1) :: Matrix float32
	// xy^2: square(%3) :: Matrix float32
	// An error occurs: Type inference error. Op: + false. Children: [Matrix float32, Matrix float64], OpType:Matrix a → Matrix a → Matrix a: Unable to unify while inferring type of + false: Unification Fail: float64 ~ float32 cannot be unified
	// An error occurs: Type inference error. Op: + false. Children: [Matrix float32, Matrix float64], OpType:Matrix a → Matrix a → Matrix a: Unable to unify while inferring type of + false: Unification Fail: float64 ~ float32 cannot be unified
	// ab: + false(%0, %1) :: Matrix float32
	// sqab: square(%3) :: Matrix float32
	// An error occurs: Type inference error. Op: + false. Children: [Matrix float32, Matrix float64], OpType:Matrix a → Matrix a → Matrix a: Unable to unify while inferring type of + false: Unification Fail: float64 ~ float32 cannot be unified

}
