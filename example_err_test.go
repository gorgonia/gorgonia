package gorgonia_test

import (
	"fmt"

	. "gorgonia.org/gorgonia"
)

// Gorgonia provides an API that is fairly idiomatic - most of the functions in in the API return (T, error).
// This is useful for many cases, such as an interactive shell for deep learning.
// However, it must also be acknowledged that this makes composing functions together a bit cumbersome.
//
// To that end, Gorgonia provides two alternative methods. First, the `Lift` based functions; Second the `Must` function
func Example_errorHandling() {
	// Lift
	g := NewGraph()
	x := NewMatrix(g, Float32, WithShape(2, 3), WithInit(RangedFrom(0)), WithName("a"))
	y := NewMatrix(g, Float32, WithShape(3, 2), WithInit(ValuesOf(float32(2))), WithName("b"))
	z := NewMatrix(g, Float32, WithShape(2, 1), WithInit(Zeroes()), WithName("bias"))
	wrong := NewMatrix(g, Float64, WithShape(2, 3), WithInit(RangedFrom(0)), WithName("wrong"))

	// Different LiftXXX functions exist for different API signatures
	// A good way to do this is to have some instantiated functions at the top level of the package
	mul := Lift2(Mul)
	add := Lift2(Add)
	addB := Lift2Broadcast(BroadcastAdd)
	sq := Lift1(Square)
	sm := Lift1Axial(SoftMax)

	nn := sm(sq(addB(mul(x, y), z, nil, []byte{1}))) // OK
	nnPlusWrong := add(nn, wrong)                    // Wrong types. Will Error
	fmt.Printf("nn: %v\nAn error occurs: %v\n", nn, nnPlusWrong.Err())

	// Must()
	h := NewGraph()
	a := NewMatrix(h, Float32, WithShape(2, 3), WithInit(RangedFrom(0)), WithName("a"))
	b := NewMatrix(h, Float32, WithShape(3, 2), WithInit(ValuesOf(float32(2))), WithName("b"))
	c := NewMatrix(h, Float32, WithShape(2, 1), WithInit(RangedFrom(0)), WithName("c"))
	wrong2 := NewMatrix(h, Float64, WithShape(2, 3), WithInit(RangedFrom(0)), WithName("wrong"))

	// This is OK
	nn2 := Must(SoftMax(
		Must(Square(
			Must(BroadcastAdd(
				Must(Mul(a, b)),
				c,
				nil, []byte{1},
			)),
		)),
	))
	fmt.Printf("nn2: %v\n", nn2)

	defer func() {
		if r := recover(); r != nil {
			fmt.Printf("An error occurs (caught by recover()): %v\n", r)
		}
	}()
	nn2PlusWrong := Must(Add(nn2, wrong2))
	_ = nn2PlusWrong

	// Output:
	// nn: ÷ false(%9, %d) :: Matrix float32
	// An error occurs: Type inference error. Op: + false. Children: [Matrix float32, Matrix float64], OpType:Matrix a → Matrix a → Matrix a: Unable to unify while inferring type of + false: Unification Fail: float64 ~ float32 cannot be unified
	// nn2: ÷ false(%9, %d) :: Matrix float32
	// An error occurs (caught by recover()): Type inference error. Op: + false. Children: [Matrix float32, Matrix float64], OpType:Matrix a → Matrix a → Matrix a: Unable to unify while inferring type of + false: Unification Fail: float64 ~ float32 cannot be unified

}
