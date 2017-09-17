package tensor_test

import (
	"fmt"

	"gorgonia.org/tensor"
)

// In this example, we want to handle basic tensor operations for arbitray types (slicing, stacking, transposing)

// LongStruct is a type that is an arbitrarily long struct
type LongStruct struct {
	a, b, c, d, e uint64
}

// Format implements fmt.Formatter for easier-to-read output of data
func (ls LongStruct) Format(s fmt.State, c rune) {
	fmt.Fprintf(s, "{a: %d, b: %d, c: %d, d: %d, e: %d}", ls.a, ls.b, ls.c, ls.d, ls.e)
}

type s int

func (ss s) Start() int { return int(ss) }
func (ss s) End() int   { return int(ss) + 1 }
func (ss s) Step() int  { return 1 }

func Example_TransposeExtension() {
	// For documentation if you're reading this on godoc:
	//
	// type LongStruct struct {
	// 		a, b, c, d, e uint64
	// }

	T := tensor.New(tensor.WithShape(2, 2),
		tensor.WithBacking([]LongStruct{
			LongStruct{0, 0, 0, 0, 0},
			LongStruct{1, 1, 1, 1, 1},
			LongStruct{2, 2, 2, 2, 2},
			LongStruct{3, 3, 3, 3, 3},
		}),
	)
	fmt.Printf("Before:\n%v\n", T)
	retVal, _ := tensor.Transpose(T) // an alternative would be to use T.T(); T.Transpose()
	fmt.Printf("After:\n%v\n", retVal)

	// Output:
	// Before:
	// ⎡{a: 0, b: 0, c: 0, d: 0, e: 0}  {a: 1, b: 1, c: 1, d: 1, e: 1}⎤
	// ⎣{a: 2, b: 2, c: 2, d: 2, e: 2}  {a: 3, b: 3, c: 3, d: 3, e: 3}⎦
	//
	// After:
	// ⎡{a: 0, b: 0, c: 0, d: 0, e: 0}  {a: 2, b: 2, c: 2, d: 2, e: 2}⎤
	// ⎣{a: 2, b: 2, c: 2, d: 2, e: 2}  {a: 3, b: 3, c: 3, d: 3, e: 3}⎦
}

func Example_stackExtension() {
	// For documentation if you're reading this on godoc:
	//
	// type LongStruct struct {
	// a, b, c, d, e uint64
	// }

	T := tensor.New(tensor.WithShape(2, 2),
		tensor.WithBacking([]LongStruct{
			LongStruct{0, 0, 0, 0, 0},
			LongStruct{1, 1, 1, 1, 1},
			LongStruct{2, 2, 2, 2, 2},
			LongStruct{3, 3, 3, 3, 3},
		}),
	)
	S, _ := T.Slice(nil, s(1)) // s is a type that implements tensor.Slice
	T2 := tensor.New(tensor.WithShape(2, 2),
		tensor.WithBacking([]LongStruct{
			LongStruct{10, 10, 10, 10, 10},
			LongStruct{11, 11, 11, 11, 11},
			LongStruct{12, 12, 12, 12, 12},
			LongStruct{13, 13, 13, 13, 13},
		}),
	)
	S2, _ := T2.Slice(nil, s(0))

	// an alternative would be something like this
	// T3, _ := S.(*tensor.Dense).Stack(1, S2.(*tensor.Dense))
	T3, _ := tensor.Stack(1, S, S2)
	fmt.Printf("Stacked:\n%v", T3)

	// Output:
	// Stacked:
	// ⎡     {a: 1, b: 1, c: 1, d: 1, e: 1}  {a: 10, b: 10, c: 10, d: 10, e: 10}⎤
	// ⎣     {a: 3, b: 3, c: 3, d: 3, e: 3}  {a: 12, b: 12, c: 12, d: 12, e: 12}⎦
}
