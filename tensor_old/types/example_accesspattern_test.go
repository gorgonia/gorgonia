package types_test

import (
	"fmt"

	. "github.com/chewxy/gorgonia/tensor/f64"
	t "github.com/chewxy/gorgonia/tensor/types"
)

// note: rs is a simple Slice implementation. You should implement your own
type rs struct {
	start, end int
}

func (s rs) Start() int { return s.start }
func (s rs) End() int   { return s.end }
func (s rs) Step() int  { return 1 }

func doc(s string) {
	fmt.Println(s)
}

// Whilst FlatIterator is mostly used for internal implementations of Tensor types,
// it is good to know the underlying reason for doing things.
//
// The rationale for the FlatIterator can be rather simply illustrated.
func ExampleFlatIterator_rationale() {
	doc("Let's say we have a 5x5 Matrix:")
	A := NewTensor(WithBacking(RangeFloat64(0, 25)), WithShape(5, 5))
	fmt.Printf("A: %+v\n", A)

	doc("Now we slice it A[1:3, 2:4]")
	B, _ := A.Slice(rs{1, 3}, rs{2, 4})
	fmt.Printf("B: %+v\n", B)

	doc("This seems nice. But we need to be aware of the underlying data.")
	doc("The underlying data of a Tensor is a flat contiguous slice.")
	fmt.Printf("A.data: %v\n", A.Data())
	fmt.Printf("B.data %v\n", B.Data())

	fmt.Println()

	doc("Access to the data of a Tensor is controlled by the *AP")
	fmt.Printf("A: %v\n", A.AP)
	fmt.Printf("B: %v\n", B.AP)
	fmt.Println()

	doc("In particular, note the stride of B.")
	doc("At B[0, 1] (value == 8), the next item is not 1 element away, it is 5.")
	doc("It can get hairy trying to figure out what the next element is, and mistakes can be easily made.")
	doc("hence the need for the iterator.\n")

	doc("So now we'll use an illustrative example. Bear in mind that the index returned is for the underlying data")
	it := t.NewFlatIterator(B.AP)
	bd := B.Data().([]float64)
	fmt.Printf("Next coordinate: %v - ", it.Coord())

	var next int
	var err error
	for next, err = it.Next(); err == nil; next, err = it.Next() {
		fmt.Printf("index: %v. Value: %v\nNext coordinate: %v - ", next, bd[next], it.Coord())
	}

	if _, ok := err.(NoOpError); err != nil && !ok {
		panic(err) // oops, something reaaaally wrong has happened
	}
	fmt.Println("Overflowed. Done iterating")

	// Output:
	// Let's say we have a 5x5 Matrix:
	// A: Matrix (5, 5) [5 1]
	// ⎡ 0   1   2   3   4⎤
	// ⎢ 5   6   7   8   9⎥
	// ⎢10  11  12  13  14⎥
	// ⎢15  16  17  18  19⎥
	// ⎣20  21  22  23  24⎦

	// Now we slice it A[1:3, 2:4]
	// B: Matrix (2, 2) [5 1]
	// ⎡ 7   8⎤
	// ⎣12  13⎦

	// This seems nice. But we need to be aware of the underlying data.
	// The underlying data of a Tensor is a flat contiguous slice.
	// A.data: [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]
	// B.data [7 8 9 10 11 12 13]

	// Access to the data of a Tensor is controlled by the *AP
	// A: Shape: (5, 5), Stride: [5 1], Dims: 2, Lock: true
	// B: Shape: (2, 2), Stride: [5 1], Dims: 2, Lock: true

	// In particular, note the stride of B.
	// At B[0, 1] (value == 8), the next item is not 1 element away, it is 5.
	// It can get hairy trying to figure out what the next element is, and mistakes can be easily made.
	// hence the need for the iterator.

	// So now we'll use an illustrative example. Bear in mind that the index returned is for the underlying data
	// Next coordinate: [0 0] - index: 0. Value: 7
	// Next coordinate: [0 1] - index: 1. Value: 8
	// Next coordinate: [1 0] - index: 5. Value: 12
	// Next coordinate: [1 1] - index: 6. Value: 13
	// Next coordinate: [0 0] - Overflowed. Done iterating
	// want:
	// A: Matrix (5, 5) [5 1]
	// ⎡ 0   1   2   3   4⎤
	// ⎢ 5   6   7   8   9⎥
	// ⎢10  11  12  13  14⎥
	// ⎢15  16  17  18  19⎥
	// ⎣20  21  22  23  24⎦

	// B: Matrix (2, 2) [5 1]
	// ⎡ 7   8⎤
	// ⎣12  13⎦

	// A.data: [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]
	// B.data [7 8 9 10 11 12 13]

	// A: Shape: (5, 5), Stride: [5 1], Dims: 2, Lock: true
	// B: Shape: (2, 2), Stride: [5 1], Dims: 2, Lock: true

	// Next coordinate: [0 0] - index: 0. Value: 7
	// Next coordinate: [0 1] - index: 1. Value: 8
	// Next coordinate: [1 0] - index: 5. Value: 12
	// Next coordinate: [1 1] - index: 6. Value: 13
	// Next coordinate: [0 0] - Overflowed. Done iterating

}
