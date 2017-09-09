package tensor

import "fmt"

func Example_iterator() {
	fmt.Println("Row Major")
	T := New(WithShape(2, 3), Of(Float64))
	it := IteratorFromDense(T)

	for i, err := it.Start(); err == nil; i, err = it.Next() {
		fmt.Printf("i: %d, coord: %v\n", i, it.Coord())
	}
	/*
		// FOR WHEN COL MAJOR IS SUPPORTED
		fmt.Println("Col Major")
		T = New(WithShape(2, 3), Of(Float64), AsFortran())
		it = IteratorFromDense(T)

		for i, err := it.Start(); err == nil; i, err = it.Next() {
			fmt.Printf("i: %d, coord: %v\n", i, it.Coord())
		}
	*/

	// Output:
	// Row Major
	// i: 0, coord: [0 1]
	// i: 1, coord: [0 2]
	// i: 2, coord: [1 0]
	// i: 3, coord: [1 1]
	// i: 4, coord: [1 2]
	// i: 5, coord: [0 0]
}
