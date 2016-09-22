package types

import "fmt"

func ExampleFlatIterator_Coord() {
	ap := NewAP(Shape{2, 2}, []int{2, 1})
	it := NewFlatIterator(ap)

	// current coordinate is (0, 0)
	fmt.Printf("Initial Coordinate: %v\n", it.Coord())
	next, _ := it.Next()
	fmt.Printf("index for previous coordinate: %d\n", next)
	fmt.Printf("Current Coordinates: %v\n", it.Coord())

	// Output:
	// Initial Coordinate: [0 0]
	// index for previous coordinate: 0
	// Current Coordinates: [0 1]
}
