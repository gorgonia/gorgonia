package tensor

import "fmt"

func Example_basics() {
	a := New(WithShape(2, 2), WithBacking([]int{1, 2, 3, 4}))
	fmt.Printf("a:\n%v\n", a)

	b := New(WithBacking(Range(Float32, 0, 24)), WithShape(2, 3, 4))
	fmt.Printf("b:\n%1.1f", b)

	// Accessing data
	x, _ := b.At(0, 1, 2) // in Numpy syntax: b[0,1,2]
	fmt.Printf("x: %v\n\n", x)

	// Setting data
	b.SetAt(float32(1000), 0, 1, 2)
	fmt.Printf("b:\n%v", b)

	// Output:
	// a:
	// ⎡1  2⎤
	// ⎣3  4⎦
	//
	// b:
	// ⎡ 0.0   1.0   2.0   3.0⎤
	// ⎢ 4.0   5.0   6.0   7.0⎥
	// ⎣ 8.0   9.0  10.0  11.0⎦
	//
	// ⎡12.0  13.0  14.0  15.0⎤
	// ⎢16.0  17.0  18.0  19.0⎥
	// ⎣20.0  21.0  22.0  23.0⎦
	//
	// x: 6
	//
	// b:
	// ⎡   0     1     2     3⎤
	// ⎢   4     5  1000     7⎥
	// ⎣   8     9    10    11⎦
	//
	// ⎡  12    13    14    15⎤
	// ⎢  16    17    18    19⎥
	// ⎣  20    21    22    23⎦
}
