package tensor

import "fmt"

func Example_Sum() {
	T := New(WithBacking([]float64{0, 1, 2, 3}), WithShape(2, 2))
	fmt.Printf("T:\n%v\n", T)

	// sum along axis 0
	summed, _ := Sum(T, 0)
	fmt.Printf("Summed:\n%v\n", summed)

	// to keep dims, simply reshape
	summed.Reshape(1, 2)
	fmt.Printf("Summed (Kept Dims - Shape: %v):\n%v\n\n", summed.Shape(), summed)

	// summing along multiple axes
	summed, _ = Sum(T, 1, 0)
	fmt.Printf("Summed along (1, 0): %v", summed)

	// Output:
	// T:
	// ⎡0  1⎤
	// ⎣2  3⎦
	//
	// Summed:
	// [2  4]
	// Summed (Kept Dims - Shape: (1, 2)):
	// R[2  4]
	//
	// Summed along (1, 0): 6
}

func Example_Argmax() {
	T := New(WithBacking([]float64{0, 100, 200, 3}), WithShape(2, 2))
	fmt.Printf("T:\n%v\n", T)

	// argmax along the x-axis
	am, _ := Argmax(T, 0)
	fmt.Printf("Argmax: %v\n", am)
	fmt.Printf("Argmax is %T of %v", am, am.Dtype())

	// Output:
	// T:
	// ⎡  0  100⎤
	// ⎣200    3⎦
	//
	// Argmax: [1  0]
	// Argmax is *tensor.Dense of int
}

func Example_Argmin() {
	T := New(WithBacking([]float64{0, 100, 200, 3}), WithShape(2, 2))
	fmt.Printf("T:\n%v\n", T)

	// argmax along the x-axis
	am, _ := Argmin(T, 0)
	fmt.Printf("Argmin: %v\n", am)
	fmt.Printf("Argmin is %T of %v", am, am.Dtype())

	// Output:
	// T:
	// ⎡  0  100⎤
	// ⎣200    3⎦
	//
	// Argmin: [0  1]
	// Argmin is *tensor.Dense of int
}
