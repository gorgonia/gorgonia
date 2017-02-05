package tensor

import "fmt"

func ExampleDense_Slice() {
	var T Tensor
	T = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	fmt.Printf("T:\n%v\n", T)

	// T[0:2, 0:2]
	T, _ = T.Slice(makeRS(0, 2), makeRS(0, 2)) // makeRS is an unexported function that creates a Slice.
	fmt.Printf("T[0:2, 0:2]:\n%v\n", T)

	// T[:, 1]
	T, _ = T.Slice(nil, ss(1)) // ss is unexported
	fmt.Printf("T[:, 1]:\n%v\n", T)

	// Output:
	// T:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
	//
	// T[0:2, 0:2]:
	// ⎡0  1⎤
	// ⎣3  4⎦
	//
	// T[:, 1]:
	// [1  4]
}

// Slicing works on one dimensional arrays too:
func ExampleDense_Slice_oneDimension() {
	var T Tensor
	T = New(WithBacking(Range(Float64, 0, 9)))
	fmt.Printf("T:\n%v\n\n", T)

	T, _ = T.Slice(makeRS(0, 5))
	fmt.Printf("T[0:5]:\n%v\n", T)

	// Output:
	// T:
	// [0  1  2  3  ... 5  6  7  8]
	//
	// T[0:5]:
	// [0  1  2  3  4]

}

// Any modifications to the sliced value modifies the original slice as well
func ExampleDense_Slice_viewMutation() {
	var T, V Tensor
	T = New(WithBacking(Range(Int, 0, 16)), WithShape(4, 4))
	fmt.Printf("T:\n%v\n", T)
	V, _ = T.Slice(makeRS(1, 3), makeRS(1, 3))
	fmt.Printf("V:\n%v\n", V)

	// Now we modify V's 0th value
	V.(*Dense).Set(0, 1000)
	fmt.Printf("V[0] = 1000:\n%v\n", V)
	fmt.Printf("T is also mutated:\n%v", T)

	// Output:
	// T:
	// ⎡ 0   1   2   3⎤
	// ⎢ 4   5   6   7⎥
	// ⎢ 8   9  10  11⎥
	// ⎣12  13  14  15⎦
	//
	// V:
	// ⎡ 5   6⎤
	// ⎣ 9  10⎦
	//
	// V[0] = 1000:
	// ⎡1000     6⎤
	// ⎣   9    10⎦
	//
	// T is also mutated:
	// ⎡   0     1     2     3⎤
	// ⎢   4  1000     6     7⎥
	// ⎢   8     9    10    11⎥
	// ⎣  12    13    14    15⎦
	//
}
