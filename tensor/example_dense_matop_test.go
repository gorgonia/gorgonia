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
	T, _ = T.(Slicer).Slice(nil, ss(1)) // ss is unexported
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

func ExampleDense_Hstack() {
	var T, T1, T2, T3 *Dense
	var err error
	T = New(WithBacking(Range(Float64, 0, 4)), WithShape(2, 2))
	T1 = New(WithBacking([]float64{1000, 2000}), WithShape(2, 1))

	// Simple example
	if T2, err = T.Hstack(T1); err == nil {
		fmt.Printf("T.Hstack(T1):\n%v\n", T2)
	}

	// This fails, because they are not the same shape
	T1.Reshape(2)
	if _, err = T.Hstack(T1); err != nil {
		fmt.Printf("Error: %v\n\n", err)
	}

	// You can stack more than one, as long as all the tensors have the same shape
	T1.Reshape(2, 1)
	T3 = T1.Clone().(*Dense)
	if T2, err = T.Hstack(T1, T3); err == nil {
		fmt.Printf("T.Hstack(T1, T3):\n%v\n", T2)
	}

	// Compatible shapes can be stacked
	T1 = New(Of(Float64), WithShape(2, 3))
	if T2, err = T.Hstack(T1); err == nil {
		fmt.Printf("Hstacking (2,2) with (2,3):\n%v\n", T2)
	}

	// Special attention to vectors - vectors can only be stacked with vectors
	T = New(WithBacking([]float64{1000, 2000}))
	T1 = New(WithBacking([]float64{0, 1}), WithShape(1, 2))
	if _, err = T.Hstack(T1); err != nil {
		fmt.Printf("Hstacking (2) with (1,2): %v\n", err)
	}

	// Now let's look at failure conditions, or unhandled situations

	// Incompatible shapes cannot be stacked
	T1.Reshape(3, 2)
	if _, err = T.Hstack(T1); err != nil {
		fmt.Printf("Hstacking (2,2) with (3,2): %v\n", err)
	}

	// Obviously you can't stack a scalar onto tensors (or the other way around)
	T1 = New(FromScalar(1.0))
	if _, err = T.Hstack(T1); err != nil {
		fmt.Printf("Hstacking a scalar onto a tensor: %v\n", err)
	}
	if _, err = T1.Hstack(T); err != nil {
		fmt.Printf("Hstacking a tensor onto a scalar: %v\n", err)
	}

	// Output:
	// T.Hstack(T1):
	// ⎡   0     1  1000⎤
	// ⎣   2     3  2000⎦
	//
	// Error: Failed to perform Concat: Unable to find new shape that results from concatenation: Dimension mismatch. Expected 2, got 1
	//
	// T.Hstack(T1, T3):
	// ⎡   0     1  1000  1000⎤
	// ⎣   2     3  2000  2000⎦
	//
	// Hstacking (2,2) with (2,3):
	// ⎡0  1  0  0  0⎤
	// ⎣2  3  0  0  0⎦
	//
	// Hstacking (2) with (1,2): Failed to perform Concat: Unable to find new shape that results from concatenation: Dimension mismatch. Expected 1, got 2
	// Hstacking (2,2) with (3,2): Failed to perform Concat: Unable to find new shape that results from concatenation: Dimension mismatch. Expected 1, got 2
	// Hstacking a scalar onto a tensor: Tensor has to be at least 1 dimensions
	// Hstacking a tensor onto a scalar: Tensor has to be at least 1 dimensions
}

func ExampleDense_Vstack() {
	var T, T1, T2, T3 *Dense
	var err error

	T = New(WithBacking(Range(Float64, 0, 4)), WithShape(2, 2))
	T1 = New(WithBacking([]float64{1000, 2000}), WithShape(1, 2))

	// Simple example
	if T2, err = T.Vstack(T1); err == nil {
		fmt.Printf("T.Vstack(T1):\n%v\n", T2)
	} else {
		fmt.Printf("%+v", err)
	}

	// You can stack more than one, as long as all the tensors have the same shape
	T3 = T1.Clone().(*Dense)
	if T2, err = T.Vstack(T1, T3); err == nil {
		fmt.Printf("T.Vstack(T1, T3):\n%v\n", T2)
	}

	// Let's look at failure conditions
	// All tensors must be at least 2D
	T.Reshape(4)
	if _, err = T.Vstack(T1); err != nil {
		fmt.Printf("Vstacking (4) with (1, 2): %v\n", err)
	}
	if _, err = T1.Vstack(T); err != nil {
		fmt.Printf("Vstacking (1, 2) with (4): %v\n", err)
	}

	// Output:
	// T.Vstack(T1):
	// ⎡   0     1⎤
	// ⎢   2     3⎥
	// ⎣1000  2000⎦
	//
	// T.Vstack(T1, T3):
	// ⎡   0     1⎤
	// ⎢   2     3⎥
	// ⎢1000  2000⎥
	// ⎣1000  2000⎦
	//
	// Vstacking (4) with (1, 2): Tensor has to be at least 2 dimensions
	// Vstacking (1, 2) with (4): Tensor has to be at least 2 dimensions
}
