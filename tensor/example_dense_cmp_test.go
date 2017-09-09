package tensor

import "fmt"

// Comparison functions return a Tensor of bool by default. To return the same type, simply pass in the AsSameType function option
func ExampleDense_Gt_basic() {
	var T1, T2, T3, V *Dense
	var sliced Tensor
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T3, _ = T1.Gt(T2)
	fmt.Println("Basic operations are safe\n=========================\nT3 = T1 > T2")
	fmt.Printf("T3:\n%v\n", T3)

	// To return the same type, use the AsSameType function option
	T3, _ = T1.Gt(T2, AsSameType())
	fmt.Println("Returning same type\n===================")
	fmt.Printf("T3 (Returns Same Type):\n%v\n", T3)

	// Sliced tensors are safe too
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 1, 5)), WithShape(2, 2))
	T3, _ = V.Gt(T2)
	fmt.Printf("Safe slicing\n============\nT3:\n%v\nT1 remains unchanged:\n%v\n", T3, T1)

	// Simliarly for tensors that return the same type
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 1, 5)), WithShape(2, 2))
	T3, _ = V.Gt(T2, AsSameType()) // AsSameType returns a tensor of the same type
	fmt.Printf("Safe slicing (Same type)\n========================\nT3:\n%v\nT1 remains unchanged:\n%v\n", T3, T1)

	// Output:
	// Basic operations are safe
	// =========================
	// T3 = T1 > T2
	// T3:
	// ⎡false  false  false⎤
	// ⎢false  false  false⎥
	// ⎣false  false  false⎦
	//
	// Returning same type
	// ===================
	// T3 (Returns Same Type):
	// ⎡0  0  0⎤
	// ⎢0  0  0⎥
	// ⎣0  0  0⎦
	//
	// Safe slicing
	// ============
	// T3:
	// ⎡false  false⎤
	// ⎣false  false⎦
	//
	// T1 remains unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
	//
	// Safe slicing (Same type)
	// ========================
	// T3:
	// ⎡0  0⎤
	// ⎣0  0⎦
	//
	// T1 remains unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
}

// If the UseUnsafe function option is passed into the call, the assumption is made that it will be returning the same type
func ExampleDense_Gt_unsafe() {
	var T1, T2, V *Dense
	var sliced Tensor
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T1.Gt(T2, UseUnsafe())
	fmt.Printf("Unsafe operation\n================\nT1:\n%v\n", T1)

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 1, 5)), WithShape(2, 2))
	V.Gt(T2, UseUnsafe())
	fmt.Printf("Unsafe operation, with a sliced Tensor\n======================================\nT1:\n%v", T1)

	// Output:
	// Unsafe operation
	// ================
	// T1:
	// ⎡0  0  0⎤
	// ⎢0  0  0⎥
	// ⎣0  0  0⎦
	//
	// Unsafe operation, with a sliced Tensor
	// ======================================
	// T1:
	// ⎡0  0  2⎤
	// ⎢0  0  5⎥
	// ⎣6  7  8⎦
}

// The WithReuse function option can be used to pass in reuse tensors. But be sure to also use the AsSameType() function option
// or else funny results will happen
func ExampleDense_Gt_reuse() {
	var T1, T2, T3, V *Dense
	var sliced Tensor
	// The reuse tensor is a Tensor of bools...
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T3 = New(WithBacking([]bool{
		true, false, true,
		false, true, false,
		true, false, true}), WithShape(3, 3))
	T1.Gt(T2, WithReuse(T3)) // note that AsSameType is not used here
	fmt.Printf("Default behaviour: Reuse tensor is expected to be of Bools\n==========================================================\nT3:\n%v\n", T3)

	// If you want to use a Reuse tensor of the same type, then besure to also pass in the AsSameType() flag
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T3 = New(WithBacking(Range(Float64, 100, 109)), WithShape(3, 3)) // The reuse tensor is a Tensor of Float64...
	T1.Gt(T2, WithReuse(T3), AsSameType())                           // AsSameType is used to return float64s
	fmt.Printf("Reuse With Same Type\n=====================\nT3:\n%v\n", T3)

	// Slicing is similar:
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 0, 4)), WithShape(2, 2))
	T3 = New(WithBacking([]bool{true, true, true, true}), WithShape(2, 2))
	V.Gt(T2, WithReuse(T3))
	fmt.Printf("Reuse on sliced tensors\n======================\nT3\n%v\n", T3)

	// Again, bear in mind same types
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 0, 4)), WithShape(2, 2))
	T3 = New(WithBacking(Range(Float64, 100, 104)), WithShape(2, 2))
	V.Gt(T2, WithReuse(T3), AsSameType())
	fmt.Printf("Reuse on sliced tensors (same type)\n=================================\nT3\n%v\n", T3)

	// Output:
	// Default behaviour: Reuse tensor is expected to be of Bools
	// ==========================================================
	// T3:
	// ⎡false  false  false⎤
	// ⎢false  false  false⎥
	// ⎣false  false  false⎦
	//
	// Reuse With Same Type
	// =====================
	// T3:
	// ⎡0  0  0⎤
	// ⎢0  0  0⎥
	// ⎣0  0  0⎦
	//
	// Reuse on sliced tensors
	// ======================
	// T3
	// ⎡false  false⎤
	// ⎣ true   true⎦
	//
	// Reuse on sliced tensors (same type)
	// =================================
	// T3
	// ⎡0  0⎤
	// ⎣1  1⎦
}

/* GTE */

// Comparison functions return a Tensor of bool by default. To return the same type, simply pass in the AsSameType function option
func ExampleDense_Gte_basic() {
	var T1, T2, T3, V *Dense
	var sliced Tensor
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T3, _ = T1.Gte(T2)
	fmt.Println("Basic operations are safe\n=========================\nT3 = T1 >= T2")
	fmt.Printf("T3:\n%v\n", T3)

	// To return the same type, use the AsSameType function option
	T3, _ = T1.Gte(T2, AsSameType())
	fmt.Println("Returning same type\n===================")
	fmt.Printf("T3 (Returns Same Type):\n%v\n", T3)

	// Sliced tensors are safe too
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 1, 5)), WithShape(2, 2))
	T3, _ = V.Gte(T2)
	fmt.Printf("Safe slicing\n============\nT3:\n%v\nT1 remains unchanged:\n%v\n", T3, T1)

	// Simliarly for tensors that return the same type
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 1, 5)), WithShape(2, 2))
	T3, _ = V.Gte(T2, AsSameType()) // AsSameType returns a tensor of the same type
	fmt.Printf("Safe slicing (Same type)\n========================\nT3:\n%v\nT1 remains unchanged:\n%v\n", T3, T1)

	// Output:
	// Basic operations are safe
	// =========================
	// T3 = T1 >= T2
	// T3:
	// ⎡true  true  true⎤
	// ⎢true  true  true⎥
	// ⎣true  true  true⎦
	//
	// Returning same type
	// ===================
	// T3 (Returns Same Type):
	// ⎡1  1  1⎤
	// ⎢1  1  1⎥
	// ⎣1  1  1⎦
	//
	// Safe slicing
	// ============
	// T3:
	// ⎡false  false⎤
	// ⎣ true   true⎦
	//
	// T1 remains unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
	//
	// Safe slicing (Same type)
	// ========================
	// T3:
	// ⎡0  0⎤
	// ⎣1  1⎦
	//
	// T1 remains unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
}

// If the UseUnsafe function option is passed into the call, the assumption is made that it will be returning the same type
func ExampleDense_Gte_unsafe() {
	var T1, T2, V *Dense
	var sliced Tensor
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T1.Gte(T2, UseUnsafe())
	fmt.Printf("Unsafe operation\n================\nT1:\n%v\n", T1)

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 1, 5)), WithShape(2, 2))
	V.Gte(T2, UseUnsafe())
	fmt.Printf("Unsafe operation, with a sliced Tensor\n======================================\nT1:\n%v", T1)

	// Output:
	// Unsafe operation
	// ================
	// T1:
	// ⎡1  1  1⎤
	// ⎢1  1  1⎥
	// ⎣1  1  1⎦
	//
	// Unsafe operation, with a sliced Tensor
	// ======================================
	// T1:
	// ⎡0  0  2⎤
	// ⎢1  1  5⎥
	// ⎣6  7  8⎦
}

// The WithReuse function option can be used to pass in reuse tensors. But be sure to also use the AsSameType() function option
// or else funny results will happen
func ExampleDense_Gte_reuse() {
	var T1, T2, T3, V *Dense
	var sliced Tensor
	// The reuse tensor is a Tensor of bools...
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T3 = New(WithBacking([]bool{
		true, false, true,
		false, true, false,
		true, false, true}), WithShape(3, 3))
	T1.Gte(T2, WithReuse(T3)) // note that AsSameType is not used here
	fmt.Printf("Default behaviour: Reuse tensor is expected to be of Bools\n==========================================================\nT3:\n%v\n", T3)

	// If you want to use a Reuse tensor of the same type, then besure to also pass in the AsSameType() flag
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T3 = New(WithBacking(Range(Float64, 100, 109)), WithShape(3, 3)) // The reuse tensor is a Tensor of Float64...
	T1.Gte(T2, WithReuse(T3), AsSameType())                          // AsSameType is used to return float64s
	fmt.Printf("Reuse With Same Type\n=====================\nT3:\n%v\n", T3)

	// Slicing is similar:
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 0, 4)), WithShape(2, 2))
	T3 = New(WithBacking([]bool{true, true, true, true}), WithShape(2, 2))
	V.Gte(T2, WithReuse(T3))
	fmt.Printf("Reuse on sliced tensors\n======================\nT3\n%v\n", T3)

	// Again, bear in mind same types
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 0, 4)), WithShape(2, 2))
	T3 = New(WithBacking(Range(Float64, 100, 104)), WithShape(2, 2))
	V.Gte(T2, WithReuse(T3), AsSameType())
	fmt.Printf("Reuse on sliced tensors (same type)\n=================================\nT3\n%v\n", T3)

	// Output:
	// Default behaviour: Reuse tensor is expected to be of Bools
	// ==========================================================
	// T3:
	// ⎡true  true  true⎤
	// ⎢true  true  true⎥
	// ⎣true  true  true⎦
	//
	// Reuse With Same Type
	// =====================
	// T3:
	// ⎡1  1  1⎤
	// ⎢1  1  1⎥
	// ⎣1  1  1⎦
	//
	// Reuse on sliced tensors
	// ======================
	// T3
	// ⎡true  true⎤
	// ⎣true  true⎦
	//
	// Reuse on sliced tensors (same type)
	// =================================
	// T3
	// ⎡1  1⎤
	// ⎣1  1⎦
}

/* LT */

// Comparison functions return a Tensor of bool by default. To return the same type, simply pass in the AsSameType function option
func ExampleDense_Lt_basic() {
	var T1, T2, T3, V *Dense
	var sliced Tensor
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T3, _ = T1.Lt(T2)
	fmt.Println("Basic operations are safe\n=========================\nT3 = T1 < T2")
	fmt.Printf("T3:\n%v\n", T3)

	// To return the same type, use the AsSameType function option
	T3, _ = T1.Lt(T2, AsSameType())
	fmt.Println("Returning same type\n===================")
	fmt.Printf("T3 (Returns Same Type):\n%v\n", T3)

	// Sliced tensors are safe too
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 1, 5)), WithShape(2, 2))
	T3, _ = V.Lt(T2)
	fmt.Printf("Safe slicing\n============\nT3:\n%v\nT1 remains unchanged:\n%v\n", T3, T1)

	// Simliarly for tensors that return the same type
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 1, 5)), WithShape(2, 2))
	T3, _ = V.Lt(T2, AsSameType()) // AsSameType returns a tensor of the same type
	fmt.Printf("Safe slicing (Same type)\n========================\nT3:\n%v\nT1 remains unchanged:\n%v\n", T3, T1)

	// Output:
	// Basic operations are safe
	// =========================
	// T3 = T1 < T2
	// T3:
	// ⎡false  false  false⎤
	// ⎢false  false  false⎥
	// ⎣false  false  false⎦
	//
	// Returning same type
	// ===================
	// T3 (Returns Same Type):
	// ⎡0  0  0⎤
	// ⎢0  0  0⎥
	// ⎣0  0  0⎦
	//
	// Safe slicing
	// ============
	// T3:
	// ⎡ true   true⎤
	// ⎣false  false⎦
	//
	// T1 remains unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
	//
	// Safe slicing (Same type)
	// ========================
	// T3:
	// ⎡1  1⎤
	// ⎣0  0⎦
	//
	// T1 remains unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
}

// If the UseUnsafe function option is passed into the call, the assumption is made that it will be returning the same type
func ExampleDense_Lt_unsafe() {
	var T1, T2, V *Dense
	var sliced Tensor
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T1.Lt(T2, UseUnsafe())
	fmt.Printf("Unsafe operation\n================\nT1:\n%v\n", T1)

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 1, 5)), WithShape(2, 2))
	V.Lt(T2, UseUnsafe())
	fmt.Printf("Unsafe operation, with a sliced Tensor\n======================================\nT1:\n%v", T1)

	// Output:
	// Unsafe operation
	// ================
	// T1:
	// ⎡0  0  0⎤
	// ⎢0  0  0⎥
	// ⎣0  0  0⎦
	//
	// Unsafe operation, with a sliced Tensor
	// ======================================
	// T1:
	// ⎡1  1  2⎤
	// ⎢0  0  5⎥
	// ⎣6  7  8⎦
}

// The WithReuse function option can be used to pass in reuse tensors. But be sure to also use the AsSameType() function option
// or else funny results will happen
func ExampleDense_Lt_reuse() {
	var T1, T2, T3, V *Dense
	var sliced Tensor
	// The reuse tensor is a Tensor of bools...
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T3 = New(WithBacking([]bool{
		true, false, true,
		false, true, false,
		true, false, true}), WithShape(3, 3))
	T1.Lt(T2, WithReuse(T3)) // note that AsSameType is not used here
	fmt.Printf("Default behaviour: Reuse tensor is expected to be of Bools\n==========================================================\nT3:\n%v\n", T3)

	// If you want to use a Reuse tensor of the same type, then besure to also pass in the AsSameType() flag
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T3 = New(WithBacking(Range(Float64, 100, 109)), WithShape(3, 3)) // The reuse tensor is a Tensor of Float64...
	T1.Lt(T2, WithReuse(T3), AsSameType())                           // AsSameType is used to return float64s
	fmt.Printf("Reuse With Same Type\n=====================\nT3:\n%v\n", T3)

	// Slicing is similar:
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 0, 4)), WithShape(2, 2))
	T3 = New(WithBacking([]bool{true, true, true, true}), WithShape(2, 2))
	V.Lt(T2, WithReuse(T3))
	fmt.Printf("Reuse on sliced tensors\n======================\nT3\n%v\n", T3)

	// Again, bear in mind same types
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 0, 4)), WithShape(2, 2))
	T3 = New(WithBacking(Range(Float64, 100, 104)), WithShape(2, 2))
	V.Lt(T2, WithReuse(T3), AsSameType())
	fmt.Printf("Reuse on sliced tensors (same type)\n=================================\nT3\n%v\n", T3)

	// Output:
	// Default behaviour: Reuse tensor is expected to be of Bools
	// ==========================================================
	// T3:
	// ⎡false  false  false⎤
	// ⎢false  false  false⎥
	// ⎣false  false  false⎦
	//
	// Reuse With Same Type
	// =====================
	// T3:
	// ⎡0  0  0⎤
	// ⎢0  0  0⎥
	// ⎣0  0  0⎦
	//
	// Reuse on sliced tensors
	// ======================
	// T3
	// ⎡false  false⎤
	// ⎣false  false⎦
	//
	// Reuse on sliced tensors (same type)
	// =================================
	// T3
	// ⎡0  0⎤
	// ⎣0  0⎦
}

/* LTE */
// Comparison functions return a Tensor of bool by default. To return the same type, simply pass in the AsSameType function option
func ExampleDense_Lte_basic() {
	var T1, T2, T3, V *Dense
	var sliced Tensor
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T3, _ = T1.Lte(T2)
	fmt.Println("Basic operations are safe\n=========================\nT3 = T1 <= T2")
	fmt.Printf("T3:\n%v\n", T3)

	// To return the same type, use the AsSameType function option
	T3, _ = T1.Lte(T2, AsSameType())
	fmt.Println("Returning same type\n===================")
	fmt.Printf("T3 (Returns Same Type):\n%v\n", T3)

	// Sliced tensors are safe too
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 1, 5)), WithShape(2, 2))
	T3, _ = V.Lte(T2)
	fmt.Printf("Safe slicing\n============\nT3:\n%v\nT1 remains unchanged:\n%v\n", T3, T1)

	// Simliarly for tensors that return the same type
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 1, 5)), WithShape(2, 2))
	T3, _ = V.Lte(T2, AsSameType()) // AsSameType returns a tensor of the same type
	fmt.Printf("Safe slicing (Same type)\n========================\nT3:\n%v\nT1 remains unchanged:\n%v\n", T3, T1)

	// Output:
	// Basic operations are safe
	// =========================
	// T3 = T1 <= T2
	// T3:
	// ⎡true  true  true⎤
	// ⎢true  true  true⎥
	// ⎣true  true  true⎦
	//
	// Returning same type
	// ===================
	// T3 (Returns Same Type):
	// ⎡1  1  1⎤
	// ⎢1  1  1⎥
	// ⎣1  1  1⎦
	//
	// Safe slicing
	// ============
	// T3:
	// ⎡true  true⎤
	// ⎣true  true⎦
	//
	// T1 remains unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
	//
	// Safe slicing (Same type)
	// ========================
	// T3:
	// ⎡1  1⎤
	// ⎣1  1⎦
	//
	// T1 remains unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
}

// If the UseUnsafe function option is passed into the call, the assumption is made that it will be returning the same type
func ExampleDense_Lte_unsafe() {
	var T1, T2, V *Dense
	var sliced Tensor
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T1.Lte(T2, UseUnsafe())
	fmt.Printf("Unsafe operation\n================\nT1:\n%v\n", T1)

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 1, 5)), WithShape(2, 2))
	V.Lte(T2, UseUnsafe())
	fmt.Printf("Unsafe operation, with a sliced Tensor\n======================================\nT1:\n%v", T1)

	// Output:
	// Unsafe operation
	// ================
	// T1:
	// ⎡1  1  1⎤
	// ⎢1  1  1⎥
	// ⎣1  1  1⎦
	//
	// Unsafe operation, with a sliced Tensor
	// ======================================
	// T1:
	// ⎡1  1  2⎤
	// ⎢1  1  5⎥
	// ⎣6  7  8⎦
}

// The WithReuse function option can be used to pass in reuse tensors. But be sure to also use the AsSameType() function option
// or else funny results will happen
func ExampleDense_Lte_reuse() {
	var T1, T2, T3, V *Dense
	var sliced Tensor
	// The reuse tensor is a Tensor of bools...
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T3 = New(WithBacking([]bool{
		true, false, true,
		false, true, false,
		true, false, true}), WithShape(3, 3))
	T1.Lte(T2, WithReuse(T3)) // note that AsSameType is not used here
	fmt.Printf("Default behaviour: Reuse tensor is expected to be of Bools\n==========================================================\nT3:\n%v\n", T3)

	// If you want to use a Reuse tensor of the same type, then besure to also pass in the AsSameType() flag
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T3 = New(WithBacking(Range(Float64, 100, 109)), WithShape(3, 3)) // The reuse tensor is a Tensor of Float64...
	T1.Lte(T2, WithReuse(T3), AsSameType())                          // AsSameType is used to return float64s
	fmt.Printf("Reuse With Same Type\n=====================\nT3:\n%v\n", T3)

	// Slicing is similar:
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 0, 4)), WithShape(2, 2))
	T3 = New(WithBacking([]bool{true, true, true, true}), WithShape(2, 2))
	V.Lte(T2, WithReuse(T3))
	fmt.Printf("Reuse on sliced tensors\n======================\nT3\n%v\n", T3)

	// Again, bear in mind same types
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 0, 4)), WithShape(2, 2))
	T3 = New(WithBacking(Range(Float64, 100, 104)), WithShape(2, 2))
	V.Lte(T2, WithReuse(T3), AsSameType())
	fmt.Printf("Reuse on sliced tensors (same type)\n=================================\nT3\n%v\n", T3)

	// Output:
	// Default behaviour: Reuse tensor is expected to be of Bools
	// ==========================================================
	// T3:
	// ⎡true  true  true⎤
	// ⎢true  true  true⎥
	// ⎣true  true  true⎦
	//
	// Reuse With Same Type
	// =====================
	// T3:
	// ⎡1  1  1⎤
	// ⎢1  1  1⎥
	// ⎣1  1  1⎦
	//
	// Reuse on sliced tensors
	// ======================
	// T3
	// ⎡ true   true⎤
	// ⎣false  false⎦
	//
	// Reuse on sliced tensors (same type)
	// =================================
	// T3
	// ⎡1  1⎤
	// ⎣0  0⎦
}

/* ELEQ */

// Comparison functions return a Tensor of bool by default. To return the same type, simply pass in the AsSameType function option
func ExampleDense_ElEq_basic() {
	var T1, T2, T3, V *Dense
	var sliced Tensor
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T3, _ = T1.ElEq(T2)
	fmt.Println("Basic operations are safe\n=========================\nT3 = T1 == T2")
	fmt.Printf("T3:\n%v\n", T3)

	// To return the same type, use the AsSameType function option
	T3, _ = T1.ElEq(T2, AsSameType())
	fmt.Println("Returning same type\n===================")
	fmt.Printf("T3 (Returns Same Type):\n%v\n", T3)

	// Sliced tensors are safe too
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 1, 5)), WithShape(2, 2))
	T3, _ = V.ElEq(T2)
	fmt.Printf("Safe slicing\n============\nT3:\n%v\nT1 remains unchanged:\n%v\n", T3, T1)

	// Simliarly for tensors that return the same type
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 1, 5)), WithShape(2, 2))
	T3, _ = V.ElEq(T2, AsSameType()) // AsSameType returns a tensor of the same type
	fmt.Printf("Safe slicing (Same type)\n========================\nT3:\n%v\nT1 remains unchanged:\n%v\n", T3, T1)

	// Output:
	// Basic operations are safe
	// =========================
	// T3 = T1 == T2
	// T3:
	// ⎡true  true  true⎤
	// ⎢true  true  true⎥
	// ⎣true  true  true⎦
	//
	// Returning same type
	// ===================
	// T3 (Returns Same Type):
	// ⎡1  1  1⎤
	// ⎢1  1  1⎥
	// ⎣1  1  1⎦
	//
	// Safe slicing
	// ============
	// T3:
	// ⎡false  false⎤
	// ⎣ true   true⎦
	//
	// T1 remains unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
	//
	// Safe slicing (Same type)
	// ========================
	// T3:
	// ⎡0  0⎤
	// ⎣1  1⎦
	//
	// T1 remains unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
}

// If the UseUnsafe function option is passed into the call, the assumption is made that it will be returning the same type
func ExampleDense_ElEq_unsafe() {
	var T1, T2, V *Dense
	var sliced Tensor
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T1.ElEq(T2, UseUnsafe())
	fmt.Printf("Unsafe operation\n================\nT1:\n%v\n", T1)

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 1, 5)), WithShape(2, 2))
	V.ElEq(T2, UseUnsafe())
	fmt.Printf("Unsafe operation, with a sliced Tensor\n======================================\nT1:\n%v", T1)

	// Output:
	// Unsafe operation
	// ================
	// T1:
	// ⎡1  1  1⎤
	// ⎢1  1  1⎥
	// ⎣1  1  1⎦
	//
	// Unsafe operation, with a sliced Tensor
	// ======================================
	// T1:
	// ⎡0  0  2⎤
	// ⎢1  1  5⎥
	// ⎣6  7  8⎦
}

// The WithReuse function option can be used to pass in reuse tensors. But be sure to also use the AsSameType() function option
// or else funny results will happen
func ExampleDense_ElEq_reuse() {
	var T1, T2, T3, V *Dense
	var sliced Tensor
	// The reuse tensor is a Tensor of bools...
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T3 = New(WithBacking([]bool{
		true, false, true,
		false, true, false,
		true, false, true}), WithShape(3, 3))
	T1.ElEq(T2, WithReuse(T3)) // note that AsSameType is not used here
	fmt.Printf("Default behaviour: Reuse tensor is expected to be of Bools\n==========================================================\nT3:\n%v\n", T3)

	// If you want to use a Reuse tensor of the same type, then besure to also pass in the AsSameType() flag
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T3 = New(WithBacking(Range(Float64, 100, 109)), WithShape(3, 3)) // The reuse tensor is a Tensor of Float64...
	T1.ElEq(T2, WithReuse(T3), AsSameType())                         // AsSameType is used to return float64s
	fmt.Printf("Reuse With Same Type\n=====================\nT3:\n%v\n", T3)

	// Slicing is similar:
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 0, 4)), WithShape(2, 2))
	T3 = New(WithBacking([]bool{true, true, true, true}), WithShape(2, 2))
	V.ElEq(T2, WithReuse(T3))
	fmt.Printf("Reuse on sliced tensors\n======================\nT3\n%v\n", T3)

	// Again, bear in mind same types
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 0, 4)), WithShape(2, 2))
	T3 = New(WithBacking(Range(Float64, 100, 104)), WithShape(2, 2))
	V.ElEq(T2, WithReuse(T3), AsSameType())
	fmt.Printf("Reuse on sliced tensors (same type)\n=================================\nT3\n%v\n", T3)

	// Output:
	// Default behaviour: Reuse tensor is expected to be of Bools
	// ==========================================================
	// T3:
	// ⎡true  true  true⎤
	// ⎢true  true  true⎥
	// ⎣true  true  true⎦
	//
	// Reuse With Same Type
	// =====================
	// T3:
	// ⎡1  1  1⎤
	// ⎢1  1  1⎥
	// ⎣1  1  1⎦
	//
	// Reuse on sliced tensors
	// ======================
	// T3
	// ⎡ true   true⎤
	// ⎣false  false⎦
	//
	// Reuse on sliced tensors (same type)
	// =================================
	// T3
	// ⎡1  1⎤
	// ⎣0  0⎦
}

/* ELNE */

// Comparison functions return a Tensor of bool by default. To return the same type, simply pass in the AsSameType function option
func ExampleDense_ElNe_basic() {
	var T1, T2, T3, V *Dense
	var sliced Tensor
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T3, _ = T1.ElNe(T2)
	fmt.Println("Basic operations are safe\n=========================\nT3 = T1 != T2")
	fmt.Printf("T3:\n%v\n", T3)

	// To return the same type, use the AsSameType function option
	T3, _ = T1.ElNe(T2, AsSameType())
	fmt.Println("Returning same type\n===================")
	fmt.Printf("T3 (Returns Same Type):\n%v\n", T3)

	// Sliced tensors are safe too
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 1, 5)), WithShape(2, 2))
	T3, _ = V.ElNe(T2)
	fmt.Printf("Safe slicing\n============\nT3:\n%v\nT1 remains unchanged:\n%v\n", T3, T1)

	// Simliarly for tensors that return the same type
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 1, 5)), WithShape(2, 2))
	T3, _ = V.ElNe(T2, AsSameType()) // AsSameType returns a tensor of the same type
	fmt.Printf("Safe slicing (Same type)\n========================\nT3:\n%v\nT1 remains unchanged:\n%v\n", T3, T1)

	// Output:
	// Basic operations are safe
	// =========================
	// T3 = T1 != T2
	// T3:
	// ⎡false  false  false⎤
	// ⎢false  false  false⎥
	// ⎣false  false  false⎦
	//
	// Returning same type
	// ===================
	// T3 (Returns Same Type):
	// ⎡0  0  0⎤
	// ⎢0  0  0⎥
	// ⎣0  0  0⎦
	//
	// Safe slicing
	// ============
	// T3:
	// ⎡ true   true⎤
	// ⎣false  false⎦
	//
	// T1 remains unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
	//
	// Safe slicing (Same type)
	// ========================
	// T3:
	// ⎡1  1⎤
	// ⎣0  0⎦
	//
	// T1 remains unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
}

// If the UseUnsafe function option is passed into the call, the assumption is made that it will be returning the same type
func ExampleDense_ElNe_unsafe() {
	var T1, T2, V *Dense
	var sliced Tensor
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T1.ElNe(T2, UseUnsafe())
	fmt.Printf("Unsafe operation\n================\nT1:\n%v\n", T1)

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 1, 5)), WithShape(2, 2))
	V.ElNe(T2, UseUnsafe())
	fmt.Printf("Unsafe operation, with a sliced Tensor\n======================================\nT1:\n%v", T1)

	// Output:
	// Unsafe operation
	// ================
	// T1:
	// ⎡0  0  0⎤
	// ⎢0  0  0⎥
	// ⎣0  0  0⎦
	//
	// Unsafe operation, with a sliced Tensor
	// ======================================
	// T1:
	// ⎡1  1  2⎤
	// ⎢0  0  5⎥
	// ⎣6  7  8⎦
}

// The WithReuse function option can be used to pass in reuse tensors. But be sure to also use the AsSameType() function option
// or else funny results will happen
func ExampleDense_ElNe_reuse() {
	var T1, T2, T3, V *Dense
	var sliced Tensor
	// The reuse tensor is a Tensor of bools...
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T3 = New(WithBacking([]bool{
		true, false, true,
		false, true, false,
		true, false, true}), WithShape(3, 3))
	T1.ElNe(T2, WithReuse(T3)) // note that AsSameType is not used here
	fmt.Printf("Default behaviour: Reuse tensor is expected to be of Bools\n==========================================================\nT3:\n%v\n", T3)

	// If you want to use a Reuse tensor of the same type, then besure to also pass in the AsSameType() flag
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T3 = New(WithBacking(Range(Float64, 100, 109)), WithShape(3, 3)) // The reuse tensor is a Tensor of Float64...
	T1.ElNe(T2, WithReuse(T3), AsSameType())                         // AsSameType is used to return float64s
	fmt.Printf("Reuse With Same Type\n=====================\nT3:\n%v\n", T3)

	// Slicing is similar:
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 0, 4)), WithShape(2, 2))
	T3 = New(WithBacking([]bool{true, true, true, true}), WithShape(2, 2))
	V.ElNe(T2, WithReuse(T3))
	fmt.Printf("Reuse on sliced tensors\n======================\nT3\n%v\n", T3)

	// Again, bear in mind same types
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 0, 4)), WithShape(2, 2))
	T3 = New(WithBacking(Range(Float64, 100, 104)), WithShape(2, 2))
	V.ElNe(T2, WithReuse(T3), AsSameType())
	fmt.Printf("Reuse on sliced tensors (same type)\n=================================\nT3\n%v\n", T3)

	// Output:
	// Default behaviour: Reuse tensor is expected to be of Bools
	// ==========================================================
	// T3:
	// ⎡false  false  false⎤
	// ⎢false  false  false⎥
	// ⎣false  false  false⎦
	//
	// Reuse With Same Type
	// =====================
	// T3:
	// ⎡0  0  0⎤
	// ⎢0  0  0⎥
	// ⎣0  0  0⎦
	//
	// Reuse on sliced tensors
	// ======================
	// T3
	// ⎡false  false⎤
	// ⎣ true   true⎦
	//
	// Reuse on sliced tensors (same type)
	// =================================
	// T3
	// ⎡0  0⎤
	// ⎣1  1⎦
}
