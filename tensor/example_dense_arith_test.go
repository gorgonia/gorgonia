package tensor

import "fmt"

// By default, arithmetic operations are safe
func ExampleDense_Add_basic() {
	var T1, T2, T3, V *Dense
	var sliced Tensor
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 10, 19)), WithShape(3, 3))
	T3, _ = T1.Add(T2)
	fmt.Printf("Default operation is safe\n==========================\nT3 = T1 + T2\nT3:\n%v\nT1 is unchanged:\n%v\n", T3, T1)

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))
	T3, _ = V.Add(T2)
	fmt.Printf("Default operation is safe (sliced operations)\n=============================================\nT3 = T1[0:2, 0:2] + T2\nT3:\n%v\nT1 is unchanged:\n%v\n", T3, T1)

	// Output:
	// Default operation is safe
	// ==========================
	// T3 = T1 + T2
	// T3:
	// ⎡10  12  14⎤
	// ⎢16  18  20⎥
	// ⎣22  24  26⎦
	//
	// T1 is unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
	//
	// Default operation is safe (sliced operations)
	// =============================================
	// T3 = T1[0:2, 0:2] + T2
	// T3:
	// ⎡10  12⎤
	// ⎣15  17⎦
	//
	// T1 is unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
}

// To perform unsafe operations, use the `UseUnsafe` function option
func ExampleDense_Add_unsafe() {
	var T1, T2, T3, V *Dense
	var sliced Tensor

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 10, 19)), WithShape(3, 3))
	T3, _ = T1.Add(T2, UseUnsafe())
	fmt.Printf("Unsafe Operation\n================\nT3 = T1 + T2\nT1 == T3: %t\nT1:\n%v", T1 == T3, T1)

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))

	V.Add(T2, UseUnsafe()) // unsafe overwrites the data in T1
	fmt.Printf("Unsafe Operation on sliced Tensors\n==================================\nV = T1[0:2, 0:2] + T2\nV:\n%v\n", V)
	fmt.Printf("Naturally, T1 is mutated too:\n%v", T1)

	// Output:
	// Unsafe Operation
	// ================
	// T3 = T1 + T2
	// T1 == T3: true
	// T1:
	// ⎡10  12  14⎤
	// ⎢16  18  20⎥
	// ⎣22  24  26⎦
	// Unsafe Operation on sliced Tensors
	// ==================================
	// V = T1[0:2, 0:2] + T2
	// V:
	// ⎡10  12⎤
	// ⎣15  17⎦
	//
	// Naturally, T1 is mutated too:
	// ⎡10  12   2⎤
	// ⎢15  17   5⎥
	// ⎣ 6   7   8⎦
}

// An optional reuse tensor can also be specified with the WithReuse function option
func ExampleDense_Add_reuse() {
	var T1, V, T2, Reuse, T3 *Dense
	var sliced Tensor

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 10, 19)), WithShape(3, 3))
	Reuse = New(WithBacking(Range(Float64, 100, 109)), WithShape(3, 3))
	T3, _ = T1.Add(T2, WithReuse(Reuse))
	fmt.Printf("Reuse tensor passed in\n======================\nT3 == Reuse: %t\nT3:\n%v\n", T3 == Reuse, T3)

	// You can also use it on operations on sliced tensors - note your reuse tensor has to be the same shape as the result
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))
	Reuse = New(WithBacking(Range(Float64, 100, 104)), WithShape(2, 2)) // same shape as result
	T3, _ = V.Add(T2, WithReuse(Reuse))
	fmt.Printf("Reuse tensor passed in (sliced tensor)\n======================================\nT3 == Reuse: %t\nT3:\n%v", T3 == Reuse, T3)

	// Output:
	// Reuse tensor passed in
	// ======================
	// T3 == Reuse: true
	// T3:
	// ⎡10  12  14⎤
	// ⎢16  18  20⎥
	// ⎣22  24  26⎦
	//
	// Reuse tensor passed in (sliced tensor)
	// ======================================
	// T3 == Reuse: true
	// T3:
	// ⎡10  12⎤
	// ⎣15  17⎦
}

// Incrementing a tensor is also a function option provided by the package
func ExampleDense_Add_incr() {
	var T1, T2, T3, Incr, V *Dense
	var sliced Tensor

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 10, 19)), WithShape(3, 3))
	Incr = New(WithBacking([]float64{100, 100, 100, 100, 100, 100, 100, 100, 100}), WithShape(3, 3))
	T3, _ = T1.Add(T2, WithIncr(Incr))
	fmt.Printf("Incr tensor passed in\n======================\nIncr += T1 + T2\nIncr == T3: %t\nT3:\n%v\n", Incr == T3, T3)

	// Operations on sliced tensor is also allowed. Note that your Incr tensor has to be the same shape as the result
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))
	Incr = New(WithBacking([]float64{100, 100, 100, 100}), WithShape(2, 2))
	T3, _ = V.Add(T2, WithIncr(Incr))
	fmt.Printf("Incr tensor passed in (sliced tensor)\n======================================\nIncr += T1 + T2\nIncr == T3: %t\nT3:\n%v\n", Incr == T3, T3)

	// Output:
	// Incr tensor passed in
	// ======================
	// Incr += T1 + T2
	// Incr == T3: true
	// T3:
	// ⎡110  112  114⎤
	// ⎢116  118  120⎥
	// ⎣122  124  126⎦
	//
	// Incr tensor passed in (sliced tensor)
	// ======================================
	// Incr += T1 + T2
	// Incr == T3: true
	// T3:
	// ⎡110  112⎤
	// ⎣115  117⎦
}

/* SUB */

// By default, arithmetic operations are safe
func ExampleDense_Sub_basic() {
	var T1, T2, T3, V *Dense
	var sliced Tensor
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 10, 19)), WithShape(3, 3))
	T3, _ = T1.Sub(T2)
	fmt.Printf("Default operation is safe\n==========================\nT3 = T1 - T2\nT3:\n%v\nT1 is unchanged:\n%v\n", T3, T1)

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))
	T3, _ = V.Sub(T2)
	fmt.Printf("Default operation is safe (sliced operations)\n=============================================\nT3 = T1[0:2, 0:2] + T2\nT3:\n%v\nT1 is unchanged:\n%v\n", T3, T1)

	// Output:
	// Default operation is safe
	// ==========================
	// T3 = T1 - T2
	// T3:
	// ⎡-10  -10  -10⎤
	// ⎢-10  -10  -10⎥
	// ⎣-10  -10  -10⎦
	//
	// T1 is unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
	//
	// Default operation is safe (sliced operations)
	// =============================================
	// T3 = T1[0:2, 0:2] + T2
	// T3:
	// ⎡-10  -10⎤
	// ⎣ -9   -9⎦
	//
	// T1 is unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
}

// To perform unsafe operations, use the `UseUnsafe` function option
func ExampleDense_Sub_unsafe() {
	var T1, T2, T3, V *Dense
	var sliced Tensor

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 10, 19)), WithShape(3, 3))
	T3, _ = T1.Sub(T2, UseUnsafe())
	fmt.Printf("Unsafe Operation\n================\nT3 = T1 - T2\nT1 == T3: %t\nT1:\n%v", T1 == T3, T1)

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))

	V.Sub(T2, UseUnsafe()) // unsafe overwrites the data in T1
	fmt.Printf("Unsafe Operation on sliced Tensors\n==================================\nV = T1[0:2, 0:2] + T2\nV:\n%v\n", V)
	fmt.Printf("Naturally, T1 is mutated too:\n%v", T1)

	// Output:
	// Unsafe Operation
	// ================
	// T3 = T1 - T2
	// T1 == T3: true
	// T1:
	// ⎡-10  -10  -10⎤
	// ⎢-10  -10  -10⎥
	// ⎣-10  -10  -10⎦
	// Unsafe Operation on sliced Tensors
	// ==================================
	// V = T1[0:2, 0:2] + T2
	// V:
	// ⎡-10  -10⎤
	// ⎣ -9   -9⎦
	//
	// Naturally, T1 is mutated too:
	// ⎡-10  -10    2⎤
	// ⎢ -9   -9    5⎥
	// ⎣  6    7    8⎦
}

// An optional reuse tensor can also be specified with the WithReuse function option
func ExampleDense_Sub_reuse() {
	var T1, V, T2, Reuse, T3 *Dense
	var sliced Tensor

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 10, 19)), WithShape(3, 3))
	Reuse = New(WithBacking(Range(Float64, 100, 109)), WithShape(3, 3))
	T3, _ = T1.Sub(T2, WithReuse(Reuse))
	fmt.Printf("Reuse tensor passed in\n======================\nT3 == Reuse: %t\nT3:\n%v\n", T3 == Reuse, T3)

	// You can also use it on operations on sliced tensors - note your reuse tensor has to be the same shape as the result
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))
	Reuse = New(WithBacking(Range(Float64, 100, 104)), WithShape(2, 2)) // same shape as result
	T3, _ = V.Sub(T2, WithReuse(Reuse))
	fmt.Printf("Reuse tensor passed in (sliced tensor)\n======================================\nT3 == Reuse: %t\nT3:\n%v", T3 == Reuse, T3)

	// Output:
	// Reuse tensor passed in
	// ======================
	// T3 == Reuse: true
	// T3:
	// ⎡-10  -10  -10⎤
	// ⎢-10  -10  -10⎥
	// ⎣-10  -10  -10⎦
	//
	// Reuse tensor passed in (sliced tensor)
	// ======================================
	// T3 == Reuse: true
	// T3:
	// ⎡-10  -10⎤
	// ⎣ -9   -9⎦
}

// Incrementing a tensor is also a function option provided by the package
func ExampleDense_Sub_incr() {
	var T1, T2, T3, Incr, V *Dense
	var sliced Tensor

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 10, 19)), WithShape(3, 3))
	Incr = New(WithBacking([]float64{100, 100, 100, 100, 100, 100, 100, 100, 100}), WithShape(3, 3))
	T3, _ = T1.Sub(T2, WithIncr(Incr))
	fmt.Printf("Incr tensor passed in\n======================\nIncr += T1 - T2\nIncr == T3: %t\nT3:\n%v\n", Incr == T3, T3)

	// Operations on sliced tensor is also allowed. Note that your Incr tensor has to be the same shape as the result
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))
	Incr = New(WithBacking([]float64{100, 100, 100, 100}), WithShape(2, 2))
	T3, _ = V.Sub(T2, WithIncr(Incr))
	fmt.Printf("Incr tensor passed in (sliced tensor)\n======================================\nIncr += T1 + T2\nIncr == T3: %t\nT3:\n%v\n", Incr == T3, T3)

	// Output:
	// Incr tensor passed in
	// ======================
	// Incr += T1 - T2
	// Incr == T3: true
	// T3:
	// ⎡90  90  90⎤
	// ⎢90  90  90⎥
	// ⎣90  90  90⎦
	//
	// Incr tensor passed in (sliced tensor)
	// ======================================
	// Incr += T1 + T2
	// Incr == T3: true
	// T3:
	// ⎡90  90⎤
	// ⎣91  91⎦
}

/* MUL */

// By default, arithmetic operations are safe
func ExampleDense_Mul_basic() {
	var T1, T2, T3, V *Dense
	var sliced Tensor
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 10, 19)), WithShape(3, 3))
	T3, _ = T1.Mul(T2)
	fmt.Printf("Default operation is safe\n==========================\nT3 = T1 × T2\nT3:\n%v\nT1 is unchanged:\n%v\n", T3, T1)

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))
	T3, _ = V.Mul(T2)
	fmt.Printf("Default operation is safe (sliced operations)\n=============================================\nT3 = T1[0:2, 0:2] × T2\nT3:\n%v\nT1 is unchanged:\n%v\n", T3, T1)

	// Output:
	// Default operation is safe
	// ==========================
	// T3 = T1 × T2
	// T3:
	// ⎡  0   11   24⎤
	// ⎢ 39   56   75⎥
	// ⎣ 96  119  144⎦
	//
	// T1 is unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
	//
	// Default operation is safe (sliced operations)
	// =============================================
	// T3 = T1[0:2, 0:2] × T2
	// T3:
	// ⎡ 0  11⎤
	// ⎣36  52⎦
	//
	// T1 is unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
}

// To perform unsafe operations, use the `UseUnsafe` function option
func ExampleDense_Mul_unsafe() {
	var T1, T2, T3, V *Dense
	var sliced Tensor

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 10, 19)), WithShape(3, 3))
	T3, _ = T1.Mul(T2, UseUnsafe())
	fmt.Printf("Unsafe Operation\n================\nT3 = T1 × T2\nT1 == T3: %t\nT1:\n%v", T1 == T3, T1)

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))

	V.Mul(T2, UseUnsafe()) // unsafe overwrites the data in T1
	fmt.Printf("Unsafe Operation on sliced Tensors\n==================================\nV = T1[0:2, 0:2] × T2\nV:\n%v\n", V)
	fmt.Printf("Naturally, T1 is mutated too:\n%v", T1)

	// Output:
	// Unsafe Operation
	// ================
	// T3 = T1 × T2
	// T1 == T3: true
	// T1:
	// ⎡  0   11   24⎤
	// ⎢ 39   56   75⎥
	// ⎣ 96  119  144⎦
	// Unsafe Operation on sliced Tensors
	// ==================================
	// V = T1[0:2, 0:2] × T2
	// V:
	// ⎡ 0  11⎤
	// ⎣36  52⎦
	//
	// Naturally, T1 is mutated too:
	// ⎡ 0  11   2⎤
	// ⎢36  52   5⎥
	// ⎣ 6   7   8⎦
}

// An optional reuse tensor can also be specified with the WithReuse function option
func ExampleDense_Mul_reuse() {
	var T1, V, T2, Reuse, T3 *Dense
	var sliced Tensor

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 10, 19)), WithShape(3, 3))
	Reuse = New(WithBacking(Range(Float64, 100, 109)), WithShape(3, 3))
	T3, _ = T1.Mul(T2, WithReuse(Reuse))
	fmt.Printf("Reuse tensor passed in\n======================\nT3 == Reuse: %t\nT3:\n%v\n", T3 == Reuse, T3)

	// You can also use it on operations on sliced tensors - note your reuse tensor has to be the same shape as the result
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))
	Reuse = New(WithBacking(Range(Float64, 100, 104)), WithShape(2, 2)) // same shape as result
	T3, _ = V.Mul(T2, WithReuse(Reuse))
	fmt.Printf("Reuse tensor passed in (sliced tensor)\n======================================\nT3 == Reuse: %t\nT3:\n%v", T3 == Reuse, T3)

	// Output:
	// Reuse tensor passed in
	// ======================
	// T3 == Reuse: true
	// T3:
	// ⎡  0   11   24⎤
	// ⎢ 39   56   75⎥
	// ⎣ 96  119  144⎦
	//
	// Reuse tensor passed in (sliced tensor)
	// ======================================
	// T3 == Reuse: true
	// T3:
	// ⎡ 0  11⎤
	// ⎣36  52⎦
}

// Incrementing a tensor is also a function option provided by the package
func ExampleDense_Mul_incr() {
	var T1, T2, T3, Incr, V *Dense
	var sliced Tensor

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 10, 19)), WithShape(3, 3))
	Incr = New(WithBacking([]float64{100, 100, 100, 100, 100, 100, 100, 100, 100}), WithShape(3, 3))
	T3, _ = T1.Mul(T2, WithIncr(Incr))
	fmt.Printf("Incr tensor passed in\n======================\nIncr += T1 × T2\nIncr == T3: %t\nT3:\n%v\n", Incr == T3, T3)

	// Operations on sliced tensor is also allowed. Note that your Incr tensor has to be the same shape as the result
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))
	Incr = New(WithBacking([]float64{100, 100, 100, 100}), WithShape(2, 2))
	T3, _ = V.Mul(T2, WithIncr(Incr))
	fmt.Printf("Incr tensor passed in (sliced tensor)\n======================================\nIncr += T1 × T2\nIncr == T3: %t\nT3:\n%v\n", Incr == T3, T3)

	// Output:
	// Incr tensor passed in
	// ======================
	// Incr += T1 × T2
	// Incr == T3: true
	// T3:
	// ⎡100  111  124⎤
	// ⎢139  156  175⎥
	// ⎣196  219  244⎦
	//
	// Incr tensor passed in (sliced tensor)
	// ======================================
	// Incr += T1 × T2
	// Incr == T3: true
	// T3:
	// ⎡100  111⎤
	// ⎣136  152⎦
}

/* DIV */

// By default, arithmetic operations are safe
func ExampleDense_Div_basic() {
	var T1, T2, T3, V *Dense
	var sliced Tensor
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 10, 19)), WithShape(3, 3))
	T3, _ = T1.Div(T2)
	fmt.Printf("Default operation is safe\n==========================\nT3 = T1 ÷ T2\nT3:\n%1.1v\nT1 is unchanged:\n%1.1v\n", T3, T1)

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))
	T3, _ = V.Div(T2)
	fmt.Printf("Default operation is safe (sliced operations)\n=============================================\nT3 = T1[0:2, 0:2] ÷ T2\nT3:\n%1.1v\nT1 is unchanged:\n%1.1v\n", T3, T1)

	// Output:
	// Default operation is safe
	// ==========================
	// T3 = T1 ÷ T2
	// T3:
	// ⎡   0  0.09   0.2⎤
	// ⎢ 0.2   0.3   0.3⎥
	// ⎣ 0.4   0.4   0.4⎦
	//
	// T1 is unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
	//
	// Default operation is safe (sliced operations)
	// =============================================
	// T3 = T1[0:2, 0:2] ÷ T2
	// T3:
	// ⎡   0  0.09⎤
	// ⎣ 0.2   0.3⎦
	//
	// T1 is unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
}

// To perform unsafe operations, use the `UseUnsafe` function option
func ExampleDense_Div_unsafe() {
	var T1, T2, T3, V *Dense
	var sliced Tensor

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 10, 19)), WithShape(3, 3))
	T3, _ = T1.Div(T2, UseUnsafe())
	fmt.Printf("Unsafe Operation\n================\nT3 = T1 ÷ T2\nT1 == T3: %t\nT1:\n%1.1v", T1 == T3, T1)

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))

	V.Div(T2, UseUnsafe()) // unsafe overwrites the data in T1
	fmt.Printf("Unsafe Operation on sliced Tensors\n==================================\nV = T1[0:2, 0:2] ÷ T2\nV:\n%1.1v\n", V)
	fmt.Printf("Naturally, T1 is mutated too:\n%1.1v", T1)

	// Output:
	// Unsafe Operation
	// ================
	// T3 = T1 ÷ T2
	// T1 == T3: true
	// T1:
	// ⎡   0  0.09   0.2⎤
	// ⎢ 0.2   0.3   0.3⎥
	// ⎣ 0.4   0.4   0.4⎦
	// Unsafe Operation on sliced Tensors
	// ==================================
	// V = T1[0:2, 0:2] ÷ T2
	// V:
	// ⎡   0  0.09⎤
	// ⎣ 0.2   0.3⎦
	//
	// Naturally, T1 is mutated too:
	// ⎡   0  0.09     2⎤
	// ⎢ 0.2   0.3     5⎥
	// ⎣   6     7     8⎦
}

// An optional reuse tensor can also be specified with the WithReuse function option
func ExampleDense_Div_reuse() {
	var T1, V, T2, Reuse, T3 *Dense
	var sliced Tensor

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 10, 19)), WithShape(3, 3))
	Reuse = New(WithBacking(Range(Float64, 100, 109)), WithShape(3, 3))
	T3, _ = T1.Div(T2, WithReuse(Reuse))
	fmt.Printf("Reuse tensor passed in\n======================\nT3 == Reuse: %t\nT3:\n%1.1v\n", T3 == Reuse, T3)

	// You can also use it on operations on sliced tensors - note your reuse tensor has to be the same shape as the result
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))
	Reuse = New(WithBacking(Range(Float64, 100, 104)), WithShape(2, 2)) // same shape as result
	T3, _ = V.Div(T2, WithReuse(Reuse))
	fmt.Printf("Reuse tensor passed in (sliced tensor)\n======================================\nT3 == Reuse: %t\nT3:\n%1.1v", T3 == Reuse, T3)

	// Output:
	// Reuse tensor passed in
	// ======================
	// T3 == Reuse: true
	// T3:
	// ⎡   0  0.09   0.2⎤
	// ⎢ 0.2   0.3   0.3⎥
	// ⎣ 0.4   0.4   0.4⎦
	//
	// Reuse tensor passed in (sliced tensor)
	// ======================================
	// T3 == Reuse: true
	// T3:
	// ⎡   0  0.09⎤
	// ⎣ 0.2   0.3⎦
}

// Incrementing a tensor is also a function option provided by the package
func ExampleDense_Div_incr() {
	var T1, T2, T3, Incr, V *Dense
	var sliced Tensor

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 10, 19)), WithShape(3, 3))
	Incr = New(WithBacking([]float64{100, 100, 100, 100, 100, 100, 100, 100, 100}), WithShape(3, 3))
	T3, _ = T1.Div(T2, WithIncr(Incr))
	fmt.Printf("Incr tensor passed in\n======================\nIncr += T1 ÷ T2\nIncr == T3: %t\nT3:\n%1.5v\n", Incr == T3, T3)

	// Operations on sliced tensor is also allowed. Note that your Incr tensor has to be the same shape as the result
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))
	Incr = New(WithBacking([]float64{100, 100, 100, 100}), WithShape(2, 2))
	T3, _ = V.Div(T2, WithIncr(Incr))
	fmt.Printf("Incr tensor passed in (sliced tensor)\n======================================\nIncr += T1 ÷ T2\nIncr == T3: %t\nT3:\n%1.5v\n", Incr == T3, T3)

	// Output:
	// Incr tensor passed in
	// ======================
	// Incr += T1 ÷ T2
	// Incr == T3: true
	// T3:
	// ⎡   100  100.09  100.17⎤
	// ⎢100.23  100.29  100.33⎥
	// ⎣100.38  100.41  100.44⎦
	//
	// Incr tensor passed in (sliced tensor)
	// ======================================
	// Incr += T1 ÷ T2
	// Incr == T3: true
	// T3:
	// ⎡   100  100.09⎤
	// ⎣100.25  100.31⎦
}

/* POW */

// By default, arithmetic operations are safe
func ExampleDense_Pow_basic() {
	var T1, T2, T3, V *Dense
	var sliced Tensor
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 10, 19)), WithShape(3, 3))
	T3, _ = T1.Pow(T2)
	fmt.Printf("Default operation is safe\n==========================\nT3 = T1 ^ T2\nT3:\n%1.1v\nT1 is unchanged:\n%v\n", T3, T1)

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))
	T3, _ = V.Pow(T2)
	fmt.Printf("Default operation is safe (sliced operations)\n=============================================\nT3 = T1[0:2, 0:2] ^ T2\nT3:\n%1.1v\nT1 is unchanged:\n%v\n", T3, T1)

	// Output:
	// Default operation is safe
	// ==========================
	// T3 = T1 ^ T2
	// T3:
	// ⎡    0      1  4e+03⎤
	// ⎢2e+06  3e+08  3e+10⎥
	// ⎣3e+12  2e+14  2e+16⎦
	//
	// T1 is unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
	//
	// Default operation is safe (sliced operations)
	// =============================================
	// T3 = T1[0:2, 0:2] ^ T2
	// T3:
	// ⎡    0      1⎤
	// ⎣5e+05  7e+07⎦
	//
	// T1 is unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
}

// To perform unsafe operations, use the `UseUnsafe` function option
func ExampleDense_Pow_unsafe() {
	var T1, T2, T3, V *Dense
	var sliced Tensor

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 10, 19)), WithShape(3, 3))
	T3, _ = T1.Pow(T2, UseUnsafe())
	fmt.Printf("Unsafe Operation\n================\nT3 = T1 ^ T2\nT1 == T3: %t\nT1:\n%1.1v\n", T1 == T3, T1)

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))

	V.Pow(T2, UseUnsafe()) // unsafe overwrites the data in T1
	fmt.Printf("Unsafe Operation on sliced Tensors\n==================================\nV = T1[0:2, 0:2] ^ T2\nV:\n%1.1v\n", V)
	fmt.Printf("Naturally, T1 is mutated too:\n%1.1v", T1)

	// Output:
	// Unsafe Operation
	// ================
	// T3 = T1 ^ T2
	// T1 == T3: true
	// T1:
	// ⎡    0      1  4e+03⎤
	// ⎢2e+06  3e+08  3e+10⎥
	// ⎣3e+12  2e+14  2e+16⎦
	//
	// Unsafe Operation on sliced Tensors
	// ==================================
	// V = T1[0:2, 0:2] ^ T2
	// V:
	// ⎡    0      1⎤
	// ⎣5e+05  7e+07⎦
	//
	// Naturally, T1 is mutated too:
	// ⎡    0      1      2⎤
	// ⎢5e+05  7e+07      5⎥
	// ⎣    6      7      8⎦
}

// An optional reuse tensor can also be specified with the WithReuse function option
func ExampleDense_Pow_reuse() {
	var T1, V, T2, Reuse, T3 *Dense
	var sliced Tensor

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 10, 19)), WithShape(3, 3))
	Reuse = New(WithBacking(Range(Float64, 100, 109)), WithShape(3, 3))
	T3, _ = T1.Pow(T2, WithReuse(Reuse))
	fmt.Printf("Reuse tensor passed in\n======================\nT3 == Reuse: %t\nT3:\n%1.1v\n", T3 == Reuse, T3)

	// You can also use it on operations on sliced tensors - note your reuse tensor has to be the same shape as the result
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))
	Reuse = New(WithBacking(Range(Float64, 100, 104)), WithShape(2, 2)) // same shape as result
	T3, _ = V.Pow(T2, WithReuse(Reuse))
	fmt.Printf("Reuse tensor passed in (sliced tensor)\n======================================\nT3 == Reuse: %t\nT3:\n%1.v", T3 == Reuse, T3)

	// Output:
	// Reuse tensor passed in
	// ======================
	// T3 == Reuse: true
	// T3:
	// ⎡    0      1  4e+03⎤
	// ⎢2e+06  3e+08  3e+10⎥
	// ⎣3e+12  2e+14  2e+16⎦
	//
	// Reuse tensor passed in (sliced tensor)
	// ======================================
	// T3 == Reuse: true
	// T3:
	// ⎡    0      1⎤
	// ⎣5e+05  7e+07⎦
}

// Incrementing a tensor is also a function option provided by the package
func ExampleDense_Pow_incr() {
	var T1, T2, T3, Incr, V *Dense
	var sliced Tensor

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 10, 19)), WithShape(3, 3))
	Incr = New(WithBacking([]float64{100, 100, 100, 100, 100, 100, 100, 100, 100}), WithShape(3, 3))
	T3, _ = T1.Pow(T2, WithIncr(Incr))
	fmt.Printf("Incr tensor passed in\n======================\nIncr += T1 ^ T2\nIncr == T3: %t\nT3:\n%1.5v\n", Incr == T3, T3)

	// Operations on sliced tensor is also allowed. Note that your Incr tensor has to be the same shape as the result
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))
	Incr = New(WithBacking([]float64{100, 100, 100, 100}), WithShape(2, 2))
	T3, _ = V.Pow(T2, WithIncr(Incr))
	fmt.Printf("Incr tensor passed in (sliced tensor)\n======================================\nIncr += T1 ^ T2\nIncr == T3: %t\nT3:\n%1.5v\n", Incr == T3, T3)

	// Output:
	// Incr tensor passed in
	// ======================
	// Incr += T1 ^ T2
	// Incr == T3: true
	// T3:
	// ⎡       100         101        4196⎤
	// ⎢1.5944e+06  2.6844e+08  3.0518e+10⎥
	// ⎣2.8211e+12  2.3263e+14  1.8014e+16⎦
	//
	// Incr tensor passed in (sliced tensor)
	// ======================================
	// Incr += T1 ^ T2
	// Incr == T3: true
	// T3:
	// ⎡       100         101⎤
	// ⎣5.3154e+05  6.7109e+07⎦
}

/* MOD */

// By default, arithmetic operations are safe
func ExampleDense_Mod_basic() {
	var T1, T2, T3, V *Dense
	var sliced Tensor
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 10, 19)), WithShape(3, 3))
	T3, _ = T1.Mod(T2)
	fmt.Printf("Default operation is safe\n==========================\nT3 = T1 %% T2\nT3:\n%v\nT1 is unchanged:\n%v\n", T3, T1)

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))
	T3, _ = V.Mod(T2)
	fmt.Printf("Default operation is safe (sliced operations)\n=============================================\nT3 = T1[0:2, 0:2] %% T2\nT3:\n%v\nT1 is unchanged:\n%v\n", T3, T1)

	// Output:
	// Default operation is safe
	// ==========================
	// T3 = T1 % T2
	// T3:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
	//
	// T1 is unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
	//
	// Default operation is safe (sliced operations)
	// =============================================
	// T3 = T1[0:2, 0:2] % T2
	// T3:
	// ⎡0  1⎤
	// ⎣3  4⎦
	//
	// T1 is unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
}

// To perform unsafe operations, use the `UseUnsafe` function option
func ExampleDense_Mod_unsafe() {
	var T1, T2, T3, V *Dense
	var sliced Tensor

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 10, 19)), WithShape(3, 3))
	T3, _ = T1.Mod(T2, UseUnsafe())
	fmt.Printf("Unsafe Operation\n================\nT3 = T1 %% T2\nT1 == T3: %t\nT1:\n%v\n", T1 == T3, T1)

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))

	V.Mod(T2, UseUnsafe()) // unsafe overwrites the data in T1
	fmt.Printf("Unsafe Operation on sliced Tensors\n==================================\nV = T1[0:2, 0:2] %% T2\nV:\n%v\n", V)
	fmt.Printf("Naturally, T1 is mutated too:\n%v", T1)

	// Output:
	// Unsafe Operation
	// ================
	// T3 = T1 % T2
	// T1 == T3: true
	// T1:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
	//
	// Unsafe Operation on sliced Tensors
	// ==================================
	// V = T1[0:2, 0:2] % T2
	// V:
	// ⎡0  1⎤
	// ⎣3  4⎦
	//
	// Naturally, T1 is mutated too:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
}

// An optional reuse tensor can also be specified with the WithReuse function option
func ExampleDense_Mod_reuse() {
	var T1, V, T2, Reuse, T3 *Dense
	var sliced Tensor

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 10, 19)), WithShape(3, 3))
	Reuse = New(WithBacking(Range(Float64, 100, 109)), WithShape(3, 3))
	T3, _ = T1.Mod(T2, WithReuse(Reuse))
	fmt.Printf("Reuse tensor passed in\n======================\nT3 == Reuse: %t\nT3:\n%v\n", T3 == Reuse, T3)

	// You can also use it on operations on sliced tensors - note your reuse tensor has to be the same shape as the result
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))
	Reuse = New(WithBacking(Range(Float64, 100, 104)), WithShape(2, 2)) // same shape as result
	T3, _ = V.Mod(T2, WithReuse(Reuse))
	fmt.Printf("Reuse tensor passed in (sliced tensor)\n======================================\nT3 == Reuse: %t\nT3:\n%v", T3 == Reuse, T3)

	// Output:
	// Reuse tensor passed in
	// ======================
	// T3 == Reuse: true
	// T3:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
	//
	// Reuse tensor passed in (sliced tensor)
	// ======================================
	// T3 == Reuse: true
	// T3:
	// ⎡0  1⎤
	// ⎣3  4⎦
}

// Incrementing a tensor is also a function option provided by the package
func ExampleDense_Mod_incr() {
	var T1, T2, T3, Incr, V *Dense
	var sliced Tensor

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 10, 19)), WithShape(3, 3))
	Incr = New(WithBacking([]float64{100, 100, 100, 100, 100, 100, 100, 100, 100}), WithShape(3, 3))
	T3, _ = T1.Mod(T2, WithIncr(Incr))
	fmt.Printf("Incr tensor passed in\n======================\nIncr += T1 %% T2\nIncr == T3: %t\nT3:\n%v\n", Incr == T3, T3)

	// Operations on sliced tensor is also allowed. Note that your Incr tensor has to be the same shape as the result
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))
	Incr = New(WithBacking([]float64{100, 100, 100, 100}), WithShape(2, 2))
	T3, _ = V.Mod(T2, WithIncr(Incr))
	fmt.Printf("Incr tensor passed in (sliced tensor)\n======================================\nIncr += T1 %% T2\nIncr == T3: %t\nT3:\n%v\n", Incr == T3, T3)

	// Output:
	// Incr tensor passed in
	// ======================
	// Incr += T1 % T2
	// Incr == T3: true
	// T3:
	// ⎡100  101  102⎤
	// ⎢103  104  105⎥
	// ⎣106  107  108⎦
	//
	// Incr tensor passed in (sliced tensor)
	// ======================================
	// Incr += T1 % T2
	// Incr == T3: true
	// T3:
	// ⎡100  101⎤
	// ⎣103  104⎦
}

/* TENSOR-SCALAR AND VICE VERSA OPERATIONS */

/* ADD */

// By default, arithmetic operations are safe
func ExampleDense_AddScalar_basic() {
	var T1, T3, V *Dense
	var sliced Tensor
	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	T3, _ = T1.AddScalar(float32(5), true)
	fmt.Printf("Default operation is safe (tensor is left operand)\n==========================\nT3 = T1 + 5\nT3:\n%v\nT1 is unchanged:\n%v\n", T3, T1)

	T3, _ = T1.AddScalar(float32(5), false)
	fmt.Printf("Default operation is safe (tensor is right operand)\n==========================\nT3 = 5 + T1\nT3:\n%v\nT1 is unchanged:\n%v\n", T3, T1)

	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(nil, makeRS(1, 3))
	V = sliced.(*Dense)
	T3, _ = V.AddScalar(float32(5), true)
	fmt.Printf("Default operation is safe (sliced operations - tensor is left operand)\n=============================================\nT3 = T1[:, 1:3] + 5\nT3:\n%v\nT1 is unchanged:\n%v\n", T3, T1)

	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(nil, makeRS(1, 3))
	V = sliced.(*Dense)
	T3, _ = V.AddScalar(float32(5), false)
	fmt.Printf("Default operation is safe (sliced operations - tensor is right operand)\n=============================================\nT3 = 5 + T1[:, 1:3]\nT3:\n%v\nT1 is unchanged:\n%v\n", T3, T1)

	// Output:
	// Default operation is safe (tensor is left operand)
	// ==========================
	// T3 = T1 + 5
	// T3:
	// ⎡ 5   6   7⎤
	// ⎢ 8   9  10⎥
	// ⎣11  12  13⎦
	//
	// T1 is unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
	//
	// Default operation is safe (tensor is right operand)
	// ==========================
	// T3 = 5 + T1
	// T3:
	// ⎡ 5   6   7⎤
	// ⎢ 8   9  10⎥
	// ⎣11  12  13⎦
	//
	// T1 is unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
	//
	// Default operation is safe (sliced operations - tensor is left operand)
	// =============================================
	// T3 = T1[:, 1:3] + 5
	// T3:
	// ⎡ 6   7⎤
	// ⎢ 9  10⎥
	// ⎣12  13⎦
	//
	// T1 is unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
	//
	// Default operation is safe (sliced operations - tensor is right operand)
	// =============================================
	// T3 = 5 + T1[:, 1:3]
	// T3:
	// ⎡ 6   7⎤
	// ⎢ 9  10⎥
	// ⎣12  13⎦
	//
	// T1 is unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
}

func ExampleDense_AddScalar_unsafe() {
	var T1, T3, V *Dense
	var sliced Tensor
	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	T3, _ = T1.AddScalar(float32(5), true, UseUnsafe())
	fmt.Printf("Operation is unsafe (tensor is left operand)\n==========================\nT3 = T1 + 5\nT3:\n%v\nT3 == T1: %t\nT1 is changed:\n%v\n", T3, T3 == T1, T1)

	T3, _ = T1.AddScalar(float32(5), false, UseUnsafe())
	fmt.Printf("Operation is unsafe (tensor is right operand)\n==========================\nT3 = 5 + T1\nT3:\n%v\nT3 == T1: %t\nT1 is changed:\n%v\n", T3, T3 == T1, T1)

	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(nil, makeRS(0, 2))
	V = sliced.(*Dense)
	T3, _ = V.AddScalar(float32(5), true, UseUnsafe())
	fmt.Printf("Operation is unsafe (sliced operations - tensor is left operand)\n=============================================\nT3 = T1[:, 0:2] + 5\nT3:\n%v\nsliced == T3: %t\nT1 is changed:\n%v\n", T3, sliced == T3, T1)

	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(nil, makeRS(0, 2))
	V = sliced.(*Dense)
	T3, _ = V.AddScalar(float32(5), false, UseUnsafe())
	fmt.Printf("Operation is unsafe (sliced operations - tensor is right operand)\n=============================================\nT3 = 5 + T1[:, 0:2]\nT3:\n%v\nsliced == T3: %t\nT1 is changed:\n%v\n", T3, sliced == T3, T1)

	// Output:
	// Operation is unsafe (tensor is left operand)
	// ==========================
	// T3 = T1 + 5
	// T3:
	// ⎡ 5   6   7⎤
	// ⎢ 8   9  10⎥
	// ⎣11  12  13⎦
	//
	// T3 == T1: true
	// T1 is changed:
	// ⎡ 5   6   7⎤
	// ⎢ 8   9  10⎥
	// ⎣11  12  13⎦
	//
	// Operation is unsafe (tensor is right operand)
	// ==========================
	// T3 = 5 + T1
	// T3:
	// ⎡10  11  12⎤
	// ⎢13  14  15⎥
	// ⎣16  17  18⎦
	//
	// T3 == T1: true
	// T1 is changed:
	// ⎡10  11  12⎤
	// ⎢13  14  15⎥
	// ⎣16  17  18⎦
	//
	// Operation is unsafe (sliced operations - tensor is left operand)
	// =============================================
	// T3 = T1[:, 0:2] + 5
	// T3:
	// ⎡ 5   6⎤
	// ⎢ 8   9⎥
	// ⎣11  12⎦
	//
	// sliced == T3: true
	// T1 is changed:
	// ⎡ 5   6   2⎤
	// ⎢ 8   9   5⎥
	// ⎣11  12   8⎦
	//
	// Operation is unsafe (sliced operations - tensor is right operand)
	// =============================================
	// T3 = 5 + T1[:, 0:2]
	// T3:
	// ⎡ 5   6⎤
	// ⎢ 8   9⎥
	// ⎣11  12⎦
	//
	// sliced == T3: true
	// T1 is changed:
	// ⎡ 5   6   2⎤
	// ⎢ 8   9   5⎥
	// ⎣11  12   8⎦
}

// Reuse tensors may be used, with the WithReuse() function option.
func ExampleDense_AddScalar_reuse() {
	var T1, V, Reuse, T3 *Dense
	var sliced Tensor

	// Tensor is left operand
	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	Reuse = New(WithBacking(Range(Float32, 100, 109)), WithShape(3, 3))
	T3, _ = T1.AddScalar(float32(5), true, WithReuse(Reuse))
	fmt.Printf("Reuse tensor passed in (Tensor is left operand)\n======================\nT3 == Reuse: %t\nT3:\n%v\n", T3 == Reuse, T3)

	// Tensor is right operand
	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	Reuse = New(WithBacking(Range(Float32, 100, 109)), WithShape(3, 3))
	T3, _ = T1.AddScalar(float32(5), false, WithReuse(Reuse))
	fmt.Printf("Reuse tensor passed in (Tensor is right operand)\n======================\nT3 == Reuse: %t\nT3:\n%v\n", T3 == Reuse, T3)

	// Tensor is left operand
	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	Reuse = New(WithBacking(Range(Float32, 100, 104)), WithShape(2, 2)) // same shape as result
	T3, _ = V.AddScalar(float32(5), true, WithReuse(Reuse))
	fmt.Printf("Reuse tensor passed in (sliced tensor - Tensor is left operand)\n======================================\nT3 == Reuse: %t\nT3:\n%v\n", T3 == Reuse, T3)

	// Tensor is left operand
	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	Reuse = New(WithBacking(Range(Float32, 100, 104)), WithShape(2, 2)) // same shape as result
	T3, _ = V.AddScalar(float32(5), false, WithReuse(Reuse))
	fmt.Printf("Reuse tensor passed in (sliced tensor - Tensor is left operand)\n======================================\nT3 == Reuse: %t\nT3:\n%v", T3 == Reuse, T3)

	// Output:
	// Reuse tensor passed in (Tensor is left operand)
	// ======================
	// T3 == Reuse: true
	// T3:
	// ⎡ 5   6   7⎤
	// ⎢ 8   9  10⎥
	// ⎣11  12  13⎦
	//
	// Reuse tensor passed in (Tensor is right operand)
	// ======================
	// T3 == Reuse: true
	// T3:
	// ⎡ 5   6   7⎤
	// ⎢ 8   9  10⎥
	// ⎣11  12  13⎦
	//
	// Reuse tensor passed in (sliced tensor - Tensor is left operand)
	// ======================================
	// T3 == Reuse: true
	// T3:
	// ⎡5  6⎤
	// ⎣8  9⎦
	//
	// Reuse tensor passed in (sliced tensor - Tensor is left operand)
	// ======================================
	// T3 == Reuse: true
	// T3:
	// ⎡5  6⎤
	// ⎣8  9⎦
}

// Incrementing a tensor is also a function option provided by the package
func ExampleDense_AddScalar_incr() {
	var T1, T3, Incr, V *Dense
	var sliced Tensor

	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	Incr = New(WithBacking([]float32{100, 100, 100, 100, 100, 100, 100, 100, 100}), WithShape(3, 3))
	T3, _ = T1.AddScalar(float32(5), true, WithIncr(Incr))
	fmt.Printf("Incr tensor passed in\n======================\nIncr += T1 + T2\nIncr == T3: %t\nT3:\n%v\n", Incr == T3, T3)

	// Operations on sliced tensor is also allowed. Note that your Incr tensor has to be the same shape as the result
	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	Incr = New(WithBacking([]float32{100, 100, 100, 100}), WithShape(2, 2))
	T3, _ = V.AddScalar(float32(5), true, WithIncr(Incr))
	fmt.Printf("Incr tensor passed in (sliced tensor)\n======================================\nIncr += T1 + T2\nIncr == T3: %t\nT3:\n%v\n", Incr == T3, T3)

	// Output:
	// Incr tensor passed in
	// ======================
	// Incr += T1 + T2
	// Incr == T3: true
	// T3:
	// ⎡105  106  107⎤
	// ⎢108  109  110⎥
	// ⎣111  112  113⎦
	//
	// Incr tensor passed in (sliced tensor)
	// ======================================
	// Incr += T1 + T2
	// Incr == T3: true
	// T3:
	// ⎡105  106⎤
	// ⎣108  109⎦
}

// By default, arithmetic operations are safe
func ExampleDense_SubScalar_basic() {
	var T1, T3, V *Dense
	var sliced Tensor
	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	T3, _ = T1.SubScalar(float32(5), true)
	fmt.Printf("Default operation is safe (tensor is left operand)\n==========================\nT3 = T1 - 5\nT3:\n%v\nT1 is unchanged:\n%v\n", T3, T1)

	T3, _ = T1.SubScalar(float32(5), false)
	fmt.Printf("Default operation is safe (tensor is right operand)\n==========================\nT3 = 5 - T1\nT3:\n%v\nT1 is unchanged:\n%v\n", T3, T1)

	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(nil, makeRS(1, 3))
	V = sliced.(*Dense)
	T3, _ = V.SubScalar(float32(5), true)
	fmt.Printf("Default operation is safe (sliced operations - tensor is left operand)\n=============================================\nT3 = T1[:, 1:3] + 5\nT3:\n%v\nT1 is unchanged:\n%v\n", T3, T1)

	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(nil, makeRS(1, 3))
	V = sliced.(*Dense)
	T3, _ = V.SubScalar(float32(5), false)
	fmt.Printf("Default operation is safe (sliced operations - tensor is right operand)\n=============================================\nT3 = 5 - T1[:, 1:3]\nT3:\n%v\nT1 is unchanged:\n%v\n", T3, T1)

	// Output:
	// Default operation is safe (tensor is left operand)
	// ==========================
	// T3 = T1 - 5
	// T3:
	// ⎡-5  -4  -3⎤
	// ⎢-2  -1   0⎥
	// ⎣ 1   2   3⎦
	//
	// T1 is unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
	//
	// Default operation is safe (tensor is right operand)
	// ==========================
	// T3 = 5 - T1
	// T3:
	// ⎡ 5   4   3⎤
	// ⎢ 2   1   0⎥
	// ⎣-1  -2  -3⎦
	//
	// T1 is unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
	//
	// Default operation is safe (sliced operations - tensor is left operand)
	// =============================================
	// T3 = T1[:, 1:3] + 5
	// T3:
	// ⎡-4  -3⎤
	// ⎢-1   0⎥
	// ⎣ 2   3⎦
	//
	// T1 is unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
	//
	// Default operation is safe (sliced operations - tensor is right operand)
	// =============================================
	// T3 = 5 - T1[:, 1:3]
	// T3:
	// ⎡ 4   3⎤
	// ⎢ 1   0⎥
	// ⎣-2  -3⎦
	//
	// T1 is unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦

}

func ExampleDense_SubScalar_unsafe() {
	var T1, T3, V *Dense
	var sliced Tensor
	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	T3, _ = T1.SubScalar(float32(5), true, UseUnsafe())
	fmt.Printf("Operation is unsafe (tensor is left operand)\n==========================\nT3 = T1 - 5\nT3:\n%v\nT3 == T1: %t\nT1 is changed:\n%v\n", T3, T3 == T1, T1)

	T3, _ = T1.SubScalar(float32(5), false, UseUnsafe())
	fmt.Printf("Operation is unsafe (tensor is right operand)\n==========================\nT3 = 5 - T1\nT3:\n%v\nT3 == T1: %t\nT1 is changed:\n%v\n", T3, T3 == T1, T1)

	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(nil, makeRS(0, 2))
	V = sliced.(*Dense)
	T3, _ = V.SubScalar(float32(5), true, UseUnsafe())
	fmt.Printf("Operation is unsafe (sliced operations - tensor is left operand)\n=============================================\nT3 = T1[:, 0:2] + 5\nT3:\n%v\nsliced == T3: %t\nT1 is changed:\n%v\n", T3, sliced == T3, T1)

	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(nil, makeRS(0, 2))
	V = sliced.(*Dense)
	T3, _ = V.SubScalar(float32(5), false, UseUnsafe())
	fmt.Printf("Operation is unsafe (sliced operations - tensor is right operand)\n=============================================\nT3 = 5 - T1[:, 0:2]\nT3:\n%v\nsliced == T3: %t\nT1 is changed:\n%v\n", T3, sliced == T3, T1)

	// Output:
	// Operation is unsafe (tensor is left operand)
	// ==========================
	// T3 = T1 - 5
	// T3:
	// ⎡-5  -4  -3⎤
	// ⎢-2  -1   0⎥
	// ⎣ 1   2   3⎦
	//
	// T3 == T1: true
	// T1 is changed:
	// ⎡-5  -4  -3⎤
	// ⎢-2  -1   0⎥
	// ⎣ 1   2   3⎦
	//
	// Operation is unsafe (tensor is right operand)
	// ==========================
	// T3 = 5 - T1
	// T3:
	// ⎡10   9   8⎤
	// ⎢ 7   6   5⎥
	// ⎣ 4   3   2⎦
	//
	// T3 == T1: true
	// T1 is changed:
	// ⎡10   9   8⎤
	// ⎢ 7   6   5⎥
	// ⎣ 4   3   2⎦
	//
	// Operation is unsafe (sliced operations - tensor is left operand)
	// =============================================
	// T3 = T1[:, 0:2] + 5
	// T3:
	// ⎡-5  -4⎤
	// ⎢-2  -1⎥
	// ⎣ 1   2⎦
	//
	// sliced == T3: true
	// T1 is changed:
	// ⎡-5  -4   2⎤
	// ⎢-2  -1   5⎥
	// ⎣ 1   2   8⎦
	//
	// Operation is unsafe (sliced operations - tensor is right operand)
	// =============================================
	// T3 = 5 - T1[:, 0:2]
	// T3:
	// ⎡ 5   4⎤
	// ⎢ 2   1⎥
	// ⎣-1  -2⎦
	//
	// sliced == T3: true
	// T1 is changed:
	// ⎡ 5   4   2⎤
	// ⎢ 2   1   5⎥
	// ⎣-1  -2   8⎦
}

// Reuse tensors may be used, with the WithReuse() function option.
func ExampleDense_SubScalar_reuse() {
	var T1, V, Reuse, T3 *Dense
	var sliced Tensor

	// Tensor is left operand
	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	Reuse = New(WithBacking(Range(Float32, 100, 109)), WithShape(3, 3))
	T3, _ = T1.SubScalar(float32(5), true, WithReuse(Reuse))
	fmt.Printf("Reuse tensor passed in (Tensor is left operand)\n======================\nT3 == Reuse: %t\nT3:\n%v\n", T3 == Reuse, T3)

	// Tensor is right operand
	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	Reuse = New(WithBacking(Range(Float32, 100, 109)), WithShape(3, 3))
	T3, _ = T1.SubScalar(float32(5), false, WithReuse(Reuse))
	fmt.Printf("Reuse tensor passed in (Tensor is right operand)\n======================\nT3 == Reuse: %t\nT3:\n%v\n", T3 == Reuse, T3)

	// Tensor is left operand
	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	Reuse = New(WithBacking(Range(Float32, 100, 104)), WithShape(2, 2)) // same shape as result
	T3, _ = V.SubScalar(float32(5), true, WithReuse(Reuse))
	fmt.Printf("Reuse tensor passed in (sliced tensor - Tensor is left operand)\n======================================\nT3 == Reuse: %t\nT3:\n%v\n", T3 == Reuse, T3)

	// Tensor is left operand
	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	Reuse = New(WithBacking(Range(Float32, 100, 104)), WithShape(2, 2)) // same shape as result
	T3, _ = V.SubScalar(float32(5), false, WithReuse(Reuse))
	fmt.Printf("Reuse tensor passed in (sliced tensor - Tensor is left operand)\n======================================\nT3 == Reuse: %t\nT3:\n%v", T3 == Reuse, T3)

	// Output:
	// Reuse tensor passed in (Tensor is left operand)
	// ======================
	// T3 == Reuse: true
	// T3:
	// ⎡-5  -4  -3⎤
	// ⎢-2  -1   0⎥
	// ⎣ 1   2   3⎦
	//
	// Reuse tensor passed in (Tensor is right operand)
	// ======================
	// T3 == Reuse: true
	// T3:
	// ⎡ 5   4   3⎤
	// ⎢ 2   1   0⎥
	// ⎣-1  -2  -3⎦
	//
	// Reuse tensor passed in (sliced tensor - Tensor is left operand)
	// ======================================
	// T3 == Reuse: true
	// T3:
	// ⎡-5  -4⎤
	// ⎣-2  -1⎦
	//
	// Reuse tensor passed in (sliced tensor - Tensor is left operand)
	// ======================================
	// T3 == Reuse: true
	// T3:
	// ⎡5  4⎤
	// ⎣2  1⎦
}

// Incrementing a tensor is also a function option provided by the package
func ExampleDense_SubScalar_incr() {
	var T1, T3, Incr, V *Dense
	var sliced Tensor

	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	Incr = New(WithBacking([]float32{100, 100, 100, 100, 100, 100, 100, 100, 100}), WithShape(3, 3))
	T3, _ = T1.SubScalar(float32(5), true, WithIncr(Incr))
	fmt.Printf("Incr tensor passed in\n======================\nIncr += T1 - T2\nIncr == T3: %t\nT3:\n%v\n", Incr == T3, T3)

	// Operations on sliced tensor is also allowed. Note that your Incr tensor has to be the same shape as the result
	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	Incr = New(WithBacking([]float32{100, 100, 100, 100}), WithShape(2, 2))
	T3, _ = V.SubScalar(float32(5), true, WithIncr(Incr))
	fmt.Printf("Incr tensor passed in (sliced tensor)\n======================================\nIncr += T1 - T2\nIncr == T3: %t\nT3:\n%v\n", Incr == T3, T3)

	// Output:
	// Incr tensor passed in
	// ======================
	// Incr += T1 - T2
	// Incr == T3: true
	// T3:
	// ⎡ 95   96   97⎤
	// ⎢ 98   99  100⎥
	// ⎣101  102  103⎦
	//
	// Incr tensor passed in (sliced tensor)
	// ======================================
	// Incr += T1 - T2
	// Incr == T3: true
	// T3:
	// ⎡95  96⎤
	// ⎣98  99⎦
}

/* MUL */

// By default, arithmetic operations are safe
func ExampleDense_MulScalar_basic() {
	var T1, T3, V *Dense
	var sliced Tensor
	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	T3, _ = T1.MulScalar(float32(5), true)
	fmt.Printf("Default operation is safe (tensor is left operand)\n==========================\nT3 = T1 * 5\nT3:\n%v\nT1 is unchanged:\n%v\n", T3, T1)

	T3, _ = T1.MulScalar(float32(5), false)
	fmt.Printf("Default operation is safe (tensor is right operand)\n==========================\nT3 = 5 * T1\nT3:\n%v\nT1 is unchanged:\n%v\n", T3, T1)

	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(nil, makeRS(1, 3))
	V = sliced.(*Dense)
	T3, _ = V.MulScalar(float32(5), true)
	fmt.Printf("Default operation is safe (sliced operations - tensor is left operand)\n=============================================\nT3 = T1[:, 1:3] + 5\nT3:\n%v\nT1 is unchanged:\n%v\n", T3, T1)

	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(nil, makeRS(1, 3))
	V = sliced.(*Dense)
	T3, _ = V.MulScalar(float32(5), false)
	fmt.Printf("Default operation is safe (sliced operations - tensor is right operand)\n=============================================\nT3 = 5 * T1[:, 1:3]\nT3:\n%v\nT1 is unchanged:\n%v\n", T3, T1)

	// Output:
	// Default operation is safe (tensor is left operand)
	// ==========================
	// T3 = T1 * 5
	// T3:
	// ⎡ 0   5  10⎤
	// ⎢15  20  25⎥
	// ⎣30  35  40⎦
	//
	// T1 is unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
	//
	// Default operation is safe (tensor is right operand)
	// ==========================
	// T3 = 5 * T1
	// T3:
	// ⎡ 0   5  10⎤
	// ⎢15  20  25⎥
	// ⎣30  35  40⎦
	//
	// T1 is unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
	//
	// Default operation is safe (sliced operations - tensor is left operand)
	// =============================================
	// T3 = T1[:, 1:3] + 5
	// T3:
	// ⎡ 5  10⎤
	// ⎢20  25⎥
	// ⎣35  40⎦
	//
	// T1 is unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
	//
	// Default operation is safe (sliced operations - tensor is right operand)
	// =============================================
	// T3 = 5 * T1[:, 1:3]
	// T3:
	// ⎡ 5  10⎤
	// ⎢20  25⎥
	// ⎣35  40⎦
	//
	// T1 is unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
}

func ExampleDense_MulScalar_unsafe() {
	var T1, T3, V *Dense
	var sliced Tensor
	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	T3, _ = T1.MulScalar(float32(5), true, UseUnsafe())
	fmt.Printf("Operation is unsafe (tensor is left operand)\n==========================\nT3 = T1 * 5\nT3:\n%v\nT3 == T1: %t\nT1 is changed:\n%v\n", T3, T3 == T1, T1)

	T3, _ = T1.MulScalar(float32(5), false, UseUnsafe())
	fmt.Printf("Operation is unsafe (tensor is right operand)\n==========================\nT3 = 5 * T1\nT3:\n%v\nT3 == T1: %t\nT1 is changed:\n%v\n", T3, T3 == T1, T1)

	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(nil, makeRS(0, 2))
	V = sliced.(*Dense)
	T3, _ = V.MulScalar(float32(5), true, UseUnsafe())
	fmt.Printf("Operation is unsafe (sliced operations - tensor is left operand)\n=============================================\nT3 = T1[:, 0:2] + 5\nT3:\n%v\nsliced == T3: %t\nT1 is changed:\n%v\n", T3, sliced == T3, T1)

	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(nil, makeRS(0, 2))
	V = sliced.(*Dense)
	T3, _ = V.MulScalar(float32(5), false, UseUnsafe())
	fmt.Printf("Operation is unsafe (sliced operations - tensor is right operand)\n=============================================\nT3 = 5 * T1[:, 0:2]\nT3:\n%v\nsliced == T3: %t\nT1 is changed:\n%v\n", T3, sliced == T3, T1)

	// Output:
	// Operation is unsafe (tensor is left operand)
	// ==========================
	// T3 = T1 * 5
	// T3:
	// ⎡ 0   5  10⎤
	// ⎢15  20  25⎥
	// ⎣30  35  40⎦
	//
	// T3 == T1: true
	// T1 is changed:
	// ⎡ 0   5  10⎤
	// ⎢15  20  25⎥
	// ⎣30  35  40⎦
	//
	// Operation is unsafe (tensor is right operand)
	// ==========================
	// T3 = 5 * T1
	// T3:
	// ⎡  0   25   50⎤
	// ⎢ 75  100  125⎥
	// ⎣150  175  200⎦
	//
	// T3 == T1: true
	// T1 is changed:
	// ⎡  0   25   50⎤
	// ⎢ 75  100  125⎥
	// ⎣150  175  200⎦
	//
	// Operation is unsafe (sliced operations - tensor is left operand)
	// =============================================
	// T3 = T1[:, 0:2] + 5
	// T3:
	// ⎡ 0   5⎤
	// ⎢15  20⎥
	// ⎣30  35⎦
	//
	// sliced == T3: true
	// T1 is changed:
	// ⎡ 0   5   2⎤
	// ⎢15  20   5⎥
	// ⎣30  35   8⎦
	//
	// Operation is unsafe (sliced operations - tensor is right operand)
	// =============================================
	// T3 = 5 * T1[:, 0:2]
	// T3:
	// ⎡ 0   5⎤
	// ⎢15  20⎥
	// ⎣30  35⎦
	//
	// sliced == T3: true
	// T1 is changed:
	// ⎡ 0   5   2⎤
	// ⎢15  20   5⎥
	// ⎣30  35   8⎦
}

// Reuse tensors may be used, with the WithReuse() function option.
func ExampleDense_MulScalar_reuse() {
	var T1, V, Reuse, T3 *Dense
	var sliced Tensor

	// Tensor is left operand
	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	Reuse = New(WithBacking(Range(Float32, 100, 109)), WithShape(3, 3))
	T3, _ = T1.MulScalar(float32(5), true, WithReuse(Reuse))
	fmt.Printf("Reuse tensor passed in (Tensor is left operand)\n======================\nT3 == Reuse: %t\nT3:\n%v\n", T3 == Reuse, T3)

	// Tensor is right operand
	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	Reuse = New(WithBacking(Range(Float32, 100, 109)), WithShape(3, 3))
	T3, _ = T1.MulScalar(float32(5), false, WithReuse(Reuse))
	fmt.Printf("Reuse tensor passed in (Tensor is right operand)\n======================\nT3 == Reuse: %t\nT3:\n%v\n", T3 == Reuse, T3)

	// Tensor is left operand
	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	Reuse = New(WithBacking(Range(Float32, 100, 104)), WithShape(2, 2)) // same shape as result
	T3, _ = V.MulScalar(float32(5), true, WithReuse(Reuse))
	fmt.Printf("Reuse tensor passed in (sliced tensor - Tensor is left operand)\n======================================\nT3 == Reuse: %t\nT3:\n%v\n", T3 == Reuse, T3)

	// Tensor is left operand
	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	Reuse = New(WithBacking(Range(Float32, 100, 104)), WithShape(2, 2)) // same shape as result
	T3, _ = V.MulScalar(float32(5), false, WithReuse(Reuse))
	fmt.Printf("Reuse tensor passed in (sliced tensor - Tensor is left operand)\n======================================\nT3 == Reuse: %t\nT3:\n%v", T3 == Reuse, T3)

	// Output:
	// Reuse tensor passed in (Tensor is left operand)
	// ======================
	// T3 == Reuse: true
	// T3:
	// ⎡ 0   5  10⎤
	// ⎢15  20  25⎥
	// ⎣30  35  40⎦
	//
	// Reuse tensor passed in (Tensor is right operand)
	// ======================
	// T3 == Reuse: true
	// T3:
	// ⎡ 0   5  10⎤
	// ⎢15  20  25⎥
	// ⎣30  35  40⎦
	//
	// Reuse tensor passed in (sliced tensor - Tensor is left operand)
	// ======================================
	// T3 == Reuse: true
	// T3:
	// ⎡ 0   5⎤
	// ⎣15  20⎦
	//
	// Reuse tensor passed in (sliced tensor - Tensor is left operand)
	// ======================================
	// T3 == Reuse: true
	// T3:
	// ⎡ 0   5⎤
	// ⎣15  20⎦
}

// Incrementing a tensor is also a function option provided by the package
func ExampleDense_MulScalar_incr() {
	var T1, T3, Incr, V *Dense
	var sliced Tensor

	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	Incr = New(WithBacking([]float32{100, 100, 100, 100, 100, 100, 100, 100, 100}), WithShape(3, 3))
	T3, _ = T1.MulScalar(float32(5), true, WithIncr(Incr))
	fmt.Printf("Incr tensor passed in\n======================\nIncr += T1 * T2\nIncr == T3: %t\nT3:\n%v\n", Incr == T3, T3)

	// Operations on sliced tensor is also allowed. Note that your Incr tensor has to be the same shape as the result
	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	Incr = New(WithBacking([]float32{100, 100, 100, 100}), WithShape(2, 2))
	T3, _ = V.MulScalar(float32(5), true, WithIncr(Incr))
	fmt.Printf("Incr tensor passed in (sliced tensor)\n======================================\nIncr += T1 * T2\nIncr == T3: %t\nT3:\n%v\n", Incr == T3, T3)

	// Output:
	// Incr tensor passed in
	// ======================
	// Incr += T1 * T2
	// Incr == T3: true
	// T3:
	// ⎡100  105  110⎤
	// ⎢115  120  125⎥
	// ⎣130  135  140⎦
	//
	// Incr tensor passed in (sliced tensor)
	// ======================================
	// Incr += T1 * T2
	// Incr == T3: true
	// T3:
	// ⎡100  105⎤
	// ⎣115  120⎦
}

// By default, arithmetic operations are safe
func ExampleDense_DivScalar_basic() {
	var T1, T3, V *Dense
	var sliced Tensor
	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	T3, _ = T1.DivScalar(float32(5), true)
	fmt.Printf("Default operation is safe (tensor is left operand)\n==========================\nT3 = T1 / 5\nT3:\n%1.1v\nT1 is unchanged:\n%1.1v\n", T3, T1)

	T3, _ = T1.DivScalar(float32(5), false)
	fmt.Printf("Default operation is safe (tensor is right operand)\n==========================\nT3 = 5 / T1\nT3:\n%1.1v\nT1 is unchanged:\n%1.1v\n", T3, T1)

	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T3, _ = V.DivScalar(float32(5), true)
	fmt.Printf("Default operation is safe (sliced operations - tensor is left operand)\n=============================================\nT3 = T1[0:2, 0:2] / 5\nT3:\n%1.1v\nT1 is unchanged:\n%1.1v\n", T3, T1)

	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T3, _ = V.DivScalar(float32(5), false)
	fmt.Printf("Default operation is safe (sliced operations - tensor is right operand)\n=============================================\nT3 = 5 / T1[0:2, 0:2]\nT3:\n%1.1v\nT1 is unchanged:\n%1.1v\n", T3, T1)

	// Output:
	// Default operation is safe (tensor is left operand)
	// ==========================
	// T3 = T1 / 5
	// T3:
	// ⎡  0  0.2  0.4⎤
	// ⎢0.6  0.8    1⎥
	// ⎣  1    1    2⎦
	//
	// T1 is unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
	//
	// Default operation is safe (tensor is right operand)
	// ==========================
	// T3 = 5 / T1
	// T3:
	// ⎡+Inf     5     2⎤
	// ⎢   2     1     1⎥
	// ⎣ 0.8   0.7   0.6⎦
	//
	// T1 is unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
	//
	// Default operation is safe (sliced operations - tensor is left operand)
	// =============================================
	// T3 = T1[0:2, 0:2] / 5
	// T3:
	// ⎡  0  0.2⎤
	// ⎣0.6  0.8⎦
	//
	// T1 is unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
	//
	// Default operation is safe (sliced operations - tensor is right operand)
	// =============================================
	// T3 = 5 / T1[0:2, 0:2]
	// T3:
	// ⎡+Inf     5⎤
	// ⎣   2     1⎦
	//
	// T1 is unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
}

// By default, arithmetic operations are safe
func ExampleDense_PowScalar_basic() {
	var T1, T3, V *Dense
	var sliced Tensor
	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	T3, _ = T1.PowScalar(float32(5), true)
	fmt.Printf("Default operation is safe (tensor is left operand)\n==========================\nT3 = T1 ^ 5\nT3:\n%v\nT1 is unchanged:\n%v\n", T3, T1)

	T3, _ = T1.PowScalar(float32(5), false)
	fmt.Printf("Default operation is safe (tensor is right operand)\n==========================\nT3 = 5 ^ T1\nT3:\n%v\nT1 is unchanged:\n%v\n", T3, T1)

	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T3, _ = V.PowScalar(float32(5), true)
	fmt.Printf("Default operation is safe (sliced operations - tensor is left operand)\n=============================================\nT3 = T1[0:2, 0:2] ^ 5\nT3:\n%v\nT1 is unchanged:\n%v\n", T3, T1)

	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T3, _ = V.PowScalar(float32(5), false)
	fmt.Printf("Default operation is safe (sliced operations - tensor is right operand)\n=============================================\nT3 = 5 ^ T1[0:2, 0:2]\nT3:\n%v\nT1 is unchanged:\n%v\n", T3, T1)

	// Output:
	// Default operation is safe (tensor is left operand)
	// ==========================
	// T3 = T1 ^ 5
	// T3:
	// ⎡    0      1     32⎤
	// ⎢  243   1024   3125⎥
	// ⎣ 7776  16807  32768⎦
	//
	// T1 is unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
	//
	// Default operation is safe (tensor is right operand)
	// ==========================
	// T3 = 5 ^ T1
	// T3:
	// ⎡     1       5      25⎤
	// ⎢   125     625    3125⎥
	// ⎣ 15625   78125  390625⎦

	// T1 is unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
	//
	// Default operation is safe (sliced operations - tensor is left operand)
	// =============================================
	// T3 = T1[0:2, 0:2] ^ 5
	// T3:
	// ⎡   0     1⎤
	// ⎣ 243  1024⎦

	// T1 is unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
	//
	// Default operation is safe (sliced operations - tensor is right operand)
	// =============================================
	// T3 = 5 ^ T1[0:2, 0:2]
	// T3:
	// ⎡  1    5⎤
	// ⎣125  625⎦
	//
	// T1 is unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
}

// By default, arithmetic operations are safe
func ExampleDense_ModScalar_basic() {
	var T1, T3, V *Dense
	var sliced Tensor
	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	T3, _ = T1.ModScalar(float32(5), true)
	fmt.Printf("Default operation is safe (tensor is left operand)\n==========================\nT3 = T1 %% 5\nT3:\n%v\nT1 is unchanged:\n%v\n", T3, T1)

	T3, _ = T1.ModScalar(float32(5), false)
	fmt.Printf("Default operation is safe (tensor is right operand)\n==========================\nT3 = 5 %% T1\nT3:\n%v\nT1 is unchanged:\n%v\n", T3, T1)

	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T3, _ = V.ModScalar(float32(5), true)
	fmt.Printf("Default operation is safe (sliced operations - tensor is left operand)\n=============================================\nT3 = T1[0:2, 0:2] %% 5\nT3:\n%v\nT1 is unchanged:\n%v\n", T3, T1)

	T1 = New(WithBacking(Range(Float32, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T3, _ = V.ModScalar(float32(5), false)
	fmt.Printf("Default operation is safe (sliced operations - tensor is right operand)\n=============================================\nT3 = 5 %% T1[0:2, 0:2]\nT3:\n%v\nT1 is unchanged:\n%v\n", T3, T1)

	// Output:
	// Default operation is safe (tensor is left operand)
	// ==========================
	// T3 = T1 % 5
	// T3:
	// ⎡0  1  2⎤
	// ⎢3  4  0⎥
	// ⎣1  2  3⎦
	//
	// T1 is unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
	//
	// Default operation is safe (tensor is right operand)
	// ==========================
	// T3 = 5 % T1
	// T3:
	// ⎡NaN    0    1⎤
	// ⎢  2    1    0⎥
	// ⎣  5    5    5⎦
	//
	// T1 is unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
	//
	// Default operation is safe (sliced operations - tensor is left operand)
	// =============================================
	// T3 = T1[0:2, 0:2] % 5
	// T3:
	// ⎡0  1⎤
	// ⎣3  4⎦
	//
	// T1 is unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
	//
	// Default operation is safe (sliced operations - tensor is right operand)
	// =============================================
	// T3 = 5 % T1[0:2, 0:2]
	// T3:
	// ⎡NaN    0⎤
	// ⎣  2    1⎦
	//
	// T1 is unchanged:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
}
