package tensor

import "fmt"

// By default, arithmetic operations are safe
func ExampleDense_Add_basic() {
	var T1, T2, T3 *Dense
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T3, _ = T1.Add(T2)
	fmt.Printf("T3:\n%v\n", T3)

	// Output:
	// T3:
	// ⎡ 0   2   4⎤
	// ⎢ 6   8  10⎥
	// ⎣12  14  16⎦
}

// To perform unsafe operations, use the `UseUnsafe` function option
func ExampleDense_Add_unsafe() {
	var T1, V, T2 *Dense
	var sliced Tensor
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))

	V.Add(T2, UseUnsafe()) // unsafe overwrites the data in T1
	fmt.Printf("V:\n%v\n", V)
	fmt.Printf("Naturally, T1 is mutated too:\n%v", T1)

	// Output:
	// V:
	// ⎡10  12⎤
	// ⎣15  17⎦
	//
	// Naturally, T1 is mutated too:
	// ⎡10  12   2⎤
	// ⎢15  17   5⎥
	// ⎣ 6   7   8⎦
}

// By default, operating on sliced tensors are safe.
func ExampleDense_Add_safe() {
	var T1, V, T2, T3 *Dense
	var sliced Tensor
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))

	T3, _ = V.Add(T2) // safe is the default
	fmt.Printf("T3:\n%v\n", T3)
	fmt.Printf("T1 is unmutated:\n%v", T1)

	// Output:
	// T3:
	// ⎡10  12⎤
	// ⎣15  17⎦
	//
	// T1 is unmutated:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
}

// An optional reuse tensor can also be specified with the WithReuse function option
func ExampleDense_Add_reuse() {
	var T1, V, T2, Reuse, T3 *Dense
	var sliced Tensor

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	Reuse = New(WithBacking(Range(Float64, 100, 104)), WithShape(2, 2))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))

	var err error
	T3, err = V.Add(T2, WithReuse(Reuse))
	if err != nil {
		fmt.Printf("Err %v", err)
	}
	fmt.Printf("Reuse is reused - %t:\n%v", T3 == Reuse, T3)

	// Output:
	// Reuse is reused - true:
	// ⎡10  12⎤
	// ⎣15  17⎦
}

// Incrementing a tensor is also a function option provided by the package
func ExampleDense_Add_incr() {
	var T1, V, T2, Incr, T3 *Dense
	var sliced Tensor

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	Incr = New(WithBacking([]float64{100, 100, 100, 100}), WithShape(2, 2))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))

	var err error
	T3, err = V.Add(T2, WithIncr(Incr))
	if err != nil {
		fmt.Printf("Err %v", err)
	}
	fmt.Printf("Incr is incremented and reused - %t:\n%v", T3 == Incr, T3)

	// Output:
	// Incr is incremented and reused - true:
	// ⎡110  112⎤
	// ⎣115  117⎦
}

// By default, arithmetic operations are safe
func ExampleDense_Sub_basic() {
	var T1, T2, T3 *Dense
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T3, _ = T1.Sub(T2)
	fmt.Printf("T3:\n%v\n", T3)

	// Output:
	// T3:
	// ⎡0  0  0⎤
	// ⎢0  0  0⎥
	// ⎣0  0  0⎦
}

// To perform unsafe operations, use the `UseUnsafe` function option
func ExampleDense_Sub_unsafe() {
	var T1, V, T2 *Dense
	var sliced Tensor
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))

	V.Sub(T2, UseUnsafe()) // unsafe overwrites the data in T1
	fmt.Printf("V:\n%v\n", V)
	fmt.Printf("Naturally, T1 is mutated too:\n%v", T1)

	// Output:
	// V:
	// ⎡-10  -10⎤
	// ⎣ -9   -9⎦
	//
	// Naturally, T1 is mutated too:
	// ⎡-10  -10    2⎤
	// ⎢ -9   -9    5⎥
	// ⎣  6    7    8⎦
}

func ExampleDense_Sub_safe() {
	var T1, V, T2, T3 *Dense
	var sliced Tensor
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))

	T3, _ = V.Sub(T2) // safe is the default
	fmt.Printf("T3:\n%v\n", T3)
	fmt.Printf("T1 is unmutated:\n%v", T1)

	// Output:
	// T3:
	// ⎡-10  -10⎤
	// ⎣ -9   -9⎦
	//
	// T1 is unmutated:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
}

func ExampleDense_Sub_reuse() {
	var T1, V, T2, Reuse, T3 *Dense
	var sliced Tensor

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	Reuse = New(WithBacking(Range(Float64, 100, 104)), WithShape(2, 2))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))

	var err error
	T3, err = V.Sub(T2, WithReuse(Reuse))
	if err != nil {
		fmt.Printf("Err %v", err)
	}
	fmt.Printf("Reuse is reused - %t:\n%v", T3 == Reuse, T3)

	// Output:
	// Reuse is reused - true:
	// ⎡-10  -10⎤
	// ⎣ -9   -9⎦
}

func ExampleDense_Sub_incr() {
	var T1, V, T2, Incr, T3 *Dense
	var sliced Tensor

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	Incr = New(WithBacking([]float64{100, 100, 100, 100}), WithShape(2, 2))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))

	var err error
	T3, err = V.Sub(T2, WithIncr(Incr))
	if err != nil {
		fmt.Printf("Err %v", err)
	}
	fmt.Printf("Incr is incremented and reused - %t:\n%v", T3 == Incr, T3)

	// Output:
	// Incr is incremented and reused - true:
	// ⎡90  90⎤
	// ⎣91  91⎦
}

// By default, arithmetic operations are safe
func ExampleDense_Mul_basic() {
	var T1, T2, T3 *Dense
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T3, _ = T1.Mul(T2)
	fmt.Printf("T3:\n%v\n", T3)

	// Output:
	// T3:
	// ⎡ 0   1   4⎤
	// ⎢ 9  16  25⎥
	// ⎣36  49  64⎦
}

func ExampleDense_Mul_unsafe() {
	var T1, V, T2 *Dense
	var sliced Tensor
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))

	V.Mul(T2, UseUnsafe()) // unsafe overwrites the data in T1
	fmt.Printf("V:\n%v\n", V)
	fmt.Printf("Naturally, T1 is mutated too:\n%v", T1)

	// Output:
	// V:
	// ⎡ 0  11⎤
	// ⎣36  52⎦
	//
	// Naturally, T1 is mutated too:
	// ⎡ 0  11   2⎤
	// ⎢36  52   5⎥
	// ⎣ 6   7   8⎦
}

func ExampleDense_Mul_safe() {
	var T1, V, T2, T3 *Dense
	var sliced Tensor
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))

	T3, _ = V.Mul(T2) // safe is the default
	fmt.Printf("T3:\n%v\n", T3)
	fmt.Printf("T1 is unmutated:\n%v", T1)

	// Output:
	// T3:
	// ⎡ 0  11⎤
	// ⎣36  52⎦
	//
	// T1 is unmutated:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
}

func ExampleDense_Mul_reuse() {
	var T1, V, T2, Reuse, T3 *Dense
	var sliced Tensor

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	Reuse = New(WithBacking(Range(Float64, 100, 104)), WithShape(2, 2))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))

	var err error
	T3, err = V.Mul(T2, WithReuse(Reuse))
	if err != nil {
		fmt.Printf("Err %v", err)
	}
	fmt.Printf("Reuse is reused - %t:\n%v", T3 == Reuse, T3)

	// Output:
	// Reuse is reused - true:
	// ⎡ 0  11⎤
	// ⎣36  52⎦
}

func ExampleDense_Mul_incr() {
	var T1, V, T2, Incr, T3 *Dense
	var sliced Tensor

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	Incr = New(WithBacking([]float64{100, 100, 100, 100}), WithShape(2, 2))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))

	var err error
	T3, err = V.Mul(T2, WithIncr(Incr))
	if err != nil {
		fmt.Printf("Err %v", err)
	}
	fmt.Printf("Incr is incremented and reused - %t:\n%v", T3 == Incr, T3)

	// Output:
	// Incr is incremented and reused - true:
	// ⎡100  111⎤
	// ⎣136  152⎦
}

// By default, arithmetic operations are safe
func ExampleDense_Div_basic() {
	var T1, T2, T3 *Dense
	var err error
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T3, _ = T1.Div(T2)
	fmt.Printf("T3:\n%v\n", T3)

	// Note that divisions of non-float types (which would typically panic in Go) returns errors
	T1 = New(WithBacking(Range(Int, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Int, 0, 9)), WithShape(3, 3))
	_, err = T1.Div(T2)
	fmt.Printf("Testing with division by 0 of type int\n")
	fmt.Printf("Error: %v", err)

	// Output:
	// T3:
	// ⎡+Inf     1     1⎤
	// ⎢   1     1     1⎥
	// ⎣   1     1     1⎦
	//
	// Testing with division by 0 of type int
	// Error: Unable to do Div(): Error in indices [0]
}

func ExampleDense_Div_unsafe() {
	var T1, V, T2 *Dense
	var sliced Tensor
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))

	V.Div(T2, UseUnsafe()) // unsafe overwrites the data in T1
	fmt.Printf("V:\n%1.1v\n", V)
	fmt.Printf("Naturally, T1 is mutated too:\n%1.1v", T1)

	// Output:
	// V:
	// ⎡   0  0.09⎤
	// ⎣ 0.2   0.3⎦
	//
	// Naturally, T1 is mutated too:
	// ⎡   0  0.09     2⎤
	// ⎢ 0.2   0.3     5⎥
	// ⎣   6     7     8⎦
}

func ExampleDense_Div_safe() {
	var T1, V, T2, T3 *Dense
	var sliced Tensor
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))

	T3, _ = V.Div(T2) // safe is the default
	fmt.Printf("T3:\n%1.1v\n", T3)
	fmt.Printf("T1 is unmutated:\n%v", T1)

	// Output:
	// T3:
	// ⎡   0  0.09⎤
	// ⎣ 0.2   0.3⎦
	//
	// T1 is unmutated:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
}

func ExampleDense_Div_reuse() {
	var T1, V, T2, Reuse, T3 *Dense
	var sliced Tensor

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	Reuse = New(WithBacking(Range(Float64, 100, 104)), WithShape(2, 2))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))

	var err error
	T3, err = V.Div(T2, WithReuse(Reuse))
	if err != nil {
		fmt.Printf("Err %v", err)
	}
	fmt.Printf("Reuse is reused - %t:\n%1.1v", T3 == Reuse, T3)

	// Output:
	// Reuse is reused - true:
	// ⎡   0  0.09⎤
	// ⎣ 0.2   0.3⎦
}

func ExampleDense_Div_incr() {
	var T1, V, T2, Incr, T3 *Dense
	var sliced Tensor

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	Incr = New(WithBacking([]float64{100, 100, 100, 100}), WithShape(2, 2))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))

	var err error
	T3, err = V.Div(T2, WithIncr(Incr))
	if err != nil {
		fmt.Printf("Err %v", err)
	}
	fmt.Printf("Incr is incremented and reused - %t:\n%1.1v", T3 == Incr, T3)

	// Output:
	// Incr is incremented and reused - true:
	// ⎡1e+02  1e+02⎤
	// ⎣1e+02  1e+02⎦
}

// By default, arithmetic operations are safe
func ExampleDense_Pow_basic() {
	var T1, T2, T3 *Dense
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T3, _ = T1.Pow(T2)
	fmt.Printf("T3:\n%v\n", T3)

	// Output:
	// T3:
	// ⎡            1              1              4⎤
	// ⎢           27            256           3125⎥
	// ⎣        46656         823543  1.6777216e+07⎦
}

func ExampleDense_Pow_unsafe() {
	var T1, V, T2 *Dense
	var sliced Tensor
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))

	V.Pow(T2, UseUnsafe()) // unsafe overwrites the data in T1
	fmt.Printf("V:\n%1.1v\n", V)
	fmt.Printf("Naturally, T1 is mutated too:\n%1.1v", T1)

	// Output:
	// V:
	// ⎡    0      1⎤
	// ⎣5e+05  7e+07⎦
	//
	// Naturally, T1 is mutated too:
	// ⎡    0      1      2⎤
	// ⎢5e+05  7e+07      5⎥
	// ⎣    6      7      8⎦
}

func ExampleDense_Pow_safe() {
	var T1, V, T2, T3 *Dense
	var sliced Tensor
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))

	T3, _ = V.Pow(T2) // safe is the default
	fmt.Printf("T3:\n%1.1v\n", T3)
	fmt.Printf("T1 is unmutated:\n%v", T1)

	// Output:
	// T3:
	// ⎡    0      1⎤
	// ⎣5e+05  7e+07⎦
	//
	// T1 is unmutated:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
}

func ExampleDense_Pow_reuse() {
	var T1, V, T2, Reuse, T3 *Dense
	var sliced Tensor

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	Reuse = New(WithBacking(Range(Float64, 100, 104)), WithShape(2, 2))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))

	var err error
	T3, err = V.Pow(T2, WithReuse(Reuse))
	if err != nil {
		fmt.Printf("Err %v", err)
	}
	fmt.Printf("Reuse is reused - %t:\n%1.1v", T3 == Reuse, T3)

	// Output:
	// Reuse is reused - true:
	// ⎡    0      1⎤
	// ⎣5e+05  7e+07⎦
}

func ExampleDense_Pow_incr() {
	var T1, V, T2, Incr, T3 *Dense
	var sliced Tensor

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	Incr = New(WithBacking([]float64{100, 100, 100, 100}), WithShape(2, 2))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))

	var err error
	T3, err = V.Pow(T2, WithIncr(Incr))
	if err != nil {
		fmt.Printf("Err %v", err)
	}
	fmt.Printf("Incr is incremented and reused - %t:\n%1.1v", T3 == Incr, T3)

	// Output:
	// Incr is incremented and reused - true:
	// ⎡1e+02  1e+02⎤
	// ⎣5e+05  7e+07⎦
}

// By default, arithmetic operations are safe
func ExampleDense_Mod_basic() {
	var T1, T2, T3 *Dense
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T2 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	T3, _ = T1.Mod(T2)
	fmt.Printf("T3:\n%v\n", T3)

	// Output:
	// T3:
	// ⎡NaN    0    0⎤
	// ⎢  0    0    0⎥
	// ⎣  0    0    0⎦
}

// To perform unsafe operations, use the `UseUnsafe` function option
func ExampleDense_Mod_unsafe() {
	var T1, V, T2 *Dense
	var sliced Tensor
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))

	V.Mod(T2, UseUnsafe()) // unsafe overwrites the data in T1
	fmt.Printf("V:\n%1.1v\n", V)
	fmt.Printf("Naturally, T1 is mutated too:\n%1.1v", T1)

	// Output:
	// V:
	// ⎡0  1⎤
	// ⎣3  4⎦
	//
	// Naturally, T1 is mutated too:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
}

// By default, operating on sliced tensors are safe.
func ExampleDense_Mod_safe() {
	var T1, V, T2, T3 *Dense
	var sliced Tensor
	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))

	T3, _ = V.Mod(T2) // safe is the default
	fmt.Printf("T3:\n%1.1v\n", T3)
	fmt.Printf("T1 is unmutated:\n%v", T1)

	// Output:
	// T3:
	// ⎡0  1⎤
	// ⎣3  4⎦
	//
	// T1 is unmutated:
	// ⎡0  1  2⎤
	// ⎢3  4  5⎥
	// ⎣6  7  8⎦
}

// An optional reuse tensor can also be specified with the WithReuse function option
func ExampleDense_Mod_reuse() {
	var T1, V, T2, Reuse, T3 *Dense
	var sliced Tensor

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	Reuse = New(WithBacking(Range(Float64, 100, 104)), WithShape(2, 2))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))

	var err error
	T3, err = V.Mod(T2, WithReuse(Reuse))
	if err != nil {
		fmt.Printf("Err %v", err)
	}
	fmt.Printf("Reuse is reused - %t:\n%1.1v", T3 == Reuse, T3)

	// Output:
	// Reuse is reused - true:
	// ⎡0  1⎤
	// ⎣3  4⎦
}

// Incrementing a tensor is also a function option provided by the package
func ExampleDense_Mod_incr() {
	var T1, V, T2, Incr, T3 *Dense
	var sliced Tensor

	T1 = New(WithBacking(Range(Float64, 0, 9)), WithShape(3, 3))
	Incr = New(WithBacking([]float64{100, 100, 100, 100}), WithShape(2, 2))
	sliced, _ = T1.Slice(makeRS(0, 2), makeRS(0, 2))
	V = sliced.(*Dense)
	T2 = New(WithBacking(Range(Float64, 10, 14)), WithShape(2, 2))

	var err error
	T3, err = V.Mod(T2, WithIncr(Incr))
	if err != nil {
		fmt.Printf("Err %v", err)
	}
	fmt.Printf("Incr is incremented and reused - %t:\n%1.1v", T3 == Incr, T3)

	// Output:
	// Incr is incremented and reused - true:
	// ⎡1e+02  1e+02⎤
	// ⎣1e+02  1e+02⎦
}
