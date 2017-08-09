package tensor

import "fmt"

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
