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
