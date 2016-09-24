package tensorf64

import "fmt"

func ExampleTensor_Sum() {
	T := NewTensor(WithShape(2, 3), WithBacking(RangeFloat64(0, 6)))
	/*
		⎡0  1  2⎤
		⎣3  4  5⎦
	*/

	// Most common use case: summing all axes
	T2, _ := T.Sum()
	fmt.Printf("Sum along every axis: %v\n", T2)

	/*
		Sum along axis 0
		↓⎡0  1  2⎤
		↓⎣3  4  5⎦
	*/
	T2, _ = T.Sum(0)
	fmt.Printf("Sum along axis 0: %v\n", T2)

	/*
		Sum along axis 1
		   ⟶
		⎡0  1  2⎤
		⎣3  4  5⎦
	*/
	T2, _ = T.Sum(1)
	fmt.Printf("Sum along axis 1: %v\n", T2)

	// T.Sum(0,1) is the same as T.Sum(1,0) is the same as T.Sum()
	T2, _ = T.Sum(1, 0)
	fmt.Printf("Sum along every axis: %v\n", T2)
	T2, _ = T.Sum(0, 1)
	fmt.Printf("Sum along every axis: %v\n", T2)

	// Output:
	// Sum along every axis: 15
	// Sum along axis 0: [3  5  7]
	// Sum along axis 1: [ 3  12]
	// Sum along every axis: 15
	// Sum along every axis: 15
}

func ExampleTensor_Max() {
	T := NewTensor(WithShape(2, 3), WithBacking(RangeFloat64(0, 6)))
	/*
		⎡0  1  2⎤
		⎣3  4  5⎦
	*/

	// Most common use case: max of the whole Tensor
	T2, _ := T.Max()
	fmt.Printf("Max along every axis: %v\n", T2)

	/*
		Max along axis 0
		↓⎡0  1  2⎤
		↓⎣3  4  5⎦
	*/
	T2, _ = T.Max(0)
	fmt.Printf("Max along axis 0: %v\n", T2)

	/*
		Max along axis 1
		   ⟶
		⎡0  1  2⎤
		⎣3  4  5⎦
	*/
	T2, _ = T.Max(1)
	fmt.Printf("Max along axis 1: %v\n", T2)

	// T.Max(0,1) is the same as T.Max(1,0) is the same as T.Max()
	T2, _ = T.Max(1, 0)
	fmt.Printf("Max along every axis: %v\n", T2)
	T2, _ = T.Max(0, 1)
	fmt.Printf("Max along every axis: %v\n", T2)

	// Output:
	// Max along every axis: 5
	// Max along axis 0: [3  4  5]
	// Max along axis 1: [2  5]
	// Max along every axis: 5
	// Max along every axis: 5
}

func ExampleTensor_Min() {
	T := NewTensor(WithShape(2, 3), WithBacking(RangeFloat64(0, 6)))
	/*
		⎡0  1  2⎤
		⎣3  4  5⎦
	*/

	// Most common use case: min of the whole Tensor
	T2, _ := T.Min()
	fmt.Printf("Min along every axis: %v\n", T2)

	/*
		Min along axis 0
		↓⎡0  1  2⎤
		↓⎣3  4  5⎦
	*/
	T2, _ = T.Min(0)
	fmt.Printf("Min along axis 0: %v\n", T2)

	/*
		Min along axis 1
		   ⟶
		⎡0  1  2⎤
		⎣3  4  5⎦
	*/
	T2, _ = T.Min(1)
	fmt.Printf("Min along axis 1: %v\n", T2)

	// T.Min(0,1) is the same as T.Min(1,0) is the same as T.Min()
	T2, _ = T.Min(1, 0)
	fmt.Printf("Min along every axis: %v\n", T2)
	T2, _ = T.Min(0, 1)
	fmt.Printf("Min along every axis: %v\n", T2)

	// Output:
	// Min along every axis: 0
	// Min along axis 0: [0  1  2]
	// Min along axis 1: [0  3]
	// Min along every axis: 0
	// Min along every axis: 0
}
