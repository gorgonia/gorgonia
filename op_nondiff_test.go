package gorgonia

import (
	"fmt"

	"gorgonia.org/tensor"
)

func ExampleDiagFlat() {
	g := NewGraph()

	// 2 dimensional
	aV := tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1, 2, 3, 4}))
	a := NodeFromAny(g, aV)
	b, err := DiagFlat(a)
	if err != nil {
		fmt.Println(err)
		return
	}
	m := NewTapeMachine(g)
	if err := m.RunAll(); err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("a:\n%v\n", a.Value())
	fmt.Printf("b:\n%v\n", b.Value())

	// 3 dimensional
	aV = tensor.New(tensor.WithShape(2, 3, 2), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}))
	a = NodeFromAny(g, aV, WithName("a'"))
	b2, err := DiagFlat(a)
	if err != nil {
		fmt.Println(err)
		return
	}
	m = NewTapeMachine(g)
	if err := m.RunAll(); err != nil {
		fmt.Println(err)
	}

	fmt.Printf("a:\n%v", a.Value())
	fmt.Printf("b:\n%v\n", b2.Value())

	// 1 dimensional
	aV = tensor.New(tensor.WithShape(2), tensor.WithBacking([]float64{1, 2}))
	a = NodeFromAny(g, aV, WithName("a''"))
	b3, err := DiagFlat(a)
	if err != nil {
		fmt.Println(err)
		return
	}
	m = NewTapeMachine(g)
	if err := m.RunAll(); err != nil {
		fmt.Println(err)
	}

	fmt.Printf("a:\n%v\n", a.Value())
	fmt.Printf("b:\n%v\n", b3.Value())

	// Scalars

	a = NodeFromAny(g, 100.0, WithName("aScalar"))
	_, err = DiagFlat(a)
	fmt.Println(err)

	// Output:
	// a:
	// ⎡1  2⎤
	// ⎣3  4⎦
	//
	// b:
	// ⎡1  0  0  0⎤
	// ⎢0  2  0  0⎥
	// ⎢0  0  3  0⎥
	// ⎣0  0  0  4⎦
	//
	// a:
	// ⎡ 1   2⎤
	// ⎢ 3   4⎥
	// ⎣ 5   6⎦
	//
	// ⎡ 7   8⎤
	// ⎢ 9  10⎥
	// ⎣11  12⎦
	//
	//
	// b:
	// ⎡ 1   0   0   0  ...  0   0   0   0⎤
	// ⎢ 0   2   0   0  ...  0   0   0   0⎥
	// ⎢ 0   0   3   0  ...  0   0   0   0⎥
	// ⎢ 0   0   0   4  ...  0   0   0   0⎥
	// .
	// .
	// .
	// ⎢ 0   0   0   0  ...  9   0   0   0⎥
	// ⎢ 0   0   0   0  ...  0  10   0   0⎥
	// ⎢ 0   0   0   0  ...  0   0  11   0⎥
	// ⎣ 0   0   0   0  ...  0   0   0  12⎦
	//
	// a:
	// [1  2]
	// b:
	// ⎡1  0⎤
	// ⎣0  2⎦
	//
	// Cannot perform DiagFlat on a scalar equivalent node

}
