package gorgonia

import (
	"fmt"
	"log"

	"gorgonia.org/tensor"
)

func ExampleSoftMax() {
	g := NewGraph()
	t := tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 3, 2, 3, 2, 1}))
	u := t.Clone().(*tensor.Dense)
	v := tensor.New(tensor.WithShape(2, 2, 3), tensor.WithBacking([]float64{
		1, 3, 2,
		4, 2, 1,

		3, 5, 3,
		2, 1, 5,
	}))

	a := NodeFromAny(g, t, WithName("a"))
	b := NodeFromAny(g, u, WithName("b"))
	c := NodeFromAny(g, v, WithName("c"))

	sm1 := Must(SoftMax(a))
	sm0 := Must(SoftMax(b, 0))
	sm := Must(SoftMax(c))
	m := NewTapeMachine(g)
	if err := m.RunAll(); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("a:\n%v\nsoftmax(a) - along last axis (default behaviour):\n%1.2f", a.Value(), sm1.Value())
	fmt.Printf("b:\n%v\nsoftmax(b) - along axis 0:\n%1.2f", b.Value(), sm0.Value())
	fmt.Printf("c %v:\n%v\nsoftmax(c) - along last axis (default behaviour) %v:\n%1.2f", c.Value().Shape(), c.Value().Data(), sm.Value().Shape(), sm.Value().Data())
	// Note: The change in format option is because of the pesky way Go examples deal with newlines and formatting of rank-N tensors in Gorgonia don't play well
	// So imagine the tensors being arranged correctly

	// Output:
	// a:
	// ⎡1  3  2⎤
	// ⎣3  2  1⎦
	//
	// softmax(a) - along last axis (default behaviour):
	// ⎡0.09  0.67  0.24⎤
	// ⎣0.67  0.24  0.09⎦
	// b:
	// ⎡1  3  2⎤
	// ⎣3  2  1⎦
	//
	// softmax(b) - along axis 0:
	// ⎡0.12  0.73  0.73⎤
	// ⎣0.88  0.27  0.27⎦
	// c (2, 2, 3):
	// [1 3 2 4 2 1 3 5 3 2 1 5]
	// softmax(c) - along last axis (default behaviour) (2, 2, 3):
	// [0.09 0.67 0.24 0.84 0.11 0.04 0.11 0.79 0.11 0.05 0.02 0.94]
}
