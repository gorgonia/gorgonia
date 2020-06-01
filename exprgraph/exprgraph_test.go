package exprgraph_test

import (
	"fmt"

	"gorgonia.org/tensor"
)

type SymbolicEngine struct {
	tensor.StdEng
	g *Graph
}

func (e *SymbolicEngine) SetGraph(g *Graph) { e.g = g }

func (e *SymbolicEngine) MatMul(a, b, preallocated tensor.Tensor) error {
	aEng := a.Engine().(*Graph) // TODO ERR
	bEng := b.Engine().(*Graph) // TODO ERR
	cEng := preallocated.Engine().(*Graph)
	aName := aEng.nameOf(a)
	bName := bEng.nameOf(b)
	cEng.Insert(preallocated)

	cName := aName + "×" + bName
	cEng.name(preallocated, cName)

}

func Example() {
	g := NewGraph()

	x := New(g, "x", tensor.WithShape(2, 3), tensor.Of(tensor.Float64))
	y := New(g, "y", tensor.WithShape(3, 2), tensor.Of(tensor.Float64))
	z := New(g, "z", tensor.WithShape(), tensor.Of(tensor.Float64))

	xy, err := MatMul(x, y)
	if err != nil {
		fmt.Println(err)
	}
	xypz, err := Add(xy, z)
	if err != nil {
		fmt.Println(err)
	}

	fmt.Printf("x:\n%v\ny:\n%v\nxy:\n%v\nxy+z", x, y, xy, xypz)
	fmt.Printf("Nodes: %v", g.Nodes())

	// Output:
	// x:
	// ⎡0  0  0⎤
	// ⎣0  0  0⎦
	//
	// y:
	// ⎡0  0⎤
	// ⎢0  0⎥
	// ⎣0  0⎦
	//
	// xy:
	// ⎡0  0⎤
	// ⎣0  0⎦
	//
	// Nodes: [x y x×y]
}
