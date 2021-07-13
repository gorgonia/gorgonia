package exprgraph_test

import (
	"fmt"

	"gorgonia.org/gorgonia/exprgraph"
	"gorgonia.org/tensor"
)

// HybridEngine creates symbolic nodes and also performs the operations immediately.
// However when it encounters a Symbolic node, the remaining operations are symbolic only.
type HybridEngine struct {
	tensor.StdEng
	g *exprgraph.Graph
}

func (e *HybridEngine) Graph() *exprgraph.Graph { return e.g }

func (e *HybridEngine) SetGraph(g *exprgraph.Graph) { e.g = g }

// HybridEngine creates symbolic nodes and also performs the operations immediately.
// However when it encounters a Symbolic node, the remaining operations are symbolic only.
func Example_hybridEngine() {
	engine := &HybridEngine{}
	g := exprgraph.NewGraph(engine)
	engine.g = g

	x := exprgraph.NewNode(g, "x", tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	y := exprgraph.NewNode(g, "y", tensor.WithShape(3, 2), tensor.WithBacking([]float64{6, 5, 4, 3, 2, 1}))
	z := exprgraph.NewNode(g, "z", tensor.WithShape(), tensor.WithBacking([]float64{1}))

	xy, err := MatMul(x, y)
	if err != nil {
		fmt.Println(err)
	}

	xypz, err := Add(xy, z)
	if err != nil {
		fmt.Println(err)
	}

	fmt.Printf("x:\n%v\ny:\n%v\nxy:\n%v\nxy+z:\n%v\n", x, y, xy, xypz)
	// Output:
	// x:
	// ⎡1  2  3⎤
	// ⎣4  5  6⎦
	//
	// y:
	// ⎡6  5⎤
	// ⎢4  3⎥
	// ⎣2  1⎦
	//
	// xy:
	// ⎡20  14⎤
	// ⎣56  41⎦
	//
	// xy+z:
	// ⎡21  15⎤
	// ⎣57  42⎦

}
