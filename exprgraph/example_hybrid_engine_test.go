package exprgraph_test

import (
	"fmt"

	"gorgonia.org/gorgonia/exprgraph"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/dense"
)

// HybridEngine creates symbolic nodes and also performs the operations immediately.
// However when it encounters a Symbolic node, the remaining operations are symbolic only.
type HybridEngine[DT tensor.Num, T tensor.Basic[DT]] struct {
	StandardEngine[DT, T]
	g *exprgraph.Graph
}

func (e *HybridEngine[DT, T]) Graph() *exprgraph.Graph { return e.g }

func (e *HybridEngine[DT, T]) SetGraph(g *exprgraph.Graph) { e.g = g }

// HybridEngine creates symbolic nodes and also performs the operations immediately.
// However when it encounters a Symbolic node, the remaining operations are symbolic only.
func Example_hybridEngine1() {
	engine := &HybridEngine[float64, *dense.Dense[float64]]{StandardEngine: dense.StdFloat64Engine[*dense.Dense[float64]]{}}
	g := exprgraph.NewGraph(engine)
	engine.g = g

	x := exprgraph.New[float64](g, "x", tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	y := exprgraph.New[float64](g, "y", tensor.WithShape(3, 2), tensor.WithBacking([]float64{6, 5, 4, 3, 2, 1}))
	z := exprgraph.New[float64](g, "z", tensor.WithShape(), tensor.WithBacking([]float64{1}))

	xy, err := MatMul[float64, *dense.Dense[float64]](x, y)
	if err != nil {
		fmt.Println(err)
	}

	xypz, err := Add[float64, *dense.Dense[float64]](xy, z)
	if err != nil {
		fmt.Println(err)
	}

	fmt.Printf("x:\n%v\ny:\n%v\nxy:\n%v\nxy+z:\n%v\n", x, y, xy, xypz)
	fmt.Printf("Symbolically:\nx:\n%s\ny:\n%s\nxy:\n%s\nxy+z:\n%s\n", x, y, xy, xypz)
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
	//
	// Symbolically:
	// x:
	// x
	// y:
	// y
	// xy:
	// x×y
	// xy+z:
	// x×y+z

}

func Example_hybridEngine_mixmatch() {
	engine := &HybridEngine[float64, *dense.Dense[float64]]{StandardEngine: dense.StdFloat64Engine[*dense.Dense[float64]]{}}
	g := exprgraph.NewGraph(engine)
	engine.g = g

	x := exprgraph.New[float64](g, "x", tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	y := exprgraph.New[float64](g, "y", tensor.WithShape(3, 2), tensor.WithBacking([]float64{6, 5, 4, 3, 2, 1}))
	z, err := exprgraph.NewSymbolic[float64](g, shapes.ScalarShape(), "z")

	if err != nil {
		fmt.Printf("Error in creataing symbolic z node %v\n", err)
		return
	}

	xy, err := MatMul[float64, *dense.Dense[float64]](x, y)
	if err != nil {
		fmt.Println(err)
	}

	xypz, err := Add[float64, *dense.Dense[float64]](xy, z)
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
	// node(4,x×y+z)

}
