package exprgraph_test

import (
	"fmt"

	"gorgonia.org/gorgonia/exprgraph"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/dense"
	stdeng "gorgonia.org/tensor/engines"
)

// SymbolicEngine is a engine that performs symbolic operations.
type SymbolicEngine struct {
	stdeng.Gen
	g *exprgraph.Graph
}

func (e *SymbolicEngine) Graph() *exprgraph.Graph { return e.g }

func (e *SymbolicEngine) SetGraph(g *exprgraph.Graph) { e.g = g }

// ExampleSymbolic demonstrates how to implement a symbolic engine that does perform symbolic operations
func Example_symbolicEngine() {
	engine := &SymbolicEngine{}
	g := exprgraph.NewGraph(engine)
	engine.g = g

	x := exprgraph.New[float64](g, "x", tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	y := exprgraph.New[float64](g, "y", tensor.WithShape(3, 2), tensor.WithBacking([]float64{6, 5, 4, 3, 2, 1}))
	z := exprgraph.New[float64](g, "z", tensor.WithShape())

	xy, err := MatMul[float64, *dense.Dense[float64]](x, y)
	if err != nil {
		fmt.Println(err)
	}
	xypz, err := Add[float64, *dense.Dense[float64]](xy, z)
	if err != nil {
		fmt.Println(err)
	}

	fmt.Printf("x:\n%s\ny:\n%s\nxy:\n%s\nxy+z:\n%s\n", x, y, xy, xypz)

	// Output:
	// x:
	//x
	//y:
	//y
	//xy:
	//x×y
	//xy+z:
	//x×y+z
}

// Example_symbolicEngineUsingTensorAPI demonstrates how the tensor package's API may be used with a symbolic engine.
func Example_symbolicEngineUsingTensorAPI() {
	g := exprgraph.NewGraph(nil)

	x := dense.New[float64](tensor.WithEngine(g), tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	y := dense.New[float64](tensor.WithEngine(g), tensor.WithShape(3, 2), tensor.WithBacking([]float64{6, 5, 4, 3, 2, 1}))
	z := dense.New[float64](tensor.WithEngine(g), tensor.WithShape(), tensor.WithBacking([]float64{1}))

	xy, err := MatMul[float64, *dense.Dense[float64]](x, y)
	if err != nil {
		fmt.Println(err)
		return
	}

	xypz, err := Add[float64, *dense.Dense[float64]](xy, z)
	if err != nil {
		fmt.Println(err)
		return
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
	// node(2,Random_2×Random_3,Random_2×Random_3)
	// xy+z:
	// node(4,Random_2×Random_3+Random_4,Random_2×Random_3+Random_4)

}
