package cuda_test

import (
	"fmt"
	"log"

	_ "net/http/pprof"

	"gorgonia.org/gorgonia/execution/engines/cuda"
	"gorgonia.org/gorgonia/exprgraph"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/dense"
)

type E[DT any, T tensor.Basic[DT]] struct {
	*cuda.Engine[DT, T]
	g *exprgraph.Graph
}

func (e *E[DT, T]) Graph() *exprgraph.Graph     { return e.g }
func (e *E[DT, T]) SetGraph(g *exprgraph.Graph) { e.g = g }

func Example() {

	e := cuda.New[float64, *dense.Dense[float64]](cuda.NewState(0))
	engine := E[float64, *dense.Dense[float64]]{Engine: e}
	go engine.Run()
	defer engine.Close()
	g := exprgraph.NewGraph(engine)
	engine.g = g
	log.Printf("XXX")

	x := exprgraph.New[float64](g, "x", tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	y := exprgraph.New[float64](g, "y", tensor.WithShape(3, 2), tensor.WithBacking([]float64{6, 5, 4, 3, 2, 1}))
	z := exprgraph.New[float64](g, "z", tensor.WithShape(), tensor.WithBacking([]float64{1}))
	xy, err := MatMul[float64, *dense.Dense[float64]](x, y)
	if err != nil {
		fmt.Printf("Matmul failed: Err: %v\n", err)
		return
	}

	xypz, err := Add[float64, *dense.Dense[float64]](xy, z)
	if err != nil {
		fmt.Printf("Add failed. Err: %v\n", err)
		return
	}

	engine.Signal()

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
