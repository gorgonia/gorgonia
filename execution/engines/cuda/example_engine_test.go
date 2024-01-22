package cuda_test

import (
	"fmt"
	"time"

	_ "net/http/pprof"

	"gorgonia.org/gorgonia/execution/engines/cuda"
	"gorgonia.org/gorgonia/exprgraph"
	"gorgonia.org/internal/debug"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/dense"
)

type E[DT any, T tensor.Basic[DT]] struct {
	*cuda.Engine[DT, T]
	g *exprgraph.Graph
}

func (e *E[DT, T]) Graph() *exprgraph.Graph     { return e.g }
func (e *E[DT, T]) SetGraph(g *exprgraph.Graph) { e.g = g }

var _ tensor.Adder[float64, *dense.Dense[float64]] = &E[float64, *dense.Dense[float64]]{}

func Example() {

	e := cuda.New[float64, *dense.Dense[float64]](cuda.NewState(0))
	engine := E[float64, *dense.Dense[float64]]{Engine: e}
	go engine.Run()
	defer engine.Done()
	g := exprgraph.NewGraph(engine)
	engine.g = g

	x := exprgraph.New[float64](g, "x", tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	y := exprgraph.New[float64](g, "y", tensor.WithShape(3, 2), tensor.WithBacking([]float64{6, 5, 4, 3, 2, 1}))
	z := exprgraph.New[float64](g, "z", tensor.WithShape(2), tensor.WithBacking([]float64{1, 1}))
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
	debug.Logf("FINAL SIGNAL")
	time.Sleep(100 * time.Millisecond) // sleep for a bit so the execution finishes and finish propagating. In real life use there is no need for this

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
