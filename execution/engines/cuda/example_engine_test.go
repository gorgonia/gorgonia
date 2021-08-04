package cuda_test

import (
	"fmt"

	_ "net/http/pprof"

	"gorgonia.org/gorgonia/execution/engines/cuda"
	"gorgonia.org/gorgonia/exprgraph"
	"gorgonia.org/tensor"
)

type E struct {
	*cuda.Engine
	g *exprgraph.Graph
}

func (e *E) Graph() *exprgraph.Graph     { return e.g }
func (e *E) SetGraph(g *exprgraph.Graph) { e.g = g }

func Example() {

	e := cuda.New(0)
	engine := E{Engine: e}
	go engine.Run()
	defer engine.Close()
	g := exprgraph.NewGraph(engine)
	engine.g = g

	x := exprgraph.NewNode(g, "x", tensor.WithShape(2, 3), tensor.Of(tensor.Float64), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	y := exprgraph.NewNode(g, "y", tensor.WithShape(3, 2), tensor.Of(tensor.Float64), tensor.WithBacking([]float64{6, 5, 4, 3, 2, 1}))
	z := exprgraph.NewNode(g, "z", tensor.WithShape(), tensor.Of(tensor.Float64), tensor.WithBacking([]float64{1}))

	xy, err := MatMul(x, y)
	if err != nil {
		fmt.Println(err)
		return
	}

	xypz, err := Add(xy, z)
	if err != nil {
		fmt.Printf("Add Error:\n%+v\n", err)
		return
	}
	engine.Signal()

	// xypz2, err := Add(xypz, z)
	// if err != nil {
	// 	fmt.Printf("AAddERr %v", err)
	// 	return
	// }

	// xypz3, err := engine.Accessible(xypz)
	// if err != nil {
	// 	fmt.Printf("Unable to copy xypz to a local variable. Err: %v\n", err)
	// 	return
	// }

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
