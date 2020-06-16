package exprgraph_test

import (
	"fmt"

	"gorgonia.org/gorgonia"
	. "gorgonia.org/gorgonia/exprgraph"
	"gorgonia.org/tensor"
)

// SymbolicEngine is a engine that performs symbolic operations.
type SymbolicEngine struct {
	tensor.StdEng
	g *Graph
}

func (e *SymbolicEngine) Graph() *Graph { return e.g }

func (e *SymbolicEngine) SetGraph(g *Graph) { e.g = g }

// HybridEngine creates symbolic nodes and also performs the operations immediately.
//
// However when it encounters a Symbolic node, the remaining operations are symbolic only.
type HybridEngine struct {
	tensor.StdEng
	g *Graph
}

func (e *HybridEngine) Graph() *Graph { return e.g }

func (e *HybridEngine) SetGraph(g *Graph) { e.g = g }

// DualEngine is an engine that does automatic differentiation
type DualEngine struct {
	tensor.StdEng
	g *Graph
}

func (e *DualEngine) Graph() *Graph { return e.g }

func (e *DualEngine) SetGraph(g *Graph) { e.g = g }

type GraphEngine interface {
	tensor.Engine
	Graph() *Graph
}

func MatMul(a, b gorgonia.Tensor) (gorgonia.Tensor, error) {
	eng := a.Engine().(GraphEngine)
	if eng == nil {
		eng = b.Engine().(GraphEngine)
	}

	g := eng.Graph()
	aid := g.ID(a)
	bid := g.ID(b)
	aname := g.NameOf(a)
	bname := g.NameOf(b)
	cname := aname + "×" + bname

	// TODO: check shapes obvs
	shp := tensor.Shape{a.Shape()[0], b.Shape()[1]}
	dt := a.Dtype()

	switch e := eng.(type) {
	case *Graph:
		retVal := NewSymbolic(e, dt, shp)
		id := e.Insert(retVal)
		e.Name(retVal, cname) // TODO: add op
		e.AddChildren(id, []NodeID{aid, bid})
		return retVal, nil
	case tensor.MatMuler:
		at := T2T(a)
		bt := T2T(b)
		prealloc := Make(g, cname, tensor.WithShape(shp...), tensor.Of(dt))
		ct := T2T(prealloc)
		if err := e.MatMul(at, bt, ct); err != nil {
			return nil, err
		}
		return prealloc, nil
	default:
		panic(fmt.Sprintf("ENGINE %T", eng))

	}
	panic("Unreachable")
}

func Add(a, b gorgonia.Tensor) (gorgonia.Tensor, error) {
	eng := a.Engine().(GraphEngine)
	if eng == nil {
		eng = b.Engine().(GraphEngine)
	}

	g := eng.Graph()
	aid := g.ID(a)
	bid := g.ID(b)
	aname := g.NameOf(a)
	bname := g.NameOf(b)
	cname := aname + "+" + bname

	// TODO: check shapes obvs
	shp := a.Shape().Clone()
	dt := a.Dtype()

	switch e := eng.(type) {
	case *Graph:
		retVal := NewSymbolic(e, dt, shp)
		id := e.Insert(retVal)
		e.Name(retVal, aname+"+"+bname) // TODO: add op
		e.AddChildren(id, []NodeID{aid, bid})
		return retVal, nil
	case tensor.Adder:
		at := T2T(a)
		bt := T2T(b)
		ct, err := e.AddScalar(at, bt, true) // note this brief example is specific to the examples. More switch cases are needed to figure out leftScalar vs rightScalar
		if err != nil {
			return nil, err
		}
		return Cons(g, cname, ct), nil
	}
	panic("NYI2")
}

func ExampleSymbolic() {
	g := New(tensor.StdEng{})

	x := Make(g, "x", tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	y := Make(g, "y", tensor.WithShape(3, 2), tensor.WithBacking([]float64{6, 5, 4, 3, 2, 1}))
	z := Make(g, "z", tensor.WithShape(), tensor.Of(tensor.Float64))

	xy, err := MatMul(x, y)
	if err != nil {
		fmt.Println(err)
	}
	xypz, err := Add(xy, z)
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

func ExampleHybrid() {
	g := New(&HybridEngine{})

	x := Make(g, "x", tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	y := Make(g, "y", tensor.WithShape(3, 2), tensor.WithBacking([]float64{6, 5, 4, 3, 2, 1}))
	z := Make(g, "z", tensor.WithShape(), tensor.WithBacking([]float64{1}))

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
