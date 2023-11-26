package exprgraph

import (
	"gorgonia.org/shapes"
	"gorgonia.org/tensor/dense"
)

type testcons struct {
	name    string
	id      int64
	shape   shapes.Shape
	backing any
	g       *Graph
}

func newTestCons() *testcons {
	return &testcons{
		name:  "test",
		shape: shapes.Shape{2, 2},
	}
}

type testopt func(*testcons)

func withID(id int64) testopt {
	return func(tc *testcons) {
		tc.id = id
	}
}

func withName(name string) testopt {
	return func(tc *testcons) {
		tc.name = name
	}
}

func withShape(shp ...int) testopt {
	return func(tc *testcons) {
		tc.shape = shapes.Shape(shp)
	}
}

func inGraph(g *Graph) testopt {
	return func(tc *testcons) {
		tc.g = g
	}
}

func newSym(opts ...testopt) *Symbolic[float32] {
	c := newTestCons()
	for _, opt := range opts {
		opt(c)
	}
	retVal, _ := NewSymbolic[float32](c.g, c.shape, c.name)
	retVal.id = NodeID(c.id)
	return retVal
}

func newVal(opts ...testopt) *Value[float64, *dense.Dense[float64]] {
	c := newTestCons()
	for _, opt := range opts {
		opt(c)
	}

	var backing any = []float64{100, 200, 3.14159, 4}
	if c.backing != nil {
		backing = c.backing
	}
	var retVal *Value[float64, *dense.Dense[float64]]
	if c.g != nil {
		retVal = NewValueInGraph[float64, *dense.Dense[float64]](c.g, c.name, dense.New[float64](dense.WithShape(c.shape...), dense.WithBacking(backing), dense.WithEngine(c.g)))
	} else {
		retVal = NewValue[float64, *dense.Dense[float64]](c.name, dense.New[float64](dense.WithShape(c.shape...), dense.WithBacking(backing)))
	}

	retVal.id = NodeID(c.id)
	return retVal

}

func newNilVal() *Value[float32, *dense.Dense[float32]] {
	return &Value[float32, *dense.Dense[float32]]{
		desc: desc{name: "test", id: 1337},
	}
}
