package gapi

import (
	"fmt"
	"reflect"
	"testing"

	"gorgonia.org/dtype"
	"gorgonia.org/gorgonia/exprgraph"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
)

func TestAdd_SymbolicSymbolic(t *testing.T) {
	g := exprgraph.NewGraph(nil)
	a, err := exprgraph.NewSymbolic(g, "a", dtype.Float64, shapes.Shape{2, 3})
	if err != nil {
		t.Fatalf("Unable to create `a`. Error: %v", err)
	}
	b, err := exprgraph.NewSymbolic(g, "b", dtype.Float64, shapes.Shape{2, 3})
	if err != nil {
		t.Fatalf("Unable to create `b`. Error: %v", err)
	}

	c, err := Add(a, b)
	if err != nil {
		t.Error(err)
	}

	if name := fmt.Sprintf("%s", c); name != "a+b" {
		t.Errorf("Expected `c`'s name to be \"a+b\". Got %q instead", name)
	}

	if !c.Shape().Eq(shapes.Shape{2, 3}) {
		t.Errorf("Expected (2,3) as the resulting shape. Got %v instead", c.Shape())
	}
}

// HybridEngine creates symbolic nodes and also performs the operations immediately.
// However when it encounters a Symbolic node, the remaining operations are symbolic only.
type HybridEngine struct {
	tensor.StdEng
	g *exprgraph.Graph
}

func (e *HybridEngine) Graph() *exprgraph.Graph { return e.g }

func (e *HybridEngine) SetGraph(g *exprgraph.Graph) { e.g = g }

func TestAdd_HybridEngines(t *testing.T) {
	engine := &HybridEngine{}
	g := exprgraph.NewGraph(engine)
	engine.g = g

	a := exprgraph.NewNode(g, "a", tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	b := exprgraph.NewNode(g, "b", tensor.WithShape(2, 3), tensor.WithBacking([]float64{6, 5, 4, 3, 2, 1}))
	c, err := Add(a, b)
	if err != nil {
		t.Error(err)
	}
	correct := []float64{7, 7, 7, 7, 7, 7}
	if !reflect.DeepEqual(correct, c.Data()) {
		t.Errorf("Expected `c` to be %v. Got %v instead", correct, c.Data())
	}
}

func TestAdd_TensorAPI(t *testing.T) {
	a := tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	b := tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{6, 5, 4, 3, 2, 1}))
	c, err := Add(a, b)
	if err != nil {
		t.Error(err)
	}
	correct := []float64{7, 7, 7, 7, 7, 7}
	if !reflect.DeepEqual(correct, c.Data()) {
		t.Errorf("Expected `c` to be %v. Got %v instead", correct, c.Data())
	}
}
