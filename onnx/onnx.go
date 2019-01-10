package onnnx

import (
	"errors"

	"gonum.org/v1/gonum/graph"
	"gorgonia.org/gorgonia/internal/engine"
)

// Graph ...
type Graph struct {
	*engine.ExprGraph
}

// ONNXGetOperationFromName ...
func (g Graph) ONNXGetOperationFromName(s string) (interface{}, error) {
	switch s {
	case "Add":
		return &Add{}, nil
	default:
		return nil, errors.New("Operator " + s + " not found")
	}
}

// ONNXApply ...
func (g Graph) ONNXApply(operation func(g graph.WeightedDirected, n graph.Node) (interface{}, error), n graph.Node) error {
	return g.ApplyOp(engine.Operation(operation), n.(*engine.Node))
}

// Add ...
type Add struct{}

// Constructor to fulfill the ONNXOperation interface
func (a *Add) Constructor() func(g graph.WeightedDirected, n graph.Node) (interface{}, error) {
	return func(g graph.WeightedDirected, n graph.Node) (interface{}, error) {
		return engine.NewAddOperation()(g, n)
	}
}
