package onnx

import (
	"errors"

	"gonum.org/v1/gonum/graph"
	"gorgonia.org/gorgonia/internal/engine"
	"gorgonia.org/gorgonia/node"
	"gorgonia.org/gorgonia/ops"
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
	oper := func(g graph.WeightedDirected, n node.Node) (ops.Op, error) {
		output, err := operation(g, n)
		return output.(ops.Op), err
	}
	return g.ApplyOp(engine.Operation(oper), n.(*engine.Node))
}
