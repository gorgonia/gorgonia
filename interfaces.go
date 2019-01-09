package gorgonia

import (
	"hash"

	"github.com/pkg/errors"
	"gonum.org/v1/gonum/graph"
)

type hashWriter interface {
	WriteHash(hash.Hash)
}

type arityer interface {
	Arity() int
}

// Operation is any op that can be applied to a node n of the graph g
//type Operation func(g graph.WeightedDirected, n node.Node) (ops.Op, error)
type Operation func(g graph.WeightedDirected, n graph.Node) (interface{}, error)

// AddOp ...
type AddOp struct{}

// Constructor to fulfill the ONNXOperation interface
func (a *AddOp) Constructor() func(g graph.WeightedDirected, n graph.Node) (interface{}, error) {
	return nil
}

// ONNXGetOperationFromName ...
func (g *ExprGraph) ONNXGetOperationFromName(s string) (interface{}, error) {
	switch s {
	case "Add":
		return &AddOp{}, nil
	default:
		return nil, errors.New("Operator " + s + " not found")
	}
}

// ONNXApply ...
func (g *ExprGraph) ONNXApply(operation func(g graph.WeightedDirected, n graph.Node) (interface{}, error), n graph.Node) error {
	return g.ApplyOp(Operation(operation), n.(*Node))
}
