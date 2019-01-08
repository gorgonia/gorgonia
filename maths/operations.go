package math

import (
	"gonum.org/v1/gonum/graph"
	"gorgonia.org/gorgonia/node"
	"gorgonia.org/gorgonia/ops"
)

// Operation is any op that can be applied to a node n of the graph g
type Operation func(g graph.DirectedWeightedBuilder, n node.Node) (ops.Op, error)

// Add ...
func Add() Operation {
	return func(g graph.DirectedWeightedBuilder, n node.Node) (ops.Op, error) {
		return nil, nil
	}

}
