package math

import (
	"gonum.org/v1/gonum/graph"
	"gorgonia.org/gorgonia"
	"gorgonia.org/gorgonia/node"
	"gorgonia.org/gorgonia/ops"
)

// Add ...
func Add() gorgonia.Operation {
	return func(g graph.DirectedWeightedBuilder, n node.Node) (ops.Op, error) {
		return nil, nil
	}

}
