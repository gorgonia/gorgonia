package operations

import "gorgonia.org/gorgonia/graph"

// Operation is any operation that can be applied on a node of the ExprGraph
type Operation func(g *graph.ExprGraph, n *graph.Node) (Op, error)

// Add ...
func Add() Operation {
	return func(g *graph.ExprGraph, n *graph.Node) (Op, error) {
		return nil, nil
	}

}
