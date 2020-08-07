package main

import "gorgonia.org/gorgonia"

type layerN interface {
	String() string
	Type() string
	ToNode(g *gorgonia.ExprGraph, input ...*gorgonia.Node) (*gorgonia.Node, error)
}
