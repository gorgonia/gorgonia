package exprgraph

import "gorgonia.org/gorgonia"

// A Lifter is any type that converts the underlying type of a Teosor.
// A typical use case is to turn a tensor into a dual value.
type Lifter interface {
	Lift(oldTensorType gorgonia.Tensor) (newTensorType gorgonia.Tensor)
}

type graphSetter interface {
	SetGraph(g *Graph)
}
