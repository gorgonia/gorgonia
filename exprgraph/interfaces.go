package exprgraph

import "gorgonia.org/gorgonia/values"

// A Lifter is any type that converts the underlying type of a Tensor.
// A typical use case is to turn a tensor into a dual value.
type Lifter interface {
	Lift(oldTensorType Tensor) (newTensorType Tensor)
}

type ValueNode interface {
	Node
	V() values.V
}

type graphSetter interface {
	SetGraph(g *Graph)
}

type valuelifter interface {
	V() values.V
	prelift() values.V

	setLifted(lifted, original values.V)
}

type valuer[T any] interface {
	Value() T
}
