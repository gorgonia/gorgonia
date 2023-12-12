package exprgraph

import (
	"gorgonia.org/gorgonia/ops"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/gorgonia/values/dual"
)

// A Lifter is any type that converts the underlying type of a Tensor.
// A typical use case is to turn a tensor into a dual value.
type Lifter interface {
	Lift(oldTensorType Tensor) (newTensorType Tensor)
}

type ValueNode interface {
	Node
	V() values.V
}

type RxNode interface {
	ValueNode

	AddWaiting()
	Waiting() int
	ZeroWaiting()

	O() ops.Desc
}

type Oper[DT any, T values.Value[DT]] interface {
	Op() ops.Op[DT, T]
}

type graphSetter interface {
	SetGraph(g *Graph)
}

type valuelifter interface {
	V() values.V
	d() dual.V // can be nil
	prelift() values.V

	setLifted(lifted, original values.V)
}

type valuer[T any] interface {
	Value() T
}
