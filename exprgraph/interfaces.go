package exprgraph

// A Lifter is any type that converts the underlying type of a Teosor.
// A typical use case is to turn a tensor into a dual value.
type Lifter interface {
	Lift(oldTensorType Tensor) (newTensorType Tensor)
}

type graphSetter interface {
	SetGraph(g *Graph)
}
