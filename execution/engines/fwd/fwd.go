// package fwd provides a tensor.Engine that performs forwards mode differentiation.
package fwd

import (
	"gorgonia.org/gorgonia/execution/engines"
	"gorgonia.org/gorgonia/exprgraph"
	"gorgonia.org/gorgonia/internal/datatypes"
	"gorgonia.org/gorgonia/values/dual"
	"gorgonia.org/tensor"
)

// Engine is a Engine that performs forwards mode differentiation
//
// Here the implementation is done by means of implementing MatMul and AddScalar
// Obviously in the real world situation, Add also needs to be implemented, but in this example
// we are not going to call Add, only AddScalar.
type Engine[DT any, T tensor.Basic[DT]] struct {
	engines.StandardEngine[DT, T]
	g *exprgraph.Graph
}

// New creates a new fwd engine.
func New[DT any, T tensor.Basic[DT]]() *Engine[DT, T] {
	return &Engine[DT, T]{
		StandardEngine: nil, // TODO
	}
}

func (e *Engine[DT, T]) BasicEng() tensor.Engine {
	//return &FwdEngine[DT, tensor.Basic[DT]]{StandardEngine: e.StandardEngine.BasicEng().(StandardEngine[DT, tensor.Basic[DT]]), g: e.g}
	return e
}

func (e *Engine[DT, T]) Graph() *exprgraph.Graph { return e.g }

func (e *Engine[DT, T]) SetGraph(g *exprgraph.Graph) { e.g = g }

func (e *Engine[DT, T]) Lift(a datatypes.Tensor) datatypes.Tensor {
	switch t := a.(type) {
	case *dual.Dual[DT, T]:
		return a
	case T:
		return dual.New[DT, T](t)
	}
	panic("Unreachable")
}
