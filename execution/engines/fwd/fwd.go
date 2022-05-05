// package fwd provides a tensor.Engine that performs forwards mode differentiation.
package fwd

import (
	"gorgonia.org/gorgonia"
	"gorgonia.org/gorgonia/exprgraph"
	"gorgonia.org/gorgonia/values/dual"
	"gorgonia.org/tensor"
)

// Engine is a tensor.Engine that performs forwards mode differentiations.
type Engine struct {
	tensor.StdEng
	g *exprgraph.Graph
}

// New creates a new Engine.
func New(g *exprgraph.Graph) *Engine {
	retVal := &Engine{g: g}
	g.Engine = retVal
	return retVal
}

// Graph returns the embedded graph.
func (e *Engine) Graph() *exprgraph.Graph { return e.g }

// SetGraph sets the graph in the engine to g.
func (e *Engine) SetGraph(g *exprgraph.Graph) { e.g = g }

// Lift implements exprgraph.Lifter. This converts values to dual.Dual.
func (e *Engine) Lift(a gorgonia.Tensor) gorgonia.Tensor {
	switch t := a.(type) {
	case *dual.Dual:
		return a
	case tensor.Tensor:
		return dual.New(t)
	}
	panic("Unreachable")
}
