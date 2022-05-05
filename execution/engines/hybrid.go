package engines

import (
	"gorgonia.org/gorgonia/exprgraph"
	"gorgonia.org/tensor"
)

// Hybrid are engines that both implement a Graph and a tensor.Engine.
type Hybrid interface {
	tensor.Engine
	Graph() *exprgraph.Graph
}

// Std is a default Hybrid engine.
type Std struct {
	tensor.StdEng
	g *exprgraph.Graph
}

func NewStd() *Std { return &Std{} }

func (e *Std) Graph() *exprgraph.Graph     { return e.g }
func (e *Std) SetGraph(g *exprgraph.Graph) { e.g = g }
