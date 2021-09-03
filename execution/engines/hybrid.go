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
