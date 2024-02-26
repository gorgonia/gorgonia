package engines

import (
	"gorgonia.org/gorgonia/exprgraph"
	"gorgonia.org/gorgonia/internal/datatypes"
	"gorgonia.org/gorgonia/ops"
	"gorgonia.org/tensor"
)

// Hybrid is an engine that contains a *exprgraph.Graph
type Hybrid interface {
	tensor.Engine
	Workhorse() tensor.Engine

	Graph() *exprgraph.Graph
}

// StandardEngine is a set of operations that must be supported by an engine in order to be used by Gorgonia.
type StandardEngine[DT any, T tensor.Basic[DT]] interface {
	tensor.Engine
	tensor.FuncOptHandler[DT]
	tensor.BLA[DT, T]
	tensor.Adder[DT, T]
}

type Queueer[DT any, T tensor.Basic[DT]] interface {
	Q(op ops.Op[DT, T], inputs []datatypes.Tensor, output datatypes.Tensor) error
}
