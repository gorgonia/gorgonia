package gapi

import (
	"gorgonia.org/gorgonia/internal/datatypes"
	"gorgonia.org/gorgonia/ops"
)

// Queueer is anything that can enqueue an op, inputs and outputs.
type Queueer interface {
	Q(op ops.Op, inputs []datatypes.Tensor, output datatypes.Tensor) error
}
