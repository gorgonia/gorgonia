package gorgonia

import (
	"hash"

	"gorgonia.org/gorgonia/internal/datatypes"
)

// Tensor represents values that are acceptable in Gorgonia. At this point, it is implemented by:
// 	- tensor.Tensor
// 	- exprgraph.Node
// 	- *exprgraph.Symbolic
//
// There is an overlap with values.Value. The reason is semantic clarity. Values are Tensors. Tensors are Values.
// There is clearly one issue with this though: *exprgraph.Symbolic.
// *Symbolic is a "fake" tensor, and cannot be lifted as a *dual.Dual.
type Tensor = datatypes.Tensor

type hashWriter interface {
	WriteHash(hash.Hash)
}

type arityer interface {
	Arity() int
}

type Tensors []Tensor
