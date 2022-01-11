package exprgraph

import (
	"gonum.org/v1/gonum/graph"
	"gorgonia.org/gorgonia/internal/datatypes"
)

type Tensor = datatypes.Tensor

// Nodelike is anything that looks like a Node.
type Nodelike = graph.Node
