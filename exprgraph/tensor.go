package exprgraph

import (
	"gonum.org/v1/gonum/graph"
	"gorgonia.org/tensor"
)

type Tensor = tensor.Desc

// Nodelike is anything that looks like a Node.
type Nodelike = graph.Node
