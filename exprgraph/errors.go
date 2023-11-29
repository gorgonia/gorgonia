package exprgraph

import (
	"errors"
	"fmt"
)

// ErrNotFoundInGraph is returned anytime an information extracted from the graph is not found.
var ErrNotFoundInGraph = errors.New("not found in graph")

// CollisionError is returned when a collision is detected during a graph operation.
type CollisionError struct {
	node int64
}

func (err CollisionError) Error() string {
	return fmt.Sprintf("collision on node %v", err.node)
}

func (err CollisionError) Node(g *Graph) Node {
	return g.nodes[err.node]
}
