package exprgraph

var (
	_ Nodelike = NodeID(0)
)

// NodeID NodeID represents a Node's ID. It does not actually implement Tensor.
// The `Tensor` interface satisfaction is solely for the purpose of being able to use it to
// retrieve nodes from the graph.
type NodeID int64

// ID returns the ID as an int64. This is used to fulfil gonum.org/gonum/graph.Node interface.
func (n NodeID) ID() int64 { return int64(n) }
