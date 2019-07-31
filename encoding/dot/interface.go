package dot

import gonumDot "gonum.org/v1/gonum/graph/encoding/dot"

// Subgrapher is any type that can represent itself as a subgraph
type subgrapher interface {
	gonumDot.Graph
	gonumDot.Attributers
}
