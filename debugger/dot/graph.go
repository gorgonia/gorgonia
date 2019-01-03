package dot

import (
	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/encoding"
	gonumDot "gonum.org/v1/gonum/graph/encoding/dot"
)

// This structures handles the toplevel graph attributes
type dotGraph struct {
	graph.Directed
	subs []gonumDot.Graph
}

// DOTAttributers to specify the top-level graph attributes for the graphviz generation
func (g dotGraph) DOTAttributers() (graph, node, edge encoding.Attributer) {
	// Create a special attribute "rank" to place the input at the same level in the graph

	graphAttributes := attributer{
		encoding.Attribute{
			Key:   "rankdir",
			Value: "TB",
		},
	}
	nodeAttributes := attributer{
		encoding.Attribute{
			Key:   "style",
			Value: "rounded",
		},
		encoding.Attribute{
			Key:   "fontsize",
			Value: "10",
		},
		encoding.Attribute{
			Key:   "shape",
			Value: "none",
		},
	}
	return graphAttributes, nodeAttributes, attributer{}
}

func (g dotGraph) Structure() []gonumDot.Graph {
	return g.subs
}
