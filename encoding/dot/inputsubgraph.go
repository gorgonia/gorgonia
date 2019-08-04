package dot

import (
	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/encoding"
	gonumDot "gonum.org/v1/gonum/graph/encoding/dot"
	internalEncoding "gorgonia.org/gorgonia/internal/encoding"
)

type inputSubGraph struct {
	name string
	graph.DirectedBuilder
	subs map[internalEncoding.Group]operatorSubGraph
}

func (g inputSubGraph) DOTID() string { return g.name }

// DOTAttributers to specify the top-level graph attributes for the graphviz generation
func (g inputSubGraph) DOTAttributers() (graph, node, edge encoding.Attributer) {
	// Create a special attribute "rank" to place the input at the same level in the graph

	graphAttributes := attributer{
		encoding.Attribute{
			Key:   "label",
			Value: g.name,
		},
		encoding.Attribute{
			Key:   "rank",
			Value: `"max"`,
		},
	}
	nodeAttributes := attributer{
		encoding.Attribute{
			Key:   "style",
			Value: `"rounded,filled"`,
		},
		encoding.Attribute{
			Key:   "shape",
			Value: "record",
		},
		encoding.Attribute{
			Key:   "fillcolor",
			Value: "yellow",
		},
	}
	return graphAttributes, nodeAttributes, attributer{}
}

// Structure fulfils the dot.Structurer interface.
func (g inputSubGraph) Structure() []gonumDot.Graph {
	output := make([]gonumDot.Graph, 0, len(g.subs))
	for _, subg := range g.subs {
		output = append(output, subg)
	}
	return output
}
