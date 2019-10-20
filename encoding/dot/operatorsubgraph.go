package dot

import (
	"strconv"

	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/encoding"
	gonumDot "gonum.org/v1/gonum/graph/encoding/dot"
	internalEncoding "gorgonia.org/gorgonia/internal/encoding"
)

type operatorSubGraph struct {
	name string
	id   int
	graph.DirectedBuilder
	subs map[internalEncoding.Group]operatorSubGraph
}

func (g operatorSubGraph) DOTID() string {
	return "cluster_" + g.name + strconv.Itoa(g.id)
}

// DOTAttributers to specify the top-level graph attributes for the graphviz generation
func (g operatorSubGraph) DOTAttributers() (graph, node, edge encoding.Attributer) {
	// Create a special attribute "rank" to place the input at the same level in the graph

	graphAttributes := attributer{
		encoding.Attribute{
			Key:   "label",
			Value: g.name,
		},
		encoding.Attribute{
			Key:   "color",
			Value: "coral",
		},
		encoding.Attribute{
			Key:   "style",
			Value: "filled",
		},
		encoding.Attribute{
			Key:   "nodeset",
			Value: "0.5",
		},
		encoding.Attribute{
			Key:   "ranksep",
			Value: `"1.2 equally"`,
		},
	}
	nodeAttributes := attributer{
		encoding.Attribute{
			Key:   "style",
			Value: `"rounded,filled"`,
		},
		encoding.Attribute{
			Key:   "fillcolor",
			Value: "white",
		},
		encoding.Attribute{
			Key:   "shape",
			Value: "Mrecord",
		},
	}
	return graphAttributes, nodeAttributes, attributer{}
}

// Structure fulfils the dot.Structurer interface.
func (g operatorSubGraph) Structure() []gonumDot.Graph {
	output := make([]gonumDot.Graph, 0, len(g.subs))
	for _, subg := range g.subs {
		output = append(output, subg)
	}
	return output
}
