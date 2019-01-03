package dot

import (
	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/encoding"
)

type exprSubGraph struct {
	name string
	graph.DirectedBuilder
}

func (g exprSubGraph) DOTID() string { return "cluster_" + g.name }

// DOTAttributers to specify the top-level graph attributes for the graphviz generation
func (g exprSubGraph) DOTAttributers() (graph, node, edge encoding.Attributer) {
	// Create a special attribute "rank" to place the input at the same level in the graph

	graphAttributes := attributer{
		encoding.Attribute{
			Key:   "label",
			Value: g.name,
		},
		encoding.Attribute{
			Key:   "color",
			Value: "lightgray",
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
