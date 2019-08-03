package dot

import (
	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/encoding"
	gonumDot "gonum.org/v1/gonum/graph/encoding/dot"
	"gonum.org/v1/gonum/graph/simple"
	"gorgonia.org/gorgonia"
	internalEncoding "gorgonia.org/gorgonia/internal/encoding"
)

var (
	subGraphs = map[internalEncoding.Group]subgrapher{
		internalEncoding.ConstantCluster: constantSubGraph{
			DirectedBuilder: simple.NewDirectedGraph(),
			name:            "Constants",
		},
		internalEncoding.InputCluster: inputSubGraph{
			DirectedBuilder: simple.NewDirectedGraph(),
			name:            "Inputs",
		},
		internalEncoding.ExprGraphCluster: exprSubGraph{
			DirectedBuilder: simple.NewDirectedGraph(),
			name:            "ExprGraph",
		},
		internalEncoding.UndefinedCluster: exprSubGraph{
			DirectedBuilder: simple.NewDirectedGraph(),
			name:            "ExprGraph",
		},
	}
)

type attributer []encoding.Attribute

func (a attributer) Attributes() []encoding.Attribute { return a }

func generateDotGraph(g *gorgonia.ExprGraph) (graph.Graph, error) {
	dg := simple.NewDirectedGraph()
	copyGraph(dg, g)
	nodes := dg.Nodes()

	for nodes.Next() {
		n := nodes.Node()
		if _, ok := n.(internalEncoding.Grouper); ok {
			for _, group := range n.(internalEncoding.Grouper).Groups() {
				if subgrapher, ok := subGraphs[group]; ok {
					subgrapher.(graph.DirectedBuilder).AddNode(n)
				}
			}
		}
	}
	subs := make([]gonumDot.Graph, 0, len(subGraphs))
	for _, g := range subGraphs {
		subs = append(subs, g)
	}
	return dotGraph{
		Directed: dg,
		subs:     subs,
	}, nil
}
