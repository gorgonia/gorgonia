package dot

import (
	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/encoding"
	gonumDot "gonum.org/v1/gonum/graph/encoding/dot"
	"gonum.org/v1/gonum/graph/simple"
	"gorgonia.org/gorgonia"
	gencoding "gorgonia.org/gorgonia/encoding"
)

var (
	subGraphs = map[gencoding.GroupID]subgrapher{
		gencoding.ConstantCluster: constantSubGraph{
			DirectedBuilder: simple.NewDirectedGraph(),
			name:            "Constants",
		},
		gencoding.InputCluster: inputSubGraph{
			DirectedBuilder: simple.NewDirectedGraph(),
			name:            "Inputs",
		},
		gencoding.ExprGraphCluster: exprSubGraph{
			DirectedBuilder: simple.NewDirectedGraph(),
			name:            "ExprGraph",
		},
		gencoding.UndefinedCluster: exprSubGraph{
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
		if _, ok := n.(gencoding.Grouper); ok {
			group := n.(gencoding.Grouper).Group()
			if subgrapher, ok := subGraphs[group]; ok {
				n := &node{
					n: n.(*gorgonia.Node),
				}
				subgrapher.(graph.DirectedBuilder).AddNode(n)
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
