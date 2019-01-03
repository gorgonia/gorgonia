package dot

import (
	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/encoding"
	gonumDot "gonum.org/v1/gonum/graph/encoding/dot"
	"gonum.org/v1/gonum/graph/simple"
	"gorgonia.org/gorgonia/debugger"
)

var (
	subGraphs = map[debugger.GroupID]DotSubgrapher{
		debugger.ConstantCluster: constantSubGraph{
			DirectedBuilder: simple.NewDirectedGraph(),
			name:            "Constants",
		},
		debugger.InputCluster: inputSubGraph{
			DirectedBuilder: simple.NewDirectedGraph(),
			name:            "Inputs",
		},
		debugger.ExprGraphCluster: exprSubGraph{
			DirectedBuilder: simple.NewDirectedGraph(),
			name:            "ExprGraph",
		},
		debugger.UndefinedCluster: exprSubGraph{
			DirectedBuilder: simple.NewDirectedGraph(),
			name:            "ExprGraph",
		},
	}
)

type attributer []encoding.Attribute

func (a attributer) Attributes() []encoding.Attribute { return a }

func generateDotGraph(g graph.Directed) (graph.Graph, error) {
	dg := simple.NewDirectedGraph()
	graph.Copy(dg, g)
	nodes := g.Nodes()
	for nodes.Next() {
		n := nodes.Node()
		if _, ok := n.(debugger.Grouper); ok {
			group := n.(debugger.Grouper).Group()
			if subgrapher, ok := subGraphs[group]; ok {
				subgrapher.(graph.DirectedBuilder).AddNode(n)
			}
		}
	}
	subs := make([]gonumDot.Graph, 0)
	for _, g := range subGraphs {
		subs = append(subs, g)
	}
	return dotGraph{
		Directed: dg,
		subs:     subs,
	}, nil
}
