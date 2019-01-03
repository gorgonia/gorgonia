package dot

import (
	"gonum.org/v1/gonum/graph"
	gonumDot "gonum.org/v1/gonum/graph/encoding/dot"
)

// Marshal the graph in a dot (graphviz)
// This methods also generates the subgraphs
func Marshal(g graph.Directed) ([]byte, error) {
	dg, err := generateDotGraph(g)
	if err != nil {
		return nil, err
	}
	return gonumDot.Marshal(dg, "", "", "\t")
}
